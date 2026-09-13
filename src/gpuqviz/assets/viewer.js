/* gpuqviz 交互式播放器：three.js 布洛赫球回放 + 播放控制 + 实时量子状态面板。
 * 插值算法与 Python 端 gpuqviz/interpolate.py 保持同构。
 * 黄金用例（与 tests/test_interpolate_js.py 对齐）：
 *   slerp([0,0,1],[0,0,-1], 0.5) -> [0,0,0]（对跖点：绕 +x 轴旋转 90° -> [1,0,0]？）
 *   本实现选择绕轴 u = normalize(cross([0,0,1],[1,0,0])) = [0,1,0]，
 *   w=0.5 时结果 = [0,1,0]。
 */
window.GPUQVIZ_VIEWER = (function () {
  "use strict";

  // ---------- 数值工具（与 interpolate.py 同构） ----------
  function slerp(a, b, w) {
    let dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    dot = Math.max(-1, Math.min(1, dot));
    const theta = Math.acos(dot);
    const sinT = Math.sin(theta);
    if (sinT < 1e-8) {
      if (dot < -1 + 1e-6) {
        // 对跖点：绕与 a 垂直的轴匀速旋转 180°（Python 端同款回退）
        const ref = Math.abs(a[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
        let axis = [
          ref[0] - (ref[0]*a[0] + ref[1]*a[1] + ref[2]*a[2]) * a[0],
          ref[1] - (ref[0]*a[0] + ref[1]*a[1] + ref[2]*a[2]) * a[1],
          ref[2] - (ref[0]*a[0] + ref[1]*a[1] + ref[2]*a[2]) * a[2],
        ];
        const n = Math.hypot(...axis) || 1;
        axis = [axis[0]/n, axis[1]/n, axis[2]/n];
        const om = Math.PI * w;
        return [
          Math.cos(om)*a[0] + Math.sin(om)*axis[0],
          Math.cos(om)*a[1] + Math.sin(om)*axis[1],
          Math.cos(om)*a[2] + Math.sin(om)*axis[2],
        ];
      }
      return [(1-w)*a[0] + w*b[0], (1-w)*a[1] + w*b[1], (1-w)*a[2] + w*b[2]];
    }
    const s0 = Math.sin((1-w)*theta) / sinT;
    const s1 = Math.sin(w*theta) / sinT;
    return [s0*a[0] + s1*b[0], s0*a[1] + s1*b[1], s0*a[2] + s1*b[2]];
  }

  function lerpState(re0, im0, re1, im1, w) {
    const n = re0.length;
    const re = new Array(n), im = new Array(n);
    let norm = 0;
    for (let i = 0; i < n; i++) {
      re[i] = (1-w)*re0[i] + w*re1[i];
      im[i] = (1-w)*im0[i] + w*im1[i];
      norm += re[i]*re[i] + im[i]*im[i];
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < n; i++) { re[i] /= norm; im[i] /= norm; }
    return { re, im };
  }

  // ---------- 播放器主体 ----------
  function Init(viewportId) {
    const D = window.GPUQVIZ_DATA;
    const meta = D.meta;
    const nQ = meta.n_qubits, nKeys = meta.n_keys;
    const dim = 1 << nQ;

    // --- DOM 引用 ---
    const vp = document.getElementById(viewportId);
    const playBtn = document.getElementById("playBtn");
    const speedSel = document.getElementById("speed");
    const scrub = document.getElementById("scrub");
    const timeLabel = document.getElementById("timeLabel");
    const probRows = document.getElementById("probRows");
    const blochList = document.getElementById("blochList");

    // --- 状态面板构建 ---
    const rowEls = [];
    for (let i = 0; i < dim; i++) {
      // qiskit 小端序标签：|q_{n-1} ... q_0>
      let label = "";
      for (let b = nQ - 1; b >= 0; b--) label += ((i >> b) & 1).toString();
      const row = document.createElement("div");
      row.className = "probRow";
      row.innerHTML =
        '<span class="label">|' + label + "&#1022;</span>" +
        '<div class="barTrack"><div class="bar"></div></div>' +
        '<span class="nums"></span>';
      probRows.appendChild(row);
      rowEls.push({ bar: row.querySelector(".bar"), nums: row.querySelector(".nums") });
    }

    // --- three.js 场景 ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e14);
    const camera = new THREE.PerspectiveCamera(45, 2, 0.1, 200);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    vp.appendChild(renderer.domElement);

    // 每球一组
    const spacing = 3.0, radius = 1.2;
    const groups = [];
    for (let i = 0; i < nQ; i++) {
      const g = new THREE.Group();
      const cx = (i - (nQ - 1) / 2) * spacing;
      g.position.set(cx, 0, 0);

      // 半透明球壳
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 48, 32),
        new THREE.MeshPhongMaterial({
          color: 0x8ca6d9, transparent: true, opacity: 0.16,
          side: THREE.DoubleSide, depthWrite: false,
        }));
      g.add(sphere);

      // 赤道环
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius, 0.006 * radius, 8, 96),
        new THREE.MeshBasicMaterial({ color: 0x4d5c74 }));
      ring.rotation.x = Math.PI / 2;
      g.add(ring);

      // XYZ 轴 + 极点标注
      const axisMat = new THREE.LineBasicMaterial({ color: 0x737e92 });
      const mkAxis = (dir) => {
        const geo = new THREE.BufferGeometry().setFromPoints([
          dir.clone().multiplyScalar(-radius * 1.15),
          dir.clone().multiplyScalar(radius * 1.15),
        ]);
        g.add(new THREE.Line(geo, axisMat));
      };
      mkAxis(new THREE.Vector3(1, 0, 0));
      mkAxis(new THREE.Vector3(0, 1, 0));
      mkAxis(new THREE.Vector3(0, 0, 1));

      // 态矢量：箭头线 + 端点小球
      const arrowGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, radius)]);
      const arrow = new THREE.Line(
        arrowGeo, new THREE.LineBasicMaterial({ color: 0x4dc5f2 }));
      const tip = new THREE.Mesh(
        new THREE.SphereGeometry(radius * 0.07, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0x4dc5f2 }));
      g.add(arrow); g.add(tip);

      groups.push({ group: g, arrow, tip });
      scene.add(g);
    }

    // 光照
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(4, 6, 8);
    scene.add(dir);

    // --- 简易轨道相机（拖拽旋转 / 滚轮缩放） ---
    let orbit = { az: 0.6, el: 0.4, dist: Math.max(5, spacing * nQ * 1.35) };
    function updateCamera() {
      camera.position.set(
        orbit.dist * Math.cos(orbit.el) * Math.sin(orbit.az),
        -orbit.dist * Math.cos(orbit.el) * Math.cos(orbit.az),
        orbit.dist * Math.sin(orbit.el));
      camera.lookAt(0, 0, 0);
      camera.aspect = vp.clientWidth / Math.max(vp.clientHeight, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(vp.clientWidth, vp.clientHeight);
    }
    let dragging = false, lastX = 0, lastY = 0;
    vp.addEventListener("pointerdown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
    window.addEventListener("pointerup", () => { dragging = false; });
    window.addEventListener("pointermove", e => {
      if (!dragging) return;
      orbit.az += (e.clientX - lastX) * 0.006;
      orbit.el = Math.max(-1.4, Math.min(1.4, orbit.el + (e.clientY - lastY) * 0.006));
      lastX = e.clientX; lastY = e.clientY;
    });
    vp.addEventListener("wheel", e => {
      e.preventDefault();
      orbit.dist = Math.max(2, Math.min(60, orbit.dist * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });

    // --- 播放状态 ---
    let t = 0, playing = true, speed = 1, lastTs = null;
    const total = meta.duration;
    playBtn.textContent = "⏸";  // 初始即播放，按钮符号与状态一致

    function keyFramesAt(t) {
      const pos = Math.min(Math.max(t / total, 0), 0.999999) * (nKeys - 1);
      const i = Math.floor(pos), w = pos - i, j = Math.min(i + 1, nKeys - 1);
      return { i, j, w };
    }

    function updatePanel(state) {
      for (let i = 0; i < dim; i++) {
        const p = state.re[i] * state.re[i] + state.im[i] * state.im[i];
        const amp = Math.sqrt(p);
        const ph = Math.atan2(state.im[i], state.re[i]) * 180 / Math.PI;
        rowEls[i].bar.style.width = (p * 100).toFixed(2) + "%";
        rowEls[i].nums.textContent =
          p.toFixed(3) + "  amp=" + amp.toFixed(3) + "  ∠" + ph.toFixed(1) + "°";
      }
      const lines = [];
      for (let q = 0; q < nQ; q++) {
        const v = state.bloch[q];
        lines.push("q" + q + ": (" + v[0].toFixed(3) + ", " + v[1].toFixed(3) + ", " + v[2].toFixed(3) + ")");
      }
      blochList.textContent = lines.join("\n");
    }

    function renderFrame() {
      const { i, w } = keyFramesAt(t);
      // 态矢量插值 + renormalize
      const st = lerpState(D.states_re[i], D.states_im[i], D.states_re[i + 1], D.states_im[i + 1], w);
      // Bloch slerp
      const bloch = [];
      for (let q = 0; q < nQ; q++) {
        const a = D.bloch[i][q], b = D.bloch[i + 1][q];
        bloch.push(slerp(a, b, w));
        const v = bloch[q];
        const norm = Math.hypot(v[0], v[1], v[2]);
        const x = norm > 1e-9 ? v[0] / norm : 0;
        const y = norm > 1e-9 ? v[1] / norm : 0;
        const z = norm > 1e-9 ? v[2] / norm : 0;
        const grp = groups[q];
        const pos = grp.arrow.geometry.attributes.position;
        pos.setXYZ(1, x * radius, y * radius, z * radius);
        pos.needsUpdate = true;
        grp.tip.position.set(x * radius, y * radius, z * radius);
      }
      st.bloch = bloch;
      updatePanel(st);

      // 控件同步
      scrub.value = String(Math.round((t / total) * 1000));
      timeLabel.textContent = t.toFixed(2) + " / " + total.toFixed(2) + " s";
    }

    function tick(ts) {
      if (lastTs !== null && playing) {
        t += ((ts - lastTs) / 1000) * speed;
        if (t >= total) t = 0;  // 循环播放
        if (t < 0) t = 0;
      }
      lastTs = ts;
      renderFrame();
      updateCamera();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }

    // --- 控件事件 ---
    function setPlaying(p) {
      playing = p;
      playBtn.textContent = p ? "⏸" : "⏵";
    }
    playBtn.addEventListener("click", () => setPlaying(!playing));
    speedSel.addEventListener("change", () => { speed = parseFloat(speedSel.value); });
    scrub.addEventListener("input", () => {
      t = (parseInt(scrub.value, 10) / 1000) * total;
      renderFrame();
    });
    window.addEventListener("keydown", e => {
      if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
      else if (e.code === "ArrowRight") { playing = false; playBtn.textContent = "⏵"; t = Math.min(total - 1e-6, t + 1 / meta.fps); renderFrame(); }
      else if (e.code === "ArrowLeft") { playing = false; playBtn.textContent = "⏵"; t = Math.max(0, t - 1 / meta.fps); renderFrame(); }
    });

    new ResizeObserver(updateCamera).observe(vp);
    updateCamera();
    requestAnimationFrame(tick);
  }

  return { Init, slerp, lerpState };
})();
