/* gpuqviz 浏览器端量子态模拟器。
 *
 * 纯 JS 实现态矢量演化 + Bloch 向量计算 + payload 构建。
 * 输出的 window.GPUQVIZ_DATA 与 Python 端 export_html.build_payload 格式完全一致，
 * 可直接喂给 viewer.js 的 Init() 获得相同的 3D Bloch 球 + 电路图 + 状态面板体验。
 *
 * 支持 2-8 qubit（n=8 时态矢量 256 个复数，运算瞬时完成）。
 * 门集：H X Y Z S T RX RY RZ CX CZ SWAP
 */

window.GPUQVIZ_SIM = (function () {
  "use strict";

  var SQRT2_INV = 1 / Math.sqrt(2);
  var PI = Math.PI;

  // ─── 2×2 复数门矩阵：[[re0, im0, re1, im1], [re2, im2, re3, im3]] ───
  // 每个元素为 [re, im] 二元组

  function c(re, im) { return [re, im || 0]; }
  function cmul(a, b) { return [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]]; }
  function cconj(a) { return [a[0], -a[1]]; }

  // 常用 2×2 门矩阵（行优先，每元素 [re, im]）
  var GATES = {
    H: [[c(SQRT2_INV,0), c(SQRT2_INV,0)],
        [c(SQRT2_INV,0), c(-SQRT2_INV,0)]],
    X: [[c(0,0), c(1,0)],
        [c(1,0), c(0,0)]],
    Y: [[c(0,0), c(0,-1)],
        [c(0,1), c(0,0)]],
    Z: [[c(1,0), c(0,0)],
        [c(0,0), c(-1,0)]],
    S: [[c(1,0), c(0,0)],
        [c(0,0), c(0,1)]],
    T: [[c(1,0), c(0,0)],
        [c(0,0), c(SQRT2_INV,SQRT2_INV)]]  // e^{iπ/4} = (1+i)/√2
  };

  function rxM(theta) {
    var c1 = Math.cos(theta/2), s = Math.sin(theta/2);
    return [[c(c1,0), c(0,-s)],
            [c(0,-s), c(c1,0)]];
  }
  function ryM(theta) {
    var c1 = Math.cos(theta/2), s = Math.sin(theta/2);
    return [[c(c1,0), c(-s,0)],
            [c(s,0),  c(c1,0)]];
  }
  function rzM(theta) {
    var ePos = [Math.cos(theta/2), Math.sin(theta/2)];
    var eNeg = [Math.cos(theta/2), -Math.sin(theta/2)];
    return [[eNeg, c(0,0)],
            [c(0,0), ePos]];
  }

  // 根据门名 + 参数获取 2×2 矩阵
  function gateMatrix(name, params) {
    name = name.toUpperCase();
    if (GATES[name]) return GATES[name];
    if (name === "RX") return rxM(params[0] || PI/2);
    if (name === "RY") return ryM(params[0] || PI/2);
    if (name === "RZ") return rzM(params[0] || PI/2);
    return null;  // CX/CZ/SWAP 由专用函数处理
  }

  // ─── 态矢量操作 ───
  // state = { re: Float64Array(2^n), im: Float64Array(2^n) }

  function newState(n) {
    var dim = 1 << n;
    var st = { re: new Float64Array(dim), im: new Float64Array(dim) };
    st.re[0] = 1;  // |0...0⟩
    return st;
  }

  function cloneState(st) {
    return { re: new Float64Array(st.re), im: new Float64Array(st.im) };
  }

  // 单量子门应用到 qubit q
  function applySingle(st, M, q, n) {
    var re = st.re, im = st.im;
    var bit = 1 << q;
    for (var i = 0; i < re.length; i++) {
      if (i & bit) continue;  // 只处理 bit=0 的项，配对项是 i|bit
      var j = i | bit;
      // [a'_i, a'_j] = M * [a_i, a_j]
      var aRe = re[i], aIm = im[i];
      var bRe = re[j], bIm = im[j];
      // M = [[m00, m01], [m10, m11]]
      // a'_i   = m00*a_i + m01*a_j
      // a'_j   = m10*a_i + m11*a_j
      var m00 = M[0][0], m01 = M[0][1], m10 = M[1][0], m11 = M[1][1];
      // 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
      var niRe = m00[0]*aRe - m00[1]*aIm + m01[0]*bRe - m01[1]*bIm;
      var niIm = m00[0]*aIm + m00[1]*aRe + m01[0]*bIm + m01[1]*bRe;
      var njRe = m10[0]*aRe - m10[1]*aIm + m11[0]*bRe - m11[1]*bIm;
      var njIm = m10[0]*aIm + m10[1]*aRe + m11[0]*bIm + m11[1]*bRe;
      re[i] = niRe; im[i] = niIm;
      re[j] = njRe; im[j] = njIm;
    }
  }

  // 受控门：control qubit 控制 target qubit 上的 2×2 矩阵
  function applyControlled(st, M, ctrl, tgt, n) {
    var re = st.re, im = st.im;
    var cBit = 1 << ctrl, tBit = 1 << tgt;
    for (var i = 0; i < re.length; i++) {
      // 需要控制位 = 1 且目标位 = 0
      if (!(i & cBit)) continue;
      if (i & tBit) continue;
      var j = i | tBit;
      var aRe = re[i], aIm = im[i];
      var bRe = re[j], bIm = im[j];
      var m00 = M[0][0], m01 = M[0][1], m10 = M[1][0], m11 = M[1][1];
      var niRe = m00[0]*aRe - m00[1]*aIm + m01[0]*bRe - m01[1]*bIm;
      var niIm = m00[0]*aIm + m00[1]*aRe + m01[0]*bIm + m01[1]*bRe;
      var njRe = m10[0]*aRe - m10[1]*aIm + m11[0]*bRe - m11[1]*bIm;
      var njIm = m10[0]*aIm + m10[1]*aRe + m11[0]*bIm + m11[1]*bRe;
      re[i] = niRe; im[i] = niIm;
      re[j] = njRe; im[j] = njIm;
    }
  }

  // SWAP(q1, q2)
  function applySwap(st, q1, q2, n) {
    if (q1 === q2) return;
    var re = st.re, im = st.im;
    var b1 = 1 << q1, b2 = 1 << q2;
    for (var i = 0; i < re.length; i++) {
      // 只处理 bit q1=1, q2=0 的项（配对项是 i ^ b1 ^ b2，bit q1=0, q2=1）
      if (!(i & b1)) continue;
      if (i & b2) continue;
      var j = i ^ b1 ^ b2;
      var tRe = re[i], tIm = im[i];
      re[i] = re[j]; im[i] = im[j];
      re[j] = tRe; im[j] = tIm;
    }
  }

  // ─── Bloch 向量 ───
  // 对 qubit q 求约化密度矩阵的 Pauli 期望值
  // x = 2 Re(ρ_01), y = 2 Im(ρ_01), z = ρ_00 - ρ_11
  // ρ_01 = Σ_{i: bit q=0} a_i * conj(a_{i|(1<<q)})

  function blochVector(st, q, n) {
    var re = st.re, im = st.im;
    var bit = 1 << q;
    var rho01_re = 0, rho01_im = 0;
    var rho00 = 0, rho11 = 0;

    for (var i = 0; i < re.length; i++) {
      var p = re[i]*re[i] + im[i]*im[i];  // |a_i|²
      if (i & bit) {
        rho11 += p;
      } else {
        rho00 += p;
        var j = i | bit;
        // ρ_01 += a_i * conj(a_j) = a_i * (re_j - i*im_j)
        // = (re_i + i*im_i)(re_j - i*im_j)
        // = (re_i*re_j + im_i*im_j) + i*(im_i*re_j - re_i*im_j)
        rho01_re += re[i]*re[j] + im[i]*im[j];
        rho01_im += im[i]*re[j] - re[i]*im[j];
      }
    }

    return [2*rho01_re, 2*rho01_im, rho00 - rho11];
  }

  // ─── 门标签生成（与 Python _gate_label 一致）───

  function gateLabel(name, params) {
    name = name.toUpperCase();
    if (params && params.length > 0) {
      var parts = [];
      for (var i = 0; i < params.length; i++) {
        var p = params[i];
        if (Math.abs(p) < 1e-10) parts.push("0");
        else if (Math.abs(p - PI) < 1e-10) parts.push("π");
        else if (Math.abs(p + PI) < 1e-10) parts.push("-π");
        else if (Math.abs(p - PI/2) < 1e-10) parts.push("π/2");
        else if (Math.abs(p + PI/2) < 1e-10) parts.push("-π/2");
        else if (Math.abs(p - 2*PI) < 1e-10) parts.push("2π");
        else if (Math.abs(p * 4 / PI - Math.round(p * 4 / PI)) < 1e-6) {
          var num = Math.round(p * 4 / PI);
          if (num === 1) parts.push("π/4");
          else if (num === -1) parts.push("-π/4");
          else parts.push(num + "π/4");
        } else {
          parts.push(p.toFixed(2));
        }
      }
      return name + "(" + parts.join(",") + ")";
    }
    // CX/CZ 的标签
    if (name === "CX" || name === "CNOT") return "CX";
    return name;
  }

  // ─── 构建完整 payload（与 export_html.build_payload 同构）───

  function buildPayload(circuit, n, fps, duration, title) {
    var nOps = circuit.length;
    var nKeys = nOps + 1;  // 初始态 + 每个门后的态

    // 逐门演化，保存关键帧
    var keyStates = [];
    var st = newState(n);
    keyStates.push(cloneState(st));

    for (var g = 0; g < nOps; g++) {
      applyGate(st, circuit[g], n);
      keyStates.push(cloneState(st));
    }

    // 构建 states_re / states_im / bloch
    var dim = 1 << n;
    var states_re = [];
    var states_im = [];
    var bloch = [];

    for (var k = 0; k < nKeys; k++) {
      var ks = keyStates[k];
      // 转为普通数组（viewer.js 用下标访问）
      var sr = new Array(dim);
      var si = new Array(dim);
      for (var i = 0; i < dim; i++) {
        sr[i] = ks.re[i];
        si[i] = ks.im[i];
      }
      states_re.push(sr);
      states_im.push(si);

      // 每个关键帧所有 qubit 的 Bloch 向量
      var frameBloch = [];
      for (var q = 0; q < n; q++) {
        frameBloch.push(blochVector(ks, q, n));
      }
      bloch.push(frameBloch);
    }

    // 构建 circuit info
    var gates = [];
    for (var col = 0; col < nOps; col++) {
      var gate = circuit[col];
      gates.push({
        name: gate.name.toUpperCase(),
        targets: gate.targets.slice(),
        controls: (gate.controls || []).slice(),
        params: (gate.params || []).slice(),
        label: gateLabel(gate.name, gate.params),
        col: col
      });
    }

    // active_gates: 每个关键帧对应的活跃门索引
    var active_gates = [];
    for (var k2 = 0; k2 < nKeys; k2++) {
      if (nKeys <= 1) {
        active_gates.push(0);
      } else {
        var idx = Math.min(Math.floor(k2 / (nKeys - 1) * nOps), nOps - 1);
        active_gates.push(idx);
      }
    }

    // gate_times: 每个门的跳转时间
    var gate_times = [];
    if (nOps === 1) {
      gate_times = [0];
    } else {
      for (var j = 0; j < nOps; j++) {
        gate_times.push(j / (nOps - 1) * duration);
      }
      gate_times[0] = 0;
      if (gate_times[nOps - 1] > duration) gate_times[nOps - 1] = duration;
    }

    return {
      meta: {
        title: title || "量子电路模拟",
        fps: fps || 30,
        duration: duration || 4,
        n_qubits: n,
        n_keys: nKeys,
        colormap: "viridis",
        generated_by: "gpuqviz-playground"
      },
      states_re: states_re,
      states_im: states_im,
      bloch: bloch,
      circuit: {
        n_qubits: n,
        n_ops: nOps,
        gates: gates,
        active_gates: active_gates,
        gate_times: gate_times
      }
    };
  }

  // ─── 应用一个门到态矢量 ───

  function applyGate(st, gate, n) {
    var name = gate.name.toUpperCase();
    var targets = gate.targets;
    var controls = gate.controls || [];
    var params = gate.params || [];

    if (name === "SWAP") {
      applySwap(st, targets[0], targets[1], n);
      return;
    }

    if (controls.length > 0) {
      // 受控门
      var M = gateMatrix(name, params);
      if (!M) {
        // CX/CZ 的目标门矩阵
        if (name === "CX" || name === "CNOT") M = GATES.X;
        else if (name === "CZ") M = GATES.Z;
        else return;  // 未知受控门
      }
      applyControlled(st, M, controls[0], targets[0], n);
      return;
    }

    // 单量子门
    var M2 = gateMatrix(name, params);
    if (M2) {
      applySingle(st, M2, targets[0], n);
    }
  }

  // ─── P4.2：密度矩阵噪声模拟（n ≤ 4，dim ≤ 16）───
  // ρ = { dim, re: Float64Array(dim*dim), im: Float64Array(dim*dim) }（行主序）

  function newDensity(n) {
    var dim = 1 << n;
    var rho = { dim: dim, re: new Float64Array(dim * dim), im: new Float64Array(dim * dim) };
    rho.re[0] = 1;  // |0...0⟩⟨0...0|
    return rho;
  }

  function cloneDensity(rho) {
    return { dim: rho.dim, re: new Float64Array(rho.re), im: new Float64Array(rho.im) };
  }

  // 2×2 块变换：ρ'[x⊕a, y⊕c] = Σ M[a,p] ρ[x⊕p, y⊕r] M*[c,r]（单 qubit 门 on q）
  function applySingleDensity(rho, M, q) {
    var dim = rho.dim, re = rho.re, im = rho.im;
    var bit = 1 << q;
    var nre = new Float64Array(re), nim = new Float64Array(im);
    for (var x = 0; x < dim; x++) {
      if (x & bit) continue;
      for (var y = 0; y < dim; y++) {
        if (y & bit) continue;
        var i0 = x, i1 = x | bit, j0 = y, j1 = y | bit;
        for (var a = 0; a < 2; a++) {
          for (var c = 0; c < 2; c++) {
            var oRe = 0, oIm = 0;
            for (var pp = 0; pp < 2; pp++) {
              for (var r = 0; r < 2; r++) {
                var ia = (a ? i1 : i0), ip = (pp ? i1 : i0);
                var jc = (c ? j1 : j0), jr = (r ? j1 : j0);
                var bRe = re[ip * dim + jr], bIm = im[ip * dim + jr];
                var m1re = M[a][pp][0], m1im = M[a][pp][1];
                var m2re = M[c][r][0], m2im = -M[c][r][1];  // 共轭
                var tRe = m1re * bRe - m1im * bIm, tIm = m1re * bIm + m1im * bRe;
                oRe += tRe * m2re - tIm * m2im;
                oIm += tRe * m2im + tIm * m2re;
              }
            }
            var oi = ia * dim + jc;
            nre[oi] = oRe; nim[oi] = oIm;
          }
        }
      }
    }
    rho.re = nre; rho.im = nim;
  }

  function applyControlledDensity(rho, M, ctrl, tgt) {
    // 控制位=1 的 (row,col) 块作用 M（target 为块内位），控制位=0 不变
    var dim = rho.dim, re = rho.re, im = rho.im;
    var cBit = 1 << ctrl, tBit = 1 << tgt;
    var nre = new Float64Array(re), nim = new Float64Array(im);
    for (var x = 0; x < dim; x++) {
      if ((x & cBit) === 0) continue;
      for (var y = 0; y < dim; y++) {
        if ((y & cBit) === 0) continue;
        if ((x & tBit) === (y & tBit)) continue;  // target 位相同的块不受此门影响
        var i0 = x & ~tBit, i1 = i0 | tBit, j0 = y & ~tBit, j1 = j0 | tBit;
        for (var a = 0; a < 2; a++) {
          for (var c = 0; c < 2; c++) {
            var oRe = 0, oIm = 0;
            for (var pp = 0; pp < 2; pp++) {
              for (var r = 0; r < 2; r++) {
                var ia = (a ? i1 : i0), ip = (pp ? i1 : i0);
                var jc = (c ? j1 : j0), jr = (r ? j1 : j0);
                var bRe = re[ip * dim + jr], bIm = im[ip * dim + jr];
                var m1re = M[a][pp][0], m1im = M[a][pp][1];
                var m2re = M[c][r][0], m2im = -M[c][r][1];
                var tRe = m1re * bRe - m1im * bIm, tIm = m1re * bIm + m1im * bRe;
                oRe += tRe * m2re - tIm * m2im;
                oIm += tRe * m2im + tIm * m2re;
              }
            }
            var oi = ia * dim + jc;
            nre[oi] = oRe; nim[oi] = oIm;
          }
        }
      }
    }
    rho.re = nre; rho.im = nim;
  }

  function applySwapDensity(rho, q1, q2) {
    var dim = rho.dim, re = rho.re, im = rho.im;
    var nre = new Float64Array(re), nim = new Float64Array(im);
    var b1 = 1 << q1, b2 = 1 << q2;
    function sw(i) {
      return (i & ~b1 & ~b2) | ((i & b1) ? b2 : 0) | ((i & b2) ? b1 : 0);
    }
    for (var i = 0; i < dim; i++) {
      var j = sw(i);
      for (var c = 0; c < dim; c++) {
        var d = sw(c);
        nre[i * dim + c] = re[j * dim + d];
        nim[i * dim + c] = im[j * dim + d];
      }
    }
    rho.re = nre; rho.im = nim;
  }

  // Kraus 通道（单 qubit）：ρ' = Σ_k K ρ K†
  function applyKraus1q(rho, kraus, q) {
    var acc = null;
    for (var k = 0; k < kraus.length; k++) {
      var tmp = cloneDensity(rho);
      applySingleDensity(tmp, kraus[k], q);
      if (acc === null) { acc = tmp; }
      else {
        for (var i = 0; i < acc.re.length; i++) {
          acc.re[i] += tmp.re[i]; acc.im[i] += tmp.im[i];
        }
      }
    }
    rho.re = acc.re; rho.im = acc.im;
  }

  // depolarizing(λ)：E(ρ) = (1-λ)ρ + λ I/2（qiskit 同约定）
  function depolarizingKraus(lam) {
    var s0 = Math.sqrt(Math.max(1 - 3 * lam / 4, 0)), s1 = Math.sqrt(lam / 4);
    // 每个条目 = [re, im] 对（统一复数表示，禁止裸数字）
    return [
      [[[s0, 0], [0, 0]], [[0, 0], [s0, 0]]],                     // √(1-3λ/4)·I
      [[[0, 0], [s1, 0]], [[s1, 0], [0, 0]]],                     // √(λ/4)·X
      [[[0, 0], [0, -s1]], [[0, s1], [0, 0]]],                    // √(λ/4)·Y
      [[[s1, 0], [0, 0]], [[0, 0], [-s1, 0]]],                    // √(λ/4)·Z
    ];
  }

  // dephasing(γ)：相位阻尼，off-diag × √(1-γ)
  function dephasingKraus(gamma) {
    var s0 = Math.sqrt(1 - gamma);
    return [
      [[[1, 0], [0, 0]], [[0, 0], [s0, 0]]],
      [[[0, 0], [0, 0]], [[0, 0], [Math.sqrt(gamma), 0]]],
    ];
  }

  // 密度矩阵的 Bloch 向量（对角 + 配对非对角）
  function blochVectorDensity(rho, q) {
    var dim = rho.dim, re = rho.re, im = rho.im;
    var bit = 1 << q;
    var r00 = 0, r11 = 0, r01re = 0, r01im = 0;
    for (var i = 0; i < dim; i++) {
      if (i & bit) continue;
      var j = i | bit;
      r00 += re[i * dim + i];
      r11 += re[j * dim + j];
      r01re += re[i * dim + j];
      r01im += im[i * dim + j];
    }
    return [2 * r01re, 2 * r01im, r00 - r11];
  }

  function densityDiag(rho) {
    var dim = rho.dim, out = new Array(dim);
    for (var i = 0; i < dim; i++) {
      out[i] = Math.max(rho.re[i * dim + i], 0);
    }
    return out;
  }

  function applyGateDensity(rho, gate, n) {
    var name = gate.name.toUpperCase();
    if (name === "BARRIER") return;
    var targets = gate.targets || [], controls = gate.controls || [];
    if (name === "SWAP") { applySwapDensity(rho, targets[0], targets[1]); return; }
    var M = gateMatrix(name, gate.params || []);
    if (controls.length > 0) {
      if (!M) {
        if (name === "CX" || name === "CNOT") M = GATES.X;
        else if (name === "CZ") M = GATES.Z;
        else throw new Error("density mode: unsupported controlled gate " + name);
      }
      applyControlledDensity(rho, M, controls[0], targets[0]);
      return;
    }
    if (!M) throw new Error("density mode: unsupported gate " + name);
    applySingleDensity(rho, M, targets[0]);
  }

  // 含噪 payload：states 置 null（viewer 降级路径），probs/bloch 来自 ρ
  function buildPayloadNoisy(circuit, n, fps, duration, title, noise) {
    if (n > 4) return null;  // 噪声模拟仅支持 n ≤ 4
    var lam = (noise && noise.depolarizing) || 0;
    var gam = (noise && noise.dephasing) || 0;
    var nOps = circuit.length;
    var nKeys = nOps + 1;

    var rho = newDensity(n);
    var keyRhos = [cloneDensity(rho)];
    for (var g = 0; g < nOps; g++) {
      applyGateDensity(rho, circuit[g], n);
      var qs = (circuit[g].targets || []).concat(circuit[g].controls || []);
      for (var qi = 0; qi < qs.length; qi++) {
        if (lam > 0) applyKraus1q(rho, depolarizingKraus(lam), qs[qi]);
        if (gam > 0) applyKraus1q(rho, dephasingKraus(gam), qs[qi]);
      }
      keyRhos.push(cloneDensity(rho));
    }

    var bloch = [], probs = [];
    for (var k = 0; k < nKeys; k++) {
      var fr = [];
      for (var q2 = 0; q2 < n; q2++) fr.push(blochVectorDensity(keyRhos[k], q2));
      bloch.push(fr);
      probs.push(densityDiag(keyRhos[k]));
    }

    var gates = [];
    for (var col = 0; col < nOps; col++) {
      var gate = circuit[col];
      gates.push({
        name: gate.name.toUpperCase(),
        targets: gate.targets.slice(),
        controls: (gate.controls || []).slice(),
        params: (gate.params || []).slice(),
        label: gateLabel(gate.name, gate.params),
        col: col
      });
    }
    var active_gates = [];
    for (var k2 = 0; k2 < nKeys; k2++) {
      active_gates.push(nOps === 0 ? 0
        : Math.min(Math.floor(k2 / Math.max(nKeys - 1, 1) * nOps), nOps - 1));
    }
    var gate_times = [];
    if (nOps <= 1) { gate_times = [0]; }
    else {
      for (var j = 0; j < nOps; j++) gate_times.push(j / (nOps - 1) * duration);
      gate_times[0] = 0;
    }

    return {
      meta: {
        title: (title || "量子电路模拟") + "（含噪）",
        fps: fps || 30, duration: duration || 4, n_qubits: n, n_keys: nKeys,
        colormap: "viridis", generated_by: "gpuqviz-playground", noisy: true
      },
      states_re: null, states_im: null,
      probs: probs, bloch: bloch,
      circuit: { n_qubits: n, n_ops: nOps, gates: gates,
                 active_gates: active_gates, gate_times: gate_times }
    };
  }

  // ─── 公开 API ───

  return {
    GATES: GATES,
    gateMatrix: gateMatrix,
    gateLabel: gateLabel,
    newState: newState,
    applyGate: applyGate,
    blochVector: blochVector,
    buildPayload: buildPayload,
    buildPayloadNoisy: buildPayloadNoisy,
    blochVectorDensity: blochVectorDensity
  };
})();
