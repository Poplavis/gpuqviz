/* gpuqviz 量子电路编辑器。
 *
 * 提供可视化电路编辑 UI：门面板 + 电路图 SVG + 拖放交互。
 * 用户构建电路后调用 runSimulation() 构建 payload 并初始化 viewer.js。
 *
 * 依赖：quantum-sim.js (GPUQVIZ_SIM), viewer.js (GPUQVIZ_VIEWER)
 */

window.GPUQVIZ_COMPOSER = (function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var MIN_Q = 2, MAX_Q = 6;

  // 门定义
  var GATE_DEFS = [
    { name: "H",  label: "H",  type: "single" },
    { name: "X",  label: "X",  type: "single" },
    { name: "Y",  label: "Y",  type: "single" },
    { name: "Z",  label: "Z",  type: "single" },
    { name: "S",  label: "S",  type: "single" },
    { name: "T",  label: "T",  type: "single" },
    { name: "RX", label: "RX", type: "param" },
    { name: "RY", label: "RY", type: "param" },
    { name: "RZ", label: "RZ", type: "param" },
    { name: "CX", label: "CX", type: "controlled" },
    { name: "CZ", label: "CZ", type: "controlled" },
    { name: "SWAP", label: "SW", type: "swap" }
  ];

  // 预设电路
  var PRESETS = {
    Bell: [
      { name: "H",  targets: [0], controls: [], params: [] },
      { name: "CX", targets: [1], controls: [0], params: [] }
    ],
    GHZ: [
      { name: "H",  targets: [0], controls: [], params: [] },
      { name: "CX", targets: [1], controls: [0], params: [] },
      { name: "CX", targets: [2], controls: [1], params: [] }
    ],
    Superposition: function (n) {
      var gates = [];
      for (var i = 0; i < n; i++) gates.push({ name: "H", targets: [i], controls: [], params: [] });
      return gates;
    },
    Grover3: [
      { name: "H",  targets: [0], controls: [], params: [] },
      { name: "H",  targets: [1], controls: [], params: [] },
      { name: "H",  targets: [2], controls: [], params: [] },
      // Oracle |111⟩
      { name: "H",  targets: [2], controls: [], params: [] },
      { name: "CX", targets: [2], controls: [0], params: [] },
      { name: "CX", targets: [2], controls: [1], params: [] },
      { name: "H",  targets: [2], controls: [], params: [] },
      // Diffuser
      { name: "H",  targets: [0], controls: [], params: [] },
      { name: "H",  targets: [1], controls: [], params: [] },
      { name: "H",  targets: [2], controls: [], params: [] },
      { name: "X",  targets: [0], controls: [], params: [] },
      { name: "X",  targets: [1], controls: [], params: [] },
      { name: "X",  targets: [2], controls: [], params: [] },
      { name: "H",  targets: [2], controls: [], params: [] },
      { name: "CX", targets: [2], controls: [0], params: [] },
      { name: "CX", targets: [2], controls: [1], params: [] },
      { name: "H",  targets: [2], controls: [], params: [] },
      { name: "X",  targets: [0], controls: [], params: [] },
      { name: "X",  targets: [1], controls: [], params: [] },
      { name: "X",  targets: [2], controls: [], params: [] },
      { name: "H",  targets: [0], controls: [], params: [] },
      { name: "H",  targets: [1], controls: [], params: [] },
      { name: "H",  targets: [2], controls: [], params: [] }
    ]
  };

  // ─── 编辑器状态 ───

  var state = {
    nQubits: 3,
    circuit: [],       // [{name, targets, controls, params}]
    selectedGate: null, // 当前选中的门类型
    pendingTarget: null, // 受控门等待选择 target
    selectedCol: 0      // 当前编辑的列
  };

  var svgEl = null;  // 电路图 SVG 元素

  // ─── SVG 工具 ───

  function svg(tag, attrs) {
    var el = document.createElementNS(SVG_NS, tag);
    if (attrs) for (var k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  // ─── 电路图渲染 ───

  function renderCircuit() {
    var container = document.getElementById("composerCircuit");
    if (!container) return;
    container.innerHTML = "";

    var n = state.nQubits;
    var nCols = Math.max(state.circuit.length + 1, 4);
    var colW = 56, rowH = 36, ml = 36, mt = 12, boxS = 26;
    var svgW = ml + (nCols + 1) * colW;
    var svgH = mt + n * rowH + 10;

    svgEl = svg("svg", { width: svgW, height: svgH });
    container.appendChild(svgEl);

    var yOf = function (q) { return mt + q * rowH + rowH / 2; };

    // 量子线 + 标签
    for (var q = 0; q < n; q++) {
      var y = yOf(q);
      svgEl.appendChild(svg("text", {
        x: ml - 6, y: y + 4, fill: "#8a94a6", "font-size": 12,
        "font-family": "Consolas,monospace", "text-anchor": "end"
      })).textContent = "q" + q;
      svgEl.appendChild(svg("line", {
        x1: ml, y1: y, x2: svgW - 6, y2: y,
        stroke: "#3d4858", "stroke-width": 2
      }));

      // 点击量子线添加门
      (function (qubit, y) {
        var hitArea = svg("rect", {
          x: ml, y: y - rowH / 2 + 2, width: svgW - ml - 6, height: rowH - 4,
          fill: "transparent", cursor: "pointer"
        });
        hitArea.addEventListener("click", function (e) {
          var rect = svgEl.getBoundingClientRect();
          var cx = e.clientX - rect.left;
          var col = Math.round((cx - ml) / colW - 1);
          onQubitClick(qubit, Math.max(0, col));
        });
        svgEl.appendChild(hitArea);
      })(q, y);
    }

    // 渲染已放置的门
    state.circuit.forEach(function (gate, col) {
      var cx = ml + (col + 1) * colW;
      renderGate(gate, cx, yOf, boxS, col);
    });
  }

  function renderGate(gate, cx, yOf, boxS, col) {
    var grp = svg("g", { "data-col": col, style: "cursor:pointer;" });
    svgEl.appendChild(grp);

    var isCtrl = gate.controls && gate.controls.length > 0;
    var allQ = (gate.targets || []).concat(gate.controls || []);

    if (gate.name === "SWAP") {
      for (var i = 0; i < gate.targets.length; i++) {
        var y = yOf(gate.targets[i]);
        grp.appendChild(svg("line", { x1: cx-6, y1: y-6, x2: cx+6, y2: y+6, stroke: "#4dc5f2", "stroke-width": 2 }));
        grp.appendChild(svg("line", { x1: cx-6, y1: y+6, x2: cx+6, y2: y-6, stroke: "#4dc5f2", "stroke-width": 2 }));
      }
      if (gate.targets.length === 2) {
        grp.appendChild(svg("line", { x1: cx, y1: yOf(gate.targets[0]), x2: cx, y2: yOf(gate.targets[1]), stroke: "#4dc5f2", "stroke-width": 1.5 }));
      }
    } else if (isCtrl) {
      // 受控门
      for (var c = 0; c < gate.controls.length; c++) {
        grp.appendChild(svg("circle", { cx: cx, cy: yOf(gate.controls[c]), r: 4, fill: "#4dc5f2" }));
      }
      var tgt = gate.targets[0];
      var ty = yOf(tgt);
      if (gate.name === "CX" || gate.name === "CNOT") {
        grp.appendChild(svg("circle", { cx: cx, cy: ty, r: 10, fill: "none", stroke: "#4dc5f2", "stroke-width": 1.5 }));
        grp.appendChild(svg("line", { x1: cx-10, y1: ty, x2: cx+10, y2: ty, stroke: "#4dc5f2", "stroke-width": 1.5 }));
        grp.appendChild(svg("line", { x1: cx, y1: ty-10, x2: cx, y2: ty+10, stroke: "#4dc5f2", "stroke-width": 1.5 }));
      } else {
        grp.appendChild(svg("rect", { x: cx-boxS/2, y: ty-boxS/2, width: boxS, height: boxS, fill: "#1a2130", stroke: "#4dc5f2", "stroke-width": 1.5, rx: 4, ry: 4 }));
        grp.appendChild(svg("text", { x: cx, y: ty, fill: "#e6ebf5", "font-size": 12, "font-family": "Consolas,monospace", "text-anchor": "middle", "dominant-baseline": "central" }))
          .textContent = window.GPUQVIZ_SIM.gateLabel(gate.name, gate.params);
      }
      var minY = Math.min.apply(null, allQ.map(yOf));
      var maxY = Math.max.apply(null, allQ.map(yOf));
      grp.appendChild(svg("line", { x1: cx, y1: minY, x2: cx, y2: maxY, stroke: "#4dc5f2", "stroke-width": 1.5 }));
    } else {
      // 单量子门
      var q = gate.targets[0];
      var y2 = yOf(q);
      grp.appendChild(svg("rect", { x: cx-boxS/2, y: y2-boxS/2, width: boxS, height: boxS, fill: "#1a2130", stroke: "#4dc5f2", "stroke-width": 1.5, rx: 4, ry: 4 }));
      grp.appendChild(svg("text", { x: cx, y: y2, fill: "#e6ebf5", "font-size": 12, "font-family": "Consolas,monospace", "text-anchor": "middle", "dominant-baseline": "central" }))
        .textContent = window.GPUQVIZ_SIM.gateLabel(gate.name, gate.params);
    }

    // 点击删除门
    grp.addEventListener("click", function (e) {
      e.stopPropagation();
      removeGate(col);
    });
  }

  // ─── 交互逻辑 ───

  function onQubitClick(qubit, col) {
    if (!state.selectedGate) return;

    var def = GATE_DEFS.find(function (g) { return g.name === state.selectedGate; });
    if (!def) return;

    if (def.type === "single") {
      // 单量子门：直接放置
      addGate({
        name: state.selectedGate,
        targets: [qubit],
        controls: [],
        params: []
      });
      clearSelection();
    } else if (def.type === "param") {
      // 参数门：默认 π/2 直接放置，角度用下方滑杆实时调节（P4.2）
      addGate({
        name: state.selectedGate,
        targets: [qubit],
        controls: [],
        params: [Math.PI / 2]
      });
      clearSelection();
    } else if (def.type === "controlled") {
      // 受控门：先选 control，再选 target
      if (state.pendingTarget === null) {
        // 第一次点击：选 control
        state.pendingTarget = qubit;
        updateStatus("已选控制位 q" + qubit + "，请点击目标位", "var(--accent2)");
      } else {
        // 第二次点击：选 target
        if (qubit === state.pendingTarget) {
          updateStatus("控制位和目标位不能相同", "var(--accent2)");
        } else {
          addGate({
            name: state.selectedGate,
            targets: [qubit],
            controls: [state.pendingTarget],
            params: []
          });
        }
        state.pendingTarget = null;
        clearSelection();
      }
    } else if (def.type === "swap") {
      // SWAP：先选第一个 qubit，再选第二个
      if (state.pendingTarget === null) {
        state.pendingTarget = qubit;
        updateStatus("已选第一个 qubit q" + qubit + "，请点击第二个", "var(--accent2)");
      } else {
        if (qubit !== state.pendingTarget) {
          addGate({
            name: "SWAP",
            targets: [state.pendingTarget, qubit],
            controls: [],
            params: []
          });
        }
        state.pendingTarget = null;
        clearSelection();
      }
    }
  }

  function addGate(gate) {
    state.circuit.push(gate);
    renderCircuit();
    updateCircuitList();
    refreshParamSliders();
    updateStatus("已添加 " + gate.name + " 门", "var(--green)");
  }

  function removeGate(col) {
    state.circuit.splice(col, 1);
    renderCircuit();
    updateCircuitList();
    refreshParamSliders();
  }

  function clearSelection() {
    state.selectedGate = null;
    state.pendingTarget = null;
    document.querySelectorAll(".gate-btn.selected").forEach(function (b) {
      b.classList.remove("selected");
    });
  }

  function clearCircuit() {
    state.circuit = [];
    clearSelection();
    renderCircuit();
    updateCircuitList();
    updateStatus("已清空电路", "var(--dim)");
  }

  function setQubits(n) {
    state.nQubits = Math.max(MIN_Q, Math.min(MAX_Q, n));
    document.getElementById("qubitNum").textContent = state.nQubits;
    renderCircuit();
    updateCircuitList();
  }

  function loadPreset(name) {
    var preset = PRESETS[name];
    if (!preset) return;
    var gates = typeof preset === "function" ? preset(state.nQubits) : preset;
    state.circuit = gates.map(function (g) {
      return {
        name: g.name,
        targets: g.targets.slice(),
        controls: (g.controls || []).slice(),
        params: (g.params || []).slice()
      };
    });
    // 调整 qubit 数
    var maxQ = 0;
    state.circuit.forEach(function (g) {
      g.targets.forEach(function (t) { if (t > maxQ) maxQ = t; });
      (g.controls || []).forEach(function (c) { if (c > maxQ) maxQ = c; });
    });
    if (maxQ + 1 > state.nQubits) {
      state.nQubits = Math.min(MAX_Q, maxQ + 1);
      document.getElementById("qubitNum").textContent = state.nQubits;
    }
    renderCircuit();
    updateCircuitList();
    updateStatus("已加载预设：" + name, "var(--green)");
  }

  // ─── 电路列表文本 ───

  function updateCircuitList() {
    var el = document.getElementById("circuitList");
    if (!el) return;
    if (state.circuit.length === 0) {
      el.innerHTML = '<span style="color:var(--dim)">（空电路）</span>';
      return;
    }
    var lines = [];
    for (var q = 0; q < state.nQubits; q++) {
      var gates = [];
      state.circuit.forEach(function (g) {
        var involved = g.targets.indexOf(q) >= 0 ||
          (g.controls && g.controls.indexOf(q) >= 0);
        if (involved) {
          var label = window.GPUQVIZ_SIM.gateLabel(g.name, g.params);
          if (g.controls && g.controls.indexOf(q) >= 0) label = "●" + label;
          gates.push(label);
        } else {
          gates.push("─");
        }
      });
      lines.push('<span style="color:var(--accent)">q' + q + "</span>: " + gates.join(" → "));
    }
    el.innerHTML = lines.join("<br>");
  }

  function updateStatus(msg, color) {
    var el = document.getElementById("composerStatus");
    if (!el) return;
    el.textContent = msg;
    el.style.color = color || "var(--dim)";
  }

  // ─── 旋转门参数滑杆（P4.2 实时联动）───

  var paramSliderTimer = 0;
  var paramSliderTimer2 = null;

  function refreshParamSliders() {
    var box = document.getElementById("paramSliders");
    if (!box) return;
    box.innerHTML = "";
    var rotIdx = [];
    state.circuit.forEach(function (g, i) {
      if (["RX", "RY", "RZ"].indexOf(g.name.toUpperCase()) >= 0) rotIdx.push(i);
    });
    if (rotIdx.length === 0) {
      box.innerHTML = '<span style="color:var(--dim);font-size:0.8rem;">'
        + "电路中无旋转门（放置 RX/RY/RZ 后可在此调角度）</span>";
      return;
    }
    rotIdx.forEach(function (gi) {
      var g = state.circuit[gi];
      var angle = (g.params && g.params[0] !== undefined) ? g.params[0] : Math.PI / 2;
      var wrap = document.createElement("div");
      wrap.style.marginBottom = "6px";
      var label = document.createElement("span");
      label.style.cssText = "font-size:0.8rem;color:var(--dim);";
      label.textContent = g.name.toUpperCase() + "[q" + g.targets[0] + "] θ=" +
        angle.toFixed(2);
      var slider = document.createElement("input");
      slider.type = "range";
      slider.min = "0"; slider.max = (2 * Math.PI).toFixed(2); slider.step = "0.01";
      slider.value = String(angle);
      slider.style.cssText = "width:100%;accent-color:var(--accent);";
      slider.addEventListener("input", function () {
        g.params = [parseFloat(slider.value)];
        label.textContent = g.name.toUpperCase() + "[q" + g.targets[0]
          + "] θ=" + g.params[0].toFixed(2);
        // 节流 200ms：拖动即演化，同时避免高频重建 GL 上下文
        var now = Date.now();
        if (now - paramSliderTimer > 200) {
          paramSliderTimer = now;
          runSimulation();
          updateCircuitList();
        } else {
          if (paramSliderTimer2) clearTimeout(paramSliderTimer2);
          paramSliderTimer2 = setTimeout(function () {
            paramSliderTimer = Date.now();
            runSimulation();
            updateCircuitList();
          }, 220);
        }
      });
      wrap.appendChild(label);
      wrap.appendChild(slider);
      box.appendChild(wrap);
    });
  }

  // ─── 噪声控件（P4.2）───

  function initNoiseControls() {
    var toggle = document.getElementById("noiseToggle");
    var sliders = document.getElementById("noiseSliders");
    var dep = document.getElementById("noiseDep");
    var deph = document.getElementById("noiseDeph");
    var depVal = document.getElementById("noiseDepVal");
    var dephVal = document.getElementById("noiseDephVal");
    if (!toggle) return;
    state.noiseEnabled = false;
    state.noiseDep = 0.05;
    state.noiseDeph = 0.05;
    toggle.addEventListener("change", function () {
      state.noiseEnabled = toggle.checked;
      sliders.style.display = toggle.checked ? "block" : "none";
      if (state.circuit.length > 0) runSimulation();
    });
    dep.addEventListener("input", function () {
      state.noiseDep = parseFloat(dep.value);
      depVal.textContent = state.noiseDep.toFixed(2);
      if (state.noiseEnabled && state.circuit.length > 0) runSimulation();
    });
    deph.addEventListener("input", function () {
      state.noiseDeph = parseFloat(deph.value);
      dephVal.textContent = state.noiseDeph.toFixed(2);
      if (state.noiseEnabled && state.circuit.length > 0) runSimulation();
    });
  }

  // ─── 运行模拟 ───

  function runSimulation() {
    if (state.circuit.length === 0) {
      updateStatus("请先添加门再运行模拟", "var(--accent2)");
      return;
    }

    updateStatus("正在模拟…", "var(--accent)");

    // 计算 duration（每个门 0.5s）
    var duration = Math.max(2, state.circuit.length * 0.5);
    var fps = 30;

    // 构建 payload（噪声开启且 n ≤ 4 时走密度矩阵路径）
    var payload;
    if (state.noiseEnabled && state.nQubits <= 4) {
      payload = window.GPUQVIZ_SIM.buildPayloadNoisy(
        state.circuit, state.nQubits, fps, duration, "量子电路实验室",
        { depolarizing: state.noiseDep, dephasing: state.noiseDeph }
      );
    } else {
      payload = window.GPUQVIZ_SIM.buildPayload(
        state.circuit, state.nQubits, fps, duration, "量子电路实验室"
      );
    }

    // 需要先清理 viewer 可能的旧 canvas
    var vp = document.getElementById("viewport");
    if (vp) {
      // 移除旧的 canvas
      var canvases = vp.querySelectorAll("canvas");
      canvases.forEach(function (c) { c.remove(); });
    }

    // 设置 payload 并初始化 viewer
    window.GPUQVIZ_DATA = payload;
    window.GPUQVIZ_VIEWER.Init("viewport");

    var noiseTag = (state.noiseEnabled && state.nQubits <= 4) ? " · 含噪" : "";
    updateStatus("模拟完成 · " + state.circuit.length + " 门 · " + state.nQubits
                 + " qubit" + noiseTag, "var(--green)");
    refreshParamSliders();
  }

  // ─── 初始化 ───

  function init() {
    // 门面板
    var palette = document.getElementById("gatePalette");
    if (palette) {
      GATE_DEFS.forEach(function (def) {
        var btn = document.createElement("button");
        btn.className = "gate-btn";
        btn.textContent = def.label;
        btn.setAttribute("data-gate", def.name);
        btn.addEventListener("click", function () {
          // 切换选中
          var wasSelected = btn.classList.contains("selected");
          clearSelection();
          if (!wasSelected) {
            btn.classList.add("selected");
            state.selectedGate = def.name;
            var hint = def.type === "controlled" ? "（选控制位 → 选目标位）"
                     : def.type === "swap" ? "（选两个 qubit）"
                     : def.type === "param" ? "（点击 qubit 放置，可设角度）"
                     : "（点击 qubit 放置）";
            updateStatus("已选 " + def.label + " 门" + hint, "var(--accent)");
          }
        });
        palette.appendChild(btn);
      });
    }

    // qubit 数控制
    var decBtn = document.getElementById("qubitDec");
    var incBtn = document.getElementById("qubitInc");
    if (decBtn) decBtn.addEventListener("click", function () { setQubits(state.nQubits - 1); });
    if (incBtn) incBtn.addEventListener("click", function () { setQubits(state.nQubits + 1); });

    // 预设按钮
    var presetContainer = document.getElementById("presetButtons");
    if (presetContainer) {
      Object.keys(PRESETS).forEach(function (name) {
        var btn = document.createElement("button");
        btn.className = "preset-btn";
        btn.textContent = name;
        btn.addEventListener("click", function () { loadPreset(name); });
        presetContainer.appendChild(btn);
      });
    }

    // 运行 / 清空按钮
    var runBtn = document.getElementById("runBtn");
    var clrBtn = document.getElementById("clearBtn");
    if (runBtn) runBtn.addEventListener("click", runSimulation);
    if (clrBtn) clrBtn.addEventListener("click", clearCircuit);

    // 噪声控件
    initNoiseControls();

    // 初始渲染
    renderCircuit();
    updateCircuitList();

    // 默认加载 Bell 态并自动运行
    loadPreset("Bell");
    runSimulation();
  }

  return { init: init, runSimulation: runSimulation, clearCircuit: clearCircuit };
})();
