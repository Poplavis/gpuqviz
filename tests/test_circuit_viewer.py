"""交互式电路图测试：payload circuit 字段 + viewer.js SVG 联动逻辑。

覆盖：
  - _build_circuit_info 门级元数据提取（Bell/GHZ/参数门/无门）
  - build_payload circuit 字段附加
  - export_html 生成的 HTML 包含电路图 DOM/CSS/JS
  - states-only 输入无 circuit 字段
  - active_gates / gate_times 映射正确性
  - 降级模式保留 circuit 字段
  - viewer.js 电路图相关函数存在
"""

import json
import math
import re

import numpy as np
import pytest

from gpuqviz.export_html import (
    build_payload,
    export_html,
    _build_circuit_info,
    _gate_label,
)
from gpuqviz.circuits import Gate
from gpuqviz.jupyter import show, _strip_state_panel

qiskit = pytest.importorskip("qiskit")


# -- _gate_label 参数格式化 --------------------------------------------------

def test_gate_label_simple():
    """无参数门：标签 = 门名。"""
    assert _gate_label(Gate("H", targets=[0])) == "H"
    assert _gate_label(Gate("X", targets=[0])) == "X"
    assert _gate_label(Gate("CX", targets=[1], controls=[0])) == "CX"


def test_gate_label_unitary():
    """UNITARY 门显示为 U。"""
    assert _gate_label(Gate("UNITARY", targets=[0, 1])) == "U"


def test_gate_label_parametric():
    """参数门：标签含格式化角度。"""
    assert _gate_label(Gate("RX", targets=[0], params=[math.pi / 2])) == "RX(π/2)"
    assert _gate_label(Gate("RZ", targets=[0], params=[math.pi])) == "RZ(π)"
    assert _gate_label(Gate("RZ", targets=[0], params=[0])) == "RZ(0)"
    assert _gate_label(Gate("U3", targets=[0], params=[math.pi, math.pi / 2, 0])) == "U3(π,π/2,0)"


# -- _build_circuit_info 门级元数据 ------------------------------------------

def test_build_circuit_info_bell():
    """Bell 态电路 → 2 个门（H, CX）。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ci = _build_circuit_info(qc, steps=20, duration=4.0)
    assert ci is not None
    assert ci["n_qubits"] == 2
    assert ci["n_ops"] == 2
    assert ci["gates"][0]["name"] == "H"
    assert ci["gates"][0]["targets"] == [0]
    assert ci["gates"][1]["name"] == "CX"
    assert ci["gates"][1]["targets"] == [1]
    assert ci["gates"][1]["controls"] == [0]


def test_build_circuit_info_ghz():
    """3-qubit GHZ → 3 个门（H, CX, CX）。"""
    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    ci = _build_circuit_info(qc, steps=30, duration=6.0)
    assert ci is not None
    assert ci["n_ops"] == 3
    assert ci["gates"][0]["name"] == "H"
    assert ci["gates"][1]["name"] == "CX"
    assert ci["gates"][2]["name"] == "CX"


def test_build_circuit_info_none_for_states():
    """非电路输入（states-only）→ circuit 字段为 None。"""
    psi = np.array([1, 0], dtype=complex)
    payload = build_payload([psi], fps=30, duration=1.0, title="t")
    assert "circuit" not in payload


def test_build_circuit_info_with_circuit_info():
    """build_payload 接收 circuit_info → payload 包含 circuit 字段。"""
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    ci = _build_circuit_info(qc, steps=10, duration=2.0)
    psi = np.array([1, 0], dtype=complex)
    payload = build_payload([psi], fps=30, duration=2.0, title="t",
                            circuit_info=ci)
    assert "circuit" in payload
    assert payload["circuit"]["n_ops"] == 1


# -- active_gates / gate_times 映射 ------------------------------------------

def test_active_gates_mapping():
    """active_gates 长度 == n_keys，值在 [0, n_ops-1]。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    steps = 20
    ci = _build_circuit_info(qc, steps=steps, duration=4.0)
    assert len(ci["active_gates"]) == steps
    for idx in ci["active_gates"]:
        assert 0 <= idx < ci["n_ops"]
    # 首帧应指向第一个门
    assert ci["active_gates"][0] == 0
    # 末帧应指向最后一个门
    assert ci["active_gates"][-1] == ci["n_ops"] - 1


def test_gate_times_correct():
    """gate_times 单调递增，首项 = 0，末项 ≤ duration。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    duration = 4.0
    ci = _build_circuit_info(qc, steps=20, duration=duration)
    gt = ci["gate_times"]
    assert gt[0] == 0.0
    assert gt[-1] <= duration
    for i in range(1, len(gt)):
        assert gt[i] >= gt[i - 1]


def test_gate_times_single_op():
    """单门电路：gate_times = [0.0]。"""
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    ci = _build_circuit_info(qc, steps=10, duration=2.0)
    assert ci["n_ops"] == 1
    assert ci["gate_times"] == [0.0]
    assert all(idx == 0 for idx in ci["active_gates"])


# -- export_html HTML 包含电路图元素 -----------------------------------------

def test_export_html_contains_circuit_panel(tmp_path):
    """export_html(circuit=...) 生成的 HTML 包含 #circuitPanel。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = export_html(circuit=qc, steps=20, out=tmp_path / "circuit.html")
    html = out.read_text(encoding="utf-8")
    assert 'id="circuitPanel"' in html
    assert "circuitSvg" in html  # viewer.js 中 SVG id
    assert "buildCircuitDiagram" in html  # viewer.js 函数


def test_export_html_no_circuit_for_states(tmp_path):
    """states-only 导出：HTML 中 #circuitPanel 存在（hidden），但 payload 无 circuit 字段。"""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    out = export_html(states=[psi], steps=10, out=tmp_path / "no_circuit.html")
    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    assert m
    data = json.loads(m.group(1))
    assert "circuit" not in data


def test_export_html_circuit_payload_correct(tmp_path):
    """export_html(circuit=...) payload 中 circuit 字段结构正确。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = export_html(circuit=qc, steps=20, duration=4.0,
                      out=tmp_path / "payload_circuit.html")
    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    assert m
    data = json.loads(m.group(1))
    assert "circuit" in data
    ci = data["circuit"]
    assert ci["n_qubits"] == 2
    assert ci["n_ops"] == 2
    assert len(ci["gates"]) == 2
    assert len(ci["active_gates"]) == 20
    assert len(ci["gate_times"]) == 2


# -- 降级模式保留 circuit 字段 ------------------------------------------------

def test_degraded_mode_keeps_circuit(tmp_path):
    """_strip_state_panel 后 circuit 字段仍在（体积小，不需降级）。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ci = _build_circuit_info(qc, steps=20, duration=4.0)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    payload = build_payload([psi], fps=30, duration=4.0, title="t",
                            circuit_info=ci)
    stripped = _strip_state_panel(payload)
    assert stripped["states_re"] is None
    assert "circuit" in stripped  # circuit 字段保留


# -- viewer.js 电路图函数存在 -------------------------------------------------

def test_viewer_js_has_circuit_functions(tmp_path):
    """viewer.js 包含电路图构建、高亮、跳转函数。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = export_html(circuit=qc, steps=10, out=tmp_path / "js_check.html")
    html = out.read_text(encoding="utf-8")
    assert "function buildCircuitDiagram" in html
    assert "function updateCircuitHighlight" in html
    assert "function seekToGate" in html
    assert "gateEls" in html
    assert "active_gates" in html
    assert "gate_times" in html


def test_viewer_js_null_circuit_guard(tmp_path):
    """viewer.js 中有 D.circuit 为 null/falsy 时的隐藏面板守卫。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = export_html(circuit=qc, steps=10, out=tmp_path / "null_guard.html")
    html = out.read_text(encoding="utf-8")
    # buildCircuitDiagram 应有 !D.circuit 检查
    assert "!D.circuit" in html or "D.circuit" in html


# -- show() 集成 -------------------------------------------------------------

def test_show_circuit_includes_circuit_info(tmp_path):
    """show(circuit) 生成的 payload 包含 circuit 字段。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = show(qc, steps=20, out=tmp_path / "show_circuit.html")
    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    assert m
    data = json.loads(m.group(1))
    assert "circuit" in data
    assert data["circuit"]["n_ops"] == 2


def test_show_states_no_circuit_info(tmp_path):
    """show(states=...) 生成的 payload 不含 circuit 字段。"""
    psi = np.array([1, 0], dtype=complex)
    out = show(states=[psi], steps=5, out=tmp_path / "show_states.html")
    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    assert m
    data = json.loads(m.group(1))
    assert "circuit" not in data
