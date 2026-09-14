"""S9 Jupyter 交互集成测试：show() 回退写文件 + payload 降级 + nbconvert smoke。"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from gpuqviz.jupyter import show, _is_notebook, _payload_size, _strip_state_panel

qiskit = pytest.importorskip("qiskit")


# -- 非 notebook 回退写文件 --------------------------------------------------

def test_show_circuit_writes_html(tmp_path):
    """非 notebook 环境：show(circuit) 写 HTML 文件并返回 Path。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = show(qc, steps=10, out=tmp_path / "test.html")
    assert isinstance(out, Path)
    assert out.exists()
    assert out.stat().st_size > 500_000  # three.js 内联

    html = out.read_text(encoding="utf-8")
    assert "GPUQVIZ_DATA" in html
    assert "GPUQVIZ_VIEWER" in html
    assert "Copyright 2010-2023 Three.js Authors" in html  # three.js 内联


def test_show_states_writes_html(tmp_path):
    """非 notebook 环境：show(states=...) 写 HTML 文件。"""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    out = show(states=[psi], steps=5, out=tmp_path / "states.html")
    assert out.exists()
    assert out.stat().st_size > 500_000


def test_show_requires_input():
    """show() 无输入时报错。"""
    with pytest.raises(ValueError, match="provide exactly one"):
        show()


def test_show_html_payload_correct(tmp_path):
    """show() 生成的 HTML payload 数值正确。"""
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    out = show(qc, steps=4, out=tmp_path / "payload.html")
    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    assert m, "payload block not found"
    data = json.loads(m.group(1))
    assert data["meta"]["n_qubits"] == 1
    assert data["meta"]["n_keys"] == 4
    assert len(data["bloch"]) == 4
    assert len(data["bloch"][0]) == 1


# -- payload 降级 -----------------------------------------------------------

def test_payload_size_calculation():
    """_payload_size 正确估算 JSON 序列化字节数。"""
    payload = {"a": [1.0, 2.0, 3.0], "b": "hello"}
    size = _payload_size(payload)
    assert size == len(json.dumps(payload).encode("utf-8"))


def test_strip_state_panel():
    """_strip_state_panel 置 states_re/im 为 null 并打标记。"""
    payload = {
        "meta": {"n_qubits": 2},
        "states_re": [[1.0, 0.0]],
        "states_im": [[0.0, 0.0]],
        "bloch": [[[1, 0, 0]]],
    }
    stripped = _strip_state_panel(payload)
    assert stripped["states_re"] is None
    assert stripped["states_im"] is None
    assert stripped["meta"]["state_panel_stripped"] is True
    assert stripped["bloch"] is not None  # Bloch 保留


def test_show_degrades_large_payload(tmp_path):
    """超 8MB payload 自动降级：states_re → null，bloch 保留。"""
    # 10 qubit × 200 帧 全非零 → ~8.7MB
    states = []
    for i in range(200):
        s = np.ones(1024, dtype=complex) / np.sqrt(1024)
        s *= np.exp(1j * i * 0.1)
        states.append(s)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = show(states=states, steps=200, out=tmp_path / "degraded.html")

    # 应有降级警告
    strip_warnings = [x for x in w if "stripping" in str(x.message).lower()]
    assert len(strip_warnings) >= 1, "expected payload stripping warning"

    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    assert m
    data = json.loads(m.group(1))
    assert data["states_re"] is None
    assert data["states_im"] is None
    assert data["meta"]["state_panel_stripped"] is True
    assert data["bloch"] is not None
    # 降级后文件应远小于 8MB
    assert out.stat().st_size < 2 * 1024 * 1024


def test_show_no_degradation_for_small_payload(tmp_path):
    """3 qubit 小 payload 不触发降级。"""
    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = show(qc, steps=30, out=tmp_path / "small.html")
    strip_warnings = [x for x in w if "stripping" in str(x.message).lower()]
    assert len(strip_warnings) == 0, "unexpected degradation warning for small payload"

    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    data = json.loads(m.group(1))
    assert data["states_re"] is not None
    assert data["meta"].get("state_panel_stripped", False) is False


# -- viewer.js 降级兼容 -----------------------------------------------------

def test_viewer_js_handles_null_states(tmp_path):
    """降级后 viewer.js 中有 null guard 处理 states_re=null。"""
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = show(qc, steps=10, out=tmp_path / "guard.html")
    html = out.read_text(encoding="utf-8")
    # viewer.js 应包含 null 检查
    assert "if (!D.states_re)" in html or "!D.states_re" in html


# -- 环境检测 ---------------------------------------------------------------

def test_is_notebook_in_terminal():
    """终端/脚本环境中 _is_notebook() 返回 False。"""
    assert _is_notebook() is False


# -- nbconvert smoke 测试 ---------------------------------------------------

def test_nbconvert_smoke(tmp_path):
    """nbconvert 执行最小 notebook：gpuqviz.show(circuit) 写 HTML 文件。

    jupyter/nbconvert 未安装时 skip。
    """
    nbformat = pytest.importorskip("nbformat")
    nbconvert = pytest.importorskip("nbconvert")
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "from qiskit import QuantumCircuit\n"
            "import gpuqviz\n"
            "from pathlib import Path\n"
            "qc = QuantumCircuit(2)\n"
            "qc.h(0); qc.cx(0, 1)\n"
            f"out_path = Path(r'{tmp_path / 'nbconv.html'}')\n"
            "gpuqviz.show(qc, steps=10, out=out_path)\n"
            "assert out_path.exists(), f'file missing: {out_path}'\n"
            "assert out_path.stat().st_size > 500_000, 'file too small'\n"
            "print('nbconvert smoke: OK')\n"
        )
    ]

    ep = ExecutePreprocessor(timeout=60, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(tmp_path)}})

    # 验证文件确实写了
    out_file = tmp_path / "nbconv.html"
    assert out_file.exists()
    assert out_file.stat().st_size > 500_000
