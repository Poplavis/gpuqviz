"""交互式 HTML 播放器导出测试。"""

import json
import re

import numpy as np
import pytest

from gpuqviz.export_html import build_payload, export_html

qiskit = pytest.importorskip("qiskit")


def test_build_payload_values():
    # |+> 单 qubit：Bloch (1,0,0)，概率均 0.5
    psi = np.array([1, 1]) / np.sqrt(2)
    payload = build_payload([psi], fps=30, duration=1.0, title="t")
    assert payload["meta"]["n_qubits"] == 1
    assert payload["bloch"][0][0] == pytest.approx([1, 0, 0], abs=1e-9)
    assert payload["states_re"][0][0] == pytest.approx(1 / np.sqrt(2))


def test_build_payload_rejects_mixed_dims():
    with pytest.raises(ValueError, match="share one dimension"):
        build_payload([np.array([1, 0]), np.array([1, 0, 0, 0])], 30, 1.0, "t")


def test_build_payload_rejects_too_many_qubits():
    big = np.zeros(2**11, dtype=complex)
    big[0] = 1
    with pytest.raises(ValueError, match="11 qubits"):
        build_payload([big], 30, 1.0, "t")


def test_export_html_structure(tmp_path):
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = export_html(circuit=qc, steps=20, fps=30, duration=2.0,
                      title="测试标题", out=tmp_path / "viewer.html")
    html = out.read_text(encoding="utf-8")

    # 结构标记齐全
    assert "GPUQVIZ_DATA" in html and "GPUQVIZ_VIEWER" in html
    assert "Copyright 2010-2023 Three.js Authors" in html  # three.js 已内联
    assert "测试标题" in html
    # payload 是合法 JSON 且数值正确
    m = re.search(r"window\.GPUQVIZ_DATA = (\{.*?\});\n", html, re.S)
    assert m, "payload block not found"
    data = json.loads(m.group(1))
    assert data["meta"]["n_keys"] == 20 and data["meta"]["n_qubits"] == 2
    assert len(data["bloch"]) == 20 and len(data["bloch"][0]) == 2
    # 无 CDN 依赖（内联模式）
    assert "cdn.jsdelivr.net" not in html
    assert out.stat().st_size > 500_000  # three.js 内联后体积下限


def test_export_html_online_mode(tmp_path):
    out = export_html(states=[np.array([1, 0], dtype=complex)], steps=5,
                      fps=10, duration=0.5, out=tmp_path / "v.html",
                      embed_three=False)
    html = out.read_text(encoding="utf-8")
    assert "cdn.jsdelivr.net" in html
    assert out.stat().st_size < 200_000
