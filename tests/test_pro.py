"""ProVisualizer 门面对拍：report 数值 ≡ analysis 模块直算 + 导出冒烟。"""

import numpy as np
import pytest

from gpuqviz.pro import ProVisualizer

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402

TOL = 1e-10


def _bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _bell_psi():
    return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


# --------------------------------------------------------------------------- #
# 门来源与构造
# --------------------------------------------------------------------------- #

def test_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        ProVisualizer()
    with pytest.raises(ValueError, match="exactly one"):
        ProVisualizer(circuit=_bell_circuit(), gates=[])


def test_template_source_with_bindings():
    from gpuqviz.parameters import CircuitTemplate, Parameter

    tmpl = CircuitTemplate(1, [
        {"name": "RY", "targets": [0], "controls": [],
         "params": [Parameter("theta")]},
    ])
    viz = ProVisualizer(template=tmpl, bindings={"theta": np.pi / 2})
    psi = viz.final_state()
    assert abs(psi[0] - np.sqrt(0.5)) < TOL
    assert abs(psi[1] - np.sqrt(0.5)) < TOL


# --------------------------------------------------------------------------- #
# report 数值 ≡ 直算
# --------------------------------------------------------------------------- #

def test_report_counts_reproducible():
    viz = ProVisualizer(circuit=_bell_circuit(),
                        title="量子态 — 测试").with_shots(shots=2000, seed=7)
    c1 = viz.report().counts
    c2 = ProVisualizer(circuit=_bell_circuit(),
                       title="量子态 — 测试").with_shots(shots=2000, seed=7) \
        .report().counts
    assert c1.counts == c2.counts
    assert set(c1.counts) == {"00", "11"}


def test_report_entanglement_and_pauli():
    viz = (ProVisualizer(circuit=_bell_circuit())
           .analyze(pauli=["IZ", "ZZ"], entanglement=True))
    rep = viz.report()
    # Bell：S(q0)=S(q1)=1 bit，MI=2，N=0.5
    assert abs(rep.entanglement.single_entropy[0] - 1.0) < TOL
    assert abs(rep.entanglement.mutual_info[0, 1] - 2.0) < TOL
    assert abs(rep.entanglement.negativity[0, 1] - 0.5) < TOL
    # ⟨Z₀⟩=0（"IZ" 左起 = q1）；Bell Φ+ 的 ⟨Z⊗Z⟩ = +1
    assert abs(rep.pauli["IZ"]) < TOL
    assert abs(rep.pauli["ZZ"] - 1.0) < TOL


def test_report_fidelity_target():
    viz = ProVisualizer(circuit=_bell_circuit()).analyze(
        fidelity_to=_bell_psi())
    assert abs(viz.report().fidelity_target - 1.0) < TOL


def test_report_dict_serializable():
    viz = (ProVisualizer(circuit=_bell_circuit())
           .with_shots(shots=500, seed=1)
           .analyze(pauli=["ZZ"], entanglement=True))
    d = viz.report().to_dict()
    assert d["entanglement"]["strongest_pair"] == [0, 1]
    assert set(d["counts"]["counts"]) == {"00", "11"}
    assert d["pauli"]["ZZ"] == pytest.approx(1.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# 噪声路径
# --------------------------------------------------------------------------- #

def test_noisy_report_shrinks_purity():
    """含噪末态纯度 < 1（depolarizing 收缩，Tr ρ² 语义）。"""
    from gpuqviz.analysis.metrics import purity

    assert purity(_bell_psi()) == pytest.approx(1.0, abs=1e-10)
    viz_noisy = (ProVisualizer(circuit=_bell_circuit())
                 .with_noise(("depolarizing", 0.1))
                 .analyze())
    noisy_purity = purity(viz_noisy.final_state())
    assert noisy_purity < 0.95
    assert noisy_purity > 0.5  # λ=0.1：收缩温和


# --------------------------------------------------------------------------- #
# 导出冒烟
# ---------------------------------------------------------------------------

@pytest.mark.hosted_runner_flaky
def test_export_video_smoke(tmp_path):

    viz = ProVisualizer(circuit=_bell_circuit())
    out = viz.export_video(tmp_path / "pro.mp4", duration=1.0, fps=30)
    assert out.exists() and out.stat().st_size > 1000
    # 状态 npz 与视频同目录
    assert (tmp_path / "pro_states.npz").exists()


def test_export_frame_smoke(tmp_path, monkeypatch):
    import gpuqviz.backends as _backends

    monkeypatch.setattr(_backends, "_cached", "cpu")
    viz = ProVisualizer(circuit=_bell_circuit())
    out = viz.export_frame(tmp_path / "pro.png", t=1.0, scale=1)
    assert out.exists() and out.stat().st_size > 1000
