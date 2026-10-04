"""参数化电路对拍：qiskit 逐值 Statevector（1e-10）+ 解析公式 + 绑定错误。"""

import numpy as np
import pytest

from gpuqviz.parameters import CircuitTemplate, Parameter, sweep

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import (  # noqa: E402
    SparsePauliOp,
    Statevector as QkSV,
)

TOL = 1e-10


def _template() -> CircuitTemplate:
    """RY(θ) on q0 → CX(0,1) → RY(θ/2 固定) on q1 的 2-qubit 模板。"""
    return CircuitTemplate(2, [
        {"name": "RY", "targets": [0], "controls": [], "params": [Parameter("theta")]},
        {"name": "CX", "targets": [1], "controls": [0], "params": []},
        {"name": "RY", "targets": [1], "controls": [], "params": [0.5]},
    ])


def _qiskit_ref(theta: float) -> np.ndarray:
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.ry(0.5, 1)
    return np.asarray(QkSV.from_instruction(qc).data)


def test_bind_matches_qiskit_per_value():
    tmpl = _template()
    for theta in (0.0, 0.3, np.pi / 2, 2.7):
        psi = tmpl.evolve(theta=theta)[-1]
        assert np.allclose(psi, _qiskit_ref(theta), atol=TOL), f"θ={theta}"


def test_sweep_matches_qiskit():
    values = np.linspace(0, np.pi, 9)
    result = sweep(_template(), "theta", values)
    assert result.param == "theta" and len(result) == 9
    for k, v in enumerate(values):
        assert np.allclose(result.states[k], _qiskit_ref(v), atol=TOL), f"θ={v}"


def test_sweep_observables_analytic():
    """⟨Z₀⟩ 与 ⟨Z₀Z₁⟩ 的解析公式对拍。

    RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩ → ⟨Z₀⟩ = cos θ；
    CX 纠缠后 ⟨Z₀Z₁⟩ = cos θ（q1 附加 RY(0.5) 不影响 Z₀Z₁? —— 不对，
    RY 改变 q1 的 z 分量 → 用完整公式验算：手动 qiskit 对拍）。
    """
    values = np.linspace(0, np.pi, 5)
    result = sweep(_template(), "theta", values, observables=["IZ", "ZZ"])
    # ⟨Z₀⟩ = "IZ"（位串左起为最高位 q1）→ q0 无后续门，⟨Z₀⟩ = cos θ
    for k, v in enumerate(values):
        assert abs(result.observables["IZ"][k] - np.cos(v)) < 1e-9
    # ⟨Z₀Z₁⟩ 对拍 qiskit
    for k, v in enumerate(values):
        qc = QuantumCircuit(2)
        qc.ry(v, 0)
        qc.cx(0, 1)
        qc.ry(0.5, 1)
        ref = QkSV.from_instruction(qc).expectation_value(SparsePauliOp("ZZ")).real
        assert abs(result.observables["ZZ"][k] - ref) < TOL


def test_sweep_entropy_structure():
    """θ=0：|00⟩ 直积态 → 熵 0；θ=π：q0 = |1⟩ 仍直积 → 熵 0。"""
    values = np.array([0.0, np.pi])
    result = sweep(_template(), "theta", values)
    assert np.allclose(result.entropy, 0.0, atol=1e-9)
    assert np.allclose(result.purity, 1.0, atol=1e-9)


def test_binding_errors():
    tmpl = _template()
    with pytest.raises(ValueError, match="unbound parameters"):
        tmpl.bind()
    with pytest.raises(ValueError, match="unknown parameters"):
        tmpl.bind(theta=1.0, extra=2.0)
    with pytest.raises(ValueError, match="not in template"):
        sweep(tmpl, "phi", [0.0, 1.0])


def test_sweep_empty_values_rejected():
    with pytest.raises(ValueError, match="at least one"):
        sweep(_template(), "theta", [])


def test_sweep_csv_export(tmp_path):
    result = sweep(_template(), "theta", np.linspace(0, 1, 3),
                   observables=["IZ"])
    text = result.to_csv(tmp_path / "sweep.csv")
    lines = text.strip().split("\n")
    assert lines[0] == "theta,purity,S_q0,S_q1,IZ"
    assert len(lines) == 4
    d = result.to_dict()
    assert set(d["observables"]) == {"IZ"}


def test_parameter_positions_can_hold_float_and_param_mix():
    """同门多参数：符号与常量混合（U3(θ, π/2, 0)）。"""
    tmpl = CircuitTemplate(1, [
        {"name": "U3", "targets": [0], "controls": [],
         "params": [Parameter("theta"), np.pi / 2, 0.0]},
    ])
    qc = QuantumCircuit(1)
    qc.u(np.pi / 4, np.pi / 2, 0.0, 0)
    psi = tmpl.evolve(theta=np.pi / 4)[-1]
    assert np.allclose(psi, np.asarray(QkSV.from_instruction(qc).data),
                       atol=TOL)
