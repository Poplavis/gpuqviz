"""条件门 / 中途测量对拍：qiskit 投影算符（1e-10）+ 传态四分支 + Aer IfElseOp。"""

import numpy as np
import pytest

from gpuqviz.circuits import (Condition, Gate, evolve_gates_branches)

qiskit = pytest.importorskip("qiskit")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # noqa: E402
from qiskit.circuit.controlflow import IfElseOp  # noqa: E402
from qiskit.quantum_info import Statevector as QkSV  # noqa: E402

TOL = 1e-10


def _projector_qubit(n: int, qubit: int, value: int) -> np.ndarray:
    """|value⟩⟨value|_qubit ⊗ I（qiskit Operator 构造，独立参考路径）。"""
    qc = QuantumCircuit(n)
    if value:
        qc.x(qubit)
    psi0 = QkSV.from_instruction(qc)
    return np.outer(np.asarray(psi0.data), np.asarray(psi0.data).conj())


# --------------------------------------------------------------------------- #
# 塌缩语义
# --------------------------------------------------------------------------- #

def test_collapse_matches_projector():
    """H|0⟩ 测 q0→1：塌缩态 ≡ 归一化的 P₁|+0⟩（qiskit 投影算符参考）。"""
    qc = QuantumCircuit(2)
    qc.h(0)
    psi = np.asarray(QkSV.from_instruction(qc).data)  # |+0⟩ 塌缩参考态

    gates = [Gate(name="H", targets=[0]),
             Gate(name="MEASURE", targets=[0], params=[0])]
    ev = evolve_gates_branches(2, gates, branch={0: 1})
    ours = ev.frames[-1]
    P = _projector_qubit(2, 0, 1)
    ref = P @ psi
    ref /= np.linalg.norm(ref)
    assert np.allclose(ours, ref, atol=TOL)
    # 概率记录 = 0.5（|+⟩ 的 q0=1 分量）
    assert abs(ev.measurements[0][2] - 0.5) < TOL


def test_collapse_n1_edge():
    """单 qubit 塌缩（n=1 边界路径）。"""
    gates = [Gate(name="H", targets=[0]),
             Gate(name="MEASURE", targets=[0], params=[0])]
    ev = evolve_gates_branches(1, gates, branch={0: 1})
    assert np.allclose(ev.frames[-1], [0, 1], atol=TOL)
    assert abs(ev.measurements[0][2] - 0.5) < TOL


def test_reset_prepares_ground():
    """H|0⟩（叠加态）后 RESET：q0 塌缩回 |0⟩。

    纯态模型的 RESET = 向 |0⟩ 分支投影（与 Aer reset 的耗散通道不同，
    从确定的 |1⟩ 出发零概率——该极限由噪声模块的 amplitude_damping(1.0) 覆盖）。
    """
    gates = [Gate(name="H", targets=[0]), Gate(name="RESET", targets=[0])]
    ev = evolve_gates_branches(2, gates, branch={})
    assert np.allclose(ev.frames[-1], [1, 0, 0, 0], atol=TOL)


def test_zero_probability_branch_raises():
    """X|0⟩=|1⟩ 后测 q0=0：零概率分支必须报错。"""
    gates = [Gate(name="X", targets=[0]),
             Gate(name="MEASURE", targets=[0], params=[0])]
    with pytest.raises(ValueError, match="zero probability"):
        evolve_gates_branches(1, gates, branch={0: 0})


def test_missing_clbit_raises():
    with pytest.raises(ValueError, match="branch missing clbit"):
        evolve_gates_branches(
            1, [Gate(name="MEASURE", targets=[0], params=[0])], branch={})
    gates = [Gate(name="X", targets=[0], condition=Condition(clbit=0, value=1))]
    with pytest.raises(ValueError, match="does not specify"):
        evolve_gates_branches(1, gates, branch={})


# --------------------------------------------------------------------------- #
# 条件门
# --------------------------------------------------------------------------- #

def _cond_gates(apply_value: int) -> list[Gate]:
    return [
        Gate(name="X", targets=[0]),
        Gate(name="MEASURE", targets=[0], params=[0]),
        Gate(name="CX", targets=[1], controls=[0],
             condition=Condition(clbit=0, value=apply_value)),
    ]


def test_conditional_applies_when_true():
    """X|0⟩=|1⟩ → 确定测得 1 → 条件 CX 施加 → |11>。"""
    ev = evolve_gates_branches(2, _cond_gates(1), branch={0: 1})
    assert np.allclose(ev.frames[-1], [0, 0, 0, 1], atol=TOL)
    assert abs(ev.measurements[0][2] - 1.0) < TOL


def test_conditional_skips_when_false():
    """条件值设为 0：测量结果必为 1 → 条件不成立 → CX 跳过 → |10>。"""
    ev = evolve_gates_branches(2, _cond_gates(0), branch={0: 1})
    assert np.allclose(ev.frames[-1], [0, 1, 0, 0], atol=TOL)  # |01>：q0=1


# --------------------------------------------------------------------------- #
# 隐形传态：四分支终点态不变（P3.2 的标志性用例）
# --------------------------------------------------------------------------- #

def _teleport_gates() -> list[Gate]:
    """ψ = RZ(0.4)·RY(0.9)|0⟩ 在 q0；Bell 对 q1-q2；
    Bell 测量 q0,q1 → c0,c1；条件修正 X(q2)←c1，Z(q2)←c0。"""
    return [
        Gate(name="RY", targets=[0], params=[0.9]),
        Gate(name="RZ", targets=[0], params=[0.4]),
        Gate(name="H", targets=[1]),
        Gate(name="CX", targets=[2], controls=[1]),
        Gate(name="CX", targets=[1], controls=[0]),
        Gate(name="H", targets=[0]),
        Gate(name="MEASURE", targets=[0], params=[0]),
        Gate(name="MEASURE", targets=[1], params=[1]),
        Gate(name="X", targets=[2], condition=Condition(clbit=1, value=1)),
        Gate(name="Z", targets=[2], condition=Condition(clbit=0, value=1)),
    ]


@pytest.mark.parametrize("b0", [0, 1])
@pytest.mark.parametrize("b1", [0, 1])
def test_teleportation_all_branches(b0, b1):
    """任意测量分支下 q2 的约化态都等于初始 ψ（隐形传态的物理判据）。"""
    ev = evolve_gates_branches(3, _teleport_gates(),
                               branch={0: b0, 1: b1})
    final = ev.frames[-1]
    from gpuqviz.state import partial_trace

    rho2 = partial_trace(np.outer(final, final.conj()), [2], 3)
    # 初始 ψ（无噪声）：RZ(0.4)RY(0.9)|0>
    qc = QuantumCircuit(1)
    qc.ry(0.9, 0)
    qc.rz(0.4, 0)
    psi_ref = np.asarray(QkSV.from_instruction(qc).data)
    ref_rho = np.outer(psi_ref, psi_ref.conj())
    assert np.allclose(rho2, ref_rho, atol=1e-10), f"branch=({b0},{b1})"


def test_teleportation_branch_probabilities():
    """四分支概率合计 = 1（Bell 测量完备性）。"""
    total = 0.0
    for b0 in (0, 1):
        for b1 in (0, 1):
            ev = evolve_gates_branches(3, _teleport_gates(),
                                       branch={0: b0, 1: b1})
            p01 = ev.measurements[0][2]
            p1 = ev.measurements[1][2]
            # 条件概率链：P(b0)·P(b1|b0)
            total += p01 * p1
    assert abs(total - 1.0) < 1e-10


# --------------------------------------------------------------------------- #
# Aer IfElseOp 对拍（确定概率分支）
# --------------------------------------------------------------------------- #

def test_conditional_matches_aer_deterministic_branch():
    """X|0⟩ → 测量（结果确定为 1）→ 条件 CX：与 Aer IfElseOp 末态一致。"""
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import SaveDensityMatrix

    qreg, creg = QuantumRegister(2, "q"), ClassicalRegister(1, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.x(qreg[0])
    qc.measure(qreg[0], creg[0])
    true_body = QuantumCircuit(qreg, creg)
    true_body.cx(qreg[0], qreg[1])
    qc.append(IfElseOp((creg[0], 1), true_body, None),
              [qreg[0], qreg[1]], [creg[0]])
    qc.append(SaveDensityMatrix(2), [0, 1])

    sim = AerSimulator(method="density_matrix")
    ref = np.asarray(sim.run(qc, shots=1).result().data(0)["density_matrix"])

    ev = evolve_gates_branches(2, _cond_gates(1), branch={0: 1})
    ours_rho = np.outer(ev.frames[-1], ev.frames[-1].conj())
    assert np.allclose(ours_rho, ref, atol=1e-9)
