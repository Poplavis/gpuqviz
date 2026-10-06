"""Shor 周期查找 / HHL 线性求解的交叉验证（容差 1e-10，docs/conventions.md §6）。

覆盖：
- Shor（N=15, a∈{2,4}）：QPE 峰位置/峰概率精确值、经典后处理得到因子、
  两引擎（qiskit cswap 链 vs numpy UNITARY 矩阵）全态一致；
- HHL（对角 A）：P(ancilla=1) 与理论 Σ(C/λ_k)²|b_k|²/‖b‖² 精确一致、
  ancilla=1 分支的 input 态 ≡ A⁻¹|b⟩（fidelity 1-1e-10）、clock 解缠、
  两引擎全态一致、含负振幅的 b。
- MCRY 门回归（MCRY = MC + RY 前缀剥离）：受控旋转矩阵 vs qiskit 算符。
"""

import math

import numpy as np
import pytest

from gpuqviz.algorithms._hhl import (
    _prep_unitary,
    hhl_classical_solution,
    hhl_conditional_state,
    hhl_circuit,
    hhl_eigenvalues,
)
from gpuqviz.algorithms._shor import (
    _iqft_matrix,
    continued_fraction_period,
    factor_from_period,
    shor_period_finding,
)
from gpuqviz.circuits import Gate, apply_gate

TOL = 1e-10


# --------------------------------------------------------------------------- #
# Shor：QPE 峰 + 经典后处理
# --------------------------------------------------------------------------- #

def _evolve(gates: list[Gate], n_qubits: int) -> np.ndarray:
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[0] = 1.0
    for g in gates:
        state = apply_gate(state, g)
    return state


@pytest.mark.parametrize("a,t_bits,r", [(2, 6, 4), (4, 6, 2)])
def test_shor_peaks_exact(a, t_bits, r):
    """周期 r 整除 2^t → 峰精确落在 s = m·2^t/r，峰概率恰为 1/r。"""
    state = _evolve(shor_period_finding(15, a, t_bits, engine="numpy"),
                    t_bits + 4)
    probs = (np.abs(state) ** 2).reshape(1 << 4, -1).sum(axis=0)
    period = (1 << t_bits) // r  # 峰间距
    expected = {m * period for m in range(r)}
    # 峰位置与概率精确
    for m in range(r):
        assert probs[m * period] == pytest.approx(1.0 / r, abs=TOL)
    # 非峰位置概率为 0
    off = probs.copy()
    for s in expected:
        off[s] = 0.0
    assert np.abs(off).max() == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize("a", [2, 4])
def test_shor_classical_postprocessing(a):
    """采样峰 → 连分数周期 → gcd 因子。峰概率质量 1（r 整除 2^t，无采样失败）。"""
    state = _evolve(shor_period_finding(15, a, 6, engine="numpy"), 10)
    probs = (np.abs(state) ** 2).reshape(16, 64).sum(axis=0)
    rng = np.random.default_rng(7)
    samples = rng.choice(64, size=16, p=probs / probs.sum())
    factors = None
    for s in samples:
        r = continued_fraction_period(int(s), 6, 15)
        factors = factor_from_period(15, a, r)
        if factors is not None:
            break
    assert factors is not None
    assert sorted(factors) == [3, 5]


def test_iqft_matrix_matches_qiskit():
    """_iqft_matrix ≡ qiskit QFT(inverse=True, do_swaps=True)（LSB 约定一致）。"""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFT
    from qiskit.quantum_info import Operator

    for bits in (2, 3, 4):
        qc = QuantumCircuit(bits)
        qc.append(QFT(bits, inverse=True, do_swaps=True), range(bits))
        assert np.allclose(_iqft_matrix(bits),
                           np.asarray(Operator(qc).data), atol=TOL)


# --------------------------------------------------------------------------- #
# Shor：两引擎对拍
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("a", [2, 4])
def test_shor_engines_agree(a):
    """qiskit cswap 链 ≡ numpy UNITARY 矩阵（全态 1e-10；算法可达子空间内严格等价）。"""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    qc = shor_period_finding(15, a, 6, engine="qiskit")
    sv = np.asarray(Statevector(qc.remove_final_measurements(inplace=False)).data)
    np_state = _evolve(shor_period_finding(15, a, 6, engine="numpy"), 10)
    assert np.abs(sv - np_state).max() == pytest.approx(0.0, abs=TOL)


# --------------------------------------------------------------------------- #
# HHL：解 + 概率 + 解缠
# --------------------------------------------------------------------------- #

def test_hhl_solution_fidelity():
    """ancilla=1 分支的 input 态 ≡ A⁻¹|b⟩ 归一化（fidelity 1-1e-10）。"""
    b = np.array([1.0, 0.5, 2.0, 1.0])
    gates, meta = hhl_circuit(b, clock_bits=3, C=0.6, engine="numpy")
    state = _evolve(gates, 6)
    sol, p1 = hhl_conditional_state(state, n_input=2, clock_bits=3)
    classical = hhl_classical_solution(b, np.array(meta["lambdas"])[:4])
    assert abs(np.vdot(sol, classical)) ** 2 == pytest.approx(1.0, abs=TOL)


def test_hhl_ancilla_probability():
    """P(ancilla=1) = Σ (C/λ_k)²·|b_k|²/‖b‖²（条件旋转的精确理论值）。"""
    b = np.array([1.0, 0.5, 2.0, 1.0])
    C = 0.6
    gates, meta = hhl_circuit(b, clock_bits=3, C=C, engine="numpy")
    state = _evolve(gates, 6)
    _, p1 = hhl_conditional_state(state, n_input=2, clock_bits=3)
    lam = np.array(meta["lambdas"])[:4]
    bn = b / np.linalg.norm(b)
    assert p1 == pytest.approx(float(np.sum((C / lam) ** 2 * bn ** 2)), abs=TOL)


def test_hhl_clock_disentangled():
    """uncompute 后 clock 寄存器回到 |0…0⟩（否则解退相干成等幅混合）。"""
    b = np.array([1.0, 0.5, 2.0, 1.0])
    gates, _ = hhl_circuit(b, clock_bits=3, C=0.6, engine="numpy")
    state = _evolve(gates, 6)
    probs = np.abs(state) ** 2
    # 布局：input = bit 0-1，clock = bit 2-4，ancilla = bit 5
    clock_probs = probs.reshape(2, 8, 4).sum(axis=(0, 2))  # [anc, clock, input]
    assert clock_probs[0] == pytest.approx(1.0, abs=TOL)


def test_hhl_negative_amplitudes():
    """b 含负振幅：解的相位结构一致（总体相位无意义，对齐后 fidelity 1）。"""
    b = np.array([1.0, -0.5, 0.0, 2.0])
    gates, meta = hhl_circuit(b, clock_bits=3, C=0.6, engine="numpy")
    state = _evolve(gates, 6)
    sol, _ = hhl_conditional_state(state, n_input=2, clock_bits=3)
    classical = hhl_classical_solution(b, np.array(meta["lambdas"])[:4])
    ov = np.vdot(sol, classical)
    aligned = sol * np.exp(-1j * np.angle(ov))
    assert abs(np.vdot(aligned, classical)) ** 2 == pytest.approx(1.0, abs=TOL)


def test_hhl_prep_unitary():
    """态制备矩阵是真酉矩阵且第 0 列 = |b⟩/‖b‖。"""
    rng = np.random.default_rng(3)
    v = rng.normal(size=8) + 1j * rng.normal(size=8)
    U = _prep_unitary(v)
    assert np.allclose(U.conj().T @ U, np.eye(8), atol=TOL)
    assert np.allclose(U[:, 0], v / np.linalg.norm(v), atol=TOL)


# --------------------------------------------------------------------------- #
# HHL：两引擎对拍
# --------------------------------------------------------------------------- #

def test_hhl_engines_agree():
    """qiskit 单元门分解 ≡ numpy UNITARY 矩阵（全态 1e-10）。"""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    b = np.array([1.0, 0.5, 2.0, 1.0])
    qc, _ = hhl_circuit(b, clock_bits=3, C=0.6, engine="qiskit")
    sv = np.asarray(Statevector(qc).data)
    gates, _ = hhl_circuit(b, clock_bits=3, C=0.6, engine="numpy")
    np_state = _evolve(gates, 6)
    assert np.abs(sv - np_state).max() == pytest.approx(0.0, abs=TOL)


def test_hhl_eigenvalues_exact_qpe():
    """λ_m·t·2^c/2π = m+1 ∈ ℤ → QPE 无近似误差（构造性保证）。"""
    for clock_bits in (3, 4):
        lam = hhl_eigenvalues(clock_bits, t=1.0)
        assert np.allclose(lam * (1 << clock_bits) / (2 * np.pi),
                           np.arange(1, 2 ** clock_bits + 1), atol=TOL)


# --------------------------------------------------------------------------- #
# MCRY 门回归（MCRY = MC + RY；此前被错剥成 MCR + Y）
# --------------------------------------------------------------------------- #

def test_mcry_controlled_rotation():
    """MCRY 在控制满足时施加 RY(θ)，未满足时恒等。"""
    theta = math.pi / 3
    state = np.zeros(8, dtype=np.complex128)
    state[0b110] = 1.0  # controls=1,1 / target=0
    out = apply_gate(state, Gate(name="MCRY", targets=[0], controls=[1, 2],
                                 params=[theta]))
    p1 = sum(abs(out[i]) ** 2 for i in range(8) if i & 1)
    assert p1 == pytest.approx(math.sin(theta / 2) ** 2, abs=TOL)
    # 控制不满足 → 恒等
    state0 = np.zeros(8, dtype=np.complex128)
    state0[0b000] = 1.0
    out0 = apply_gate(state0, Gate(name="MCRY", targets=[0], controls=[1, 2],
                                   params=[theta]))
    assert np.abs(out0 - state0).max() == pytest.approx(0.0, abs=TOL)


def test_mcry_matches_qiskit():
    """MCRY(θ) 矩阵 ≡ qiskit RYGate(θ).control(2)（qargs 顺序一致）。"""
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.library import RYGate
    from qiskit.quantum_info import Operator

    theta = math.pi / 5
    # 本地参考矩阵
    ref = {}
    for col in range(8):
        s = np.zeros(8, dtype=np.complex128)
        s[col] = 1.0
        ref[col] = apply_gate(s, Gate(name="MCRY", targets=[0],
                                      controls=[1, 2], params=[theta]))
    M = np.column_stack([ref[c] for c in range(8)])
    qc = qiskit.QuantumCircuit(3)
    qc.append(RYGate(theta).control(2), [1, 2, 0])
    assert np.allclose(M, np.asarray(Operator(qc).data), atol=TOL)
