"""性质测试：种子化随机电路上断言数学不变量（docs/conventions.md §6/§7）。

hypothesis 未安装时的替代方案：固定种子的随机电路批量断言。
将来引入 hypothesis 后，本文件的 `_random_circuit` 可替换为策略生成器，
断言体保持不变。
"""

import numpy as np
import pytest

from gpuqviz.analysis import exact_probs, fidelity, purity, sample_counts
from gpuqviz.state import DensityMatrix, Statevector, partial_trace

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector as QkSV

TOL = 1e-10


def _random_circuit(seed: int, n: int = 3, depth: int = 6) -> QuantumCircuit:
    rng = np.random.Generator(np.random.PCG64(seed))
    qc = QuantumCircuit(n)
    for _ in range(depth):
        q = int(rng.integers(n))
        gate = rng.integers(4)
        if gate == 0:
            qc.h(q)
        elif gate == 1:
            qc.s(q)
        elif gate == 2:
            qc.ry(float(rng.uniform(0, np.pi)), q)
        elif n > 1:
            t = int(rng.integers(n))
            if t != q:
                qc.cx(q, t)
    return qc


def _random_mixed(seed: int, n: int = 2) -> DensityMatrix:
    rng = np.random.Generator(np.random.PCG64(seed))
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(3):
        m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        q, _ = np.linalg.qr(m)
        w = rng.random(dim)
        w /= w.sum()
        rho += (q * w) @ q.conj().T
    return DensityMatrix(rho / 3)


N_CIRCUITS = 12


# --------------------------------------------------------------------------- #
# 演化不变量
# --------------------------------------------------------------------------- #

def test_probabilities_normalized():
    for seed in range(N_CIRCUITS):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed)).data))
        p = exact_probs(sv)
        assert abs(p.sum() - 1.0) <= 1e-12
        assert (p >= -1e-15).all()


def test_pure_state_total_entropy_zero():
    for seed in range(N_CIRCUITS):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed)).data))
        assert abs(sv.entropy() - 0.0) <= 1e-9


def test_single_qubit_entropy_bounds():
    """随机纯态的任一单 qubit 熵 ∈ [0, 1] bit。"""
    for seed in range(N_CIRCUITS):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed, n=4)).data))
        for q in range(4):
            s = sv.entropy([q])
            assert -1e-9 <= s <= 1.0 + 1e-9, f"seed={seed} q={q}: S={s}"


def test_reduced_trace_one_and_hermitian():
    for seed in range(N_CIRCUITS):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed, n=4)).data))
        rho = np.asarray(sv.reduced([0, 2]).data)
        assert abs(np.trace(rho).real - 1.0) <= 1e-10
        assert np.allclose(rho, rho.conj().T, atol=1e-12)


def test_reduced_purity_bounds_pure_input():
    """纯态输入的单 qubit 约化纯度 ∈ [0.5, 1]。"""
    for seed in range(N_CIRCUITS):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed, n=3)).data))
        for q in range(3):
            p = sv.reduced([q]).purity()
            assert 0.5 - 1e-10 <= p <= 1.0 + 1e-10, f"seed={seed} q={q}: {p}"


def test_partial_trace_consistency_sv_vs_dm():
    """纯态经 partial_trace(rho) 与 reduced() 结果一致。"""
    sv = Statevector(np.asarray(
        QkSV.from_instruction(_random_circuit(99, n=3)).data))
    keep = [0, 2]
    rho_full = sv.density()
    via_generic = partial_trace(rho_full, keep, 3)
    via_sv = np.asarray(sv.reduced(keep).data)
    assert np.allclose(via_generic, via_sv, atol=1e-10)


def test_mixed_state_validity():
    for seed in range(6):
        rho = _random_mixed(seed, n=3)
        assert rho.is_valid()
        assert 0 < rho.purity() <= 1.0 + 1e-10


# --------------------------------------------------------------------------- #
# 保真度不变量
# --------------------------------------------------------------------------- #

def test_fidelity_bounds_and_symmetry():
    for seed in range(8):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed, n=2)).data))
        rho = _random_mixed(seed + 100, n=2)
        f_self = fidelity(sv, sv)
        assert abs(f_self - 1.0) <= 1e-10
        f_sr = fidelity(sv, rho)
        f_rs = fidelity(rho, sv)
        assert abs(f_sr - f_rs) <= 1e-9  # 纯混保真度对称
        assert -1e-12 <= f_sr <= 1.0 + 1e-9
        f_mm = fidelity(rho, _random_mixed(seed + 200, n=2))
        assert -1e-9 <= f_mm <= 1.0 + 1e-9


def test_fidelity_invariant_under_global_phase():
    sv = Statevector(np.asarray(
        QkSV.from_instruction(_random_circuit(7, n=2)).data))
    rotated = Statevector(np.asarray(sv.data) * np.exp(1j * 0.7))
    assert abs(fidelity(sv, rotated) - 1.0) <= 1e-10


# --------------------------------------------------------------------------- #
# Bloch 语义（模长 = 单 qubit 纯度换算）
# --------------------------------------------------------------------------- #

def test_bloch_length_matches_reduced_purity():
    """|r| = sqrt(2·Tr ρ_q² − 1)（conventions.md §3 换算式）。"""
    for seed in range(N_CIRCUITS):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed, n=3)).data))
        bloch = sv.bloch()
        for q in range(3):
            r_len = np.linalg.norm(bloch[q])
            p = sv.reduced([q]).purity()
            expected = np.sqrt(max(2 * p - 1, 0.0))
            assert abs(r_len - expected) <= 1e-9, f"seed={seed} q={q}"


def test_bloch_maximally_mixed_qubit_zero_vector():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector(np.asarray(QkSV.from_instruction(qc).data))
    bloch = sv.bloch()
    assert np.allclose(bloch[0], 0, atol=1e-10)
    assert np.allclose(bloch[1], 0, atol=1e-10)


# --------------------------------------------------------------------------- #
# 采样不变量
# --------------------------------------------------------------------------- #

def test_counts_invariants():
    for seed in range(5):
        sv = Statevector(np.asarray(
            QkSV.from_instruction(_random_circuit(seed, n=3)).data))
        c = sample_counts(sv, shots=5_000, seed=seed)
        assert sum(c.counts.values()) == c.shots
        assert abs(sum(c.probs.values()) - 1.0) <= 1e-12
        # 边际后总和不变
        m = c.marginal([0])
        assert sum(m.counts.values()) == c.shots
        # 边际位串长度 = 选中 qubit 数
        assert all(len(bs) == 1 for bs in m.counts)
