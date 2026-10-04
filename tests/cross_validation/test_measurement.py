"""采样统计对拍：可复现性、边际分布、5σ 二项界、AerSimulator 卡方一致性。"""

import numpy as np
import pytest

from gpuqviz.analysis import exact_probs, sample_counts
from gpuqviz.state import Statevector

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit, transpile  # noqa: E402


def _bell() -> Statevector:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    from qiskit.quantum_info import Statevector as QkSV

    return Statevector(np.asarray(QkSV.from_instruction(qc).data))


def _random_circuit_sv(seed: int, n: int = 3, depth: int = 6) -> Statevector:
    from qiskit.quantum_info import Statevector as QkSV

    rng = np.random.Generator(np.random.PCG64(seed))
    qc = QuantumCircuit(n)
    for _ in range(depth):
        q = int(rng.integers(n))
        gate = rng.integers(4)
        if gate == 0:
            qc.h(q)
        elif gate == 1:
            qc.t(q)
        elif gate == 2:
            qc.ry(float(rng.uniform(0, np.pi)), q)
        elif n > 1:
            t = int(rng.integers(n))
            if t != q:
                qc.cx(q, t)
    return Statevector(np.asarray(QkSV.from_instruction(qc).data))


# --------------------------------------------------------------------------- #
# 可复现性与结构
# --------------------------------------------------------------------------- #

def test_same_seed_same_counts():
    a = sample_counts(_bell(), shots=1000, seed=42)
    b = sample_counts(_bell(), shots=1000, seed=42)
    assert a.counts == b.counts


def test_different_seed_differs_statistically():
    a = sample_counts(_bell(), shots=1000, seed=1)
    b = sample_counts(_bell(), shots=1000, seed=2)
    # 不是逐位相同（PCG64 不同流），但分布一致——差值应在合理幅度
    assert a.counts != b.counts


def test_counts_sum_equals_shots():
    c = sample_counts(_random_circuit_sv(5), shots=12345, seed=7)
    assert sum(c.counts.values()) == c.shots == 12345
    assert all(0 <= len(bs) == c.n_qubits for bs in c.counts)
    for bs in c.counts:
        int(bs, 2)  # 位串合法


def test_counts_rejects_zero_shots():
    with pytest.raises(ValueError, match="shots"):
        sample_counts(_bell(), shots=0, seed=1)


# --------------------------------------------------------------------------- #
# 统计正确性：5σ 二项界
# --------------------------------------------------------------------------- #

def _assert_within_5sigma(counts, exact_p, n):
    """逐基矢：|freq − p| ≤ 5·sqrt(p(1-p)/N)。p=0 的基矢必须零计数。"""
    for i in range(2 ** n):
        bs = format(i, f"0{n}b")
        freq = counts.counts.get(bs, 0) / counts.shots
        p = exact_p[i]
        if p == 0:
            assert freq == 0.0, f"{bs}: p=0 but freq={freq}"
            continue
        sigma = np.sqrt(p * (1 - p) / counts.shots)
        assert abs(freq - p) <= 5 * sigma, f"{bs}: |{freq}-{p}| > 5σ({sigma:.2e})"


def test_bell_sampling_within_5sigma():
    sv = _bell()
    p = exact_probs(sv)
    c = sample_counts(sv, shots=200_000, seed=2026)
    _assert_within_5sigma(c, p, 2)
    # Bell：只有 |00> 与 |11>，各约 0.5
    assert set(c.counts) == {"00", "11"}


def test_random_circuit_sampling_within_5sigma():
    for seed in (101, 202, 303):
        sv = _random_circuit_sv(seed)
        p = exact_probs(sv)
        c = sample_counts(sv, shots=100_000, seed=seed + 1)
        _assert_within_5sigma(c, p, 3)


# --------------------------------------------------------------------------- #
# 边际分布
# --------------------------------------------------------------------------- #

def test_marginal_bell_qubit0():
    c = sample_counts(_bell(), shots=100_000, seed=11)
    m = c.marginal([0])
    # Bell 态对 q0 是最大混合：'0'/'1' 各约 0.5
    assert m.n_qubits == 1
    f0 = m.counts.get("0", 0) / m.shots
    assert abs(f0 - 0.5) <= 5 * np.sqrt(0.25 / m.shots)
    assert sum(m.counts.values()) == m.shots


def test_marginal_matches_subspace_sampling():
    """先采样后边际 ≡ 直接对边际子系统采样（统计一致）。"""
    sv = _random_circuit_sv(77)
    full = sample_counts(sv, shots=80_000, seed=3).marginal([0, 2])
    direct = sample_counts(sv, shots=80_000, seed=3, qubits=[0, 2])
    assert full.n_qubits == direct.n_qubits == 2
    p_full = exact_probs(sv)
    # 两路经验分布都应贴近精确边际：合并卡方上界内一致
    for i in range(4):
        bs = format(i, "02b")
        f1 = full.counts.get(bs, 0) / full.shots
        f2 = direct.counts.get(bs, 0) / direct.shots
        # 位串映射：sel=[0,2] → 位串 "q2 q0"
        q2, q0 = int(bs[0]), int(bs[1])
        p_exact = sum(p for k, p in enumerate(p_full)
                      if ((k >> 2) & 1) == q2 and ((k >> 0) & 1) == q0)
        sigma = np.sqrt(max(p_exact * (1 - p_exact), 1e-12) / full.shots)
        assert abs(f1 - f2) <= 6 * sigma + 1e-9, f"bs={bs}: {f1} vs {f2}"


# --------------------------------------------------------------------------- #
# 对拍 AerSimulator
# --------------------------------------------------------------------------- #

def test_sampling_matches_aer():
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator

    n, depth, seed = 3, 6, 55
    rng = np.random.Generator(np.random.PCG64(seed))
    qc = QuantumCircuit(n, n)
    for _ in range(depth):
        q = int(rng.integers(n))
        gate = rng.integers(4)
        if gate == 0:
            qc.h(q)
        elif gate == 1:
            qc.t(q)
        elif gate == 2:
            qc.ry(float(rng.uniform(0, np.pi)), q)
        elif n > 1:
            t = int(rng.integers(n))
            if t != q:
                qc.cx(q, t)
    qc.measure(range(n), range(n))

    shots = 50_000
    sim = AerSimulator()
    result = sim.run(transpile(qc, sim), shots=shots,
                     seed_simulator=99).result()
    aer_counts = result.get_counts()
    # 位串规整（Aer 可能带空格）
    aer_counts = {bs.replace(" ", ""): int(v) for bs, v in aer_counts.items()}

    from qiskit.quantum_info import Statevector as QkSV

    sv = Statevector(np.asarray(QkSV.from_instruction(qc.remove_final_measurements(inplace=False)).data))
    ours = sample_counts(sv, shots=shots, seed=100)
    p = exact_probs(sv)

    # 双方经验频率都与精确概率在 5σ 内 ⇒ 彼此一致
    for i in range(2 ** n):
        bs = format(i, f"0{n}b")
        f_ours = ours.counts.get(bs, 0) / shots
        f_aer = aer_counts.get(bs, 0) / shots
        pi = p[i]
        if pi == 0:
            assert f_ours == 0.0 and f_aer == 0.0
            continue
        sigma = np.sqrt(pi * (1 - pi) / shots)
        assert abs(f_ours - pi) <= 5 * sigma, f"{bs} ours: |{f_ours}-{pi}|>5σ"
        assert abs(f_aer - pi) <= 5 * sigma, f"{bs} aer: |{f_aer}-{pi}|>5σ"
