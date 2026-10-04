"""纠缠量对拍：qiskit 参考实现 + 已知解析值（容差 1e-10）。"""

import numpy as np
import pytest

from gpuqviz.analysis import (entanglement_entropy, entanglement_summary,
                              mutual_information, negativity,
                              schmidt_coefficients)
from gpuqviz.state import DensityMatrix, Statevector

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import (  # noqa: E402
    DensityMatrix as QkDM,
    Statevector as QkSV,
    entropy as qk_entropy,
    partial_trace as qk_partial_trace,
)

TOL = 1e-10


def _sv_from_qc(qc: QuantumCircuit) -> Statevector:
    return Statevector(np.asarray(QkSV.from_instruction(qc).data))


def _bell() -> Statevector:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return _sv_from_qc(qc)


def _ghz(n: int = 3) -> Statevector:
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return _sv_from_qc(qc)


def _random_sv(seed: int, n: int = 3) -> Statevector:
    rng = np.random.Generator(np.random.PCG64(seed))
    qc = QuantumCircuit(n)
    for _ in range(8):
        q = int(rng.integers(n))
        g = rng.integers(3)
        if g == 0:
            qc.h(q)
        elif g == 1:
            qc.ry(float(rng.uniform(0, np.pi)), q)
        elif n > 1:
            t = int(rng.integers(n))
            if t != q:
                qc.cx(q, t)
    return _sv_from_qc(qc)


# --------------------------------------------------------------------------- #
# 解析已知值
# --------------------------------------------------------------------------- #

def test_bell_entropy_is_one_bit():
    assert abs(entanglement_entropy(_bell(), 0) - 1.0) < TOL
    assert abs(entanglement_entropy(_bell(), 1) - 1.0) < TOL


def test_bell_mutual_information_is_two_bits():
    # S_A = S_B = 1, S_AB = 0（纯态）→ I = 2（两 qubit 上限）
    assert abs(mutual_information(_bell(), 0, 1) - 2.0) < TOL


def test_bell_negativity_is_half():
    assert abs(negativity(_bell(), 0, 1) - 0.5) < TOL


def test_product_state_has_zero_entanglement():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(0.7, 1)
    sv = _sv_from_qc(qc)
    assert abs(entanglement_entropy(sv, 0)) < TOL
    assert abs(mutual_information(sv, 0, 1)) < TOL
    assert abs(negativity(sv, 0, 1)) < TOL


def test_ghz_pairwise_structure():
    sv = _ghz(3)
    for q in range(3):
        assert abs(entanglement_entropy(sv, q) - 1.0) < TOL
    for i in range(3):
        for j in range(i + 1, 3):
            # GHZ 任意对：S_i = S_j = 1, S_ij = 1 → I = 1；N = 0（可分对！）
            assert abs(mutual_information(sv, i, j) - 1.0) < TOL
            assert abs(negativity(sv, i, j)) < TOL


def test_w_state_entanglement_structure():
    """W 态：任意两 qubit 约化态为纠缠混合态 → negativity > 0。"""
    # |W> = (|100> + |010> + |001>)/√3
    psi = np.zeros(8, dtype=complex)
    psi[0b001] = psi[0b010] = psi[0b100] = 1 / np.sqrt(3)
    sv = Statevector(psi)
    assert negativity(sv, 0, 1) > 0.01
    # 单 qubit 熵 = -(2/3 log2(2/3) + 1/3 log2(1/3)) ≈ 0.9183
    expected = -(2 / 3 * np.log2(2 / 3) + 1 / 3 * np.log2(1 / 3))
    assert abs(entanglement_entropy(sv, 0) - expected) < TOL


# --------------------------------------------------------------------------- #
# qiskit 参考对拍
# --------------------------------------------------------------------------- #

def test_entanglement_entropy_against_qiskit():
    for seed in range(8):
        sv = _random_sv(seed)
        for q in range(3):
            qk_rho = qk_partial_trace(QkSV(sv.data), [r for r in range(3) if r != q])
            qk_s = qk_entropy(qk_rho)
            assert abs(entanglement_entropy(sv, q) - qk_s) < TOL, \
                f"seed={seed} q={q}"


def test_mutual_information_componentwise_against_qiskit():
    """MI 的三个组成熵分别对拍 qiskit（随机电路批量）。"""
    for seed in range(8):
        sv = _random_sv(seed)
        a, b = 0, 2
        qk = QkSV(sv.data)
        s_a = qk_entropy(qk_partial_trace(qk, [1, 2]))   # keep {0}
        s_b = qk_entropy(qk_partial_trace(qk, [0, 1]))   # keep {2}
        s_ab = qk_entropy(qk_partial_trace(qk, [1]))     # keep {0,2}
        ours = mutual_information(sv, a, b)
        assert abs(ours - (s_a + s_b - s_ab)) < TOL, f"seed={seed}"


def test_negativity_against_manual_partial_transpose():
    """negativity 对拍：qiskit 约化态 + numpy 手工部分转置（独立路径）。"""
    for seed in range(8):
        sv = _random_sv(seed)
        qk_rho = np.asarray(
            qk_partial_trace(QkSV(sv.data), [2]).data)  # keep {0,1}
        t = qk_rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
        eig = np.linalg.eigvalsh((t + t.conj().T) / 2)
        ref = max((float(np.sum(np.abs(eig))) - 1) / 2, 0.0)
        ours = negativity(sv, 0, 1)
        assert abs(ours - ref) < TOL, f"seed={seed}: {ours} vs {ref}"


def test_schmidt_vs_reduced_eigenvalues():
    """纯态 Schmidt 系数平方 = A 侧约化态特征值（数学恒等式）。"""
    for seed in range(8):
        sv = _random_sv(seed)
        for keep in ([0], [1], [0, 1]):
            s = schmidt_coefficients(sv, keep)
            assert abs(np.sum(s ** 2) - 1.0) < 1e-9
            rho = np.asarray(sv.reduced(keep).data)
            ref = np.sqrt(np.clip(np.linalg.eigvalsh(rho).real, 0, None))
            ref = np.sort(ref)[::-1]
            # Schmidt 谱长 = min(2^|A|, 2^|B|)，比 2^|A| 短时补零对齐
            s_full = np.concatenate([s, np.zeros(len(ref) - len(s))])
            # 零奇异值附近两种算法的浮点噪声量级不同（SVD ~1e-17，
            # 谱路径 ~1e-9），容差取 1e-7
            assert np.allclose(s_full, ref, atol=1e-7), f"seed={seed} keep={keep}"


def test_schmidt_bell_is_maximally_entangled():
    s = schmidt_coefficients(_bell(), [0])
    assert np.allclose(s, [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=TOL)
    s_prod = schmidt_coefficients(_sv_from_qc(QuantumCircuit(2)), [0])
    assert np.allclose(s_prod, [1.0, 0.0], atol=TOL)


def test_schmidt_rejects_mixed_and_full_set():
    rho = DensityMatrix(np.eye(4) / 4)
    with pytest.raises(TypeError):
        schmidt_coefficients(rho, [0])
    with pytest.raises(ValueError):
        schmidt_coefficients(_bell(), [0, 1])
    with pytest.raises(ValueError):
        schmidt_coefficients(_bell(), [])


# --------------------------------------------------------------------------- #
# EntanglementReport
# --------------------------------------------------------------------------- #

def test_entanglement_summary_report():
    rep = entanglement_summary(_ghz(3))
    assert rep.n_qubits == 3
    assert np.allclose(rep.single_entropy, 1.0, atol=TOL)
    # 对称矩阵 + 对角零
    assert np.allclose(rep.mutual_info, rep.mutual_info.T, atol=TOL)
    assert np.allclose(np.diag(rep.mutual_info), 0.0)
    # GHZ 最强对互信息 = 1
    i, j = rep.strongest_pair()
    assert abs(rep.mutual_info[i, j] - 1.0) < TOL
    assert abs(rep.total_correlation() - 3.0) < TOL  # 3 对 × 1 bit
    # to_dict 可序列化
    d = rep.to_dict()
    assert d["strongest_pair"] in ([0, 1], [0, 2], [1, 2])


def test_entanglement_summary_mixed_state():
    """混合态输入（密度矩阵）同样可用。"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    rho = QkDM(QkSV.from_instruction(qc)).data
    rep = entanglement_summary(DensityMatrix(np.asarray(rho)))
    assert abs(rep.mutual_info[0, 1] - 2.0) < TOL
    assert abs(rep.negativity[0, 1] - 0.5) < TOL
