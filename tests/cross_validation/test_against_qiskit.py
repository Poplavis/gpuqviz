"""精确量对拍 qiskit.quantum_info（容差 1e-10，docs/conventions.md §6）。

每个分析量与参考实现逐一对比；本文件是"没有对拍不允许合入"的载体。
"""

import numpy as np
import pytest

from gpuqviz.analysis import (exact_probs, fidelity, linear_entropy,
                              pauli_expectation, purity, state_table)
from gpuqviz.state import DensityMatrix, Statevector

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import (  # noqa: E402
    DensityMatrix as QkDensityMatrix,
    SparsePauliOp,
    Statevector as QkStatevector,
    entropy as qk_entropy,
    partial_trace as qk_partial_trace,
    state_fidelity as qk_state_fidelity,
)

TOL = 1e-10


def _bell_sv() -> Statevector:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return Statevector(np.asarray(QkStatevector.from_instruction(qc).data))


def _ghz_sv() -> Statevector:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return Statevector(np.asarray(QkStatevector.from_instruction(qc).data))


def _product_sv() -> Statevector:
    """|+⟩⊗|0⟩（2 qubit 直积态，用于与 Bell 同维对比）。"""
    qc = QuantumCircuit(2)
    qc.h(0)
    return Statevector(np.asarray(QkStatevector.from_instruction(qc).data))


def _random_mixed(seed: int, n: int = 2) -> DensityMatrix:
    """随机酉混合构造的合法密度矩阵。"""
    rng = np.random.Generator(np.random.PCG64(seed))
    dim = 2 ** n
    mats = []
    for _ in range(4):
        m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        q, _ = np.linalg.qr(m)
        w = rng.random(dim)
        w /= w.sum()
        mats.append(QkDensityMatrix((q * w) @ q.conj().T))
    rho = sum(m.data for m in mats) / len(mats)
    return DensityMatrix(rho)


# --------------------------------------------------------------------------- #
# 概率
# --------------------------------------------------------------------------- #

def test_probs_against_qiskit():
    for sv in (_bell_sv(), _ghz_sv()):
        qk_p = QkStatevector(sv.data).probabilities()
        assert np.allclose(exact_probs(sv), qk_p, atol=TOL)


def test_probs_mixed_against_qiskit():
    rho = _random_mixed(7)
    qk_p = QkDensityMatrix(rho.data).probabilities()
    assert np.allclose(exact_probs(rho), qk_p, atol=TOL)


# --------------------------------------------------------------------------- #
# Pauli 期望（位串记法对拍 SparsePauliOp）
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label", ["IZ", "ZI", "XX", "YY", "XZ", "ZZ"])
def test_pauli_expectation_pure(label):
    sv = _bell_sv()
    qk = QkStatevector(sv.data).expectation_value(SparsePauliOp(label)).real
    ours = pauli_expectation(sv, label)
    assert abs(ours - qk) < TOL


def test_pauli_expectation_mixed():
    rho = _random_mixed(11)
    for label in ("II", "IX", "ZI", "XY", "ZZ"):
        qk = QkDensityMatrix(rho.data).expectation_value(SparsePauliOp(label)).real
        assert abs(pauli_expectation(rho, label) - qk) < TOL


def test_pauli_expectation_identity_is_one():
    sv = _ghz_sv()
    assert abs(pauli_expectation(sv, "III") - 1.0) < TOL


def test_pauli_expectation_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        pauli_expectation(_bell_sv(), "ZZZ")


# --------------------------------------------------------------------------- #
# 保真度（三种类型组合，对拍 StateFidelity）
# --------------------------------------------------------------------------- #

def test_fidelity_pure_pure():
    a, b = _bell_sv(), _product_sv()
    qk = qk_state_fidelity(QkStatevector(a.data), QkStatevector(b.data))
    assert abs(fidelity(a, b) - qk) < TOL


def test_fidelity_pure_mixed():
    sv, rho = _bell_sv(), _random_mixed(3)
    qk = qk_state_fidelity(QkStatevector(sv.data), QkDensityMatrix(rho.data))
    assert abs(fidelity(sv, rho) - qk) < TOL
    # 参数顺序不影响（数学上纯混保真度对称）
    assert abs(fidelity(rho, sv) - qk) < TOL


def test_fidelity_mixed_mixed():
    ra, rb = _random_mixed(5), _random_mixed(6)
    qk = qk_state_fidelity(QkDensityMatrix(ra.data), QkDensityMatrix(rb.data))
    ours = fidelity(ra, rb)
    assert abs(ours - qk) < 1e-8  # Uhlmann 谱路径的数值噪声略高于代数路径
    assert abs(fidelity(rb, ra) - ours) < 1e-8  # 对称


def test_fidelity_self_is_one():
    sv, rho = _ghz_sv(), _random_mixed(9)
    assert abs(fidelity(sv, sv) - 1.0) < TOL
    assert abs(fidelity(rho, rho) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# 纯度 / 线性熵 / 熵
# --------------------------------------------------------------------------- #

def test_purity_against_qiskit():
    sv = _ghz_sv()
    rho = _random_mixed(13)
    assert abs(purity(sv) - 1.0) < TOL
    assert abs(purity(rho) - QkDensityMatrix(rho.data).purity()) < TOL


def test_linear_entropy_matches_one_minus_purity():
    rho = _random_mixed(17)
    assert abs(linear_entropy(rho) - (1 - purity(rho))) < TOL
    # 归一化变体：d/(d-1) 因子
    d = 4
    assert abs(linear_entropy(rho, normalized=True)
               - d / (d - 1) * linear_entropy(rho)) < TOL


def test_entropy_against_qiskit():
    """子系统约化态的 von Neumann 熵对拍 qiskit.entropy。"""
    sv = _ghz_sv()
    qk_sv = QkStatevector(sv.data)
    for keep in ([0], [1], [2], [0, 1]):
        # qiskit: partial_trace 的第二参数是要迹掉的 qubit
        traced = [q for q in range(3) if q not in keep]
        qk_rho = qk_partial_trace(qk_sv, traced)
        qk_s = qk_entropy(qk_rho)
        ours = sv.entropy(keep)
        assert abs(ours - qk_s) < TOL, f"keep={keep}: {ours} vs {qk_s}"


def test_entropy_bell_is_one_bit():
    assert abs(_bell_sv().entropy([0]) - 1.0) < TOL
    assert abs(_bell_sv().entropy([1]) - 1.0) < TOL
    # 整体纯态熵为 0
    assert abs(_bell_sv().entropy() - 0.0) < TOL


# --------------------------------------------------------------------------- #
# 约化密度矩阵对拍
# --------------------------------------------------------------------------- #

def test_reduced_density_matrix_against_qiskit():
    sv = _ghz_sv()
    qk_sv = QkStatevector(sv.data)
    for keep in ([0], [1], [2], [0, 2]):
        traced = [q for q in range(3) if q not in keep]
        qk_rho = np.asarray(qk_partial_trace(qk_sv, traced).data)
        ours = np.asarray(sv.reduced(keep).data)
        assert np.allclose(ours, qk_rho, atol=TOL), f"keep={keep}"


def test_reduced_density_matrix_mixed():
    rho = _random_mixed(19, n=3)
    traced = [2]
    qk_rho = np.asarray(qk_partial_trace(QkDensityMatrix(rho.data), traced).data)
    ours = np.asarray(rho.reduced([0, 1]).data)
    assert np.allclose(ours, qk_rho, atol=TOL)


# --------------------------------------------------------------------------- #
# 态表格
# --------------------------------------------------------------------------- #

def test_state_table_bell():
    table = state_table(_bell_sv())
    rows = {bs: (re_, im_, p, ph) for _, bs, re_, im_, p, ph in table.rows()}
    # Bell 态：|00> 与 |11> 各 0.5，振幅 1/√2 实数，相位 0
    assert abs(rows["00"][2] - 0.5) < TOL
    assert abs(rows["11"][2] - 0.5) < TOL
    assert abs(rows["00"][0] - 1 / np.sqrt(2)) < TOL
    assert rows["01"][2] == 0.0
    assert abs(rows["00"][3]) < TOL  # 相位 0
    # 排序：概率降序
    probs_order = [r[4] for r in table.rows()]
    assert probs_order == sorted(probs_order, reverse=True)


def test_state_table_csv_roundtrip():
    table = state_table(_ghz_sv())
    csv = table.to_csv()
    lines = csv.strip().split("\n")
    assert lines[0] == "index,bitstring,re,im,prob,phase_deg"
    assert len(lines) == 9  # 表头 + 8 行
    # GHZ：|000> 与 |111> 各 0.5
    assert any(",000," in ln and ",0.5," in ln for ln in lines)


def test_state_table_mixed_has_no_amplitude():
    table = state_table(_random_mixed(23))
    row = table.rows()[0]
    assert row[2] is None and row[3] is None
