"""约定契约测试（docs/conventions.md 各节逐条锁定）。"""

import numpy as np
import pytest

from gpuqviz.analysis import exact_probs
from gpuqviz.analysis.measurement import Counts
from gpuqviz.render.heatmap import state_to_image
from gpuqviz.state import Statevector

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector as QkSV  # noqa: E402


# --------------------------------------------------------------------------- #
# §1 端序与位串归属
# --------------------------------------------------------------------------- #

def test_bitstring_matches_qiskit_probabilities_dict():
    """位串 'q_{n-1}…q_0' 归属与 qiskit probabilities_dict 逐项一致。"""
    qc = QuantumCircuit(3)
    qc.x(1)          # |010>（q1=1）
    qc.h(2)          # q2 叠加 → |110> / |010>
    sv = Statevector(np.asarray(QkSV.from_instruction(qc).data))
    qk_dict = QkSV(sv.data).probabilities_dict()

    ours = {format(i, "03b"): p for i, p in enumerate(exact_probs(sv)) if p > 0}
    # qiskit 字典键即同一位串约定
    assert set(ours) == set(qk_dict), f"{set(ours)} vs {set(qk_dict)}"
    for bs, p in ours.items():
        assert abs(p - qk_dict[bs]) <= 1e-10


def test_counts_bitstring_is_qiskit_style():
    """sample_counts 的位串键与 qiskit 概率字典的键空间一致。"""
    from gpuqviz.analysis import sample_counts

    qc = QuantumCircuit(3)
    qc.x(0)   # q0=1 → 位串最右位是 1
    sv = Statevector(np.asarray(QkSV.from_instruction(qc).data))
    c = sample_counts(sv, shots=100, seed=1)
    assert set(c.counts) == {"001"}  # q2q1q0 = 001


def test_counts_marginal_keeps_relative_order():
    """marginal([0,2]) 的位串是 'q2 q0'（保持高→低相对顺序）。"""
    counts = Counts({"0110": 10, "0010": 30}, shots=40, n_qubits=4)
    m = counts.marginal([0, 2])
    # |0110>: q2=1,q0=0 → "10"；|0010>: q2=0,q0=0 → "00"
    assert m.counts == {"10": 10, "00": 30}
    assert m.shots == 40 and m.n_qubits == 2


# --------------------------------------------------------------------------- #
# §4 热图网格排列
# --------------------------------------------------------------------------- #

def test_heatmap_grid_index_mapping():
    """index i → (row = i // cols, col = i % cols)，probability 基为原始 p_i。"""
    # n=2：cols=2, rows=2；|01>（index 1）→ img[0,1] = 1
    psi = np.zeros(4, dtype=complex)
    psi[1] = 1.0
    img = state_to_image(psi, basis="probability")
    assert img.shape == (2, 2)
    assert abs(img[0, 1] - 1.0) < 1e-9
    assert abs(float(img.sum()) - 1.0) < 1e-6  # 概率基总和 = 1

    # n=3：cols = 2^ceil(3/2) = 4, rows = 2；|101>（index 5）→ img[1, 1]
    psi3 = np.zeros(8, dtype=complex)
    psi3[5] = 1.0
    img3 = state_to_image(psi3, basis="probability")
    assert img3.shape == (2, 4)
    assert abs(img3[1, 1] - 1.0) < 1e-9


def test_heatmap_probability_values_match_exact():
    psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    img = state_to_image(psi, basis="probability")
    assert np.allclose(img, [[0.25, 0.25], [0.25, 0.25]], atol=1e-12)


# --------------------------------------------------------------------------- #
# §2 全局相位 / 归一化
# --------------------------------------------------------------------------- #

def test_normalize_preserves_global_phase():
    psi = np.array([1, 1], dtype=complex) * np.exp(1j * 0.3)
    psi /= np.linalg.norm(psi)
    out = Statevector(psi).normalize()
    # 全局相位不变：逐位比值恒定
    ratio = np.asarray(out.data) / np.asarray(psi)
    assert np.allclose(ratio, ratio[0], atol=1e-12)
    assert abs(np.linalg.norm(np.asarray(out.data)) - 1.0) <= 1e-12


def test_fidelity_ignores_global_phase():
    from gpuqviz.analysis import fidelity

    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rotated = psi * np.exp(1j * 2.5)
    assert abs(fidelity(psi, rotated) - 1.0) <= 1e-12


# --------------------------------------------------------------------------- #
# §6 容差自检（作为活文档的冒烟断言）
# --------------------------------------------------------------------------- #

def test_gpu_cpu_exact_parity_if_cupy():
    """cupy 可用时：GPU/CPU 精确量逐位一致（§6）。"""
    cpy = pytest.importorskip("cupy")
    from gpuqviz.analysis import fidelity

    psi_np = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    phi_np = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    f_np = fidelity(psi_np, phi_np)
    f_cp = fidelity(cpy.asarray(psi_np), cpy.asarray(phi_np))
    assert f_np == f_cp  # 逐位相等，不是 allclose


def test_state_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        Statevector(np.zeros(5, dtype=complex))
    with pytest.raises(ValueError, match="power of 2"):
        from gpuqviz.state import DensityMatrix as DM
        DM(np.zeros((5, 5), dtype=complex))
