"""纠缠结构分析：纠缠熵、互信息、Schmidt 分解、negativity。

公式定义（docs/conventions.md §7）：
- 纠缠熵 S(ρ_A) = −Tr(ρ_A log₂ ρ_A)（von Neumann，单位 bit）
- 互信息 I(A:B) = S_A + S_B − S_AB（≥0，两 qubit 子系统上限 2 bit）
- Schmidt 系数：|ψ⟩ = Σ s_i |i_A⟩|i_B⟩，Σ s_i² = 1；
  S_A = −Σ s_i² log₂ s_i²
- negativity N(ρ_AB) = (‖ρ_AB^{T_B}‖₁ − 1) / 2（2-qubit 可分性判据，
  Bell 态 N = 0.5，可分态 N = 0）

全部量与 qiskit.quantum_info / 手工参考实现对拍
（tests/cross_validation/test_entanglement.py）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state import State, Statevector, as_state, partial_trace

__all__ = ["entanglement_entropy", "mutual_information", "negativity",
           "schmidt_coefficients", "entanglement_summary", "EntanglementReport"]


def _reduced_rho(st: State, qubits: list[int]) -> np.ndarray:
    """子系统的约化密度矩阵（输出 index = q_max…q_min）。"""
    n = st.n_qubits
    if not qubits or any(q < 0 or q >= n for q in qubits):
        raise ValueError(f"qubits {qubits} out of range for {n} qubits")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"qubits {qubits} contains duplicates")
    if isinstance(st, Statevector):
        from ..state import _sv_reduced_rho
        return _sv_reduced_rho(np.asarray(st.data), sorted(qubits), n)
    return partial_trace(np.asarray(st.data), sorted(qubits), n)


def _entropy_of_rho(rho: np.ndarray) -> float:
    eig = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    eig = np.clip(eig.real, 0.0, None)
    nz = eig[eig > 1e-12]
    return float(-np.sum(nz * np.log2(nz)))


def entanglement_entropy(state, qubit: int) -> float:
    """qubit q 与其余部分的纠缠熵 S(ρ_q)，单位 bit。

    单 qubit 约化态最大熵 1 bit（最大纠缠 / 最大混合）。
    """
    st = as_state(state)
    return _entropy_of_rho(_reduced_rho(st, [int(qubit)]))


def mutual_information(state, q_a: int, q_b: int) -> float:
    """两 qubit 互信息 I(A:B) = S_A + S_B − S_AB（单位 bit）。

    Bell 态 I = 2（最大）；乘积态 I = 0；GHZ 的任意对 I = 1。
    """
    st = as_state(state)
    a, b = int(q_a), int(q_b)
    if a == b:
        raise ValueError("mutual_information needs two distinct qubits")
    s_a = _entropy_of_rho(_reduced_rho(st, [a]))
    s_b = _entropy_of_rho(_reduced_rho(st, [b]))
    s_ab = _entropy_of_rho(_reduced_rho(st, [a, b]))
    return float(s_a + s_b - s_ab)


def negativity(state, q_a: int, q_b: int) -> float:
    """两 qubit 约化态的部分转置负性 N = (‖ρ^{T_B}‖₁ − 1)/2。

    2-qubit 态的可分性充要判据（Peres-Horodecki）：N > 0 ⟺ 纠缠。
    Bell 态 N = 0.5。
    """
    st = as_state(state)
    a, b = int(q_a), int(q_b)
    if a == b:
        raise ValueError("negativity needs two distinct qubits")
    rho = _reduced_rho(st, sorted([a, b]))
    # rho 的 index = q_max·2 + q_min（q_max 是行/列的高位因子）。
    # 对"低位 qubit"做部分转置：reshape (q_hi, q_lo, q_hi', q_lo')
    # → 转置 (q_hi, q_lo', q_hi', q_lo) = 轴 (0, 3, 2, 1)
    t = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    eig = np.linalg.eigvalsh((t + t.conj().T) / 2)  # T_B 仍 Hermit
    l1 = float(np.sum(np.abs(eig)))
    return max((l1 - 1.0) / 2.0, 0.0)


def schmidt_coefficients(state, qubits_a: list[int]) -> np.ndarray:
    """给定割 A|B 的 Schmidt 系数（降序，Σ s² = 1）。

    qubits_a 为 A 侧 qubit 集合（要求非空且非全集）。
    纠缠熵 S_A = −Σ s_i² log₂ s_i²；乘积态只有一个非零系数 1。
    """
    st = as_state(state)
    if not isinstance(st, Statevector):
        raise TypeError("schmidt_coefficients requires a pure state")
    n = st.n_qubits
    keep = sorted(set(int(q) for q in qubits_a))
    if not keep or len(keep) == n:
        raise ValueError("qubits_a must be a proper non-empty subset")
    psi = np.asarray(st.data)
    t = psi.reshape((2,) * n)
    # 轴 a ↔ qubit n-1-a：A 侧轴聚到前部（降序高位在前，输出序与
    # partial_trace 一致），B 侧其余轴随后
    a_axes = [n - 1 - q for q in sorted(keep, reverse=True)]
    b_axes = [a for a in range(n) if a not in a_axes]
    t = np.moveaxis(t, a_axes + b_axes, list(range(n)))
    m = t.reshape(2 ** len(keep), -1)
    s = np.linalg.svd(m, compute_uv=False)
    return np.sort(s)[::-1]


@dataclass(frozen=True)
class EntanglementReport:
    """一次算全的纠缠结构报告（渲染层直接消费）。"""

    n_qubits: int
    single_entropy: np.ndarray   # (n,) 每 qubit 纠缠熵
    mutual_info: np.ndarray      # (n, n) 对称，对角 0
    negativity: np.ndarray       # (n, n) 对称，对角 0（仅 i<j 有效使用）

    def strongest_pair(self) -> tuple[int, int]:
        """互信息最大的 qubit 对（平局取编号最小者）。"""
        mi = self.mutual_info.copy()
        np.fill_diagonal(mi, -np.inf)
        idx = np.unravel_index(np.argmax(mi), mi.shape)
        return int(idx[0]), int(idx[1])

    def total_correlation(self) -> float:
        """成对互信息之和（纠缠"总量"的粗度量）。"""
        mi = self.mutual_info
        iu = np.triu_indices(self.n_qubits, k=1)
        return float(mi[iu].sum())

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "single_entropy": [float(x) for x in self.single_entropy],
            "mutual_info": [[float(x) for x in row] for row in self.mutual_info],
            "negativity": [[float(x) for x in row] for row in self.negativity],
            "strongest_pair": list(self.strongest_pair()),
            "total_correlation": self.total_correlation(),
        }


def entanglement_summary(state) -> EntanglementReport:
    """全对互信息 + 每 qubit 熵 + negativity 一次算全。

    复杂度 O(n²) 次 2-qubit partial trace（n ≤ 13 由 partial_trace 保证）。
    """
    st = as_state(state)
    n = st.n_qubits
    single = np.array([_entropy_of_rho(_reduced_rho(st, [q])) for q in range(n)])
    mi = np.zeros((n, n))
    neg = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mi[i, j] = mi[j, i] = mutual_information(st, i, j)
            neg[i, j] = neg[j, i] = negativity(st, i, j)
    return EntanglementReport(n_qubits=n, single_entropy=single,
                              mutual_info=mi, negativity=neg)
