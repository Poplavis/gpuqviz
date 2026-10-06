"""通用量子度量：纯度、Pauli 期望、保真度、精确态表格。

所有函数接受 Statevector / DensityMatrix / 裸数组 / qiskit 对象
（经 gpuqviz.state.as_state 归一）。公式定义见 docs/conventions.md §7，
数值由 tests/cross_validation/test_against_qiskit.py 对拍
qiskit.quantum_info 锁定（容差 1e-10）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state import DensityMatrix, Statevector, as_state
from .measurement import exact_probs

__all__ = ["purity", "linear_entropy", "pauli_expectation", "fidelity",
           "StateTable", "state_table"]


# --------------------------------------------------------------------------- #
# 纯度 / 线性熵
# --------------------------------------------------------------------------- #

def purity(state) -> float:
    """Tr ρ²：纯态 1，k-qubit 最大混合 1/2^k。"""
    return float(as_state(state).purity())


def linear_entropy(state, normalized: bool = False) -> float:
    """线性熵 S_L = 1 − Tr ρ²（纯态 0，越大越混合）。

    normalized=True 时返回归一化变体 d/(d−1)·(1 − Tr ρ²)
    （d = Hilbert 维数，取值 [0,1]，纯态 0、最大混合 1）。
    """
    st = as_state(state)
    p = st.purity()
    s = 1.0 - p
    if normalized:
        d = 2 ** st.n_qubits
        return float(d / (d - 1) * s) if d > 1 else 0.0
    return float(s)


# --------------------------------------------------------------------------- #
# Pauli 串期望
# --------------------------------------------------------------------------- #

_PAULI_MAT = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _pauli_matrix(label: str) -> np.ndarray:
    """位串 → 2^n 酉矩阵。位串左起为 q_{n-1}（最高位），与显示约定一致，
    因此按字符串顺序 kron（kron 的左因子作用于更高位）。"""
    for ch in label:
        if ch not in _PAULI_MAT:
            raise ValueError(f"invalid Pauli character {ch!r} in {label!r} "
                             "(allowed: I, X, Y, Z)")
    m = np.array([[1.0 + 0j]])
    for ch in label:  # 左起 = q_{n-1} = 最高位 → kron 顺序即字符串顺序
        m = np.kron(m, _PAULI_MAT[ch])
    return m


def pauli_expectation(state, pauli_string: str) -> float:
    """Pauli 串期望 Tr(ρ P)，实数（P 为 Hermit）。

    位串记法："IIXZ" 表示 Z⊗作用于 q0、X 作用于 q1、其余 I——
    即**左起字符对应最高位 qubit**（与位串显示约定一致，qiskit
    SparsePauliOp 相同记法）。

    0.9.0 T1-1：不构造稠密 2^n×2^n Pauli 矩阵——
    - 纯态：逐位 batched GEMM 作用单比特算符，O(n·2^n)；
    - 混态：ρ 张量的行/列轴逐对收缩（合并轴累计在前的布局
      [merged | rows | cols]，第 pos 步行轴 = pos、列轴 = n+pos），
      总成本 O(4^n)，原实现 O(4^n) 矩阵构造 + O(8^n) 矩阵乘。
    """
    st = as_state(state)
    n = st.n_qubits
    if len(pauli_string) != n:
        raise ValueError(f"pauli_string length {len(pauli_string)} != "
                         f"n_qubits {n}")
    for ch in pauli_string:
        if ch not in _PAULI_MAT:
            raise ValueError(f"invalid Pauli character {ch!r} in {pauli_string!r} "
                             "(allowed: I, X, Y, Z)")

    if isinstance(st, DensityMatrix):
        # 布局不变式：T = [merged(pos 个) | 剩余行轴 | 剩余列轴]，
        # 行块起点 = pos、列块起点 = n（列块起点恒为 n：pos + (n-pos)）
        T = np.asarray(st.data).reshape((2,) * (2 * n))
        for pos, ch in enumerate(pauli_string):
            M = _PAULI_MAT[ch]
            T = np.moveaxis(T, (pos, n), (0, 1))
            last = pos == n - 1
            res = np.einsum("ji,ijr->" if last else "ji,ijr->jr",
                            M, T.reshape(2, 2, -1), optimize=True)
            if last:
                return float(np.real(res))
            T = res.reshape((2,) * (2 * n - pos - 1))
        raise AssertionError("unreachable")

    psi = np.asarray(st.data).reshape(-1)
    out = psi
    for pos, ch in enumerate(pauli_string):
        if ch == "I":
            continue
        q = n - 1 - pos
        left = 1 << (n - 1 - q)
        right = 1 << q
        out = np.matmul(_PAULI_MAT[ch], out.reshape(left, 2, right)).reshape(-1)
    return float(np.real(np.vdot(psi, out)))


# --------------------------------------------------------------------------- #
# 保真度
# --------------------------------------------------------------------------- #

def _sqrtm_hermit(h: np.ndarray) -> np.ndarray:
    """Hermit 矩阵的谱平方根（特征值裁到 ≥0 抵御数值噪声）。"""
    eig, vec = np.linalg.eigh((h + h.conj().T) / 2)
    eig = np.clip(eig.real, 0.0, None)
    return (vec * np.sqrt(eig)) @ vec.conj().T


def fidelity(state_a, state_b) -> float:
    """态保真度 F，定义按输入类型分派（与 qiskit StateFidelity 一致）：

    - 纯 ⊗ 纯：F = |⟨ψ|φ⟩|²
    - 纯 ⊗ 混：F = ⟨ψ|ρ|ψ⟩
    - 混 ⊗ 混：F = (Tr √(√ρ_A ρ_B √ρ_A))²  （Uhlmann）
    """
    a = as_state(state_a)
    b = as_state(state_b)
    if isinstance(a, Statevector) and isinstance(b, Statevector):
        return float(abs(a.inner(b)) ** 2)
    if isinstance(a, Statevector) != isinstance(b, Statevector):
        sv, dm = (a, b) if isinstance(a, Statevector) else (b, a)
        psi = np.asarray(sv.data)
        rho = np.asarray(dm.data)
        return float(np.real(psi.conj() @ rho @ psi))
    # 混混：Uhlmann
    ra = np.asarray(a.data)
    rb = np.asarray(b.data)
    sa = _sqrtm_hermit(ra)
    m = sa @ rb @ sa
    eig = np.linalg.eigvalsh((m + m.conj().T) / 2)
    eig = np.clip(eig.real, 0.0, None)
    return float(np.sum(np.sqrt(eig)) ** 2)


# --------------------------------------------------------------------------- #
# 精确态表格（专家核对理论的"原始账本"）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StateTable:
    """逐基矢的精确数值表：|基矢⟩ / 振幅 / 概率 / 相位。

    混合态无振幅概念，re/im 列为 None、相位为 NaN。
    """

    n_qubits: int
    index: np.ndarray          # (dim,) int
    bitstring: tuple[str, ...]  # (dim,) "q_{n-1}…q_0"
    re: np.ndarray | None      # (dim,) float | None（混合态）
    im: np.ndarray | None
    prob: np.ndarray           # (dim,) float
    phase_deg: np.ndarray | None

    def __len__(self) -> int:
        return len(self.index)

    def rows(self, min_prob: float = 0.0) -> list[tuple]:
        """按概率降序的行： (index, bitstring, re, im, prob, phase_deg)。

        min_prob 过滤掉概率低于阈值的行（大空间常用）。
        """
        order = np.argsort(-self.prob)
        out = []
        for i in order:
            if self.prob[i] < min_prob:
                break
            out.append((int(self.index[i]), self.bitstring[i],
                        None if self.re is None else float(self.re[i]),
                        None if self.im is None else float(self.im[i]),
                        float(self.prob[i]),
                        None if self.phase_deg is None else float(self.phase_deg[i])))
        return out

    def to_dict(self, min_prob: float = 0.0) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "rows": [
                {"index": idx, "bitstring": bs,
                 "amplitude": None if re_ is None else complex(re_, im_ or 0.0).__repr__(),
                 "re": re_, "im": im_, "prob": p, "phase_deg": ph}
                for idx, bs, re_, im_, p, ph in self.rows(min_prob)
            ],
        }

    def to_csv(self, path=None, min_prob: float = 0.0) -> str:
        """CSV：index,bitstring,re,im,prob,phase_deg。"""
        lines = ["index,bitstring,re,im,prob,phase_deg"]
        for idx, bs, re_, im_, p, ph in self.rows(min_prob):
            fmt = lambda v: "" if v is None else f"{v:.12g}"  # noqa: E731
            lines.append(f"{idx},{bs},{fmt(re_)},{fmt(im_)},{fmt(p)},{fmt(ph)}")
        text = "\n".join(lines) + "\n"
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text


def state_table(state) -> StateTable:
    """态 → 全基矢精确表格（振幅/概率/相位，排序见 rows()）。"""
    st = as_state(state)
    dim = 2 ** st.n_qubits
    n = st.n_qubits
    p = exact_probs(st)
    if isinstance(st, Statevector):
        amp = np.asarray(st.data)
        re, im = amp.real.copy(), amp.imag.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            phase = np.degrees(np.arctan2(im, re))
        phase = np.where(np.abs(amp) > 0, phase, np.nan)
    else:
        re = im = None
        phase = None
    return StateTable(
        n_qubits=n,
        index=np.arange(dim),
        bitstring=tuple(format(i, f"0{n}b") for i in range(dim)),
        re=re, im=im, prob=p, phase_deg=phase,
    )
