"""State 统一抽象：Statevector / DensityMatrix 双态基类。

约定（详见 docs/conventions.md）：
- 小端序：qubit 0 为最低有效位，位串显示 q_{n-1}…q_0；
- 输入原样存储，不做隐式归一化（可核对性优先）；
- 精确量容差 1e-10，由 tests/cross_validation/ 对拍 qiskit 锁定。

本模块是分析层（gpuqviz.analysis）的数据基座：所有度量函数同时接受
Statevector / DensityMatrix / 裸 numpy 数组（鸭子类型，内部经 `as_state`
归一）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

try:  # cupy 可选：GPU 数组后端
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

__all__ = ["Statevector", "DensityMatrix", "as_state", "partial_trace"]


# --------------------------------------------------------------------------- #
# 数组工具
# --------------------------------------------------------------------------- #

def _xp(arr):
    """按输入类型选择数组后端（cupy 输入留 GPU，numpy 走 CPU）。"""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def _as_complex(arr, label: str) -> "np.ndarray":
    """裸数组 → 本后端复数数组（cupy 输入保留在 GPU）。"""
    xp = _xp(arr)
    return xp.asarray(arr).astype(xp.complex128, copy=False)


def _dim_to_n(dim: int) -> int:
    n = int(round(np.log2(dim)))
    if 2**n != dim:
        raise ValueError(f"dimension {dim} is not a power of 2")
    return n


def _expect_dim(name: str, arr, n: int | None, shape_tail: tuple) -> int:
    dim = int(arr.shape[0])
    got = _dim_to_n(dim)
    if n is not None and got != n:
        raise ValueError(f"{name} has {got} qubits, expected {n}")
    return got


# --------------------------------------------------------------------------- #
# 约化密度矩阵（partial trace）
# --------------------------------------------------------------------------- #

def partial_trace(rho, keep: list[int], n: int) -> "np.ndarray":
    """对密度矩阵求子系统的约化密度矩阵（小端序）。

    ρ_A = Tr_rest ρ。输出矩阵的 index = q_{max}·2^{k-1} + … + q_{min}
    （保留 qubit 降序为高位→低位，与 qiskit partial_trace 一致，
    由 tests/cross_validation/test_against_qiskit.py 锁定）。

    约定：reshape 后轴 a ↔ qubit n-1-a（见 evolve.py 注释）。
    n ≤ 13（einsum 字母上限），更大规模走 Phase 5 的分块路径。
    """
    if n > 13:
        raise NotImplementedError(
            f"partial_trace supports n<=13 (got {n}); larger systems need the "
            "Phase-5 blocked path")
    import string
    letters = string.ascii_lowercase + string.ascii_uppercase
    keep = sorted(set(keep), reverse=True)   # 输出顺序：q_max…q_min
    traced = set(range(n)) - set(keep)
    R = rho.reshape((2,) * n + (2,) * n)
    # 轴位置 a (0..n-1) ↔ qubit n-1-a：给每个 qubit 分配字母
    row = [None] * n
    col = [None] * n
    for q in range(n):
        row[n - 1 - q] = letters[q]
        col[n - 1 - q] = letters[q] if q in traced else letters[26 + q]
    spec = ("".join(row) + "".join(col) + "->"
            + "".join(letters[q] for q in keep)          # kept row 轴
            + "".join(letters[26 + q] for q in keep))    # kept col 轴
    out = np.einsum(spec, R, optimize=True)
    d = 2 ** len(keep)
    return np.asarray(out).reshape(d, d)


def _sv_reduced_rho(psi, keep: list[int], n: int) -> "np.ndarray":
    """纯态约化密度矩阵：把保留 qubit 的轴聚到前部，(T @ T†) 批量外积。

    输出 index = q_{max}…q_{min}（与 partial_trace / qiskit 一致）。
    """
    keep_desc = sorted(set(keep), reverse=True)
    t = psi.reshape((2,) * n)
    # 轴 a ↔ qubit n-1-a：qubit q 在轴 n-1-q。
    # moveaxis 后新轴 0 是 MSB → 必须先移最大 qubit。
    t = np.moveaxis(t, [n - 1 - q for q in keep_desc], list(range(len(keep_desc))))
    k = len(keep_desc)
    T = t.reshape(2 ** k, -1)
    return T @ T.conj().T


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class State(ABC):
    """态基类：不可变值对象，保留输入数组后端（numpy/cupy）。"""

    data: "np.ndarray" = field(repr=False)

    @property
    def n_qubits(self) -> int:
        return _dim_to_n(int(self.data.shape[0]))

    # -- 统一接口 ----------------------------------------------------------- #

    @abstractmethod
    def probs(self) -> "np.ndarray":
        """计算基概率分布 p_i（Σ p_i = 1，float64）。"""

    @abstractmethod
    def density(self) -> "np.ndarray":
        """完整密度矩阵 (2^n, 2^n)，Hermit、迹 1。"""

    @abstractmethod
    def reduced(self, qubits: list[int]) -> "DensityMatrix":
        """给定 qubit 子集的约化密度矩阵。"""

    @abstractmethod
    def purity(self) -> float:
        """Tr ρ²：纯态 1，最大混合 1/2^k。"""

    def entropy(self, qubits: list[int] | None = None) -> float:
        """von Neumann 熵 S = −Tr(ρ log₂ ρ)，单位 bit。

        qubits=None 时对整体求（纯态恒 0）；指定子集时对该子系统
        的约化密度矩阵求。
        """
        if qubits is None:
            rho = self.density()
        else:
            rho = self.reduced(qubits).data
        xp = _xp(rho)
        eig = np.linalg.eigvalsh(np.asarray(cp.asnumpy(rho) if xp is cp else rho))
        eig = np.clip(eig.real, 0.0, None)
        nz = eig[eig > 1e-12]
        return float(-np.sum(nz * np.log2(nz)))

    def bloch(self) -> "np.ndarray":
        """逐 qubit Bloch 向量 (n, 3)，语义见 conventions.md §3。"""
        from .evolve import bloch_vectors

        return np.asarray(bloch_vectors([self]))[0]

    def to_dict(self) -> dict:
        """JSON 可序列化摘要（data 无 numpy 类型）。"""
        return {
            "kind": type(self).__name__,
            "n_qubits": self.n_qubits,
            "purity": self.purity(),
            "entropy": self.entropy(),
            "provenance": getattr(self, "provenance", "exact"),
        }


# --------------------------------------------------------------------------- #
# Statevector
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Statevector(State):
    """纯态：1D 复数向量，‖ψ‖ 由输入保证（不隐式归一化）。

    provenance（0.8.0 审计 #4）：数据来源语义。"exact" = 精确演化/精确约化；
    "mps_chi=N" = MPS χ 截断产物（数值为近似，渲染层据此标注）。
    """

    data: "np.ndarray" = field(repr=False)
    provenance: str = field(default="exact", repr=False)

    def __post_init__(self):
        arr = _as_complex(self.data, "statevector")
        if arr.ndim != 1:
            raise ValueError(f"Statevector expects 1D data, got shape {arr.shape}")
        _dim_to_n(int(arr.shape[0]))
        object.__setattr__(self, "data", arr)

    # -- 构造 --------------------------------------------------------------- #

    @classmethod
    def from_int(cls, index: int, n_qubits: int) -> "Statevector":
        """计算基态 |index⟩（小端序：index 的二进制位即 qubit 位）。"""
        if not 0 <= index < 2 ** n_qubits:
            raise ValueError(f"index {index} out of range for {n_qubits} qubits")
        psi = np.zeros(2 ** n_qubits, dtype=np.complex128)
        psi[index] = 1.0
        return cls(psi)

    @classmethod
    def from_label(cls, label: str) -> "Statevector":
        """位串标签（qiskit 风格，左起 q_{n-1}，右端 q0）→ 计算基态。"""
        return cls.from_int(int(label, 2), len(label))

    # -- 统一接口 ----------------------------------------------------------- #

    def probs(self) -> "np.ndarray":
        amp = self.data
        return np.abs(np.asarray(amp)) ** 2

    def density(self) -> "np.ndarray":
        psi = np.asarray(self.data).reshape(-1, 1)
        return np.asarray(psi @ psi.conj().T)

    def reduced(self, qubits: list[int]) -> "DensityMatrix":
        n = self.n_qubits
        keep = list(qubits) if qubits else list(range(n))
        rho = _sv_reduced_rho(np.asarray(self.data), keep, n)
        return DensityMatrix(rho)

    def purity(self) -> float:
        """Tr ρ²，ρ = ψψ† ⇒ 纯度 = ‖ψ‖⁴（归一化输入恒为 1）。

        注意不能算成 Σ p_i²——那是碰撞概率，不是纯度；
        该错误曾由 tests/cross_validation 对拍 GHZ 态时捕获。
        """
        norm2 = float(np.real(np.vdot(np.asarray(self.data), np.asarray(self.data))))
        return norm2 * norm2

    def normalize(self) -> "Statevector":
        """显式归一化（不改全局相位）。"""
        norm = float(np.linalg.norm(np.asarray(self.data)))
        if norm <= 0:
            raise ValueError("cannot normalize zero vector")
        return Statevector(np.asarray(self.data) / norm)

    # 与其他态的内积/保真度需要两侧裸数组
    def inner(self, other: "Statevector") -> complex:
        return complex(np.vdot(np.asarray(self.data), np.asarray(other.data)))


# --------------------------------------------------------------------------- #
# DensityMatrix
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DensityMatrix(State):
    """混合态：(2^n, 2^n) Hermit 半正定矩阵，迹由输入保证。

    provenance 语义同 Statevector（见其 docstring）。
    """

    data: "np.ndarray" = field(repr=False)
    provenance: str = field(default="exact", repr=False)

    def __post_init__(self):
        arr = _as_complex(self.data, "density matrix")
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"DensityMatrix expects square 2D data, got {arr.shape}")
        _dim_to_n(arr.shape[0])
        object.__setattr__(self, "data", arr)

    # -- 构造 --------------------------------------------------------------- #

    @classmethod
    def from_statevector(cls, sv: Statevector) -> "DensityMatrix":
        return cls(sv.density())

    @classmethod
    def maximally_mixed(cls, n_qubits: int) -> "DensityMatrix":
        return cls(np.eye(2 ** n_qubits, dtype=np.complex128) / 2 ** n_qubits)

    # -- 统一接口 ----------------------------------------------------------- #

    def probs(self) -> "np.ndarray":
        diag = np.real(np.diag(np.asarray(self.data)))
        return diag.astype(np.float64)

    def density(self) -> "np.ndarray":
        return np.asarray(self.data)

    def reduced(self, qubits: list[int]) -> "DensityMatrix":
        n = self.n_qubits
        keep = list(qubits) if qubits else list(range(n))
        rho = partial_trace(np.asarray(self.data), keep, n)
        return DensityMatrix(rho)

    def purity(self) -> float:
        """Tr ρ²：Tr(ρ·ρ) 双张量全收缩 O(4^n)
        （原 ρ@ρ 完整矩阵乘再取迹 O(8^n)，0.9.0 T1-2）。"""
        rho = np.asarray(self.data)
        n = int(round(np.log2(rho.shape[0])))
        T = rho.reshape((2,) * (2 * n))  # 前 n 轴 = 行（MSB-first），后 n 轴 = 列
        a = list(range(n)) + list(range(n, 2 * n))          # A 的 (行 i, 列 j)
        b = list(range(n, 2 * n)) + list(range(n))          # B 的 (行 j, 列 i)
        return float(np.real(np.einsum(T, a, T, b, optimize=True)))

    # -- 密度矩阵专属 -------------------------------------------------------- #

    def is_valid(self, tol: float = 1e-10) -> bool:
        """Hermit + 半正定 + 迹 1（契约测试与噪声演化用）。"""
        rho = np.asarray(self.data)
        hermit_ok = bool(np.allclose(rho, rho.conj().T, atol=tol))
        trace_ok = abs(float(np.real(np.trace(rho))) - 1.0) <= 1e-10
        eig = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
        psd_ok = bool(eig.min() >= -1e-10)
        return hermit_ok and trace_ok and psd_ok


# --------------------------------------------------------------------------- #
# 鸭子类型归一入口
# --------------------------------------------------------------------------- #

def as_state(obj) -> State:
    """任意态对象 → Statevector / DensityMatrix。

    接受：gpuqviz State、qiskit Statevector/DensityMatrix、
    裸 1D ndarray（视为纯态）、裸 2D 方阵（视为密度矩阵）、
    以及任何 .data 可取出的对象（qiskit 密度矩阵 .data 是矩阵，
    纯态 .data 是向量，按维数判别）。

    注意：numpy/cupy 的 ndarray 自身也有 .data（内存 buffer），
    必须先判类型再决定是否解包，否则会拿到形状为 () 的 buffer。
    """
    if isinstance(obj, State):
        return obj
    if isinstance(obj, np.ndarray) or (cp is not None and isinstance(obj, cp.ndarray)):
        raw = obj
    else:
        raw = getattr(obj, "data", obj)
    if cp is not None and isinstance(raw, cp.ndarray):
        raw = cp.asnumpy(raw)
    arr = np.asarray(raw)
    if arr.ndim == 1:
        return Statevector(arr.astype(np.complex128, copy=False))
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return DensityMatrix(arr.astype(np.complex128, copy=False))
    raise TypeError(f"cannot interpret object of shape {arr.shape} as a quantum state")
