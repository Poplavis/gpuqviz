"""测量统计：精确概率、可复现 shot 采样、Counts 结构。

约定（docs/conventions.md §1/§5）：
- 位串显示 q_{n-1}…q_0（qiskit 风格，qubit 0 在最右）；
- 采样使用 numpy PCG64(seed)，同 seed 同结果；
- 采样正确性由 tests/cross_validation/test_measurement.py 对拍
  qiskit AerSimulator（卡方一致性）锁定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # cupy 可选：P5.1 GPU 采样路径
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from ..state import State, as_state

__all__ = ["exact_probs", "sample_counts", "Counts"]


# --------------------------------------------------------------------------- #
# 精确概率
# --------------------------------------------------------------------------- #

def exact_probs(state) -> np.ndarray:
    """计算基概率分布 p_i（归一化，float64）。

    纯态 p_i = |ψ_i|²；混合态 p_i = ρ_ii。输入先经 as_state 归一。
    """
    st = as_state(state)
    p = np.asarray(st.probs(), dtype=np.float64)
    total = p.sum()
    if total <= 0:
        raise ValueError("state has zero norm")
    return p / total


# --------------------------------------------------------------------------- #
# Counts 结构
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Counts:
    """一次测量的计数结果（不可变）。

    counts: {位串: 次数}，位串为 q_{n-1}…q_0（conventions.md §1）。
    """

    counts: dict[str, int]
    shots: int
    n_qubits: int
    seed: int | None = None
    _exact: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        total = sum(self.counts.values())
        if total != self.shots:
            raise ValueError(f"counts sum {total} != shots {self.shots}")

    # -- 基础视图 ------------------------------------------------------------ #

    @property
    def probs(self) -> dict[str, float]:
        """经验频率 {位串: 次数/shots}。"""
        return {k: v / self.shots for k, v in self.counts.items()}

    def most_common(self, k: int | None = None) -> list[tuple[str, int]]:
        """按次数降序的前 k 项（k=None 全部）。"""
        items = sorted(self.counts.items(), key=lambda kv: -kv[1])
        return items if k is None else items[:k]

    # -- 分布操作 ------------------------------------------------------------ #

    def marginal(self, qubits: list[int]) -> "Counts":
        """对未选中 qubit 求和的边际分布。

        返回位串只含选中 qubit，顺序保持 q_{max}…q_{min}
        （conventions.md §1 的位串方向，选中集合内部相对位置不变）。
        """
        n = self.n_qubits
        sel = sorted(set(qubits), reverse=True)  # 拼接顺序：高位在前
        if not sel or sel[-1] < 0 or sel[0] >= n:
            raise ValueError(f"qubits {sel} out of range for {n} qubits")
        merged: dict[str, int] = {}
        for bs, c in self.counts.items():
            # 位串第 k 个字符（左起）是 qubit n-1-k
            key = "".join(bs[n - 1 - q] for q in sel)
            merged[key] = merged.get(key, 0) + c
        return Counts(merged, self.shots, len(sel), seed=self.seed)

    # -- 导出 ---------------------------------------------------------------- #

    def to_dict(self) -> dict:
        out = {
            "shots": self.shots,
            "n_qubits": self.n_qubits,
            "seed": self.seed,
            "counts": dict(self.counts),
            "probs": dict(self.probs),
        }
        if self._exact is not None:
            out["exact_probs"] = {format(i, f"0{self.n_qubits}b"): float(p)
                                  for i, p in enumerate(self._exact) if p > 0}
        return out

    def to_csv(self, path=None) -> str:
        """CSV：bitstring,count,prob。给 path 时写文件，同时返回内容。"""
        lines = ["bitstring,count,prob"]
        for bs in sorted(self.counts):
            lines.append(f"{bs},{self.counts[bs]},{self.counts[bs] / self.shots:.10g}")
        text = "\n".join(lines) + "\n"
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text


# --------------------------------------------------------------------------- #
# shot 采样
# --------------------------------------------------------------------------- #

# auto 模式下启用 GPU 采样的最小维数（2^20 = 1M；小维数走 numpy 保证
# 与既有 golden 测试同流，且省去传输开销）
_GPU_SAMPLE_MIN_DIM = 1 << 20


def _bitstring_keys(indices: np.ndarray, n: int) -> np.ndarray:
    """下标数组 → 定宽位串 numpy 数组（向量化，P5.1 大规模路径的关键）。

    Python 逐项 format() 在千万级非零 bin 时成为瓶颈（~1s/3.5M）；
    numpy 视图转换快一个数量级以上。
    """
    if indices.size == 0:
        return np.empty(0, dtype=f"U{n}")
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    bits = (((indices[:, None].astype(np.int64) >> shifts) & 1) + 48).astype(np.uint8)
    return np.ascontiguousarray(bits).view(f"S{n}").ravel().astype(str)


def _sample_gpu(p: np.ndarray, shots: int, seed: int | None) -> np.ndarray:
    """GPU 批量分类采样（P5.1）：cdf + searchsorted，全程不下 GPU。

    返回下标数组（numpy）。与 CPU 路径使用不同随机流（XORWOW vs PCG64），
    同 seed 跨后端不逐位一致——约定仅承诺精确量跨后端逐位一致
    （docs/conventions.md §6），采样统计性质相同。
    """
    import cupy as cpx

    p_gpu = cpx.asarray(p, dtype=cpx.float64)
    cdf = cpx.cumsum(p_gpu)
    cdf /= cdf[-1]
    rng = cpx.random.Generator(cpx.random.XORWOW(seed if seed is not None else 0))
    u = rng.random(shots, dtype=cpx.float64)
    idx = cpx.searchsorted(cdf, u, side="right")
    idx = cpx.minimum(idx, p_gpu.size - 1)  # 数值容差保护：u≈1 越界
    return cpx.asnumpy(idx)


def sample_counts(state, shots: int, seed: int | None = None,
                  qubits: list[int] | None = None,
                  backend: str = "auto") -> Counts:
    """计算基 shot 采样（可复现，P5.1 支持大规模 GPU 路径）。

    state: Statevector/DensityMatrix/裸数组/qiskit 对象。
    shots: 采样次数（≥1）。
    seed:  随机种子（None 时不可复现，测试与论文复现请固定 seed）。
    qubits: 只在这些 qubit 上测量（其余迹掉）；None = 全部。
    backend: "auto"（维数 ≥ 2^20 且 cupy 可用时走 GPU）/ "cpu" / "gpu"。

    CPU 实现：PCG64 一次性批量分类采样 O(shots + dim)。
    GPU 实现：概率上卡后 cdf + searchsorted（n=24、shots=10^6 亚秒级），
    随机流与 CPU 不同（约定只承诺统计一致，见 docs/conventions.md §6）。
    统计正确性由契约测试对拍 AerSimulator 锁定。
    """
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    st = as_state(state)
    p = exact_probs(st)

    if qubits is not None:
        sel = sorted(set(qubits))
        if not sel or sel[0] < 0 or sel[-1] >= st.n_qubits:
            raise ValueError(f"qubits {sel} out of range for {st.n_qubits} qubits")
        n_out = len(sel)
        # 显式位映射：完整下标 i 的第 sel[j] 位 → 紧凑下标的第 j 位
        # （允许 sel 有空洞，保持升序相对位置）
        idx = np.arange(p.size)
        bits = (idx[:, None] >> np.asarray(sel)[None, :]) & 1        # (dim, k)
        out_idx = bits @ (1 << np.arange(n_out))                     # (dim,)
        p_out = np.bincount(out_idx, weights=p, minlength=2 ** n_out)
        p = p_out / p_out.sum()
        n = n_out
    else:
        n = st.n_qubits

    use_gpu = False
    if backend == "gpu":
        if cp is None:
            raise RuntimeError("backend='gpu' requires cupy")
        use_gpu = True
    elif backend == "auto":
        use_gpu = cp is not None and p.size >= _GPU_SAMPLE_MIN_DIM
    elif backend != "cpu":
        raise ValueError(f"backend must be auto/cpu/gpu, got {backend!r}")

    if use_gpu:
        draws = _sample_gpu(p, shots, seed)
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.choice(p.size, size=shots, p=p)

    # bincount 计数（O(shots + dim)，大维数下远快于 sort-based unique）
    cnt_arr = np.bincount(draws, minlength=p.size)
    nz = np.nonzero(cnt_arr)[0]
    keys = _bitstring_keys(nz, n)
    counts = dict(zip(keys.tolist(), cnt_arr[nz].tolist()))
    return Counts(counts, shots, n, seed=seed, _exact=p)
