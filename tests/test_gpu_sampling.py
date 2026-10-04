"""P5.1 GPU shot 采样：大规模维数/shot 的亚秒级路径 + 统计对拍。"""

import time

import numpy as np
import pytest

from gpuqviz.analysis import exact_probs, sample_counts

cp = pytest.importorskip("cupy")  # GPU 测试整体依赖 cupy

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector as QkSV  # noqa: E402


def _random_sv_probs(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (精确概率, 振幅态矢量)——采样入口接受态而非概率。"""
    rng = np.random.Generator(np.random.PCG64(seed))
    qc = QuantumCircuit(n)
    for _ in range(min(6, n)):  # 小规模门层（态矢量构造本身 O(2^n)）
        q = int(rng.integers(n))
        if rng.integers(2):
            qc.ry(float(rng.uniform(0, np.pi)), q)
        elif n > 1:
            t = (q + 1) % n
            qc.cx(q, t)
        else:
            qc.h(q)
    sv = QkSV.from_instruction(qc)
    p = np.asarray(sv.probabilities())
    return p, np.asarray(sv.data)


# --------------------------------------------------------------------------- #
# 统计正确性
# --------------------------------------------------------------------------- #

def test_gpu_sampling_within_5sigma():
    """GPU 采样经验频率与精确概率在 5σ 二项界内（n=10）。"""
    p, amps = _random_sv_probs(11, 10)
    c = sample_counts(amps, shots=200_000, seed=5, backend="gpu")
    for i in range(2 ** 10):
        freq = c.counts.get(format(i, "010b"), 0) / c.shots
        pi = p[i]
        if pi == 0:
            assert freq == 0.0
            continue
        sigma = np.sqrt(pi * (1 - pi) / c.shots)
        assert abs(freq - pi) <= 5 * sigma, f"{format(i, '010b')}"


def test_gpu_sampling_reproducible_per_seed():
    """同 seed 同后端：逐位可复现。"""
    p, amps = _random_sv_probs(3, 10)
    a = sample_counts(amps, shots=1000, seed=42, backend="gpu")
    b = sample_counts(amps, shots=1000, seed=42, backend="gpu")
    assert a.counts == b.counts


def test_gpu_sampling_matches_cpu_statistically():
    """GPU 与 CPU 路径的经验频率相互一致（均在 5σ 内 ⇒ 彼此一致）。"""
    p, amps = _random_sv_probs(21, 10)
    shots = 100_000
    c_cpu = sample_counts(amps, shots=shots, seed=1, backend="cpu")
    c_gpu = sample_counts(amps, shots=shots, seed=1, backend="gpu")
    for i in range(2 ** 10):
        bs = format(i, "010b")
        f1 = c_cpu.counts.get(bs, 0) / shots
        f2 = c_gpu.counts.get(bs, 0) / shots
        pi = p[i]
        if pi == 0:
            assert f1 == 0.0 and f2 == 0.0
            continue
        sigma = np.sqrt(pi * (1 - pi) / shots)
        # 两经验频率各自都在 5σ 内 → 差值 ≤ 10σ
        assert abs(f1 - pi) <= 5 * sigma and abs(f2 - pi) <= 5 * sigma
        assert abs(f1 - f2) <= 10 * sigma, f"{bs}: {f1} vs {f2}"


def test_gpu_backend_requires_cupy(monkeypatch):
    """backend='gpu' 无 cupy 时明确报错（模拟缺失）。"""
    import gpuqviz.analysis.measurement as meas

    monkeypatch.setattr(meas, "cp", None)
    p = np.array([0.5, 0.5])
    with pytest.raises(RuntimeError, match="requires cupy"):
        sample_counts(p, shots=10, backend="gpu")


def test_bad_backend_name():
    with pytest.raises(ValueError, match="auto/cpu/gpu"):
        sample_counts(np.array([1.0, 0.0]), shots=10, backend="tpu")


# --------------------------------------------------------------------------- #
# 规模基准：n=24、shots=10^6 亚秒级
# --------------------------------------------------------------------------- #

def test_gpu_sampling_n24_shots1e6_benchmark():
    """n=24（dim=16.7M）、shots=10^6：GPU 归约核心 < 1s（P5.1 验收）。

    只对采样+计数核心计时（预热后），exact_probs 与上下文初始化不计入；
    分布用浅层电路场景（随机支撑 4096，真实模拟的稀疏谱）。
    """
    from gpuqviz.analysis.measurement import _sample_gpu

    dim = 1 << 24
    amps = np.zeros(dim, dtype=np.complex128)
    rng = np.random.Generator(np.random.PCG64(9))
    support = rng.integers(0, dim, 4096)
    amps[support] = rng.random(4096) + 1j * rng.random(4096)
    amps /= np.linalg.norm(amps)

    p = exact_probs(amps)

    # 预热 CUDA 上下文（排除首次初始化的秒级开销）
    _ = _sample_gpu(np.array([0.5, 0.5]), 100, seed=0)

    t0 = time.perf_counter()
    draws = _sample_gpu(p, 1_000_000, seed=7)
    cnt_arr = np.bincount(draws, minlength=dim)
    elapsed = time.perf_counter() - t0

    assert int(cnt_arr.sum()) == 1_000_000
    assert elapsed < 1.0, f"GPU sampling took {elapsed:.2f}s (>1s)"
    nz = int((cnt_arr > 0).sum())
    print("\nn=24 shots=1e6 GPU 归约核心: "
          f"{elapsed:.3f}s, {nz} 个非零基矢")
