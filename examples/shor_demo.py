"""Shor 周期查找演示（N=15，a=2）→ out/shor.mp4。

完整流程：
1. 构建周期查找线路（H^⊗6 → 受控模乘 ×6 → IQFT，10 qubit）；
2. numpy 演化验证 QPE 峰：s ∈ {0,16,32,48}（= 2^6/4 的倍数），峰概率 1/4；
3. 模拟测量采样 → 连分数求周期 r → gcd(a^{r/2}±1, N) → 15 = 3 × 5；
4. 全线路渲染为 Bloch 球视频。

N=15、a∈{2,4} 时受控模乘是工作寄存器的循环移位，可分解为受控 SWAP
（⟨2⟩ = {1,2,4,8} 子群）；正确性由 tests/cross_validation/test_shor_hhl.py
以 1e-10 容差锁定（与 qiskit cswap 链全态对拍）。

运行::

    python examples/shor_demo.py
"""

import time
from collections import Counter

import numpy as np

from gpuqviz import render_bloch_video
from gpuqviz.algorithms import (
    continued_fraction_period,
    factor_from_period,
    shor_period_finding,
)
from gpuqviz.circuits import apply_gate


def main() -> None:
    N, a, t_bits = 15, 2, 6
    gates = shor_period_finding(N, a, t_bits, engine="numpy")
    print(f"[shor] N={N}, a={a}, 计数 qubit={t_bits} + 工作 qubit=4, "
          f"gates={len(gates)}")

    # ---- numpy 演化：计数寄存器分布 ----
    state = np.zeros(1 << (t_bits + 4), dtype=np.complex128)
    state[0] = 1.0
    for g in gates:
        state = apply_gate(state, g)
    probs = (np.abs(state) ** 2).reshape(1 << 4, -1).sum(axis=0)

    print("\nQPE 峰（计数寄存器）:")
    for s in np.argsort(probs)[::-1][:4]:
        r_est = continued_fraction_period(int(s), t_bits, N)
        print(f"  |{int(s):0{t_bits}b}⟩  s={int(s):2d}  P={probs[s]:.4f}  "
              f"周期估计 r={r_est}")

    # ---- 模拟测量 → 经典后处理 → 因子 ----
    rng = np.random.default_rng(7)
    samples = rng.choice(1 << t_bits, size=16, p=probs / probs.sum())
    factors = None
    for s in samples:
        r = continued_fraction_period(int(s), t_bits, N)
        factors = factor_from_period(N, a, r)
        if factors is not None:
            break
    print(f"\n采样 {Counter(int(s) for s in samples)}")
    print(f"周期 r = {r} → 因子: {N} = {factors[0]} × {factors[1]}")
    assert sorted(factors) == [3, 5]

    # ---- 可视化：全线路 Bloch 球视频 ----
    t0 = time.perf_counter()
    render_bloch_video(
        circuit=shor_period_finding(N, a, t_bits, engine="qiskit"),
        steps=240,
        fps=60,
        seconds=12.0,
        out="out/shor.mp4",
        style="dark",
        title=f"Shor period finding: N={N}, a={a}",
    )
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f}s")
    print("output: out/shor.mp4")


if __name__ == "__main__":
    main()
