"""P5.2 验收 demo：20-qubit 弱纠缠电路的 MPS 规模化渲染。

brickwork 电路（RY 层 + 最近邻 CX 层 ×2）在 n=20 下：
- MPS 精确演化（χ 无限制）与 χ=8 截断的保真度对比；
- 约化分析量（Bloch 向量 / 割纠缠熵）经 BlochVectorsTrack 直接喂渲染层
  —— 全程不存在 2^20 态矢量的显式表示。

运行：python examples/mps_demo.py
输出：out/mps_demo.mp4 / out/mps_demo.png / out/mps_demo_entropies.csv
"""

import time
from pathlib import Path

import numpy as np

import gpuqviz
from gpuqviz import BlochVectorsTrack, Scene
from gpuqviz.mps import evolve_mps


def _brickwork_gates(n: int, layers: int = 2) -> list:
    from gpuqviz.circuits import Gate

    gates = []
    for q in range(n):
        gates.append(Gate(name="RY", targets=[q], params=[0.4 + 0.15 * q]))
    for layer in range(layers):
        start = layer % 2
        for q in range(start, n - 1, 2):
            gates.append(Gate(name="CX", targets=[q + 1], controls=[q]))
        for q in range(n):
            gates.append(Gate(name="RZ", targets=[q], params=[0.2 * layer + 0.1]))
    return gates


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    n = 20
    gates = _brickwork_gates(n)
    print(f"电路: {n} qubit, {len(gates)} 门（brickwork ×2 层）")

    # ── MPS 演化：精确 vs χ=8 截断 ──
    t0 = time.perf_counter()
    result_exact = evolve_mps(n, gates)  # χ 无限制
    t_exact = time.perf_counter() - t0
    bonds = [t.shape[2] for t in result_exact.frames[-1].tensors]
    print(f"精确演化: {t_exact:.2f}s, 末帧键维 {bonds}")

    t0 = time.perf_counter()
    result_chi8 = evolve_mps(n, gates, chi_max=8)
    t_chi = time.perf_counter() - t0
    bonds8 = [t.shape[2] for t in result_chi8.frames[-1].tensors]
    print(f"χ=8 演化: {t_chi:.2f}s, 末帧键维 {bonds8}")

    # 截断精度：用约化分析量（Bloch 向量）差作代理指标
    bloch_diff = np.max(np.abs(result_exact.frames[-1].bloch()
                               - result_chi8.frames[-1].bloch()))
    print(f"χ=8 vs 精确: Bloch 最大偏差 = {bloch_diff:.2e}")

    # ── 约化分析量 → 渲染层 ──
    entropies = result_exact.frames[-1]._cached_entropies()
    print(f"末帧割熵: {np.round(entropies, 3)}")
    np.savetxt(out_dir / "mps_demo_entropies.csv", entropies,
               delimiter=",", header="cut_entropy_bits", comments="")

    bloch_keys = result_exact.bloch_keys()  # (K, n, 3)
    print(f"Bloch 关键帧: {bloch_keys.shape}")
    npz = out_dir / "mps_demo_bloch.npz"
    np.savez(npz, bloch=bloch_keys)

    scene = Scene(
        width=1920, height=1080, fps=30, duration=4.0,
        background="#0b0e14",
        title="量子态演化 — 模拟结果",
        tracks=[BlochVectorsTrack(states_path="mps_demo_bloch.npz",
                                  layout="full")],
    )
    scene.save_json(out_dir / "mps_demo_scene.json")

    out_mp4 = out_dir / "mps_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render(scene, out=out_mp4, states_dir=out_dir)
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    out_png = out_dir / "mps_demo.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"中段静态图: {out_png}")

    print("\n✅ MPS 规模化验收输出完成")


if __name__ == "__main__":
    main()
