"""M1 验收 demo：Grover 搜索的 Bloch 球（上）+ 测量概率直方图（下）。

展示 HistogramTrack 与专业分析模块的衔接：
states → exact_probs（gpuqviz.analysis）→ 直方图动画。

运行：python examples/histogram_demo.py
输出：out/histogram_demo.mp4 / out/histogram_demo.png
"""

from pathlib import Path

import numpy as np

import gpuqviz
from gpuqviz import BlochTrack, HistogramTrack, Scene


def make_grover_states(n_qubits: int = 3, frames: int = 4) -> np.ndarray:
    """手工构造 Grover 演化态序列（不依赖 qiskit）：均匀叠加 → 逐轮放大目标。

    frames=4 即 r=0..3 轮：3 qubit 最优迭代数 ≈ π/4·√8 ≈ 2.2，
    r=2 时目标概率 ≈ 0.95（峰值），r=3 开始回落——完整展示振荡。
    """
    dim = 2 ** n_qubits
    target = dim - 1  # |111⟩
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    states = [psi.copy()]
    for _ in range(frames - 1):
        # 一轮 Grover 迭代：oracle（翻转目标相位）+ 扩散（绕均值反射）
        psi[target] *= -1
        mean = psi.mean()
        psi = 2 * mean - psi
        states.append(psi.copy())
    return np.array(states)


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    states = make_grover_states(n_qubits=3)
    npz = out_dir / "grover_states.npz"
    np.savez(npz, states=states)

    # 专业侧核对：末帧目标态概率（exact_probs，无采样误差）
    from gpuqviz.analysis import exact_probs
    p_final = exact_probs(states[-1])
    print(f"末帧 |111⟩ 概率（精确）: {p_final[-1]:.6f}")

    scene = Scene(
        width=1920, height=1080, fps=30, duration=4.0,
        background="#0b0e14",
        title="Grover 搜索 — 测量概率演化",
        tracks=[
            BlochTrack(states_path="grover_states.npz", trail=True,
                       layout="top"),
            HistogramTrack(states_path="grover_states.npz", top_k=6,
                           layout="bottom"),
        ],
    )
    scene.save_json(out_dir / "histogram_demo_scene.json")

    out_mp4 = out_dir / "histogram_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render(scene, out=out_mp4, states_dir=out_dir)
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    out_png = out_dir / "histogram_demo.png"
    gpuqviz.render_frame(scene=scene, t=0.9, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"末段静态帧: {out_png}")

    print("\n✅ 直方图 Track 验收输出完成")


if __name__ == "__main__":
    main()
