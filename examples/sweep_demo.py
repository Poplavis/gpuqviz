"""M4 验收 demo：RY(θ) 参数扫描 — Bloch 轨迹动画 + 可观测量曲线。

sweep 沿 θ ∈ [0, π] 扫描 RY(θ)|0⟩ → CX 电路：每个参数值独立演化并
计算全部分析量。states（逐值末态）直接作为 BlochTrack 关键帧 →
参数-布洛赫轨迹动画；⟨Z₀⟩ 与解析 cos θ 逐点核对。

运行：python examples/sweep_demo.py
输出：out/sweep_demo.mp4 / out/sweep_demo.png / out/sweep_demo.csv
"""

from pathlib import Path

import numpy as np

import gpuqviz
from gpuqviz import BlochTrack, Scene
from gpuqviz.parameters import CircuitTemplate, Parameter, sweep


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    tmpl = CircuitTemplate(2, [
        {"name": "RY", "targets": [0], "controls": [],
         "params": [Parameter("theta")]},
        {"name": "CX", "targets": [1], "controls": [0], "params": []},
    ])
    values = np.linspace(0, np.pi, 41)
    result = sweep(tmpl, "theta", values, observables=["IZ", "ZZ"])

    # 专业侧核对：⟨Z₀⟩ = cos θ（q0 无后续门）
    err = np.max(np.abs(result.observables["IZ"] - np.cos(values)))
    print(f"⟨Z₀⟩ vs cos θ 最大偏差: {err:.2e}")
    print(f"⟨Z₀Z₁⟩(θ=π/2) = {result.observables['ZZ'][len(values) // 2]:.4f}")
    print(f"θ=π/2 纠缠熵 S(q0) = {result.entropy[len(values) // 2, 0]:.4f} bit")

    csv = out_dir / "sweep_demo.csv"
    result.to_csv(csv)

    npz = out_dir / "sweep_demo_states.npz"
    np.savez(npz, states=result.states)

    scene = Scene(
        width=1920, height=1080, fps=30, duration=4.0,
        background="#0b0e14",
        title="参数演化 — 量子模拟",
        tracks=[BlochTrack(states_path="sweep_demo_states.npz",
                           trail=True, layout="full")],
    )
    scene.save_json(out_dir / "sweep_demo_scene.json")

    out_mp4 = out_dir / "sweep_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render(scene, out=out_mp4, states_dir=out_dir)
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    out_png = out_dir / "sweep_demo.png"
    gpuqviz.render_frame(scene=scene, t=0.5, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"中段静态图: {out_png}")
    print(f"扫参数据表: {csv}")

    print("\n✅ 参数扫描验收输出完成")


if __name__ == "__main__":
    main()
