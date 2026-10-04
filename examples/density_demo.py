"""M3 收尾 demo：含噪 Bell 态的 Hinton 图演化。

Depolarizing 噪声逐步抹掉 Bell 态的非对角相干性（纠缠的代数痕迹），
Hinton 图让这一过程逐帧可见：|ρ_03|/|ρ_30| 方块收缩、对角趋于均匀。

运行：python examples/density_demo.py
输出：out/density_demo.mp4 / out/density_demo.png
"""

from pathlib import Path

import numpy as np

import gpuqviz
from gpuqviz import DensityMatrixTrack, Scene
from gpuqviz.noise import depolarizing, evolve_density, tensor_channels
from gpuqviz.state import DensityMatrix


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    from gpuqviz.circuits import Gate

    gates = [Gate(name="H", targets=[0], controls=[], params=[]),
             Gate(name="CX", targets=[1], controls=[0], params=[])]
    LAM = 0.12  # 每门去极化强度

    rhos = np.stack([np.asarray(r) for r in
                     evolve_density(
                         2, gates,
                         noise=lambda g, i: tensor_channels(
                             [depolarizing(LAM)]
                             * (len(g.targets) + len(g.controls))))])

    # 专业侧核对：相干性 |ρ_03| 随门数衰减
    for k, rho in enumerate(rhos):
        label = "|00>" if k == 0 else f"门{k}"
        print(f"{label:>6}: |ρ_03|={abs(rho[0, 3]):.4f}  "
              f"纯度={DensityMatrix(rho).purity():.4f}")

    npz = out_dir / "density_demo_rhos.npz"
    np.savez(npz, states=rhos)

    scene = Scene(
        width=1920, height=1080, fps=30, duration=3.0,
        background="#0b0e14",
        title="量子态 — Hinton 图",
        tracks=[DensityMatrixTrack(states_path="density_demo_rhos.npz",
                                   layout="full")],
    )
    scene.save_json(out_dir / "density_demo_scene.json")

    out_mp4 = out_dir / "density_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render(scene, out=out_mp4, states_dir=out_dir)
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    out_png = out_dir / "density_demo.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"末帧静态图: {out_png}")

    print("\n✅ Hinton 渲染验收输出完成")


if __name__ == "__main__":
    main()
