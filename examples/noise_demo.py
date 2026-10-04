"""M3 验收 demo：理想 vs 含噪演化的双视图对比。

同一电路（H → CX → CX → RY）的两个 Bloch 视图：
- 上：理想演化（纯态，|r| = 1）；
- 下：含 depolarizing 噪声的密度矩阵演化（Bloch 矢量逐门收缩）。

渲染端零改动——"模长 = 纯度"的既有语义天然呈现噪声收缩
（noise.evolve_density 产出 ρ 关键帧 → bloch_vectors 批量 partial trace）。

运行：python examples/noise_demo.py
输出：out/noise_demo.mp4 / out/noise_demo.png
"""

from pathlib import Path

import numpy as np

import gpuqviz
from gpuqviz import BlochTrack, Scene
from gpuqviz.noise import depolarizing, evolve_density
from gpuqviz.state import DensityMatrix


def _build_gates():
    """RY 旋转 + 一次 CX：理想态各 qubit Bloch 矢量非零、部分纠缠，
    使噪声收缩与纠缠收缩在 Bloch 视图中同时可见。"""
    return [
        {"name": "RY", "targets": [0], "controls": [], "params": [1.2]},
        {"name": "RY", "targets": [1], "controls": [], "params": [0.9]},
        {"name": "RY", "targets": [2], "controls": [], "params": [0.6]},
        {"name": "CX", "targets": [1], "controls": [0], "params": []},
    ]


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    from gpuqviz.circuits import Gate

    gates = [Gate(name=g["name"], targets=g["targets"],
                  controls=g["controls"], params=g["params"])
             for g in _build_gates()]
    n_qubits = 3

    LAM1, LAM2 = 0.08, 0.12  # 单/双 qubit 门去极化强度

    def noise(gate, gi):
        nq = len(gate.targets) + len(gate.controls)
        return depolarizing(LAM1 if nq == 1 else LAM2, n_qubits=nq)

    rhos = evolve_density(n_qubits, gates, noise=noise)
    rhos = np.stack([np.asarray(r) for r in rhos])

    # 理想演化（无噪声）
    rhos_ideal = np.stack(
        [np.asarray(r) for r in evolve_density(n_qubits, gates)])

    # 专业侧核对：末帧纯度与 Bloch 模长
    dm_noisy = DensityMatrix(rhos[-1])
    dm_ideal = DensityMatrix(rhos_ideal[-1])
    b_noisy = dm_noisy.bloch()
    b_ideal = dm_ideal.bloch()
    print(f"末帧整体纯度: 理想={dm_ideal.purity():.6f}  含噪={dm_noisy.purity():.6f}")
    print(f"末帧 Bloch 模长: 理想={np.linalg.norm(b_ideal, axis=1).round(3)}  "
          f"含噪={np.linalg.norm(b_noisy, axis=1).round(3)}")

    npz_ideal = out_dir / "noise_demo_ideal.npz"
    npz_noisy = out_dir / "noise_demo_noisy.npz"
    np.savez(npz_ideal, states=rhos_ideal)
    np.savez(npz_noisy, states=rhos)

    scene = Scene(
        width=1920, height=1080, fps=30, duration=4.0,
        background="#0b0e14",
        title="量子态演化 — 模拟结果",
        tracks=[
            BlochTrack(states_path="noise_demo_ideal.npz", trail=True,
                       layout="top"),
            BlochTrack(states_path="noise_demo_noisy.npz", trail=True,
                       layout="bottom"),
        ],
    )
    scene.save_json(out_dir / "noise_demo_scene.json")

    out_mp4 = out_dir / "noise_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render(scene, out=out_mp4, states_dir=out_dir)
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    out_png = out_dir / "noise_demo.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"末帧静态图: {out_png}")

    print("\n✅ 噪声演化渲染验收输出完成")


if __name__ == "__main__":
    main()
