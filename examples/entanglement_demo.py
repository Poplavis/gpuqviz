"""M2 验收 demo：GHZ 制备的纠缠流动动画。

同一个态序列的两个互补视图：
- Bloch 球（上）：单 qubit 矢量收缩（模长 = 单 qubit 纯度）；
- 纠缠图（下）：互信息边在 H/CX 门作用时"亮起"，节点随纠缠熵长大。

这正是单看 Bloch 球会完全丢失的信息——纠缠结构的显式呈现。

运行：python examples/entanglement_demo.py
输出：out/entanglement_demo.mp4 / out/entanglement_demo.png
"""

from pathlib import Path

import numpy as np

import gpuqviz
from gpuqviz import BlochTrack, EntanglementTrack, Scene


def make_ghz_states(n_qubits: int = 3, frames: int = 60) -> np.ndarray:
    """GHZ 制备演化：|0…0⟩ → H → 逐位 CX 纠缠扩散。

    用 gpuqviz.evolve.sample_circuit（qiskit 演化，约定经对拍锁定）
    按电路深度均匀采样关键帧——H/CX 各一层，天然呈现"门作用中"的过渡。
    """
    from qiskit import QuantumCircuit

    from gpuqviz.evolve import sample_circuit

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    key = np.asarray(sample_circuit(qc, steps=4))  # |0…0⟩ → H 后 → CX 后
    t = np.linspace(0, len(key) - 1, frames)
    i0 = np.clip(np.floor(t).astype(int), 0, len(key) - 2)
    w = (t - i0).reshape(-1, 1)
    out = key[i0] * (1 - w) + key[i0 + 1] * w
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    states = make_ghz_states(n_qubits=3, frames=60)
    npz = out_dir / "ghz_states.npz"
    np.savez(npz, states=states)

    # 专业侧核对：初/末帧纠缠结构
    from gpuqviz.analysis import entanglement_summary
    rep0 = entanglement_summary(states[0])
    rep1 = entanglement_summary(states[-1])
    print(f"初帧: 单qubit熵={rep0.single_entropy.round(3)}, "
          f"总相关={rep0.total_correlation():.3f}")
    print(f"末帧: 单qubit熵={rep1.single_entropy.round(3)}, "
          f"总相关={rep1.total_correlation():.3f} (GHZ 理论值 3.0)")

    scene = Scene(
        width=1920, height=1080, fps=30, duration=4.0,
        background="#0b0e14",
        title="GHZ — 纠缠演化",
        tracks=[
            BlochTrack(states_path="ghz_states.npz", trail=True, layout="top"),
            EntanglementTrack(states_path="ghz_states.npz", layout="bottom"),
        ],
    )
    scene.save_json(out_dir / "entanglement_demo_scene.json")

    out_mp4 = out_dir / "entanglement_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render(scene, out=out_mp4, states_dir=out_dir)
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    out_png = out_dir / "entanglement_demo.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"末帧静态图: {out_png}")

    print("\n✅ 纠缠图 Track 验收输出完成")


if __name__ == "__main__":
    main()
