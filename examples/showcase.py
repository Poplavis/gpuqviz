"""效果展示：多层旋转门电路 → 相机环绕 + 布洛赫球轨迹 + 概率热图 → out/showcase.mp4

运行::

    python examples/showcase.py
"""

import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from gpuqviz import render
from gpuqviz.evolve import sample_circuit
from gpuqviz.scene import BlochTrack, Camera, HeatmapTrack, Scene


def build_circuit() -> QuantumCircuit:
    """多层小角度旋转 + 纠缠：让两个 qubit 全程保持可视化的连续运动。"""
    qc = QuantumCircuit(2)
    qc.h(0)
    for i in range(1, 7):  # 6 层小角度旋转，穿插纠缠
        qc.rx(np.pi / 6, 0)
        qc.ry(np.pi / 5, 1)
        qc.rz(np.pi / 7, 0)
        if i == 3:
            qc.cx(0, 1)          # 中途纠缠一次
        qc.rz(np.pi / 8, 1)
        qc.rx(np.pi / 9, 1)
    qc.cx(0, 1)                  # 末尾再纠缠
    return qc


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    qc = build_circuit()
    print(f"circuit depth = {qc.depth()}, qubits = {qc.num_qubits}")

    # 生成关键帧态矢量数据（Scene 的 states_path 指向它）
    key_states = np.stack(sample_circuit(qc, steps=120)).astype(np.complex64)
    npz = out_dir / "showcase_states.npz"
    np.savez(npz, states=key_states)

    scene = Scene(
        width=1920, height=1080, fps=60, duration=8.0,
        background="#0b0e14",
        title="量子态演化演示 · Bell 型电路",
        camera=Camera(azimuth=(0.0, 120.0), elevation=(18.0, 42.0), zoom=1.15),
        tracks=[
            BlochTrack(states_path=str(npz.resolve()), trail=True, layout="top"),
            HeatmapTrack(states_path=str(npz.resolve()), basis="probability",
                         layout="bottom"),
        ],
    )

    t0 = time.perf_counter()
    render(scene, out="out/showcase.mp4", quality=0.92)
    print(f"total wall time: {time.perf_counter() - t0:.1f}s -> out/showcase.mp4")


if __name__ == "__main__":
    main()
