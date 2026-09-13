"""S4 验收：Scene JSON 场景 → out/scene.mp4（含中文标题、相机环绕动画）。

运行::

    python examples/scene_demo.py
    # 等价 CLI：gpuqviz render examples/scene.json -o out/scene.mp4
"""

import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from gpuqviz import render
from gpuqviz.evolve import sample_circuit
from gpuqviz.scene import BlochTrack, Camera, HeatmapTrack, Scene


def main():
    examples_dir = Path(__file__).parent

    # 1) 生成态矢量数据（npz）
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(np.pi / 2, 1)
    key_states = np.stack(sample_circuit(qc, steps=90)).astype(np.complex64)
    npz_path = examples_dir / "bell_states.npz"
    np.savez(npz_path, states=key_states)

    # 2) 场景定义（也可直接手写等价 JSON）
    scene = Scene(
        duration=5.0, fps=60,
        background="#0b0e14",
        title="贝尔态演化",
        camera=Camera(azimuth=(0.0, 60.0), elevation=(25.0, 35.0)),
        tracks=[
            BlochTrack(states_path="bell_states.npz", trail=True, layout="top"),
            HeatmapTrack(states_path="bell_states.npz", basis="probability",
                         layout="bottom"),
        ],
    )
    scene.save_json(examples_dir / "scene.json")

    # 3) 渲染（states_path 相对 JSON 所在目录解析）
    render(scene, out="out/scene.mp4", states_dir=examples_dir)
    print("done -> out/scene.mp4")


if __name__ == "__main__":
    sys.exit(main())
