"""全场景基准：bell / ghz / scene 在 gl(GPU) 与 cpu 后端的耗时对照 → docs/benchmarks.md。

运行::

    python benchmarks/suite.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from gpuqviz import render_bloch_video, render
from gpuqviz.evolve import sample_circuit
from gpuqviz.scene import BlochTrack, HeatmapTrack, Scene

ROOT = Path(__file__).parent.parent
SPECS = [("720p", 1280, 720)]  # 软光栅在 1080p 下过慢，基准用 720p


def bench(name, fn):
    t0 = time.perf_counter()
    fn()
    return round(time.perf_counter() - t0, 1)


def main():
    qc_bell = QuantumCircuit(2)
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    qc_ghz = QuantumCircuit(3)
    qc_ghz.h(0)
    qc_ghz.cx(0, 1)
    qc_ghz.cx(1, 2)

    rows = []
    for backend in ("gl", "cpu"):
        tag = f"{backend}"
        rows.append((tag, "bell 3s@30fps",
                     bench(backend, lambda: render_bloch_video(
                         circuit=qc_bell, steps=30, fps=30, seconds=3,
                         out=f"out/bench_{backend}_bell.mp4", backend=backend))))
        if backend == "gl":  # cpu 软光栅不支持热图/scene（limited）
            states = np.stack(sample_circuit(qc_ghz, steps=30)).astype(np.complex64)
            np.savez(ROOT / "examples" / "bench_states.npz", states=states)
            scene = Scene(duration=3.0, fps=30, tracks=[
                BlochTrack(states_path="bench_states.npz", layout="top"),
                HeatmapTrack(states_path="bench_states.npz", layout="bottom"),
            ])
            rows.append((tag, "ghz split 3s@30fps",
                         bench(backend, lambda: render(
                             scene, out="out/bench_gl_ghz.mp4",
                             states_dir=ROOT / "examples"))))

    lines = [
        "# gpuqviz 基准", "",
        f"环境：消费级 NVIDIA GPU（NVENC 会话不可用，编码走 libx264）", "",
        "| 后端 | 场景 | 耗时 (s) |", "|---|---|---|",
    ]
    lines += [f"| {t} | {s} | {v} |" for t, s, v in rows]
    out_md = ROOT / "docs" / "benchmarks.md"
    out_md.parent.mkdir(exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out_md}")


if __name__ == "__main__":
    main()
