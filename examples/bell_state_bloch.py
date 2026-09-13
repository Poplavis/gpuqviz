"""S2 验收：Bell 态（H+CNOT）→ 布洛赫球动画 out/bell.mp4。

运行::

    python examples/bell_state_bloch.py
"""

import time

from qiskit import QuantumCircuit

from gpuqviz import render_bloch_video

if __name__ == "__main__":
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    t0 = time.perf_counter()
    path = render_bloch_video(
        circuit=qc,
        steps=120,
        fps=60,
        out="out/bell.mp4",
        style="dark",
        trail=True,
        seconds=6.0,
    )
    print(f"total wall time: {time.perf_counter() - t0:.2f}s, output: {path}")
