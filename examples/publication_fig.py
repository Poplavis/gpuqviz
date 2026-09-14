"""S7 示例：出版级静态图导出。

- Bell 态 bw（论文黑白）风格 3200×2000 PNG，scale=4 超采样
- 8-qubit GHZ 场景 cols=2 多行布局 PNG

运行::

    python examples/publication_fig.py
"""

from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from gpuqviz import render_frame
from gpuqviz.evolve import sample_circuit


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def ghz8_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(8)
    qc.h(0)
    for i in range(7):
        qc.cx(i, i + 1)
    return qc


def main() -> None:
    out = Path("out")
    out.mkdir(parents=True, exist_ok=True)

    # 1) Bell 态 bw 论文风格：figsize=(32, 20) → 3200×2000，scale=4 抗锯齿
    bell = sample_circuit(bell_circuit(), steps=60)
    p1 = render_frame(
        states=bell, t=1.0, out=out / "bell_bw_3200x2000.png",
        scale=4, style="bw", figsize=(32.0, 20.0),
    )
    print(f"[1] Bell bw figure -> {p1}")

    # 2) 8-qubit GHZ cols=2 多行布局
    ghz = sample_circuit(ghz8_circuit(), steps=60)
    p2 = render_frame(
        states=ghz, t=1.0, out=out / "ghz8_cols2.png",
        scale=2, style="dark", cols=2, figsize=(16.0, 12.0),
    )
    print(f"[2] GHZ-8 cols=2 figure -> {p2}")

    # 3) Bell 态 dark 风格中间时刻（t=0.5 演化中），用于论文对比图
    p3 = render_frame(
        states=bell, t=0.5, out=out / "bell_dark_mid.png",
        scale=4, style="dark", figsize=(16.0, 10.0),
    )
    print(f"[3] Bell dark mid-evolution figure -> {p3}")


if __name__ == "__main__":
    main()
