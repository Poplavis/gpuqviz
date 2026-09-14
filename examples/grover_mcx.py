"""S6 示例：含 mcx 的 3-qubit Grover 算法 → out/grover.mp4。

目标态 |101⟩（q0=1, q1=0, q2=1）。两轮 Grover 迭代后目标态振幅被放大。
全程零中间文件，验证 gpuqviz 对 mcx/复合门/多层电路的兼容性。

运行::

    python examples/grover_mcx.py
"""

import time

import numpy as np
from qiskit import QuantumCircuit

from gpuqviz import render_bloch_video
from gpuqviz.adapters import qiskit_to_gates
from gpuqviz.circuits import evolve_gates


def grover_oracle(qc: QuantumCircuit) -> None:
    """标记 |101⟩：对 q1 取反后做 MCX 踢回相位，再还原 q1。"""
    qc.x(1)
    qc.h(2)
    qc.mcx([0, 1], 2)
    qc.h(2)
    qc.x(1)


def grover_diffusion(qc: QuantumCircuit) -> None:
    """Grover 扩散算子（关于 |+⟩ 的反射）。"""
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.h(2)
    qc.mcx([0, 1], 2)
    qc.h(2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h([0, 1, 2])
    for _ in range(2):  # 两轮 Grover 迭代
        grover_oracle(qc)
        grover_diffusion(qc)
    return qc


def main() -> None:
    qc = build_circuit()
    # 先打印保真度与目标态概率，方便肉眼核对数值正确性
    n, gates = qiskit_to_gates(qc)
    final = evolve_gates(n, gates)[-1]
    p101 = float(np.abs(final[5]) ** 2)
    print(f"[grover] n_qubits={n}, gates={len(gates)}, P(|101⟩)={p101:.4f}")

    t0 = time.perf_counter()
    render_bloch_video(
        circuit=qc,
        steps=180,
        fps=60,
        seconds=9.0,
        out="out/grover.mp4",
        style="dark",
        trail=True,
    )
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")
    print("output: out/grover.mp4")


if __name__ == "__main__":
    main()
