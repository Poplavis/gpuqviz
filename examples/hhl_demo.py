"""HHL 线性求解演示（对角 A，精确 QPE）→ out/hhl.mp4。

完整流程：
1. A = diag(λ_0..λ_3)，λ_k = 2π(k+1)/8（正、非零，QPE 在 c=3 clock
   qubit 下精确：λ_k·t·2^c/2π = k+1 ∈ ℤ）；b = [1, 0.5, 2, 1]；
2. numpy 演化 6-qubit 电路（2 input + 3 clock + 1 ancilla）：
   QPE → 条件旋转 R_y(θ_m), sin(θ_m/2) = C/λ_m → uncompute；
3. 验证 P(ancilla=1) = Σ (C/λ_k)²·|b_k|²/‖b‖²，ancilla=1 分支的
   input 寄存器态 ≡ A⁻¹|b⟩ 归一化（fidelity = 1）；
4. 全线路渲染为 Bloch 球视频。

正确性由 tests/cross_validation/test_shor_hhl.py 以 1e-10 容差锁定
（含 qiskit 引擎全态对拍与含负振幅的 b）。

运行::

    python examples/hhl_demo.py
"""

import time

import numpy as np

from gpuqviz import render_bloch_video
from gpuqviz.algorithms import hhl_classical_solution, hhl_circuit, hhl_conditional_state
from gpuqviz.circuits import apply_gate


def main() -> None:
    b = np.array([1.0, 0.5, 2.0, 1.0])
    C = 0.6
    gates, meta = hhl_circuit(b, clock_bits=3, C=C, engine="numpy")
    lambdas = np.array(meta["lambdas"])[:4]
    print(f"[hhl] A = diag({lambdas.round(3)}), b = {b}, C = {C}")
    print(f"      6 qubits (2 input + 3 clock + 1 ancilla), gates={len(gates)}")

    # ---- numpy 演化 ----
    state = np.zeros(1 << 6, dtype=np.complex128)
    state[0] = 1.0
    for g in gates:
        state = apply_gate(state, g)

    sol, p1 = hhl_conditional_state(state, n_input=2, clock_bits=3)
    classical = hhl_classical_solution(b, lambdas)
    bn = b / np.linalg.norm(b)
    p_theory = float(np.sum((C / lambdas) ** 2 * bn ** 2))

    print(f"\nP(ancilla=1) = {p1:.6f}（理论 {p_theory:.6f}）")
    print(f"量子解 |x⟩ (|x_k|):  {np.abs(sol).round(4)}")
    print(f"经典 A⁻¹b 归一化:    {np.abs(classical).round(4)}")
    fid = abs(np.vdot(sol, classical)) ** 2
    print(f"fidelity = {fid:.8f}")
    assert fid > 1 - 1e-9

    # ---- 可视化 ----
    qc, _ = hhl_circuit(b, clock_bits=3, C=C, engine="qiskit")
    t0 = time.perf_counter()
    render_bloch_video(
        circuit=qc,
        steps=200,
        fps=60,
        seconds=10.0,
        out="out/hhl.mp4",
        style="dark",
        title="HHL: diag(A) linear solve",
    )
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f}s")
    print("output: out/hhl.mp4")


if __name__ == "__main__":
    main()
