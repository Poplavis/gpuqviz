"""HHL 线性求解（对角 A 演示级）：QPE + 条件旋转 + 逆计算。

数学基础（tests/cross_validation/test_shor_hhl.py 锁定）：

- A = diag(λ_0..λ_{2^n-1})，λ_k = 2π(k+1)/(2^c·t)（正、非零、
  QPE 在 c 个 clock qubit 下精确——λ_k·t·2^c/2π = k+1 ∈ ℤ）；
- b 任意（计算基叠加）；电路输出：ancilla=1 分支的 input 寄存器态
  ∝ A⁻¹|b⟩ = Σ (b_k/λ_k)|k⟩（归一化后）。

线路（2 input + c clock + 1 ancilla）：
  1. input = |b⟩（UNITARY 态制备）
  2. clock = H^⊗c
  3. QPE：controlled e^{+i·2^j·t·A}（对角 UNITARY on clock_j + input）
  4. Ancilla 旋转：MCRY(θ_m) on clock = |m⟩（X 共轭处理 0-bit）
  5. Uncompute QPE（逆序 + 共轭相位）+ H^⊗c
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["hhl", "hhl_circuit", "hhl_eigenvalues", "hhl_classical_solution",
           "hhl_conditional_state"]


def hhl(b=None, clock_bits: int = 3, C: float = 0.6, engine: str = "qiskit"):
    """注册表入口：HHL 线性求解（对角 A 演示级）。

    b 缺省为 [1, 0.5, 2, 1]；需要元数据（λ、经典参考解）时直接调
    hhl_circuit（返回 (电路, meta)）。
    """
    if b is None:
        b = np.array([1.0, 0.5, 2.0, 1.0])
    circuit, _meta = hhl_circuit(b, clock_bits=clock_bits, C=C,
                                 engine=engine)
    return circuit


def _qft_matrix(bits: int) -> np.ndarray:
    """正 QFT 矩阵：QFT[s, x] = e^{+2πi·sx/2^bits}/√2^bits（LSB-first 整数）。

    QPE 的相位回授态是 Σ_x e^{+2πi·φx}|x⟩（x = Σ 2^j·clock_j），
    uncompute 需要它把 clock 读数 |m̃⟩ 恢复成相位态，供逆受控 U 逐位抵消。
    """
    dim = 1 << bits
    s, x = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
    return np.exp(2j * np.pi * s * x / dim) / np.sqrt(dim)


def hhl_eigenvalues(clock_bits: int, t: float = 1.0) -> np.ndarray:
    """A 的特征值（对角）：λ_m = 2π(m+1)/(2^c·t)，m = 0..2^c-1。

    选择此形式使 QPE 精确：λ_m·t·2^c/(2π) = m+1 ∈ ℤ。
    """
    return np.array([2 * np.pi * (m + 1) / (2 ** clock_bits * t)
                     for m in range(2 ** clock_bits)])


def hhl_classical_solution(b, lambdas) -> np.ndarray:
    """经典参考：A⁻¹b 归一化。"""
    b = np.asarray(b, dtype=np.float64)
    sol = b / lambdas[:len(b)]
    return sol / np.linalg.norm(sol)


def hhl_conditional_state(full_psi: np.ndarray, n_input: int,
                          clock_bits: int) -> tuple[np.ndarray, float]:
    """从 HHL 全态提取 ancilla=1 分支的 input 寄存器态。

    返回 (归一化 input 态, P(ancilla=1))。
    Qubit 布局：input = qubit 0..n_input-1（LSB），clock 在后，
    ancilla = 最高位 qubit。
    """
    n_clock = clock_bits
    anc_bit = n_input + n_clock
    dim = 1 << (n_input + n_clock + 1)
    probs_1 = 0.0
    sol = np.zeros(2 ** n_input, dtype=np.complex128)
    for idx in range(dim):
        if (idx >> anc_bit) & 1:
            amp = full_psi[idx]
            inp_idx = idx & (2 ** n_input - 1)
            sol[inp_idx] += amp
            probs_1 += abs(amp) ** 2
    norm = np.linalg.norm(sol)
    if norm > 1e-12:
        sol = sol / norm
    return sol, probs_1


def _prep_unitary(v: np.ndarray) -> np.ndarray:
    """|0…0⟩ → |v⟩ 的酉矩阵（第 0 列 = v，其余列由 [v|I] 的 QR 补全）。

    numpy 引擎用 UNITARY 门携带此矩阵：初态恒为 |0…0⟩ 时仅第 0 列生效，
    但矩阵本身必须是真酉矩阵（qiskit Statevector 对拍、交叉验证的前提）。
    """
    v = np.asarray(v, dtype=np.complex128)
    n = v.size
    q, _ = np.linalg.qr(np.hstack([v.reshape(-1, 1), np.eye(n)]))
    if np.vdot(q[:, 0], v).real < 0:  # QR 首列符号对齐 v
        q[:, 0] *= -1
    return q


def hhl_circuit(b, clock_bits: int = 3, C: float = 0.6, t: float = 1.0,
                engine: str = "qiskit"):
    """HHL 线性求解电路（对角 A，精确 QPE）。

    b: (2^n_input,) 复数/实数振幅（自动归一化）。
    clock_bits: QPE 精度（c=3 对 λ ∈ {2π/8, π/4, 3π/8, π/2} 精确）。
    C: HHL 条件旋转常数（C ≤ λ_min）。
    返回 (QuantumCircuit 或 Gate 列表, dict 元数据)。

    Qubit 布局：input = qubit 0..n-1（LSB），clock = 后 c 个，ancilla = 最高位。
    """
    b = np.asarray(b, dtype=np.float64)
    n_input = int(round(np.log2(len(b))))
    dim_input = 2 ** n_input
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-12:
        raise ValueError("b is zero")
    b_normalized = b / b_norm

    lambdas = hhl_eigenvalues(clock_bits, t)
    C = min(C, float(lambdas.min()) * 0.99)  # C ≤ λ_min

    n_total = n_input + clock_bits + 1
    anc = n_total - 1
    clock_qs = list(range(n_input, n_input + clock_bits))
    input_qs = list(range(n_input))

    # QPE 对角门：U_j = exp(+i·2^j·t·A)（正相位使 clock 读出 m+1）
    def _qpe_diag_matrix(j: int) -> np.ndarray:
        """3-qubit 对角门 (clock_j, input) 的 8×8 矩阵。"""
        D = np.eye(8, dtype=np.complex128)
        for inp in range(dim_input):
            phase = np.exp(1j * (1 << j) * t * lambdas[inp])
            # index: clock_j bit + 2·inp（targets 顺序 [clock_j] + input_qs）
            for cj in (0, 1):
                idx = cj + (inp << 1)
                D[idx, idx] = (1.0 if cj == 0 else phase)
        return D

    def _ancilla_angle(m: int) -> float:
        """R_y(θ_m) 使得 sin(θ_m/2) = C/λ_m（m = clock 读数）。

        QPE 读数 r 对应特征值 λ = 2π·r/(2^c·t) = lambdas[r-1]。
        """
        lam_r = 2 * np.pi * m / (2 ** clock_bits * t)
        ratio = min(C / lam_r, 1.0) if lam_r > 1e-12 else 0.0
        return 2 * math.asin(min(max(ratio, 0.0), 1.0))

    if engine == "qiskit":
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate, RYGate

        qc = QuantumCircuit(n_total)
        # 1. input = |b⟩
        qc.append(UnitaryGate(_prep_unitary(b_normalized)), input_qs)
        # 2. clock = H^⊗c
        qc.h(clock_qs)
        # 3. QPE + 逆 QFT → clock 读出 |m̃⟩
        for j in range(clock_bits):
            D = _qpe_diag_matrix(j)
            qc.append(UnitaryGate(D), [clock_qs[j]] + input_qs)
        qc.append(UnitaryGate(_qft_matrix(clock_bits).conj().T), clock_qs)
        # 4. Ancilla rotations (X-conjugated MCRY for each clock reading m)
        for m in range(1, 2 ** clock_bits):
            theta = _ancilla_angle(m)
            if theta < 1e-10:
                continue
            zero_bits = [clock_qs[b] for b in range(clock_bits)
                         if not (m >> b) & 1]
            for q in zero_bits:
                qc.x(q)
            qc.append(RYGate(theta).control(clock_bits),
                      clock_qs + [anc])
            for q in zero_bits:
                qc.x(q)
        # 5. Uncompute QPE：正 QFT → 逆受控 U（j 逆序）→ H⊗c
        #    （顺序与 numpy 引擎一致；QFT 用显式矩阵避免分解约定歧义）
        qc.append(UnitaryGate(_qft_matrix(clock_bits)), clock_qs)
        for j in reversed(range(clock_bits)):
            D = _qpe_diag_matrix(j)
            qc.append(UnitaryGate(D.conj().T), [clock_qs[j]] + input_qs)
        qc.h(clock_qs)
        meta = {
            "lambdas": lambdas.tolist(),
            "C": C, "t": t, "n_input": n_input,
            "classical_solution": hhl_classical_solution(b, lambdas).tolist(),
        }
        return qc, meta

    # numpy：框架无关 Gate 列表
    from ..circuits import Gate

    gates: list = []

    def _unitary(M, targets):
        gates.append(Gate(name="UNITARY", matrix=M, targets=list(targets)))

    def _x(q):
        gates.append(Gate(name="X", targets=[q]))

    # 1. input = |b⟩
    _unitary(_prep_unitary(b_normalized), input_qs)

    # 2. clock = H^⊗c
    for q in clock_qs:
        gates.append(Gate(name="H", targets=[q]))

    # 3. QPE（受控 e^{+i·2^j·t·A}）+ 逆 QFT → clock 读出 |m̃⟩
    for j in range(clock_bits):
        _unitary(_qpe_diag_matrix(j), [clock_qs[j]] + input_qs)
    _unitary(_qft_matrix(clock_bits).conj().T, clock_qs)

    # 4. Ancilla rotations (X-conjugated MCRY)
    for m in range(1, 2 ** clock_bits):
        theta = _ancilla_angle(m)
        if theta < 1e-10:
            continue
        zero_bits = [clock_qs[b] for b in range(clock_bits)
                     if not (m >> b) & 1]
        for q in zero_bits:
            _x(q)
        gates.append(Gate(name="MCRY", targets=[anc], controls=clock_qs,
                          params=[theta]))
        for q in zero_bits:
            _x(q)

    # 5. Uncompute QPE：正 QFT（把 clock 读数恢复成相位态）→
    #    逆受控 U（j 逆序，相位逐位抵消）→ H⊗c（clock → |0⟩）。
    #    顺序不可颠倒：clock 处于计算基 |m̃⟩ 时先做 C† 只添相位，
    #    逆 QFT 后 clock 仍是叠加态 → 与 input 纠缠 → 解退相干成等幅混合。
    gates.append(Gate(name="UNITARY", matrix=_qft_matrix(clock_bits),
                      targets=list(clock_qs)))
    for j in reversed(range(clock_bits)):
        _unitary(_qpe_diag_matrix(j).conj().T, [clock_qs[j]] + input_qs)
    for q in clock_qs:
        gates.append(Gate(name="H", targets=[q]))

    meta = {
        "lambdas": lambdas.tolist(),
        "C": C, "t": t, "n_input": n_input,
        "classical_solution": hhl_classical_solution(b, lambdas).tolist(),
    }
    return gates, meta
