"""量子随机游走（Quantum Random Walk）电路构建器。

在 n 个位置节点上的离散时间量子游走，使用 1 个硬币 qubit + log2(n) 个位置 qubit。
硬币 qubit 做 H（公平硬币），条件移位根据硬币状态左/右移位。

为可视化目的，使用简化的循环图上的量子游走：
- 硬币 qubit：H 操作
- 位置寄存器：受控移位（条件 SWAP / 加法器）
- 循环 steps 次

对于小规模 n，用 SWAP 链模拟位置转移。
"""

from __future__ import annotations

from ._engine import (
    _check_engine,
    _numpy_gates_to_circuit,
    _qiskit_circuit,
)
from ..circuits import Gate


def quantum_walk(n: int = 3, steps: int = 3, engine: str = "qiskit"):
    """量子随机游走电路。

    Parameters
    ----------
    n : int
        位置 qubit 数（位置数 = 2^n）。n=1 时为 2 位置的最简游走。
    steps : int
        游走步数（每步 = 硬币 H + 条件移位）。
    engine : str
        "qiskit" / "numpy"。
    """
    engine = _check_engine(engine)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    # 电路结构：q0 = 硬币，q1..qn = 位置寄存器
    # 每步：H(硬币) → 条件移位
    # 条件移位：硬币=0 时右移（SWAP 链），硬币=1 时左移
    # 简化版：硬币控制位置寄存器的增量/减量
    # 对于 n=1（2 位置）：硬币控制 SWAP(q1, q1) = 无操作 → 退化为 H + CNOT
    # 对于 n>=2：用受控 SWAP 实现

    n_total = 1 + n  # q0 = 硬币, q1..qn = 位置
    coin = 0

    if engine == "qiskit":
        qc = _qiskit_circuit(n_total)
        # 初始化位置在 |0…0⟩
        for _ in range(steps):
            # 硬币翻转
            qc.h(coin)
            # 条件移位：根据硬币态移动位置
            if n == 1:
                # 2 位置：硬币=1 时翻转位置
                qc.cx(coin, 1)
            else:
                # 多位置：用受控 SWAP 链实现循环移位
                # 硬币=0：右移（q1→q2→…→qn→q1）
                # 硬币=1：左移
                # 简化：用 CNOT 链实现二进制增量/减量
                _conditional_shift_qiskit(qc, coin, n)
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use quantum_walk_pyqpanda(qubits, machine, ...)")

    # numpy 路径
    gates: list[Gate] = []
    for _ in range(steps):
        gates.append(Gate("H", targets=[coin]))
        if n == 1:
            gates.append(Gate("CX", targets=[1], controls=[coin]))
        else:
            _conditional_shift_numpy(gates, coin, n)
    return _numpy_gates_to_circuit(gates)


def _conditional_shift_qiskit(qc, coin: int, n: int):
    """条件移位：硬币控制位置寄存器的循环移位。

    硬币=|0⟩：不移动
    硬币=|1⟩：循环左移（SWAP 相邻位置 qubit）

    用受控 SWAP（cswap）实现，受控 SWAP 在适配器中经矩阵路径翻译。
    """
    pos_qubits = list(range(1, n + 1))
    # 受控 SWAP 链：硬币=1 时循环左移位置寄存器
    for i in range(n - 1):
        qc.cswap(coin, pos_qubits[i], pos_qubits[i + 1])
    if n > 1:
        qc.cswap(coin, pos_qubits[-1], pos_qubits[0])


def _conditional_shift_numpy(gates: list[Gate], coin: int, n: int):
    """numpy 路径条件移位：受控 SWAP 用显式 3-qubit 矩阵实现。"""
    from ..circuits import Gate
    import numpy as np
    # Fredkin gate (CSWAP) 8x8 矩阵：control=1 时 SWAP target1/target2
    # qubit 顺序 [control, target1, target2]，LSB = control
    fredkin = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.complex128)
    pos_qubits = list(range(1, n + 1))
    for i in range(n - 1):
        # UNITARY 门：targets = [coin, pos_i, pos_{i+1}]，LSB=coin
        gates.append(Gate("UNITARY", targets=[coin, pos_qubits[i], pos_qubits[i + 1]],
                          matrix=fredkin))
    if n > 1:
        gates.append(Gate("UNITARY", targets=[coin, pos_qubits[-1], pos_qubits[0]],
                          matrix=fredkin))
