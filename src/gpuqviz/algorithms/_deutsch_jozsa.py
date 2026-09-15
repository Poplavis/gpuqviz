"""Deutsch-Jozsa 算法电路构建器。

判断函数 f: {0,1}^n → {0,1} 是常数（所有输出相同）还是平衡（一半 0 一半 1）。

电路结构：
1. 辅助 qubit |1⟩ → H
2. 输入寄存器 H(all)
3. Oracle Uf（常数/平衡两种）
4. 输入寄存器 H(all)
5. 测量输入寄存器：全 0 = 常数，否则 = 平衡
"""

from __future__ import annotations

from ._engine import (
    _check_engine,
    _numpy_gates_to_circuit,
    _qiskit_circuit,
)
from ..circuits import Gate


def _build_balanced_oracle_qiskit(qc, n: int):
    """平衡 oracle：f(x) = x_0（第一个 qubit 的值）。
    用 CX(0, aux) 实现：aux ^= x_0。
    """
    aux = n  # 辅助 qubit 索引
    qc.cx(0, aux)


def _build_constant_oracle_qiskit(qc, n: int):
    """常数 oracle：f(x) = 0（不做任何操作）或 f(x) = 1（翻转 aux）。
    这里用 f(x) = 1：X(aux)。
    """
    aux = n
    qc.x(aux)


def deutsch_jozsa(oracle_type: str = "balanced", n: int = 3,
                  engine: str = "qiskit"):
    """Deutsch-Jozsa 算法电路。

    Parameters
    ----------
    oracle_type : str
        "balanced" 或 "constant"。
    n : int
        输入寄存器量子比特数（总 qubit = n + 1 辅助）。
    engine : str
        "qiskit" / "numpy"。
    """
    engine = _check_engine(engine)
    if oracle_type not in ("balanced", "constant"):
        raise ValueError(f"oracle_type must be 'balanced' or 'constant', got {oracle_type!r}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    n_total = n + 1
    aux = n

    if engine == "qiskit":
        qc = _qiskit_circuit(n_total)
        # 辅助 qubit |1⟩
        qc.x(aux)
        # 全部 H
        qc.h(range(n_total))
        # Oracle
        if oracle_type == "balanced":
            _build_balanced_oracle_qiskit(qc, n)
        else:
            _build_constant_oracle_qiskit(qc, n)
        # 输入寄存器 H
        qc.h(range(n))
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use deutsch_jozsa_pyqpanda(qubits, machine, ...)")

    # numpy 路径
    gates: list[Gate] = []
    gates.append(Gate("X", targets=[aux]))
    for i in range(n_total):
        gates.append(Gate("H", targets=[i]))
    if oracle_type == "balanced":
        gates.append(Gate("CX", targets=[aux], controls=[0]))
    else:
        gates.append(Gate("X", targets=[aux]))
    for i in range(n):
        gates.append(Gate("H", targets=[i]))
    return _numpy_gates_to_circuit(gates)
