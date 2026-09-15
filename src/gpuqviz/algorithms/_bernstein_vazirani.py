"""Bernstein-Vazirani 算法电路构建器。

给定一个隐藏字符串 s ∈ {0,1}^n 和函数 f(x) = s · x (mod 2)，
BV 算法用一次量子查询恢复 s。

电路结构：
1. 辅助 qubit |1⟩ → H
2. 输入寄存器 H(all)
3. Oracle Uf：对每个 s_i = 1 的 qubit 做 CX(i, aux)
4. 输入寄存器 H(all)
5. 测量输入寄存器 → 得到 |s⟩
"""

from __future__ import annotations

from ._engine import (
    _check_engine,
    _numpy_gates_to_circuit,
    _qiskit_circuit,
)
from ..circuits import Gate


def bernstein_vazirani(secret: str = "101", engine: str = "qiskit"):
    """Bernstein-Vazirani 算法电路。

    Parameters
    ----------
    secret : str
        隐藏比特串，如 "101"（长度 = n）。从左到右对应 q0, q1, …, q_{n-1}。
    engine : str
        "qiskit" / "numpy"。
    """
    engine = _check_engine(engine)
    # secret 字符串："101" → s[0]='1', s[1]='0', s[2]='1'
    # qubit i 对应 secret[i]
    n = len(secret)
    if n < 1:
        raise ValueError(f"secret must be non-empty, got {secret!r}")
    bits = [int(c) for c in secret if c in "01"]
    if len(bits) != n:
        raise ValueError(f"secret must be binary string, got {secret!r}")

    n_total = n + 1
    aux = n

    if engine == "qiskit":
        qc = _qiskit_circuit(n_total)
        qc.x(aux)
        qc.h(range(n_total))
        # Oracle：s_i = 1 时 CX(i, aux)
        for i in range(n):
            if bits[i]:
                qc.cx(i, aux)
        qc.h(range(n))
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use bernstein_vazirani_pyqpanda(qubits, machine, ...)")

    # numpy 路径
    gates: list[Gate] = []
    gates.append(Gate("X", targets=[aux]))
    for i in range(n_total):
        gates.append(Gate("H", targets=[i]))
    for i in range(n):
        if bits[i]:
            gates.append(Gate("CX", targets=[aux], controls=[i]))
    for i in range(n):
        gates.append(Gate("H", targets=[i]))
    return _numpy_gates_to_circuit(gates)
