"""Simon 算法电路构建器。

给定一个 2-to-1 函数 f，满足 f(x) = f(x ⊕ s)（s 为隐藏字符串），
Simon 算法用 O(n) 次量子查询找到 s。

电路结构（2n qubit：n 输入 + n 输出）：
1. 输入寄存器 H(all)
2. Oracle Uf：|x⟩|y⟩ → |x⟩|y ⊕ f(x)⟩
3. 输入寄存器 H(all)
4. 测量输入寄存器 → 得到 y 满足 y · s = 0 (mod 2)
5. 收集 n 个线性无关的 y 后，解线性方程组得到 s

Oracle 实现（f(x) = x ⊕ s 的简化版本，s 作用在输出寄存器）：
对每个 s_i = 1，CX(input_i, output_i) 实现 f(x) = x ⊕ s。
"""

from __future__ import annotations

from ._engine import (
    _check_engine,
    _numpy_gates_to_circuit,
    _qiskit_circuit,
)
from ..circuits import Gate


def simon(s: str = "01", n: int | None = None, engine: str = "qiskit"):
    """Simon 算法电路。

    Parameters
    ----------
    s : str
        隐藏字符串（二进制），如 "01"。长度 = n。
    n : int | None
        输入寄存器量子比特数。None 时取 len(s)。
    engine : str
        "qiskit" / "numpy"。
    """
    engine = _check_engine(engine)
    if n is None:
        n = len(s)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    bits = [int(c) for c in s if c in "01"]
    if len(bits) != len(s):
        raise ValueError(f"s must be binary string, got {s!r}")
    # 补齐到 n 位
    while len(bits) < n:
        bits.append(0)

    n_total = 2 * n  # 前 n 个输入，后 n 个输出

    if engine == "qiskit":
        qc = _qiskit_circuit(n_total)
        # 输入寄存器 H
        qc.h(range(n))
        # Oracle：f(x) = x ⊕ s
        # |x⟩|0⟩ → |x⟩|x ⊕ s⟩
        # 先复制 x 到输出：CX(input_i, output_i)
        for i in range(n):
            qc.cx(i, n + i)
        # 再叠加 s：对 s_i = 1 的输出 qubit 做 X
        for i in range(n):
            if bits[i]:
                qc.x(n + i)
        # 输入寄存器 H
        qc.h(range(n))
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use simon_pyqpanda(qubits, machine, ...)")

    # numpy 路径
    gates: list[Gate] = []
    for i in range(n):
        gates.append(Gate("H", targets=[i]))
    for i in range(n):
        gates.append(Gate("CX", targets=[n + i], controls=[i]))
    for i in range(n):
        if bits[i]:
            gates.append(Gate("X", targets=[n + i]))
    for i in range(n):
        gates.append(Gate("H", targets=[i]))
    return _numpy_gates_to_circuit(gates)
