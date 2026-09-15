"""超密编码（Superdense Coding）电路构建器。

Alice 通过发送 1 个 qubit 给 Bob，传递 2 经典比特信息。
利用共享 Bell pair。

电路结构（2 qubit，q0=Alice, q1=Bob）：
1. 制备 Bell pair：H(q0) → CX(q0, q1)
2. Alice 编码（根据 message）：
   - "00"：不做操作
   - "01"：Z(q0)
   - "10"：X(q0)
   - "11"：ZX(q0) = iY(q0)
3. Bob 解码：CX(q0, q1) → H(q0)
4. 测量 → 得到 message
"""

from __future__ import annotations

from ._engine import (
    _check_engine,
    _numpy_gates_to_circuit,
    _pyqpanda_gate,
    _pyqpanda_prog,
    _qiskit_circuit,
)
from ..circuits import Gate


def superdense(message: str = "11", engine: str = "qiskit"):
    """超密编码电路。

    Parameters
    ----------
    message : str
        要传输的 2 比特消息："00" / "01" / "10" / "11"。
    engine : str
        "qiskit" / "pyqpanda" / "numpy"。
    """
    engine = _check_engine(engine)
    if message not in ("00", "01", "10", "11"):
        raise ValueError(f"message must be '00'/'01'/'10'/'11', got {message!r}")
    bit0, bit1 = int(message[0]), int(message[1])

    if engine == "qiskit":
        qc = _qiskit_circuit(2)
        # Bell pair
        qc.h(0)
        qc.cx(0, 1)
        # Alice 编码
        if bit1:
            qc.x(0)
        if bit0:
            qc.z(0)
        # Bob 解码
        qc.cx(0, 1)
        qc.h(0)
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use superdense_pyqpanda(qubits, machine, ...)")

    # numpy 路径
    gates: list[Gate] = []
    gates.append(Gate("H", targets=[0]))
    gates.append(Gate("CX", targets=[1], controls=[0]))
    if bit1:
        gates.append(Gate("X", targets=[0]))
    if bit0:
        gates.append(Gate("Z", targets=[0]))
    gates.append(Gate("CX", targets=[1], controls=[0]))
    gates.append(Gate("H", targets=[0]))
    return _numpy_gates_to_circuit(gates)


def superdense_pyqpanda(qubits, machine, message: str = "11"):
    """pyqpanda 路径超密编码。"""
    if message not in ("00", "01", "10", "11"):
        raise ValueError(f"message must be '00'/'01'/'10'/'11', got {message!r}")
    bit0, bit1 = int(message[0]), int(message[1])
    q0, q1 = qubits[0], qubits[1]
    prog = _pyqpanda_prog()
    _pyqpanda_gate(prog, "H", [q0])
    _pyqpanda_gate(prog, "CNOT", [q0, q1])
    if bit1:
        _pyqpanda_gate(prog, "X", [q0])
    if bit0:
        _pyqpanda_gate(prog, "Z", [q0])
    _pyqpanda_gate(prog, "CNOT", [q0, q1])
    _pyqpanda_gate(prog, "H", [q0])
    return prog
