"""量子隐形传态（Quantum Teleportation）电路构建器。

Alice 要将她持有的 qubit 0 上的未知态 |ψ⟩ 传给 Bob 的 qubit 2。
利用 Bell pair（qubit 1, 2）作为量子信道 + 经典通信。

电路结构（3 qubit，q0=Alice 数据, q1=Alice Bell, q2=Bob）：
1. 制备 Bell pair：H(q1) → CX(q1, q2)
2. Alice Bell 测量：CX(q0, q1) → H(q0)
3. 经典校正：CX(q1, q2) → CZ(q0, q2)
4. 最终 q2 = |ψ⟩

注意：态矢量模拟中不做真实测量坍缩，用受控校正门等效替代。
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


def teleportation(engine: str = "qiskit", prepare_state: str = "rx"):
    """量子隐形传态电路。

    Parameters
    ----------
    engine : str
        "qiskit" / "pyqpanda" / "numpy"。
    prepare_state : str
        q0 上制备的初始态："rx" (RX(π/3)) 或 "ry" (RY(π/4)) 或 "plus" (H)。
    """
    engine = _check_engine(engine)

    if engine == "qiskit":
        qc = _qiskit_circuit(3)
        # 在 q0 上制备未知态 |ψ⟩
        if prepare_state == "rx":
            qc.rx(2 * 3.14159265358979 / 3, 0)  # RX(2π/3)
        elif prepare_state == "ry":
            qc.ry(3.14159265358979 / 4, 0)      # RY(π/4)
        elif prepare_state == "plus":
            qc.h(0)
        else:
            qc.h(0)
        # Bell pair (q1, q2)
        qc.h(1)
        qc.cx(1, 2)
        # Alice 测量（等效：CX + H 后做受控校正，不显式测量）
        qc.cx(0, 1)
        qc.h(0)
        # 经典校正
        qc.cx(1, 2)
        qc.cz(0, 2)
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use teleportation_pyqpanda(qubits, machine, ...)")

    # numpy 路径
    gates: list[Gate] = []
    if prepare_state == "rx":
        gates.append(Gate("RX", targets=[0], params=[2 * 3.14159265358979 / 3]))
    elif prepare_state == "ry":
        gates.append(Gate("RY", targets=[0], params=[3.14159265358979 / 4]))
    else:
        gates.append(Gate("H", targets=[0]))
    # Bell pair
    gates.append(Gate("H", targets=[1]))
    gates.append(Gate("CX", targets=[2], controls=[1]))
    # Alice
    gates.append(Gate("CX", targets=[1], controls=[0]))
    gates.append(Gate("H", targets=[0]))
    # 校正
    gates.append(Gate("CX", targets=[2], controls=[1]))
    gates.append(Gate("CZ", targets=[2], controls=[0]))
    return _numpy_gates_to_circuit(gates)


def teleportation_pyqpanda(qubits, machine, prepare_state: str = "rx"):
    """pyqpanda 路径量子隐形传态。"""
    prog = _pyqpanda_prog()
    q0, q1, q2 = qubits[0], qubits[1], qubits[2]
    if prepare_state == "rx":
        _pyqpanda_gate(prog, "RX", [q0], [2 * 3.14159265358979 / 3])
    elif prepare_state == "ry":
        _pyqpanda_gate(prog, "RY", [q0], [3.14159265358979 / 4])
    else:
        _pyqpanda_gate(prog, "H", [q0])
    _pyqpanda_gate(prog, "H", [q1])
    _pyqpanda_gate(prog, "CNOT", [q1, q2])
    _pyqpanda_gate(prog, "CNOT", [q0, q1])
    _pyqpanda_gate(prog, "H", [q0])
    _pyqpanda_gate(prog, "CNOT", [q1, q2])
    _pyqpanda_gate(prog, "CZ", [q0, q2])
    return prog
