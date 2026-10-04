"""基础量子态制备电路：Bell 态、GHZ 态、均匀叠加态。

三种引擎（qiskit / pyqpanda / numpy）统一构建逻辑。
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

# --------------------------------------------------------------------------- #
# Bell 态：H(q0) + CX(q0, q1) → (|00⟩ + |11⟩)/√2
# --------------------------------------------------------------------------- #

def bell(engine: str = "qiskit"):
    """Bell 态 |Φ+⟩ = (|00⟩ + |11⟩)/√2 的制备电路。2 量子比特。"""
    engine = _check_engine(engine)
    if engine == "qiskit":
        qc = _qiskit_circuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc
    if engine == "pyqpanda":
        # gate 序列与 qiskit 路径一致；qubits 由调用方传入，这里用占位函数
        # 实际 pyqpanda 路径需通过 build_with_machine 封装
        raise NotImplementedError("use bell_pyqpanda(machine) for pyqpanda engine")
    # numpy
    return _numpy_gates_to_circuit([
        Gate("H", targets=[0]),
        Gate("CX", targets=[1], controls=[0]),
    ])


def bell_pyqpanda(qubits, machine):
    """pyqpanda 路径：传入已分配的 qubits 列表和 machine，返回 QProg。"""
    prog = _pyqpanda_prog()
    _pyqpanda_gate(prog, "H", [qubits[0]])
    _pyqpanda_gate(prog, "CX", [qubits[0], qubits[1]])
    return prog


# --------------------------------------------------------------------------- #
# GHZ 态：H(q0) + CX(q0,q1) + CX(q1,q2) + … → (|0…0⟩ + |1…1⟩)/√2
# --------------------------------------------------------------------------- #

def ghz(n: int = 3, engine: str = "qiskit"):
    """n 量子比特 GHZ 态制备电路。H(q0) + 级联 CX。"""
    engine = _check_engine(engine)
    if n < 2:
        raise ValueError(f"GHZ requires n >= 2, got {n}")
    if engine == "qiskit":
        qc = _qiskit_circuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        return qc
    if engine == "pyqpanda":
        raise NotImplementedError("use ghz_pyqpanda(qubits, machine) for pyqpanda engine")
    # numpy
    gates = [Gate("H", targets=[0])]
    for i in range(n - 1):
        gates.append(Gate("CX", targets=[i + 1], controls=[i]))
    return _numpy_gates_to_circuit(gates)


def ghz_pyqpanda(qubits, machine):
    """pyqpanda 路径 GHZ 态。"""
    n = len(qubits)
    prog = _pyqpanda_prog()
    _pyqpanda_gate(prog, "H", [qubits[0]])
    for i in range(n - 1):
        _pyqpanda_gate(prog, "CX", [qubits[i], qubits[i + 1]])
    return prog


# --------------------------------------------------------------------------- #
# 均匀叠加态：H(all) → |+⟩^n
# --------------------------------------------------------------------------- #

def superposition(n: int = 3, engine: str = "qiskit"):
    """n 量子比特均匀叠加态制备电路。对所有 qubit 做 H。"""
    engine = _check_engine(engine)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if engine == "qiskit":
        qc = _qiskit_circuit(n)
        qc.h(range(n))
        return qc
    if engine == "pyqpanda":
        raise NotImplementedError("use superposition_pyqpanda(qubits, machine)")
    # numpy
    return _numpy_gates_to_circuit([Gate("H", targets=[i]) for i in range(n)])


def superposition_pyqpanda(qubits, machine):
    """pyqpanda 路径均匀叠加态。"""
    prog = _pyqpanda_prog()
    for q in qubits:
        _pyqpanda_gate(prog, "H", [q])
    return prog
