"""引擎抽象层：算法函数按 engine 参数返回对应类型的电路对象。

三种引擎：
- qiskit（默认）：返回 qiskit.QuantumCircuit
- pyqpanda：返回 pyqpanda.QProg（需调用方传入 machine）
- numpy：返回 list[Gate]（框架无关，纯 numpy 演化）

每个算法构建函数在内部按 engine 分发，算法逻辑不变。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..circuits import Gate

ENGINE_QISKIT = "qiskit"
ENGINE_PYQPANDA = "pyqpanda"
ENGINE_NUMPY = "numpy"
_VALID_ENGINES = {ENGINE_QISKIT, ENGINE_PYQPANDA, ENGINE_NUMPY}


def _check_engine(engine: str) -> str:
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"unsupported engine {engine!r}; expected one of {sorted(_VALID_ENGINES)}"
        )
    return engine


# --------------------------------------------------------------------------- #
# qiskit 构建辅助
# --------------------------------------------------------------------------- #

def _qiskit_circuit(n_qubits: int):
    """创建 qiskit QuantumCircuit（惰性 import qiskit）。"""
    from qiskit import QuantumCircuit

    return QuantumCircuit(n_qubits)


# --------------------------------------------------------------------------- #
# pyqpanda 构建辅助
# --------------------------------------------------------------------------- #

def _pyqpanda_prog():
    """创建 pyqpanda QProg（惰性 import）。"""
    from pyqpanda import QProg

    return QProg()


def _pyqpanda_gate(prog, name: str, qubits: list, params: list[float] | None = None,
                   controls: list | None = None):
    """向 QProg 追加一个门。qubits 是已分配的 pyqpanda qubit 引用列表。"""
    import pyqpanda as pq

    name = name.upper()
    if name == "H":
        prog << pq.H(qubits[0])
    elif name == "X":
        prog << pq.X(qubits[0])
    elif name == "Y":
        prog << pq.Y(qubits[0])
    elif name == "Z":
        prog << pq.Z(qubits[0])
    elif name == "S":
        prog << pq.S(qubits[0])
    elif name == "T":
        prog << pq.T(qubits[0])
    elif name == "RX":
        prog << pq.RX(qubits[0], params[0])
    elif name == "RY":
        prog << pq.RY(qubits[0], params[0])
    elif name == "RZ":
        prog << pq.RZ(qubits[0], params[0])
    elif name in ("CX", "CNOT"):
        prog << pq.CNOT(qubits[0], qubits[1])
    elif name == "CZ":
        prog << pq.CZ(qubits[0], qubits[1])
    elif name == "SWAP":
        prog << pq.SWAP(qubits[0], qubits[1])
    elif name in ("CCX", "TOFFOLI"):
        prog << pq.Toffoli(qubits[0], qubits[1], qubits[2])
    else:
        raise ValueError(f"pyqpanda does not support gate {name}")


# --------------------------------------------------------------------------- #
# numpy 构建辅助
# --------------------------------------------------------------------------- #

def _numpy_gates_to_circuit(gates: list[Gate]) -> list[Gate]:
    """numpy 路径直接返回 Gate 列表。"""
    return list(gates)
