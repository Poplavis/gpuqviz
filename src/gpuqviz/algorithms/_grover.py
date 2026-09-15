"""Grover 搜索算法电路构建器。

统一现有两个不一致的 Grover 实现（grover_mcx.py 标记 |101⟩ + mcx；
circuit_viewer_demo.py 标记 |111⟩ + ccx），提供参数化的标准版本。

oracle = X(unmark) → H(target) → MCX(controls,target) → H(target) → X(unmark)
diffuser = H(all) → X(all) → H(target) → MCX → H(target) → X(all) → H(all)
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


def _grover_oracle_qiskit(qc, n: int, marked: int):
    """标记 |marked⟩ 态的相位 oracle（qiskit）。"""
    # 对 marked 中为 0 的 qubit 取反，使目标变成 |1…1⟩
    for i in range(n):
        if not (marked >> i) & 1:
            qc.x(i)
    # 用相位踢回：H(target) → MCX(controls, target) → H(target)
    target = n - 1
    controls = list(range(n - 1))
    qc.h(target)
    if len(controls) == 1:
        qc.cx(controls[0], target)
    else:
        qc.mcx(controls, target)
    qc.h(target)
    # 还原取反的 qubit
    for i in range(n):
        if not (marked >> i) & 1:
            qc.x(i)


def _grover_diffuser_qiskit(qc, n: int):
    """Grover 扩散算子（关于 |+⟩ 的反射）。"""
    target = n - 1
    controls = list(range(n - 1))
    qc.h(range(n))
    qc.x(range(n))
    qc.h(target)
    if len(controls) == 1:
        qc.cx(controls[0], target)
    else:
        qc.mcx(controls, target)
    qc.h(target)
    qc.x(range(n))
    qc.h(range(n))


def grover(n: int = 3, marked: int = 0b101, iterations: int | None = None,
           engine: str = "qiskit"):
    """Grover 搜索算法电路。

    Parameters
    ----------
    n : int
        量子比特数（搜索空间 2^n）。
    marked : int
        被标记的目标态（整数值，小端序 qubit 编码）。
    iterations : int | None
        Grover 迭代次数。None 时自动取 floor(π/4 · √(2^n))。
    engine : str
        "qiskit" / "pyqpanda" / "numpy"。
    """
    engine = _check_engine(engine)
    if n < 2:
        raise ValueError(f"Grover requires n >= 2, got {n}")
    if not (0 <= marked < 2**n):
        raise ValueError(f"marked={marked} out of range for {n} qubits")
    if iterations is None:
        import math
        iterations = max(1, int(math.floor(math.pi / 4 * (2**n) ** 0.5)))

    if engine == "qiskit":
        qc = _qiskit_circuit(n)
        qc.h(range(n))
        for _ in range(iterations):
            _grover_oracle_qiskit(qc, n, marked)
            _grover_diffuser_qiskit(qc, n)
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use grover_pyqpanda(qubits, machine, ...)")

    # numpy 路径
    gates: list[Gate] = []
    for i in range(n):
        gates.append(Gate("H", targets=[i]))
    for _ in range(iterations):
        _grover_oracle_numpy(gates, n, marked)
        _grover_diffuser_numpy(gates, n)
    return _numpy_gates_to_circuit(gates)


def _grover_oracle_numpy(gates: list[Gate], n: int, marked: int):
    """追加 Grover oracle 门到 gates 列表。"""
    target = n - 1
    controls = list(range(n - 1))
    for i in range(n):
        if not (marked >> i) & 1:
            gates.append(Gate("X", targets=[i]))
    gates.append(Gate("H", targets=[target]))
    if len(controls) == 1:
        gates.append(Gate("CX", targets=[target], controls=[controls[0]]))
    else:
        gates.append(Gate("MCX", targets=[target], controls=controls))
    gates.append(Gate("H", targets=[target]))
    for i in range(n):
        if not (marked >> i) & 1:
            gates.append(Gate("X", targets=[i]))


def _grover_diffuser_numpy(gates: list[Gate], n: int):
    """追加 Grover diffuser 门到 gates 列表。"""
    target = n - 1
    controls = list(range(n - 1))
    for i in range(n):
        gates.append(Gate("H", targets=[i]))
    for i in range(n):
        gates.append(Gate("X", targets=[i]))
    gates.append(Gate("H", targets=[target]))
    if len(controls) == 1:
        gates.append(Gate("CX", targets=[target], controls=[controls[0]]))
    else:
        gates.append(Gate("MCX", targets=[target], controls=controls))
    gates.append(Gate("H", targets=[target]))
    for i in range(n):
        gates.append(Gate("X", targets=[i]))
    for i in range(n):
        gates.append(Gate("H", targets=[i]))


def grover_pyqpanda(qubits, machine, marked: int = 0b101,
                    iterations: int | None = None):
    """pyqpanda 路径 Grover。"""
    import math
    n = len(qubits)
    if iterations is None:
        iterations = max(1, int(math.floor(math.pi / 4 * (2**n) ** 0.5)))

    prog = _pyqpanda_prog()
    # 初始化叠加态
    for q in qubits:
        _pyqpanda_gate(prog, "H", [q])
    for _ in range(iterations):
        # oracle
        for i in range(n):
            if not (marked >> i) & 1:
                _pyqpanda_gate(prog, "X", [qubits[i]])
        target = n - 1
        controls = list(range(n - 1))
        _pyqpanda_gate(prog, "H", [qubits[target]])
        if len(controls) == 1:
            _pyqpanda_gate(prog, "CNOT", [qubits[controls[0]], qubits[target]])
        else:
            _pyqpanda_gate(prog, "TOFFOLI", [qubits[controls[0]],
                                              qubits[controls[1]], qubits[target]])
        _pyqpanda_gate(prog, "H", [qubits[target]])
        for i in range(n):
            if not (marked >> i) & 1:
                _pyqpanda_gate(prog, "X", [qubits[i]])
        # diffuser
        for q in qubits:
            _pyqpanda_gate(prog, "H", [q])
        for q in qubits:
            _pyqpanda_gate(prog, "X", [q])
        _pyqpanda_gate(prog, "H", [qubits[target]])
        if len(controls) == 1:
            _pyqpanda_gate(prog, "CNOT", [qubits[controls[0]], qubits[target]])
        else:
            _pyqpanda_gate(prog, "TOFFOLI", [qubits[controls[0]],
                                              qubits[controls[1]], qubits[target]])
        _pyqpanda_gate(prog, "H", [qubits[target]])
        for q in qubits:
            _pyqpanda_gate(prog, "X", [q])
        for q in qubits:
            _pyqpanda_gate(prog, "H", [q])
    return prog
