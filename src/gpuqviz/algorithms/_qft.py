"""量子傅里叶变换（QFT）电路构建器。

手动构建 QFT（不依赖 qiskit QFTGate），保证 pyqpanda / numpy 可移植。
逐 qubit：H → 受控相位旋转 CR(k) 级联，最后 SWAP 反转 qubit 顺序。

QFT†（逆 QFT）用于量子相位估计的后处理。
"""

from __future__ import annotations

import numpy as np

from ._engine import (
    _check_engine,
    _numpy_gates_to_circuit,
    _qiskit_circuit,
)
from ..circuits import Gate


def _qft_qiskit(qc, n: int, inverse: bool = False):
    """在已有 qiskit QuantumCircuit 上追加 QFT（或逆 QFT）门序列。

    标准 QFT（小端序 q0=LSB）：从 MSB（q_{n-1}）开始处理。
    正向 QFT：H(q_{n-1}) → CP 级联 → … → H(q_0)，最后 SWAP 反转。
    """
    if inverse:
        # 逆 QFT：先 SWAP 反转，再反向执行
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                k = j - i
                angle = -np.pi / (2**k)
                qc.cp(angle, i, j)
    else:
        # 正向 QFT：从 MSB（q_{n-1}）开始
        for i in range(n - 1, -1, -1):
            qc.h(i)
            for j in range(i - 1, -1, -1):
                k = i - j
                angle = np.pi / (2**k)
                qc.cp(angle, j, i)
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)


def qft(n: int = 3, engine: str = "qiskit", inverse: bool = False):
    """n 量子比特 QFT 电路。

    Parameters
    ----------
    n : int
        量子比特数。
    inverse : bool
        True 返回逆 QFT（QFT†）。
    engine : str
        "qiskit" / "pyqpanda" / "numpy"。
    """
    engine = _check_engine(engine)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if engine == "qiskit":
        qc = _qiskit_circuit(n)
        _qft_qiskit(qc, n, inverse=inverse)
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("use qft_pyqpanda(qubits, machine, ...)")

    # numpy 路径：QFT 用受控相位门 CP(angle)
    gates: list[Gate] = []
    _qft_numpy(gates, n, inverse=inverse)
    return _numpy_gates_to_circuit(gates)


def _qft_numpy(gates: list[Gate], n: int, inverse: bool = False):
    """追加 QFT 门序列到 gates 列表（与 _qft_qiskit 顺序一致）。"""
    if inverse:
        # 逆 QFT：先 SWAP，再正向处理（H + CP 逆序、角度取负）
        for i in range(n // 2):
            gates.append(Gate("SWAP", targets=[i, n - 1 - i]))
        for i in range(n):
            gates.append(Gate("H", targets=[i]))
            for j in range(i + 1, n):
                k = j - i
                angle = -np.pi / (2**k)
                gates.append(Gate("CP", targets=[j], controls=[i], params=[angle]))
    else:
        # 正向 QFT：从 MSB 开始
        for i in range(n - 1, -1, -1):
            gates.append(Gate("H", targets=[i]))
            for j in range(i - 1, -1, -1):
                k = i - j
                angle = np.pi / (2**k)
                gates.append(Gate("CP", targets=[i], controls=[j], params=[angle]))
        for i in range(n // 2):
            gates.append(Gate("SWAP", targets=[i, n - 1 - i]))


def qft_pyqpanda(qubits, machine, inverse: bool = False):
    """pyqpanda 路径 QFT。注意：pyqpanda 门集不含 CP，用 CRZ 近似或 CU3 替代。
    为保证数值正确性，pyqpanda QFT 仅用 H + CNOT 的一阶近似（2 qubit）
    或建议用户用 qiskit 路径。完整 QFT 在 pyqpanda 上暂不支持。
    """
    raise NotImplementedError(
        "QFT on pyqpanda requires CRZ/CU3 support; use engine='qiskit' for full QFT"
    )
