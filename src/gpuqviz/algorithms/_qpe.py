"""量子相位估计（QPE）电路构建器。

给定一个酉算子 U 的本征态 |ψ⟩ 和对应本征值 e^{2πiφ}，
QPE 将相位 φ 编码到计数寄存器中。

电路结构：
1. 计数寄存器（n_count 个 qubit）做 H
2. 本征态寄存器保持 |ψ⟩
3. 受控 U^(2^k) 级联：计数 qubit k 控制 U 作用 2^k 次
4. 逆 QFT 作用于计数寄存器
5. 测量计数寄存器 → 得到 φ 的二进制估计

这里 U = RZ(2πθ)，本征态 |1⟩ 的本征值为 e^{-iπθ}（相位 = θ/2）。
为简化演示，用 U = P(2πθ) = diag(1, e^{i2πθ})，本征态 |1⟩ → 相位 θ。
"""

from __future__ import annotations

import numpy as np

from ._engine import (
    _check_engine,
    _numpy_gates_to_circuit,
    _qiskit_circuit,
)
from ._qft import _qft_qiskit, _qft_numpy
from ..circuits import Gate


def phase_estimation(n_count: int = 3, theta: float = 0.375,
                     engine: str = "qiskit"):
    """量子相位估计电路。

    Parameters
    ----------
    n_count : int
        计数寄存器量子比特数（精度 2^{-n_count}）。
    theta : float
        待估计相位 φ ∈ [0, 1)，U = P(2πθ) 作用于 |1⟩。
    engine : str
        "qiskit" / "numpy"（pyqpanda 暂不支持，需 CRZ）。
    """
    engine = _check_engine(engine)
    if n_count < 1:
        raise ValueError(f"n_count must be >= 1, got {n_count}")
    if not (0 <= theta < 1):
        raise ValueError(f"theta must be in [0, 1), got {theta}")

    n_total = n_count + 1  # 最后一个 qubit 是本征态寄存器
    eigenstate = n_count    # 本征态 qubit 索引

    if engine == "qiskit":
        qc = _qiskit_circuit(n_total)
        # 本征态 |1⟩
        qc.x(eigenstate)
        # 计数寄存器 H
        for i in range(n_count):
            qc.h(i)
        # 受控 U^(2^k)：计数 qubit k 控制 P(2πθ · 2^k) 作用于本征态
        for k in range(n_count):
            angle = 2 * np.pi * theta * (2**k)
            qc.cp(angle, k, eigenstate)
        # 逆 QFT 作用于计数寄存器
        _qft_qiskit(qc, n_count, inverse=True)
        return qc

    if engine == "pyqpanda":
        raise NotImplementedError("QPE on pyqpanda requires CP support; use engine='qiskit'")

    # numpy 路径
    gates: list[Gate] = []
    gates.append(Gate("X", targets=[eigenstate]))
    for i in range(n_count):
        gates.append(Gate("H", targets=[i]))
    for k in range(n_count):
        angle = 2 * np.pi * theta * (2**k)
        gates.append(Gate("CP", targets=[eigenstate], controls=[k], params=[angle]))
    _qft_numpy(gates, n_count, inverse=True)
    return _numpy_gates_to_circuit(gates)
