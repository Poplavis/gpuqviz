"""模拟器适配器：把第三方量子编程框架的电路接入 gpuqviz。

- qiskit：evolve.sample_circuit（原生支持）
- pyqpanda：sample_pyqpanda（经 ORIGINIR 转换 + numpy 演化）

pyqpanda 注意：transform_qprog_to_originir 必须使用与 prog 相同的 machine
实例，跨 machine 转换会在原生层段错误（pyqpanda 2.x/3.x 已知行为），
因此 machine 是必传参数，适配器绝不自建虚拟机。
"""

from __future__ import annotations

import numpy as np


def sample_pyqpanda(prog, steps: int, machine) -> list[np.ndarray]:
    """pyqpanda QProg → steps 个关键帧态矢量。

    machine：创建 prog 时所用的量子虚拟机实例（必传，见模块 docstring）。
    pyqpanda 为可选依赖，未安装时报错提示。
    """
    if machine is None:
        raise TypeError(
            "sample_pyqpanda requires the QVM instance that `prog` was built on "
            "(machine=qm). Cross-machine ORIGINIR transforms segfault in pyqpanda.")
    try:
        from pyqpanda import transform_qprog_to_originir
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pyqpanda is required for pyqpanda circuits: pip install gpuqviz[pyqpanda]"
        ) from e

    from ..circuits import sample_originir

    originir = transform_qprog_to_originir(prog, machine)
    return sample_originir(originir, steps)


def to_key_states(circuit, steps: int, machine=None) -> list[np.ndarray]:
    """自动识别电路类型并采样关键帧。

    qiskit QuantumCircuit 直接识别；pyqpanda QProg 需要同时传入
    machine=（创建 prog 的虚拟机实例）。
    """
    # qiskit QuantumCircuit：num_qubits 属性 + data 指令列表
    if hasattr(circuit, "num_qubits") and hasattr(circuit, "data"):
        from ..evolve import sample_circuit

        return sample_circuit(circuit, steps)
    if machine is not None:
        return sample_pyqpanda(circuit, steps, machine)
    raise TypeError(
        f"unsupported circuit type {type(circuit).__name__}: expected a qiskit "
        "QuantumCircuit, or a pyqpanda QProg together with machine=<QVM instance>"
    )
