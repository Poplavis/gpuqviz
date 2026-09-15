"""gpuqviz 内置量子算法库。

每个算法函数按 ``engine`` 参数返回对应类型的电路对象：
- ``"qiskit"``（默认）：返回 ``qiskit.QuantumCircuit``
- ``"pyqpanda"``：返回 ``pyqpanda.QProg``（需配合 machine）
- ``"numpy"``：返回 ``list[Gate]``（框架无关）

返回的电路对象可直接传入 ``render_bloch_video`` / ``export_html`` / ``show``
等可视化 API，或经 ``evolve_gates`` 做纯 numpy 演化。

用法::

    from gpuqviz.algorithms import grover, bell, qft

    qc = grover(n=3, marked=0b101, iterations=2)
    gpuqviz.export_html(circuit=qc, out="out/grover.html")

    # numpy 路径（无 qiskit 依赖）
    gates = grover(n=3, engine="numpy")
    from gpuqviz.circuits import evolve_gates
    states = evolve_gates(3, gates)

CLI 一行命令::

    gpuqviz demo --algo grover
    gpuqviz demo --algo qft --format mp4
    gpuqviz demo --list
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ._primitive import bell, ghz, superposition
from ._grover import grover
from ._qft import qft
from ._qpe import phase_estimation
from ._deutsch_jozsa import deutsch_jozsa
from ._bernstein_vazirani import bernstein_vazirani
from ._teleportation import teleportation
from ._superdense import superdense
from ._simon import simon
from ._quantum_walk import quantum_walk

__all__ = [
    "bell", "ghz", "superposition", "grover", "qft", "phase_estimation",
    "deutsch_jozsa", "bernstein_vazirani", "teleportation", "superdense",
    "simon", "quantum_walk",
    "ALGORITHM_REGISTRY", "AlgorithmSpec", "get_algorithm", "list_algorithms",
]


@dataclass
class AlgorithmSpec:
    """算法注册表条目。"""
    name: str
    description: str
    builder: Callable[..., Any]
    default_n_qubits: int
    category: str


# 算法注册表：name → spec
ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "bell": AlgorithmSpec(
        "bell", "Bell 态 |Φ+⟩ = (|00⟩+|11⟩)/√2 制备", bell, 2, "基础态"),
    "ghz": AlgorithmSpec(
        "ghz", "n 量子比特 GHZ 纠缠态 (|0…0⟩+|1…1⟩)/√2", ghz, 3, "基础态"),
    "superposition": AlgorithmSpec(
        "superposition", "均匀叠加态 H^n |0…0⟩ = |+⟩^n", superposition, 3, "基础态"),
    "grover": AlgorithmSpec(
        "grover", "Grover 搜索算法（振幅放大）", grover, 3, "搜索"),
    "qft": AlgorithmSpec(
        "qft", "量子傅里叶变换", qft, 3, "变换"),
    "phase_estimation": AlgorithmSpec(
        "phase_estimation", "量子相位估计（QPE）", phase_estimation, 4, "估计"),
    "deutsch_jozsa": AlgorithmSpec(
        "deutsch_jozsa", "Deutsch-Jozsa 算法（常数 vs 平衡判定）",
        deutsch_jozsa, 4, "查询复杂度"),
    "bernstein_vazirani": AlgorithmSpec(
        "bernstein_vazirani", "Bernstein-Vazirani 算法（恢复隐藏字符串）",
        bernstein_vazirani, 3, "查询复杂度"),
    "teleportation": AlgorithmSpec(
        "teleportation", "量子隐形传态（3 qubit Bell pair + 经典校正）",
        teleportation, 3, "通信"),
    "superdense": AlgorithmSpec(
        "superdense", "超密编码（1 qubit 传 2 经典比特）",
        superdense, 2, "通信"),
    "simon": AlgorithmSpec(
        "simon", "Simon 算法（寻找 2-to-1 函数的隐藏周期）",
        simon, 4, "查询复杂度"),
    "quantum_walk": AlgorithmSpec(
        "quantum_walk", "离散时间量子随机游走（硬币 + 条件移位）",
        quantum_walk, 3, "游走"),
}


def get_algorithm(name: str) -> AlgorithmSpec:
    """按名称获取算法注册表条目。不存在时抛 KeyError。"""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {sorted(ALGORITHM_REGISTRY)}"
        )
    return ALGORITHM_REGISTRY[name]


def list_algorithms() -> str:
    """返回格式化的算法列表字符串（供 CLI --list 输出）。"""
    lines = [
        f"{'名称':<25} {'类别':<12} {'默认qubit':<10} {'说明'}",
        f"{'─' * 25} {'─' * 12} {'─' * 10} {'─' * 40}",
    ]
    for spec in ALGORITHM_REGISTRY.values():
        lines.append(
            f"{spec.name:<25} {spec.category:<12} {spec.default_n_qubits:<10} {spec.description}"
        )
    return "\n".join(lines)
