"""gpuqviz.analysis：面向专业用户的测量计算与分析模块。

数据准确性的契约（docs/conventions.md）：
- 每个量有数学定义 docstring；
- 与 qiskit.quantum_info / AerSimulator 对拍（tests/cross_validation/）；
- 采样可复现（PCG64 seed）；
- 输出为结构化 dataclass，可导出 CSV/JSON。

子模块：
- measurement：精确概率、shot 采样、Counts
- metrics：纯度、线性熵、Pauli 期望、保真度、态表格
"""

from .entanglement import (EntanglementReport, entanglement_entropy,
                           entanglement_summary, mutual_information,
                           negativity, schmidt_coefficients)
from .measurement import Counts, exact_probs, sample_counts
from .metrics import (StateTable, fidelity, linear_entropy, pauli_expectation,
                      purity, state_table)

__all__ = [
    "exact_probs", "sample_counts", "Counts",
    "purity", "linear_entropy", "pauli_expectation", "fidelity",
    "state_table", "StateTable",
    "entanglement_entropy", "mutual_information", "negativity",
    "schmidt_coefficients", "entanglement_summary", "EntanglementReport",
]
