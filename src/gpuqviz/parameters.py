"""参数化电路与变分工作流：符号参数 → 绑定 → 扫参分析。

设计（docs/development-plan.md P3.1）：
- ``Parameter`` 占位符可出现在模板门的 params 中；
- ``CircuitTemplate.bind(**values)`` → 具体 Gate 列表（未解析参数报错）；
- ``sweep`` 沿单参数扫描：逐值演化 → 结构化分析量
  （Bloch 向量 / 每 qubit 纠缠熵 / 纯度 / Pauli 可观测量期望）。

所有数值经 tests/cross_validation/test_sweep.py 对拍 qiskit
（逐值 Statevector，1e-10）与解析公式（⟨Z⟩ = cosθ）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuits import Gate, evolve_gates

__all__ = ["Parameter", "CircuitTemplate", "sweep", "SweepResult"]


@dataclass(frozen=True)
class Parameter:
    """符号参数占位符（如 θ）。"""

    name: str

    def __repr__(self) -> str:  # 模板打印时更可读
        return self.name


def _resolve(value, bindings: dict[str, float], where: str) -> float:
    if isinstance(value, Parameter):
        if value.name not in bindings:
            raise ValueError(f"unresolved parameter {value.name!r} in {where}; "
                             f"bound: {sorted(bindings)}")
        return float(bindings[value.name])
    return float(value)


class CircuitTemplate:
    """参数化电路模板。

    gates_spec 每项 = {"name", "targets", "controls", "params"}，
    params 元素可为 float 或 :class:`Parameter`。
    """

    def __init__(self, n_qubits: int, gates_spec: list[dict]):
        self.n_qubits = int(n_qubits)
        self.gates_spec = [dict(g) for g in gates_spec]
        self.param_names = {
            p.name for g in self.gates_spec for p in g.get("params", [])
            if isinstance(p, Parameter)
        }

    def bind(self, **bindings) -> list[Gate]:
        """参数绑定 → 具体 Gate 列表。多余/缺失参数均报错。"""
        extra = set(bindings) - self.param_names
        if extra:
            raise ValueError(f"unknown parameters {sorted(extra)}; "
                             f"template declares {sorted(self.param_names)}")
        missing = self.param_names - set(bindings)
        if missing:
            raise ValueError(f"unbound parameters {sorted(missing)}")
        gates = []
        for g in self.gates_spec:
            params = [_resolve(p, bindings, f"gate {g['name']}")
                      for p in g.get("params", [])]
            gates.append(Gate(name=g["name"], targets=list(g["targets"]),
                              controls=list(g.get("controls", [])),
                              params=params))
        return gates

    def evolve(self, **bindings) -> list[np.ndarray]:
        """绑定后按门序列演化（纯态关键帧）。"""
        return evolve_gates(self.n_qubits, self.bind(**bindings))


# --------------------------------------------------------------------------- #
# 扫参
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SweepResult:
    """单参数扫描的结构化结果（渲染/导出直接消费）。"""

    param: str                       # 参数名
    values: np.ndarray               # (V,) 扫描值
    n_qubits: int
    states: np.ndarray               # (V, 2**n) 每个参数值的末态
    bloch: np.ndarray                # (V, n, 3)
    entropy: np.ndarray              # (V, n) 每 qubit 纠缠熵
    purity: np.ndarray               # (V,) 全局纯度
    observables: dict[str, np.ndarray] = field(default_factory=dict)
    # observables: {Pauli 串: (V,) 期望值}

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict:
        return {
            "param": self.param,
            "values": [float(v) for v in self.values],
            "n_qubits": self.n_qubits,
            "purity": [float(p) for p in self.purity],
            "entropy": [[float(e) for e in row] for row in self.entropy],
            "observables": {k: [float(x) for x in v]
                            for k, v in self.observables.items()},
        }

    def to_csv(self, path=None) -> str:
        """CSV：param,purity,S_q0…,<observable 列>。"""
        n = self.n_qubits
        header = (f"{self.param},purity,"
                  + ",".join(f"S_q{q}" for q in range(n))
                  + ("," + ",".join(self.observables) if self.observables else ""))
        lines = [header]
        for k in range(len(self.values)):
            row = [f"{self.values[k]:.10g}", f"{self.purity[k]:.10g}"]
            row += [f"{self.entropy[k, q]:.10g}" for q in range(n)]
            row += [f"{self.observables[name][k]:.10g}"
                    for name in self.observables]
            lines.append(",".join(row))
        text = "\n".join(lines) + "\n"
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text


def sweep(template: CircuitTemplate, param: str, values,
          observables: list[str] | None = None) -> SweepResult:
    """沿单参数扫描：逐值演化 → 结构化分析量。

    values: 可迭代的参数值。
    observables: 可选 Pauli 串列表（如 ["ZII", "IZI"]）——逐值计算期望，
    供 VQE/QAOA 的能量/关联曲线使用。
    """
    if param not in template.param_names:
        raise ValueError(f"parameter {param!r} not in template "
                         f"{sorted(template.param_names)}")
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("sweep needs at least one value")

    n = template.n_qubits
    from .analysis.metrics import pauli_expectation, purity as _purity
    from .state import Statevector

    states = np.empty((values.size, 2 ** n), dtype=np.complex128)
    bloch = np.empty((values.size, n, 3), dtype=np.float64)
    entropy = np.empty((values.size, n), dtype=np.float64)
    pur = np.empty(values.size, dtype=np.float64)
    obs = {name: np.empty(values.size, dtype=np.float64)
           for name in (observables or [])}

    for k, v in enumerate(values):
        keyframes = template.evolve(**{param: float(v)})
        psi = np.asarray(keyframes[-1])
        states[k] = psi
        sv = Statevector(psi)
        bloch[k] = sv.bloch()
        for q in range(n):
            entropy[k, q] = entanglement_entropy_safe(sv, q)
        pur[k] = _purity(sv)
        for name in obs:
            obs[name][k] = pauli_expectation(sv, name)

    return SweepResult(param=param, values=values, n_qubits=n, states=states,
                       bloch=bloch, entropy=entropy, purity=pur,
                       observables=obs)


def entanglement_entropy_safe(state, qubit: int) -> float:
    """纠缠熵（从 analysis 导入惰性化，避免模块循环）。"""
    from .analysis.entanglement import entanglement_entropy

    return entanglement_entropy(state, qubit)
