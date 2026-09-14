"""模拟器适配器：把第三方量子编程框架的电路接入 gpuqviz。

- qiskit：evolve.sample_circuit（原生支持）；qiskit_to_gates 提供框架无关
  的门级翻译（含 transpile 兜底），供 numpy 模拟器路径使用
- pyqpanda：sample_pyqpanda（经 ORIGINIR 转换 + numpy 演化）

pyqpanda 注意：transform_qprog_to_originir 必须使用与 prog 相同的 machine
实例，跨 machine 转换会在原生层段错误（pyqpanda 2.x/3.x 已知行为），
因此 machine 是必传参数，适配器绝不自建虚拟机。
"""

from __future__ import annotations

import numpy as np

# 命名门直译集合（其余门走矩阵路径 / definition 递归 / transpile 兜底）
_NAMED_1Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "id", "i"}
_NAMED_P = {"p", "u1", "u2", "u3", "u"}
_NAMED_ROT = {"rx", "ry", "rz"}
_TRANSPILE_BASIS = [
    "id", "h", "x", "y", "z", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "u", "cx", "cy", "cz", "swap",
]


class _UnsupportedGate(Exception):
    """门无法直译也无法取矩阵/展开时抛出，由上层触发 transpile 兜底。"""


def _translate_op(op, qidx: list[int], depth: int = 0) -> list:
    """单条 qiskit 指令 → Gate 列表。

    优先级：命名直译 > to_matrix/Operator 矩阵 > definition 递归。
    全局相位不影响 Bloch 向量与概率，统一忽略。
    """
    from ..circuits import Gate

    if depth > 32:
        raise _UnsupportedGate(f"definition nesting too deep: {op.name}")
    name = op.name

    # 命名门直译（保持 gate 语义可读，且覆盖 circuits.py 的命名分发路径）
    if name in _NAMED_1Q:
        return [Gate(name.upper(), targets=qidx)]
    if name in _NAMED_ROT or name in _NAMED_P:
        return [Gate(name.upper(), targets=qidx, params=list(op.params))]
    if name in ("cx", "cnot"):
        return [Gate("CX", targets=[qidx[1]], controls=[qidx[0]])]
    if name in ("cy", "cz", "ch"):
        return [Gate(name.upper(), targets=[qidx[1]], controls=[qidx[0]])]
    if name == "swap":
        return [Gate("SWAP", targets=qidx)]
    if name == "iswap":
        return [Gate("ISWAP", targets=qidx)]
    if name == "ccx":
        return [Gate("CCX", targets=[qidx[2]], controls=qidx[:2])]

    # 受控门（CRX/MCX/CU/…）：直接取整门矩阵，qargs 顺序为 controls+targets
    if getattr(op, "num_ctrl_qubits", 0):
        try:
            return [Gate("UNITARY", targets=qidx, matrix=_op_matrix(op))]
        except Exception:
            pass  # 落到 definition 递归

    # 任意酉门（UnitaryGate、自定义 Gate、库门）：整门矩阵路径
    try:
        return [Gate("UNITARY", targets=qidx, matrix=_op_matrix(op))]
    except Exception:
        pass

    # 复合门：递归展开 definition（子电路 qubit i 对应 qidx[i]）
    defn = getattr(op, "definition", None)
    if defn is not None and len(defn.data) > 0:
        out = []
        for sub in defn.data:
            sub_q = [qidx[defn.find_bit(q).index] for q in sub.qubits]
            out.extend(_translate_op(sub.operation, sub_q, depth + 1))
        return out

    raise _UnsupportedGate(f"cannot translate qiskit instruction: {name}")


def _op_matrix(op) -> np.ndarray:
    """qiskit Gate/Instruction → 2^k×2^k 酉矩阵（LSB = op 自身 qubit 0）。"""
    try:
        m = op.to_matrix()
    except Exception:
        from qiskit.quantum_info import Operator

        m = Operator(op).to_matrix()
    return np.asarray(m, dtype=np.complex128)


def _translate_circuit(circuit) -> tuple[int, list]:
    from ..circuits import Gate

    n = circuit.num_qubits
    gates: list = []
    warned_measure = False
    for inst in circuit.data:
        op = inst.operation
        qidx = [circuit.find_bit(q).index for q in inst.qubits]
        if op.name in ("measure", "reset"):
            # 态矢量演化不涉及测量/重置：按层边界处理，保证与可视化语义一致
            if not warned_measure:
                print("[gpuqviz] measure/reset 不会改变态矢量演化，按层边界处理")
                warned_measure = True
            gates.append(Gate("BARRIER"))
            continue
        if op.name == "delay":
            continue
        if op.name == "barrier":
            gates.append(Gate("BARRIER"))
            continue
        gates.extend(_translate_op(op, qidx))
    return n, gates


def qiskit_to_gates(circuit) -> tuple[int, list]:
    """qiskit QuantumCircuit → (n_qubits, Gate 列表)，框架无关路径的入口。

    兜底策略：出现无法翻译的指令时，transpile 到基础门集后重试一次。
    电路的末尾测量自动剔除；中途 measure/reset 按层边界处理。
    """
    circuit = circuit.remove_final_measurements(inplace=False)
    try:
        return _translate_circuit(circuit)
    except _UnsupportedGate as e:
        print(f"[gpuqviz] 直译失败（{e}），transpile 到基础门集后重试")
        from qiskit import transpile

        t = transpile(circuit, basis_gates=_TRANSPILE_BASIS,
                      optimization_level=0)
        return _translate_circuit(t)


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
