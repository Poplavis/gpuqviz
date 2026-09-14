"""量子演化层：qiskit 电路 / 态矢量序列 → 关键帧 Bloch 向量。

输入遵循鸭子类型：接受 qiskit Statevector、DensityMatrix 或任何 `.data`
可取出复数向量的对象。qiskit 仅在本模块内 import（软依赖）。
"""

from __future__ import annotations

import numpy as np

try:  # cupy 可选：GPU 数组后端
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

# Pauli 矩阵（实部/虚部分开供 einsum 使用，保持 dtype 一致）
_PAULI = np.array(
    [
        [[0, 1], [1, 0]],      # X
        [[0, -1j], [1j, 0]],   # Y
        [[1, 0], [0, -1]],     # Z
    ],
    dtype=np.complex128,
)


def _to_array(state) -> np.ndarray:
    """qiskit Statevector / DensityMatrix / np.ndarray → 复数 1D 向量。"""
    data = getattr(state, "data", state)
    arr = cp.asnumpy(data) if cp is not None and isinstance(data, cp.ndarray) else np.asarray(data)
    arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"state must be a flat statevector, got shape {arr.shape}")
    return arr.astype(np.complex128, copy=False)


def _xp(arr):
    """按输入类型选择数组后端（cupy 输入留 GPU，numpy 输入走 CPU）。"""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def sample_circuit(circuit, steps: int = 120) -> list[np.ndarray]:
    """把 QuantumCircuit 按深度均匀采样 steps 个关键帧，返回态矢量列表。

    实现方式：以"层"（depth 切片）为步进单位逐段演化；相邻采样点间无门的
    段直接复用上一次态矢量，避免重复仿真。
    """
    from qiskit.quantum_info import Statevector

    circuit = circuit.remove_final_measurements(inplace=False)
    n_qubits = circuit.num_qubits
    depth = circuit.depth()
    if depth == 0:
        sv = np.zeros(2**n_qubits, dtype=np.complex128)
        sv[0] = 1.0
        return [sv.copy() for _ in range(steps)]

    if steps < 2:
        steps = 2
    # 均匀切 depth 为 steps-1 段（每段至少 1 层）
    seg = max(1, depth // (steps - 1))
    boundaries = [min(i * seg, depth) for i in range(steps)]
    boundaries[-1] = depth

    # 预计算每层（把电路按指令分组到层太复杂，直接按 boundary 截断重组）
    state = Statevector.from_int(0, 2**n_qubits)
    states: list[np.ndarray] = [_to_array(state)]
    current_depth = 0
    for target_depth in boundaries[1:]:
        chunk = _slice_circuit_by_depth(circuit, current_depth, target_depth)
        if chunk.size() > 0:
            state = state.evolve(chunk)
        current_depth = target_depth
        states.append(_to_array(state))

    # 关键帧数量不足 steps 时用末帧补齐（静止段）
    while len(states) < steps:
        states.append(states[-1].copy())
    return states[:steps]


def _slice_circuit_by_depth(circuit, d0: int, d1: int):
    """提取电路中 layer depth ∈ [d0, d1) 的指令组成子电路。

    注意：不能直接把原 DAG 的 Qubit 对象塞进新电路（寄存器错位），
    必须按 qubit index 重建。
    """
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)
    # instruction index -> layer index
    inst_layer: list[int] = []
    for li, layer in enumerate(dag.layers()):
        inst_layer.extend([li] * len(layer["graph"].op_nodes()))

    sub = QuantumCircuit(circuit.num_qubits)
    warned_measure = False
    for idx, inst in enumerate(circuit.data):
        if inst.operation.name in ("measure", "reset"):
            # Statevector.evolve 不接受测量/重置指令；态矢量可视化下测量
            # 不改变演化，跳过并提示一次
            if not warned_measure:
                print("[gpuqviz] 电路含中途 measure/reset，态矢量演化中按"
                      "无操作处理（可视化结果不含测量塌缩）")
                warned_measure = True
            continue
        if d0 <= inst_layer[idx] < d1:
            qargs = [sub.qubits[circuit.find_bit(q).index] for q in inst.qubits]
            cargs = [sub.clbits[circuit.find_bit(c).index] for c in inst.clbits]
            sub.append(inst.operation, qargs, cargs)
    return sub


def bloch_vectors(states, n_qubits: int | None = None) -> "np.ndarray":
    """关键帧态矢量序列 → (steps, n_qubits, 3) Bloch 向量。

    向量化实现：把所有帧堆成批张量一次性 einsum，禁止 Python 层逐帧求迹。
    输入可以是 numpy/cupy 数组（回 GPU 计算）或 qiskit Statevector 列表。
    """
    if n_qubits is None:
        n_qubits = _infer_qubits(states[0])

    # 统一转 numpy 批张量 (S, 2**n)；GPU 数组会先拷回 CPU —— 本函数数值
    # 计算用 einsum 向量化，规模 S×4^n 在 CPU 上也足够快，S3 若成瓶颈再迁移
    batch = np.stack([_to_array(s) for s in states])  # (S, 2**n)

    # 重排成 (S, n, 2, 2^(n-1))：qubit i 的振幅对
    batch = batch.reshape((batch.shape[0], *([2] * n_qubits)))  # (S, 2, 2, ..., 2)
    # 转为 einsum 友好的形状：对每个 qubit i，把该轴拆为行/列指标
    xps = _xp(batch)
    einsum = cp.einsum if xps is cp else np.einsum

    pauli = cp.asarray(_PAULI) if xps is cp else _PAULI

    out = np.zeros((batch.shape[0], n_qubits, 3), dtype=np.float64)
    for i in range(n_qubits):
        # qiskit 小端序：reshape 后轴 1+j 对应 qubit n-1-j，
        # 因此 qubit i 位于轴 1+(n-1-i)
        moved = np.moveaxis(batch, 1 + (n_qubits - 1 - i), 1)  # (S, 2, rest...)
        moved = moved.reshape(batch.shape[0], 2, -1)  # (S, 2, R)
        rho = moved @ moved.conj().transpose(0, 2, 1)  # (S, 2, 2) 批量外积

        # <P> = Tr(ρ P)，批量 einsum
        expvals = einsum("sab,pba->sp", rho, pauli).real  # (S, 3)
        out[:, i, :] = np.asarray(cp.asnumpy(expvals) if xps is cp else expvals)

    result = cp.asarray(out) if xps is cp else out
    return result


def _infer_qubits(state_or_vec) -> int:
    data = getattr(state_or_vec, "data", state_or_vec)
    dim = int(np.prod(np.asarray(data).shape)) if np.asarray(data).ndim else int(np.asarray(data).shape[0])
    n = int(round(np.log2(dim)))
    if 2**n != dim:
        raise ValueError(f"state dimension {dim} is not a power of 2")
    return n
