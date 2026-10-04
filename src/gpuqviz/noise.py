"""噪声与开放系统：Kraus 通道库 + 密度矩阵演化。

约定（docs/conventions.md）：
- Kraus 算符为 2^k×2^k 复数矩阵列表，Σ K†K = I（trace-preserving）；
- 通道作用于 qubit 子集时，qubits 遵循 circuits._apply_matrix 的
  LSB-first 约定（qubits[0] 是矩阵最低位）；
- depolarizing(λ) 采用 qiskit 同名约定：E(ρ) = (1−λ)ρ + λ·I/2^k
  （λ 是"完全混合概率"，不是 Pauli 翻转概率）；
- 所有通道与 qiskit QuantumError.to_quantumchannel() 数值对拍
  （tests/cross_validation/test_noise.py，容差 1e-10）；
- 全电路演化与 AerSimulator(method="density_matrix") + NoiseModel
  对拍（容差 1e-6，浮点经 qiskit 内部 superop 路径）。
"""

from __future__ import annotations

import numpy as np

try:  # cupy 可选
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from .circuits import Gate, _gate_matrix

__all__ = [
    "depolarizing", "amplitude_damping", "phase_damping", "dephasing",
    "thermal_relaxation", "apply_channel", "apply_matrix_density",
    "evolve_density", "lerp_density", "tensor_channels",
]

_SQ = np.sqrt

# 单 qubit Pauli（含 I），供通道构造
_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# --------------------------------------------------------------------------- #
# 通道库（全部返回 Kraus 算符列表）
# --------------------------------------------------------------------------- #

def depolarizing(lam: float, n_qubits: int = 1) -> list[np.ndarray]:
    """去极化通道（qiskit depolarizing_error 同约定）。

    E(ρ) = (1−λ)ρ + λ·Tr[ρ]·I/2^n，λ ∈ [0, 4^n/(4^n−1)]。
    单 qubit Kraus：{√(1−3λ/4)·I, √(λ/4)·X, √(λ/4)·Y, √(λ/4)·Z}。
    Bloch 矢量收缩因子 = (1−λ)。
    """
    if not 0 <= lam <= (4 ** n_qubits) / (4 ** n_qubits - 1):
        raise ValueError(f"depolarizing param λ={lam} out of range")
    d = 1 << n_qubits
    if n_qubits == 1:
        return [
            _SQ(1 - 3 * lam / 4) * _I2,
            _SQ(lam / 4) * _X,
            _SQ(lam / 4) * _Y,
            _SQ(lam / 4) * _Z,
        ]
    # 多 qubit：以 Pauli 基构造（Σ_{P≠I} (λ/(4^n−1)) PρP + (1−λ·(4^n−1)/(4^n−1))ρ）
    paulis = [_I2, _X, _Y, _Z]
    ops = []
    import itertools

    prob_i = 1 - lam * (d * d - 1) / (d * d)
    for combo in itertools.product(paulis, repeat=n_qubits):
        m = np.array([[1.0 + 0j]])
        for p in combo:  # 左起 = 最高位（与位串显示一致）
            m = np.kron(m, p)
        w = prob_i if all(p is _I2 for p in combo) else lam / (d * d)
        ops.append(_SQ(max(w, 0.0)) * m)
    return ops


def amplitude_damping(gamma: float) -> list[np.ndarray]:
    """振幅阻尼通道（T1 弛豫，|1⟩ → |0⟩ 衰减概率 γ）。"""
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma={gamma} out of [0,1]")
    return [
        np.array([[1, 0], [0, _SQ(1 - gamma)]], dtype=np.complex128),
        np.array([[0, _SQ(gamma)], [0, 0]], dtype=np.complex128),
    ]


def phase_damping(gamma: float) -> list[np.ndarray]:
    """相位阻尼通道（相干性衰减，对角不变）。Off-diag × √(1−γ)。"""
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma={gamma} out of [0,1]")
    return [
        np.diag([1.0, _SQ(1 - gamma)]).astype(np.complex128),
        np.diag([0.0, _SQ(gamma)]).astype(np.complex128),
    ]


def dephasing(p: float) -> list[np.ndarray]:
    """相位翻转通道（Pauli-Z 以概率 p 翻转）。Off-diag × (1−2p)。"""
    if not 0 <= p <= 1:
        raise ValueError(f"p={p} out of [0,1]")
    return [_SQ(1 - p) * _I2, _SQ(p) * _Z]


def thermal_relaxation(t1: float, t2: float, time: float,
                       excited_state_population: float = 0.0) -> list[np.ndarray]:
    """T1/T2 热弛豫通道（qiskit thermal_relaxation_error 同约定，T2 ≤ 2·T1 分支）。

    组合：振幅阻尼 γ1 = 1−exp(−time/T1)（含激发态布居）
    ⊗ 纯相位阻尼 γ2 = 1−exp(−2·time·(1/T2 − 1/(2·T1)))。
    """
    if t1 <= 0 or t2 <= 0:
        raise ValueError("t1/t2 must be positive")
    if t2 > 2 * t1:
        raise NotImplementedError("T2 > 2·T1 branch (Choi composition) not supported")
    gamma1 = 1 - np.exp(-time / t1)
    gamma2 = 1 - np.exp(-2 * time * (1 / t2 - 1 / (2 * t1)))
    p1 = excited_state_population

    # 振幅阻尼（含激发布居）：K0=[[1,0],[0,√(1-γ1)]], K1=[[0,√(γ1(1-p1))],[0,0]],
    # K2=[[0,0],[√(γ1·p1),0]]
    a0 = np.array([[1, 0], [0, _SQ(1 - gamma1)]], dtype=np.complex128)
    a1 = np.array([[0, _SQ(gamma1 * (1 - p1))], [0, 0]], dtype=np.complex128)
    a2 = np.array([[0, 0], [_SQ(gamma1 * p1), 0]], dtype=np.complex128)
    amp = [a0, a1, a2] if p1 > 0 else [a0, a1]

    ph = phase_damping(gamma2)
    # 顺序组合：{P_i · A_j}（相位阻尼在前或后结果相同——两者可交换）
    ops = []
    for p in ph:
        for a in amp:
            ops.append(p @ a)
    return ops


def tensor_channels(channels: list[list[np.ndarray]]) -> list[np.ndarray]:
    """独立子通道的张量积组合。

    channels[i] 是作用于第 i 个 qubit（LSB-first）的 Kraus 算符列表，
    返回联合通道的 Kraus 列表（|channels[0]|×…×|channels[-1]| 个算符）。
    典型用途：把单 qubit 噪声挂到多 qubit 门上——
    ``tensor_channels([amplitude_damping(γ)] * n)``。
    """
    import itertools

    ops = []
    for combo in itertools.product(*channels):
        m = np.array([[1.0 + 0j]])
        for op in reversed(combo):  # channels[0] = LSB → kron 最右因子
            m = np.kron(m, op)
        ops.append(m)
    return ops


# --------------------------------------------------------------------------- #
# 密度矩阵上的门 / 通道作用原语
# --------------------------------------------------------------------------- #

def _rho_tensor(rho: np.ndarray, n: int) -> np.ndarray:
    """(d,d) → (2,)*2n 张量：row 轴 a ↔ qubit n−1−a，col 轴 n+a ↔ qubit n−1−a。"""
    return rho.reshape((2,) * n + (2,) * n)


def apply_matrix_density(rho: np.ndarray, matrix: np.ndarray,
                         qubits: list[int]) -> np.ndarray:
    """酉门作用到密度矩阵：ρ' = U ρ U†（qubits LSB-first，同 _apply_matrix）。"""
    n = int(round(np.log2(rho.shape[0])))
    k = len(qubits)
    d = 1 << k
    U = np.asarray(matrix, dtype=np.complex128)
    R = _rho_tensor(np.asarray(rho, dtype=np.complex128), n)
    # ρ 张量是大端 reshape：轴 a ↔ qubit n−1−a（row）/ n+(n−1−a)（col）。
    # 参与 qubit 按 LSB-first 逆序移到前部
    row_src = [n - 1 - q for q in reversed(qubits)]
    col_src = [n + (n - 1 - q) for q in reversed(qubits)]
    R = np.moveaxis(R, row_src + col_src, list(range(2 * k)))
    R = R.reshape(d, d, -1)  # (Q-row, Q-col, rest)
    out = np.einsum("xA,yB,ABt->xyt", U, U.conj(), R, optimize=True)
    out = out.reshape((2,) * (2 * n))  # 恢复完整张量再移轴（rest 展开为 2^(n-k) 个轴）
    out = np.moveaxis(out, list(range(2 * k)), row_src + col_src)
    return np.ascontiguousarray(out.reshape(rho.shape))


def apply_channel(rho: np.ndarray, kraus: list[np.ndarray],
                  qubits: list[int]) -> np.ndarray:
    """Kraus 通道：ρ' = Σ_k (K_k)_Q ρ (K_k)_Q†。"""
    n = int(round(np.log2(rho.shape[0])))
    k = len(qubits)
    d = 1 << k
    K = np.stack([np.asarray(m, dtype=np.complex128) for m in kraus])
    if K.shape[1:] != (d, d):
        raise ValueError(f"Kraus ops shape {K.shape[1:]} != {(d, d)}")
    R = _rho_tensor(np.asarray(rho, dtype=np.complex128), n)
    row_src = [n - 1 - q for q in reversed(qubits)]
    col_src = [n + (n - 1 - q) for q in reversed(qubits)]
    R = np.moveaxis(R, row_src + col_src, list(range(2 * k)))
    R = R.reshape(d, d, -1)
    out = np.einsum("kxA,kyB,ABt->xyt", K, K.conj(), R, optimize=True)
    out = out.reshape((2,) * (2 * n))
    out = np.moveaxis(out, list(range(2 * k)), row_src + col_src)
    return np.ascontiguousarray(out.reshape(rho.shape))


# --------------------------------------------------------------------------- #
# 含噪演化（门序列 → ρ 关键帧）
# --------------------------------------------------------------------------- #

def evolve_density(n_qubits: int, gates: list[Gate],
                   noise=None) -> list[np.ndarray]:
    """门序列演化密度矩阵：|0…0⟩⟨0…0| 出发，返回 [初态, 每个非 barrier 门后的 ρ]。

    noise: callable(gate, gate_index) -> Kraus 算符列表 | None。
    返回 None/空列表表示该门无噪声。噪声在门作用之后施加。
    """
    d = 1 << n_qubits
    rho = np.zeros((d, d), dtype=np.complex128)
    rho[0, 0] = 1.0
    frames = [rho]
    for gi, gate in enumerate(gates):
        if gate.name.upper() == "BARRIER":
            continue
        matrix, qubits = _gate_matrix(gate)
        rho = apply_matrix_density(rho, matrix, qubits)
        if noise is not None:
            ops = noise(gate, gi)
            if ops:
                rho = apply_channel(rho, ops, qubits)
        frames.append(rho)
    return frames


def lerp_density(rhos, out_frames: int) -> np.ndarray:
    """密度矩阵关键帧 (K, d, d) 线性插值 + 迹归一 → (out_frames, d, d)。"""
    xp = cp if (cp is not None and isinstance(rhos, cp.ndarray)) else np
    rhos = xp.asarray(rhos).astype(xp.complex128)
    k = rhos.shape[0]
    if k < 2:
        return xp.repeat(rhos, out_frames, axis=0)
    t = xp.linspace(0.0, 1.0, out_frames)
    pos = t * (k - 1)
    idx = xp.clip(xp.floor(pos).astype(xp.int64), 0, k - 2)
    w = (pos - idx).reshape(-1, 1, 1)
    out = (1 - w) * rhos[idx] + w * rhos[idx + 1]
    tr = xp.real(xp.trace(out, axis1=1, axis2=2))
    out = out / xp.where(xp.abs(tr) > 1e-12, tr, 1.0)[:, None, None]
    return out
