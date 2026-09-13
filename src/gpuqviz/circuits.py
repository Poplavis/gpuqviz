"""框架无关的电路表示与 numpy 态矢量演化。

适配器（qiskit / pyqpanda / …）把各自的电路翻译成 Gate 列表（或直接解析
ORIGINIR 文本），本模块用 numpy 完成逐门演化并产出关键帧态矢量序列——
使 gpuqviz 的可视化能力不绑定任何单一模拟器。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# 单比特门矩阵（计算基）
_I2 = np.eye(2, dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], np.complex128) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], np.complex128)
_Z = np.diag([1, -1]).astype(np.complex128)
_S = np.diag([1, 1j]).astype(np.complex128)
_T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(np.complex128)

_MATRICES = {"I": _I2, "H": _H, "X": _X, "Y": _Y, "Z": _Z, "S": _S, "T": _T}


@dataclass
class Gate:
    """框架无关的门：name ∈ 单比特名 或 RX/RY/RZ/CNOT/CZ/SWAP/BARRIER。

    targets/controls 为逻辑 qubit 下标（qiskit/pyqpanda 均为小端序）。
    params：旋转门角度等。
    """
    name: str
    targets: list[int] = field(default_factory=list)
    controls: list[int] = field(default_factory=list)
    params: list[float] = field(default_factory=list)


def _as_le(state: np.ndarray) -> np.ndarray:
    """重排张量轴为小端序视图：轴 i ↔ qubit i（qiskit/pyqpanda 约定）。

    numpy reshape 后轴 0 是最高位（大端序），必须翻转才能与框架的 qubit
    编号一致——这是适配器数值正确性的关键。
    """
    n = int(round(np.log2(state.shape[0])))
    return state.reshape((2,) * n).transpose(*reversed(range(n)))


def _apply_1q(state: np.ndarray, matrix: np.ndarray, target: int) -> np.ndarray:
    psi = np.moveaxis(_as_le(state), target, 0)
    psi = np.einsum("ij,j...->i...", matrix, psi)
    n = psi.ndim
    return np.moveaxis(psi, 0, target).transpose(*reversed(range(n))).reshape(-1)


def _apply_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
    psi = np.moveaxis(_as_le(state), control, 0).copy()
    # 控制态为 |1> 的半空间对 target 做 X
    psi[1] = np.moveaxis(psi[1], target - 1, 0)[::-1]
    psi[1] = np.moveaxis(psi[1], 0, target - 1)
    n = psi.ndim
    return np.moveaxis(psi, 0, control).transpose(*reversed(range(n))).reshape(-1)


def apply_gate(state: np.ndarray, gate: Gate) -> np.ndarray:
    """单个 Gate 作用到态矢量（纯函数，返回新数组）。"""
    name = gate.name.upper()
    if name == "BARRIER":
        return state
    if name in _MATRICES:
        return _apply_1q(state, _MATRICES[name], gate.targets[0])
    if name in ("RX", "RY", "RZ"):
        theta = gate.params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        if name == "RX":
            m = np.array([[c, -1j * s], [-1j * s, c]], np.complex128)
        elif name == "RY":
            m = np.array([[c, -s], [s, c]], np.complex128)
        else:
            m = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
        return _apply_1q(state, m, gate.targets[0])
    if name in ("CNOT", "CX"):
        return _apply_cx(state, gate.controls[0], gate.targets[0])
    if name == "CZ":
        c, t = gate.controls[0], gate.targets[0]
        n = int(round(np.log2(state.shape[0])))
        psi = _as_le(state).copy()
        idx = [slice(None)] * n
        idx[c] = 1
        idx[t] = 1
        psi[tuple(idx)] *= -1
        return psi.transpose(*reversed(range(n))).reshape(-1)
    if name == "SWAP":
        a, b = gate.targets
        psi = _as_le(state)
        n = psi.ndim
        axes = list(range(n))
        axes[a], axes[b] = axes[b], axes[a]
        psi = psi.transpose(axes)  # LE 视图下交换 a/b 两个 qubit 轴
        return np.ascontiguousarray(psi.transpose(*reversed(range(n)))).reshape(-1)
    raise ValueError(f"unsupported gate: {gate.name}")


def evolve_gates(n_qubits: int, gates: list[Gate]) -> list[np.ndarray]:
    """按门序列演化，返回 [初态, 每个非 barrier 门后的态] 关键帧快照。"""
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    snapshots = [state]
    for gate in gates:
        state = apply_gate(state, gate)
        if gate.name.upper() != "BARRIER":
            snapshots.append(state)
    return snapshots


def sample_snapshots(snapshots: list[np.ndarray], steps: int) -> list[np.ndarray]:
    """门级快照均匀采样 steps 个关键帧（含初态与末态，静止段复用）。"""
    if steps < 2:
        steps = 2
    m = len(snapshots)
    if m >= steps:
        idx = np.round(np.linspace(0, m - 1, steps)).astype(int)
        return [snapshots[i] for i in idx]
    out = []
    for i in range(steps):
        pos = i / (steps - 1) * (m - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, m - 1)
        w = pos - lo
        if w < 1e-12 or lo == hi:
            out.append(snapshots[lo])
        else:  # 关键帧间线性插值 + renormalize（可视化近似）
            mid = (1 - w) * snapshots[lo] + w * snapshots[hi]
            norm = np.linalg.norm(mid)
            out.append(mid / norm if norm > 1e-12 else snapshots[lo])
    return out


# ---------------- ORIGINIR 解析（pyqpanda 适配器的中间层） ----------------

_ORIGINIR_1Q = {"H", "X", "Y", "Z", "S", "T", "I", "RX", "RY", "RZ"}
_ORIGINIR_2Q = {"CNOT", "CX", "CZ", "SWAP"}
_QUBIT_RE = re.compile(r"q\[(\d+)\]")
_PARAM_RE = re.compile(r"\(([^)]*)\)")


def _parse_angle(token: str) -> float:
    token = token.strip()
    # ORIGINIR 里常见 pi 表达式
    expr = token.replace("pi", "np.pi")
    return float(eval(expr, {"np": np, "pi": np.pi}))  # noqa: S307 - 受控输入来自转换器


def parse_originir(text: str) -> tuple[int, list[Gate]]:
    """ORIGINIR 文本 → (n_qubits, Gate 列表)。"""
    n_qubits = 0
    gates: list[Gate] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        head = line.split()[0].upper()
        if head == "QINIT":
            n_qubits = max(n_qubits, int(line.split()[1]))
            continue
        if head == "CREG":
            continue
        if head == "BARRIER":
            gates.append(Gate("BARRIER"))
            continue
        name = head
        rest = line[len(line.split()[0]):].strip()
        qubits = [int(m) for m in _QUBIT_RE.findall(rest)]
        param_match = _PARAM_RE.search(rest)
        params = ([_parse_angle(p) for p in param_match.group(1).split(",")]
                  if param_match else [])
        if not qubits:
            raise ValueError(f"cannot parse ORIGINIR line: {raw!r}")
        n_qubits = max(n_qubits, max(qubits) + 1)
        if name in _ORIGINIR_1Q:
            gates.append(Gate(name, targets=[qubits[0]], params=params))
        elif name in _ORIGINIR_2Q:
            if name in ("CNOT", "CX", "CZ"):
                gates.append(Gate(name, targets=[qubits[1]], controls=[qubits[0]]))
            else:  # SWAP
                gates.append(Gate("SWAP", targets=[qubits[0], qubits[1]]))
        else:
            raise ValueError(f"unsupported ORIGINIR gate: {name}")
    if n_qubits == 0:
        raise ValueError("no QINIT found in ORIGINIR")
    return n_qubits, gates


def sample_originir(text: str, steps: int) -> list[np.ndarray]:
    """ORIGINIR 文本 → steps 个关键帧态矢量。"""
    n_qubits, gates = parse_originir(text)
    snapshots = evolve_gates(n_qubits, gates)
    return sample_snapshots(snapshots, steps)
