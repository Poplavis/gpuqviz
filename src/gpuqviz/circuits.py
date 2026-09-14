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
_S_DG = _S.conj().copy()
_T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(np.complex128)
_T_DG = _T.conj().copy()
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                 np.complex128)
_ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                  np.complex128)

_MATRICES = {
    "I": _I2, "H": _H, "X": _X, "Y": _Y, "Z": _Z, "S": _S, "T": _T,
    "SDG": _S_DG, "TDG": _T_DG,
}

# 可作为受控 base 的门名（受控前缀剥离的终止条件）
_BASE_NAMES = frozenset(_MATRICES) | {"RX", "RY", "RZ", "U", "U3", "U2"}


@dataclass
class Gate:
    """框架无关的门。

    targets/controls 为逻辑 qubit 下标（qiskit/pyqpanda 均为小端序）。
    params：旋转门角度等。
    matrix：可选的显式酉矩阵（2^k×2^k，LSB 对应 targets[0]，qiskit
    Operator 约定）——适配器无法直译的门以 UNITARY 形式携带矩阵下发了。
    """
    name: str
    targets: list[int] = field(default_factory=list)
    controls: list[int] = field(default_factory=list)
    params: list[float] = field(default_factory=list)
    matrix: np.ndarray | None = field(default=None, repr=False, compare=False)


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


def _apply_matrix(state: np.ndarray, matrix: np.ndarray, qubits: list[int]) -> np.ndarray:
    """把 2^k×2^k 酉矩阵作用到指定 qubit 集合（通用门作用原语）。

    qubits 采用 qiskit Operator 约定：qubits[0] 是矩阵的最低位（LSB）。
    受控门按 targets 在前（低位）、controls 在后（高位）拼入 qubits，
    配合 _controlled 构造的矩阵（base 作用于低位块、控制位全 1 时生效）。
    """
    k = len(qubits)
    dim = 1 << k
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (dim, dim):
        raise ValueError(f"matrix shape {matrix.shape} does not match {k} qubits")
    if k == 1:
        return _apply_1q(state, matrix, qubits[0])
    n = int(round(np.log2(state.shape[0])))
    psi = _as_le(state)  # 轴 i ↔ qubit i
    # 逆序移动参与轴：展平后 qubits[0] 位于最低位
    psi = np.moveaxis(psi, list(reversed(qubits)), list(range(k)))
    rest = psi.shape[k:]
    psi = matrix @ psi.reshape(dim, -1)
    psi = psi.reshape((2,) * k + rest)
    psi = np.moveaxis(psi, list(range(k)), list(reversed(qubits)))
    return np.ascontiguousarray(psi.transpose(*reversed(range(n)))).reshape(-1)


def _controlled(base: np.ndarray, n_ctrl: int) -> np.ndarray:
    """base（2^t×2^t）→ 多控制门矩阵：控制位为高位，全 1 时作用 base。"""
    dim = base.shape[0]
    m = np.eye((1 << n_ctrl) * dim, dtype=np.complex128)
    m[-dim:, -dim:] = base
    return m


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        np.complex128)


def _rotation(name: str, theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if name == "RX":
        return np.array([[c, -1j * s], [-1j * s, c]], np.complex128)
    if name == "RY":
        return np.array([[c, -s], [s, c]], np.complex128)
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(np.complex128)


def _gate_matrix(gate: Gate) -> tuple[np.ndarray, list[int]]:
    """Gate → (酉矩阵, qubits)。qubits 满足 _apply_matrix 的 LSB-first 约定。"""
    if gate.matrix is not None:
        return np.asarray(gate.matrix, np.complex128), list(gate.targets)
    name = gate.name.upper()
    tgt, ctrl, params = list(gate.targets), list(gate.controls), list(gate.params)

    # 纯相位类（受控与否都只在对角线加相位）
    if name in ("P", "PHASE", "U1"):
        return np.diag([1.0, np.exp(1j * params[0])]).astype(np.complex128), tgt
    if name in ("CP", "MCP", "MCPHASE"):
        k = len(tgt) + len(ctrl)
        m = np.eye(1 << k, dtype=np.complex128)
        m[-1, -1] = np.exp(1j * params[0])
        return m, tgt + ctrl

    # 控制位归一：名字剥掉 C / MC / MCR 前缀得到 base 门
    if name == "CNOT":
        name = "CX"  # 历史别名；否则会被 "C" 前缀剥成非法的 "NOT"
    base_name, n_ctrl = name, len(ctrl)
    while n_ctrl and base_name not in _BASE_NAMES:
        for pfx in ("MCR", "MC", "C"):
            if base_name.startswith(pfx) and len(base_name) > len(pfx):
                base_name = base_name[len(pfx):]
                break
        else:
            break

    if base_name in _MATRICES:
        base = _MATRICES[base_name]
    elif base_name in ("RX", "RY", "RZ"):
        base = _rotation(base_name, params[0])
    elif base_name in ("U", "U3"):
        base = _u3(params[0], params[1], params[2])
    elif base_name == "U2":
        base = _u3(np.pi / 2, params[0], params[1])
    else:
        raise ValueError(f"unsupported gate: {gate.name}")

    if n_ctrl:
        return _controlled(base, n_ctrl), tgt + ctrl
    return base, tgt


def apply_gate(state: np.ndarray, gate: Gate) -> np.ndarray:
    """单个 Gate 作用到态矢量（纯函数，返回新数组）。"""
    name = gate.name.upper()
    if name == "BARRIER":
        return state
    if name == "SWAP":
        return _apply_matrix(state, _SWAP, list(gate.targets))
    if name == "ISWAP":
        return _apply_matrix(state, _ISWAP, list(gate.targets))
    matrix, qubits = _gate_matrix(gate)
    return _apply_matrix(state, matrix, qubits)


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

_ORIGINIR_1Q = {"H", "X", "Y", "Z", "S", "T", "I", "SDG", "TDG",
                "RX", "RY", "RZ", "U1", "U2", "U3", "P"}
_ORIGINIR_2Q = {"CNOT", "CX", "CZ", "SWAP", "ISWAP", "CRZ", "CH", "CU3"}
_ORIGINIR_3Q = {"TOFFOLI", "CCX"}
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
        if name in _ORIGINIR_3Q:  # TOFFOLI q[0], q[1], q[2]：前两个为控制位
            gates.append(Gate("CCX", targets=[qubits[2]], controls=qubits[:2]))
        elif name in _ORIGINIR_1Q:
            gates.append(Gate(name, targets=[qubits[0]], params=params))
        elif name in _ORIGINIR_2Q:
            if name in ("CNOT", "CX", "CZ", "CRZ", "CH", "CU3"):
                gates.append(Gate(name, targets=[qubits[1]], controls=[qubits[0]],
                                  params=params))
            else:  # SWAP / ISWAP
                gates.append(Gate(name, targets=[qubits[0], qubits[1]]))
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
