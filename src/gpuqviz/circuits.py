"""框架无关的电路表示与 numpy 态矢量演化。

适配器（qiskit / pyqpanda / …）把各自的电路翻译成 Gate 列表（或直接解析
ORIGINIR 文本），本模块用 numpy 完成逐门演化并产出关键帧态矢量序列——
使 gpuqviz 的可视化能力不绑定任何单一模拟器。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

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


@dataclass(frozen=True)
class Condition:
    """经典条件：经典位 clbit 的值 == value 时门才施加（P3.2 确定性反馈）。"""

    clbit: int
    value: int = 1


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
    condition: Condition | None = field(default=None, compare=False)


def _as_le(state: np.ndarray) -> np.ndarray:
    """重排张量轴为小端序视图：轴 i ↔ qubit i（qiskit/pyqpanda 约定）。

    numpy reshape 后轴 0 是最高位（大端序），必须翻转才能与框架的 qubit
    编号一致——这是适配器数值正确性的关键。
    """
    n = int(round(np.log2(state.shape[0])))
    return state.reshape((2,) * n).transpose(*reversed(range(n)))


def _apply_1q(state: np.ndarray, matrix: np.ndarray, target: int) -> np.ndarray:
    """单比特门：小端序平铺索引中 qubit q 的位权重是 2^q，恰为
    reshape(2^{n-1-q}, 2, 2^q) 的中间轴步长——单次 einsum 一步到位，
    无轴重排拷贝（0.9.0 T1-4）。"""
    dim = state.shape[0]
    n = int(round(np.log2(dim)))
    left = 1 << (n - 1 - target)
    right = 1 << target
    psi = np.asarray(state).reshape(left, 2, right)
    # 批量 GEMM：(2×2) @ (left, 2, right) 广播到每个低位块，单次拷贝
    out = np.matmul(matrix, psi)
    return out.reshape(-1)


def _apply_matrix(state: np.ndarray, matrix: np.ndarray, qubits: list[int]) -> np.ndarray:
    """把 2^k×2^k 酉矩阵作用到指定 qubit 集合（通用门作用原语）。

    qubits 采用 qiskit Operator 约定：qubits[0] 是矩阵的最低位（LSB）。
    受控门按 targets 在前（低位）、controls 在后（高位）拼入 qubits，
    配合 _controlled 构造的矩阵（base 作用于低位块、控制位全 1 时生效）。

    0.9.0 T1-4：矩阵指数分解为 2^k 张量 + 整数字母 einsum 单遍收缩，
    消除 moveaxis/transpose 的多次全态拷贝（原实现 3 次，现 1 次）。
    """
    k = len(qubits)
    dim = 1 << k
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (dim, dim):
        raise ValueError(f"matrix shape {matrix.shape} does not match {k} qubits")
    if k == 1:
        return _apply_1q(state, matrix, qubits[0])
    n = int(round(np.log2(state.shape[0])))
    if n + k > 52:
        raise ValueError(f"too many qubits ({n}) + targets ({k}) for einsum letters")
    m6 = matrix.reshape((2,) * k + (2,) * k)
    # M6 轴序：前 k 轴 = 输出 MSB→LSB = qubits[k-1]→qubits[0]；后 k 轴 = 输入同序
    m_subs = [n + qubits.index(qubits[k - 1 - i]) for i in range(k)]
    m_subs += [qubits[k - 1 - i] for i in range(k)]
    # psi 轴 p ↔ qubit n-1-p；输入字母 = qubit 编号本身，输出字母 = n + targets 序号
    psi_subs = [n - 1 - p for p in range(n)]
    out_subs = [n + qubits.index(n - 1 - p) if (n - 1 - p) in qubits
                else (n - 1 - p) for p in range(n)]
    psi = np.asarray(state).reshape((2,) * n)
    out = np.einsum(m6, m_subs, psi, psi_subs, out_subs, optimize=True)
    return out.reshape(-1)


def _apply_controlled(state: np.ndarray, base: np.ndarray,
                      targets: list[int], controls: list[int]) -> np.ndarray:
    """受控门：控制位切片（零拷贝 strided 视图）只作用 control=1 半态。

    替代稠密 _controlled 矩阵路径——不再构造 2^(nc+t) 矩阵（MCX 9 控制
    时是 1024×1024 稠密块），且张量流量减半（只触碰 control=1 的振幅）。
    base 约定同 _gate_matrix：targets[0] 是 LSB。
    """
    n = int(round(np.log2(state.shape[0])))
    t = len(targets)
    dim_t = 1 << t
    base = np.asarray(base, dtype=np.complex128)
    if base.shape != (dim_t, dim_t):
        raise ValueError(f"base shape {base.shape} does not match {t} targets")
    if set(targets) & set(controls):
        raise ValueError("targets and controls must be disjoint")
    psi_t = np.asarray(state).reshape((2,) * n)  # 轴 p ↔ qubit n-1-p
    # 控制位固定为 1：切片视图，零拷贝
    sel = tuple(1 if (n - 1 - p) in controls else slice(None) for p in range(n))
    block = psi_t[sel]  # 轴序 = 非控制 qubit 降序
    # 目标轴在 block 中的位置（tensordot 需 MSB-first：targets[k-1] 先收缩）
    def _block_pos(q: int) -> int:
        return sum(1 for c in range(n) if c not in controls and c > q)
    tpos = [_block_pos(q) for q in reversed(targets)]  # MSB-first
    base6 = base.reshape((2,) * t + (2,) * t)  # 列索引分解为 t 个轴（MSB-first）
    out = np.tensordot(base6, block,
                       axes=(list(range(t, 2 * t)), tpos))  # (2^t, rest...)
    rest_dims = [block.shape[p] for p in range(block.ndim) if p not in tpos]
    out = out.reshape((2,) * t + tuple(rest_dims))
    # 前 t 轴 = 目标 MSB→LSB = targets[k-1]→targets[0]，moveaxis 回 block 轴序
    dest = [_block_pos(q) for q in targets]  # targets[0]..targets[k-1] 的位置
    src = list(range(t))  # out 前 t 轴顺序 = targets[k-1]..targets[0]
    out = np.moveaxis(out, src, list(reversed(dest)))
    # 纯函数契约：apply_gate 不得修改输入（快照持有各关键帧数组）
    new_state = state.copy()
    new_state.reshape((2,) * n)[sel] = out
    return new_state


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

    base, tgt = _base_matrix_of(gate)
    if ctrl:
        return _controlled(base, len(ctrl)), tgt + ctrl
    return base, tgt


def _base_matrix_of(gate: Gate) -> tuple[np.ndarray, list[int]]:
    """受控门 → (base 酉矩阵, targets)：剥掉 MC/C 前缀解析 base 门，
    不包装控制位（控制位切片路径用，0.9.0 T1-3）。"""
    name = gate.name.upper()
    tgt, ctrl, params = list(gate.targets), list(gate.controls), list(gate.params)

    # 相位族：CP = 受控 P——base 是 targets 上的对角相位门，
    # phase 仅在全部 targets+controls 为 1 时生效（与稠密实现语义一致）
    if name in ("P", "PHASE", "U1"):
        return np.diag([1.0, np.exp(1j * params[0])]).astype(np.complex128), tgt
    if name in ("CP", "MCP", "MCPHASE"):
        k = max(1, len(tgt))
        m = np.eye(1 << k, dtype=np.complex128)
        m[-1, -1] = np.exp(1j * params[0])
        return m, tgt

    # 控制位归一：名字剥掉 MC / C 前缀得到 base 门
    # （无 "MCR" 前缀：MCRY = MC + RY，先剥 3 字符会把 base 错剥成 Y）
    if name == "CNOT":
        name = "CX"  # 历史别名；否则会被 "C" 前缀剥成非法的 "NOT"
    base_name, n_ctrl = name, len(ctrl)
    while n_ctrl and base_name not in _BASE_NAMES:
        for pfx in ("MC", "C"):
            if base_name.startswith(pfx) and len(base_name) > len(pfx):
                base_name = base_name[len(pfx):]
                break
        else:
            break

    if base_name in _MATRICES:
        base = _MATRICES[base_name]
    elif base_name in ("RX", "RY", "RZ"):
        base = _rotation_cached(base_name, params[0])
    elif base_name in ("U", "U3"):
        base = _u3_cached("U3", params[0], params[1], params[2])
    elif base_name == "U2":
        base = _u3_cached("U2", np.pi / 2, params[0], params[1])
    else:
        raise ValueError(f"unsupported gate: {gate.name}")
    return base, tgt


@lru_cache(maxsize=512)
def _rotation_cached(name: str, theta: float) -> np.ndarray:
    return _rotation(name, theta)


@lru_cache(maxsize=512)
def _u3_cached(kind: str, theta: float, phi: float, lam: float) -> np.ndarray:
    if kind == "U2":
        return _u3(np.pi / 2, theta, phi)
    return _u3(theta, phi, lam)


def apply_gate(state: np.ndarray, gate: Gate) -> np.ndarray:
    """单个 Gate 作用到态矢量（纯函数，返回新数组）。

    0.9.0 T1-3：带 controls 且未携带显式矩阵的门走控制位切片路径
    （_apply_controlled），不再构造稠密 2^(nc+t) 受控矩阵。
    """
    name = gate.name.upper()
    if name == "BARRIER":
        return state
    if name == "SWAP":
        return _apply_matrix(state, _SWAP, list(gate.targets))
    if name == "ISWAP":
        return _apply_matrix(state, _ISWAP, list(gate.targets))
    if gate.controls and gate.matrix is None:
        base, targets = _base_matrix_of(gate)
        return _apply_controlled(state, base, targets, list(gate.controls))
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


# --------------------------------------------------------------------------- #
# 中途测量 / 重置 / 条件门（P3.2 确定性经典反馈）
# --------------------------------------------------------------------------- #

@dataclass
class BranchEvolution:
    """带经典反馈演化的结果。

    frames: [初态, 每个非 barrier 门后的态]（与 evolve_gates 同构）；
    measurements: [(clbit, 塌缩值, 塌缩前概率), ...] 按发生顺序。
    """

    frames: list[np.ndarray]
    measurements: list[tuple[int, int, float]]


def _collapse_statevector(psi: np.ndarray, n_qubits: int,
                          qubit: int, value: int) -> tuple[np.ndarray, float]:
    """把 qubit 投影到计算基 |value⟩ 并归一化。返回 (新态, 塌缩前概率)。

    输出保持全长 2**n（固定 qubit 位 = value，其余振幅置零后归一），
    与演化关键帧的形状契约一致。
    概率为零说明该分支不可能发生。
    """
    idx = np.arange(psi.size)
    mask = ((idx >> qubit) & 1) == value  # 小端序：位 qubit = value 的振幅
    kept = psi[mask]
    prob = float(np.vdot(kept, kept).real)
    if prob < 1e-12:
        raise ValueError(
            f"branch has zero probability: qubit {qubit} → {value}")
    out = np.zeros_like(psi)
    out[mask] = kept / np.sqrt(prob)
    return out, prob


def evolve_gates_branches(n_qubits: int, gates: list[Gate],
                          branch: dict[int, int]) -> BranchEvolution:
    """带中途测量/重置/条件门的单分支确定性演化（P3.2）。

    branch: {clbit: 期望测量结果}——MEASURE 门按分支值塌缩（非随机采样）；
    条件门（Gate.condition）当 branch[condition.clbit] == condition.value
    时施加，否则按恒等跳过。分支未覆盖的 clbit 触发报错。

    与 Aer 的对拍：tests/cross_validation/test_conditional.py
    （IfElseOp + density_matrix，确定概率分支下逐元素一致）。
    """
    state = np.zeros(2 ** n_qubits, dtype=np.complex128)
    state[0] = 1.0
    frames = [state]
    measurements: list[tuple[int, int, float]] = []

    for gate in gates:
        name = gate.name.upper()
        if name == "BARRIER":
            continue

        if name == "MEASURE":
            if len(gate.targets) != 1 or not gate.params:
                raise ValueError("MEASURE 需要 targets=[qubit], params=[clbit]")
            clbit = int(gate.params[0])
            if clbit not in branch:
                raise ValueError(
                    f"branch missing clbit {clbit} (measure needs a "
                    "deterministic branch under the P3.2 model)")
            value = int(branch[clbit])
            state, prob = _collapse_statevector(
                state, n_qubits, gate.targets[0], value)
            measurements.append((clbit, value, prob))
            frames.append(state)
            continue

        if name == "RESET":
            state, prob = _collapse_statevector(
                state, n_qubits, gate.targets[0], 0)
            frames.append(state)
            continue

        if gate.condition is not None:
            clbit = gate.condition.clbit
            if clbit not in branch:
                raise ValueError(
                    f"conditional gate {gate.name} reads clbit {clbit} "
                    "which the branch does not specify")
            if int(branch[clbit]) != gate.condition.value:
                continue  # 条件不成立 → 恒等
            state = apply_gate(state, gate)
            frames.append(state)
            continue

        state = apply_gate(state, gate)
        frames.append(state)

    return BranchEvolution(frames=frames, measurements=measurements)


def sample_snapshots(snapshots: list[np.ndarray], steps: int) -> list[np.ndarray]:
    """门级快照均匀采样 steps 个关键帧（含初态与末态，静止段复用）。"""
    if steps < 2:
        steps = 2
    m = len(snapshots)
    if m >= steps:
        idx = np.round(np.linspace(0, m - 1, steps)).astype(int)
        return [snapshots[i] for i in idx]
    # 0.9.0 T2：向量化批量插值（复用 interpolate.lerp_states 的
    # gather+renormalize 公式），替代逐帧 Python 循环
    from .interpolate import lerp_states

    frames = lerp_states(snapshots, steps)
    return [frames[i] for i in range(frames.shape[0])]


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
