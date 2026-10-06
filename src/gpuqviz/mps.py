"""MPS（矩阵乘积态）近似演化后端（P5.2）。

面向 20+ qubit 低纠缠电路的可选后端：1q 门直接收缩，2q 门经 SVD
截断到键维 χ_max；非相邻 2q 门自动 SWAP 路由。输出逐关键帧的
约化密度矩阵 / Bloch 向量 / 割纠缠熵（docs/development-plan.md P5.2
"输出约化分析量喂给现有渲染层"）。

约定（与全库一致，docs/conventions.md §1）：
- qubit 小端序；Gate 2q 矩阵 LSB = targets[0]（circuits._gate_matrix）；
- 精确模式（chi_max=None）与态矢量逐位对拍 1e-10（tests/cross_validation/
  test_mps.py）；截断模式断言误差有界。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuits import Gate, _gate_matrix

__all__ = ["MPS", "evolve_mps", "MPSResult"]

_SVD_TOL = 1e-14


class MPS:
    """有限键维矩阵乘积态。site i 的张量形状 (Dl, 2, Dr)。"""

    def __init__(self, tensors: list[np.ndarray]):
        self.tensors = tensors
        self.n = len(tensors)

    # -- 构造 --------------------------------------------------------------- #

    @classmethod
    def product(cls, n: int) -> "MPS":
        """|0…0⟩ 初始态。"""
        tensors = []
        for i in range(n):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0
            tensors.append(t)
        return cls(tensors)

    def copy(self) -> "MPS":
        return MPS([t.copy() for t in self.tensors])

    def max_bond_dim(self) -> int:
        """当前最大键维（截断程度的上界指示）。"""
        return max((max(t.shape[0], t.shape[2]) for t in self.tensors), default=1)

    @property
    def provenance(self) -> str:
        """近似语义标注（0.8.0 审计 #4）："mps_chi=N"。

        χ 逐操作指定、无单一全局值，此处报告当前最大键维；
        与精确态对拍（chi_max=None）时数值为精确。
        """
        return f"mps_chi={self.max_bond_dim()}"

    # -- 门作用 ------------------------------------------------------------- #

    def apply_1q(self, U: np.ndarray, site: int) -> None:
        """单 qubit 酉门：不改变键维。"""
        g = self.tensors[site]
        self.tensors[site] = np.einsum("ji,aib->ajb", U, g, optimize=True)

    def apply_2q_nn(self, gate_tensor: np.ndarray, site: int,
                    chi_max: int | None = None) -> None:
        """最近邻 2q 门：gate_tensor 轴 = (s_site, s_site+1, s'_site, s'_site+1)，
        即 site 是 LSB。SVD 截断到 chi_max。"""
        gi = self.tensors[site]
        gj = self.tensors[site + 1]
        theta = np.einsum("aib,bjc->aijc", gi, gj, optimize=True)
        # 作用门：out[a, p, q, c] = Σ T[p,q,i,j] θ[a,i,j,c]
        out = np.einsum("pqij,aijc->apqc", gate_tensor, theta, optimize=True)
        dl, _, _, dr = out.shape
        mat = out.reshape(dl * 2, 2 * dr)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = int(np.sum(S > _SVD_TOL * S[0]))
        if chi_max is not None:
            keep = min(keep, chi_max)
        keep = max(keep, 1)
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        self.tensors[site] = U.reshape(dl, 2, keep)
        rest = (S[:, None] * Vh).reshape(keep, 2, dr)
        self.tensors[site + 1] = rest

    def swap(self, site: int, chi_max: int | None = None) -> None:
        """相邻 SWAP（路由用）。"""
        swap4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                          [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
        self.apply_2q_nn(swap4.reshape(2, 2, 2, 2), site, chi_max=chi_max)

    # -- 正交化与谱 ---------------------------------------------------------- #

    def left_canonicalize(self, upto: int) -> None:
        """把 site 0..upto-1 扫成左正则（QR/SVD 混合，用 SVD 保数值稳定）。"""
        for site in range(min(upto, self.n - 1)):
            g = self.tensors[site]
            dl, phys, dr = g.shape
            mat = g.reshape(dl * phys, dr)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = int(np.sum(S > _SVD_TOL * (S[0] if S.size else 1.0)))
            keep = max(keep, 1)
            self.tensors[site] = U[:, :keep].reshape(dl, phys, keep)  # 左正则
            rest = (S[:keep][:, None] * Vh[:keep, :])
            nxt = self.tensors[site + 1]
            self.tensors[site + 1] = np.einsum("kb,bjc->kjc", rest, nxt,
                                               optimize=True)

    def right_canonicalize_from(self, start: int) -> None:
        """把 site start..n-1 扫成右正则（S 从左侧并入前一张量）。"""
        for site in range(self.n - 1, max(start - 1, 0), -1):
            g = self.tensors[site]
            dl, phys, dr = g.shape
            mat = g.reshape(dl, phys * dr)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = int(np.sum(S > _SVD_TOL * (S[0] if S.size else 1.0)))
            keep = max(keep, 1)
            self.tensors[site] = Vh[:keep, :].reshape(keep, phys, dr)  # 右正则
            left = (U[:, :keep] * S[:keep])  # (dl, k)
            prv = self.tensors[site - 1]
            # US 乘在右键上：左键 dl 保持不变（最左站边界约定）
            self.tensors[site - 1] = np.einsum("asb,bk->ask", prv, left,
                                               optimize=True)

    def schmidt_values(self, cut: int) -> np.ndarray:
        """割 cut|cut+1 的 Schmidt 系数（降序）。

        割在 site cut 的 (左键, phys) 与 (右键) 之间：Schmidt 矩阵 =
        Γ[cut].reshape(dl·2, dr)。需先把 site 0..cut-1 左正则化使左侧基
        正交——本方法内部完成，不改变物理态（保态的规范变换）。
        """
        # 混合正则：右侧右正则化到 cut+1，左侧左正则化到 cut-1
        # → 正交中心落在 tensors[cut]，其 (dl·phys, dr) 的奇异值即 Schmidt 谱
        self.right_canonicalize_from(cut + 1)
        self.left_canonicalize(cut)
        g = self.tensors[cut]
        mat = g.reshape(g.shape[0] * 2, g.shape[2])
        s = np.linalg.svd(mat, compute_uv=False)
        return np.sort(s)[::-1]

    def entanglement_entropy(self, cut: int) -> float:
        """割 cut|cut+1 的 von Neumann 熵（log₂，单位 bit）。"""
        s = self.schmidt_values(cut)
        s2 = np.clip(s ** 2, 0.0, None)
        nz = s2[s2 > 1e-14]
        return float(-np.sum(nz * np.log2(nz)))

    # -- 约化量 -------------------------------------------------------------- #

    def _left_env(self, site: int) -> np.ndarray:
        L = np.ones((1, 1), dtype=np.complex128)
        for j in range(site):
            g = self.tensors[j]
            L = np.einsum("ac,asb,csd->bd", L, g, g.conj(), optimize=True)
        return L

    def _right_env(self, site: int) -> np.ndarray:
        R = np.ones((1, 1), dtype=np.complex128)
        for j in range(self.n - 1, site, -1):
            g = self.tensors[j]
            R = np.einsum("bd,asb,csd->ac", R, g, g.conj(), optimize=True)
        return R

    def reduced_rho(self, site: int) -> np.ndarray:
        """单 qubit 约化密度矩阵 (2,2)（双环境收缩）。"""
        L = self._left_env(site)
        R = self._right_env(site)
        g = self.tensors[site]
        rho = np.einsum("ac,asb,ctd,bd->st", L, g, g.conj(), R, optimize=True)
        rho = (rho + rho.conj().T) / 2  # 消数值 Hermit 噪声
        tr = float(np.real(np.trace(rho)))
        return rho / tr if abs(tr) > 1e-12 else rho

    def bloch(self) -> np.ndarray:
        """全部 qubit 的 Bloch 向量 (n, 3)。"""
        out = np.zeros((self.n, 3), dtype=np.float64)
        for q in range(self.n):
            rho = self.reduced_rho(q)
            out[q, 0] = float(np.real(rho[0, 1] + rho[1, 0]))
            out[q, 1] = float(np.real(1j * (rho[1, 0] - rho[0, 1])))
            out[q, 2] = float(np.real(rho[0, 0] - rho[1, 1]))
        return out

    # -- 测试辅助 ------------------------------------------------------------ #

    def to_statevector(self) -> np.ndarray:
        """收缩为全态矢量（仅小 n 测试用）。小端序：轴 i ↔ qubit i。"""
        acc = self.tensors[0].reshape(2, -1)  # (s_q0, bond)
        for site in range(1, self.n):
            g = self.tensors[site]  # (D, 2, Dr)
            # 键轴求和，phys 轴按 qubit 序累积在前面
            acc = np.einsum("...b,bic->...ic", acc, g, optimize=True)
        # C-order reshape 最后的轴变化最快 → q0（LSB）必须排最后：
        # phys 轴（前 n-1 个）按 qubit 降序放到末尾，最终键轴放最前
        phys_axes = list(range(acc.ndim - 1))
        order = [acc.ndim - 1] + phys_axes[::-1]
        acc = acc.transpose(order)
        return acc.reshape(-1)


# --------------------------------------------------------------------------- #
# 电路演化
# --------------------------------------------------------------------------- #

@dataclass
class MPSResult:
    """MPS 演化结果：关键帧（MPS 快照）+ 路由统计。"""

    frames: list[MPS]
    n_swaps: int = 0
    meta: dict = field(default_factory=dict)

    def bloch_keys(self) -> np.ndarray:
        """逐关键帧 Bloch 向量 (K, n, 3)——渲染层直接消费。"""
        return np.stack([f.bloch() for f in self.frames])

    def entropies(self) -> np.ndarray:
        """逐关键帧的割纠缠熵 (K, n-1)（MPS 结构免费给出）。"""
        return np.stack([f._cached_entropies() for f in self.frames])


def _entropy_all(mps: MPS) -> np.ndarray:
    return np.array([mps.entanglement_entropy(c) for c in range(mps.n - 1)])


MPS._cached_entropies = _entropy_all


def _gate_tensor_2q(U4: np.ndarray, qubits: list[int]) -> tuple[np.ndarray, int]:
    """(U4, [qa, qb]) → (T[phys_site, phys_site+1, ...], site)。

    qubits LSB-first：qubits[0] 是 U4 的 LSB。若 qubits = [x, x+1]，
    site = x；若 qubits = [x+1, x]，site = x 且子系统对调。
    """
    qa, qb = qubits[0], qubits[1]
    if qb == qa + 1:
        return U4.reshape(2, 2, 2, 2), qa
    if qa == qb + 1:
        # U4 的 LSB 挂在 site+1 → 对调两个子系统轴
        return U4.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2), qb
    raise ValueError("2q gate is not nearest-neighbor; route with SWAPs first")


def evolve_mps(n_qubits: int, gates: list[Gate],
               chi_max: int | None = None) -> MPSResult:
    """门序列的 MPS 演化，返回 [初态, 每个非 barrier 门后的 MPS 快照]。

    chi_max=None 精确演化（键维随纠缠增长）；给定则 SVD 截断。
    非相邻 2q 门自动 SWAP 路由（去程 + 回程，n_swaps 统计），
    布局在门后恢复原位，保证后续门的 qubit 编号语义不变。
    """
    mps = MPS.product(n_qubits)
    frames = [mps.copy()]
    n_swaps = 0

    for gate in gates:
        name = gate.name.upper()
        if name == "BARRIER":
            continue
        if name in ("MEASURE", "RESET"):
            raise NotImplementedError(
                "MPS backend does not support measurement/reset gates")
        matrix, qubits = _gate_matrix(gate)
        if len(qubits) == 1:
            mps.apply_1q(np.asarray(matrix, np.complex128), qubits[0])
            frames.append(mps.copy())
            continue
        if len(qubits) != 2:
            raise NotImplementedError(
                f"MPS backend supports 1q/2q gates, got {len(qubits)}")

        U4 = np.asarray(matrix, np.complex128)
        qa, qb = qubits[0], qubits[1]
        if abs(qa - qb) == 0:
            raise ValueError("2q gate needs two distinct qubits")

        if abs(qa - qb) == 1:
            # reshape(2,2,2,2) 首轴 = MSB = qubits[1]（numpy 行主序拆分）：
            # T0[msb, lsb, msb', lsb']。要得到 (phys_site, phys_site+1) 轴序：
            #   qubits = [site, site+1]（LSB 在 site）→ 需 transpose(1,0,3,2)
            #   qubits = [site+1, site]（LSB 在 site+1）→ 直接 reshape
            if qb == qa + 1:
                mps.apply_2q_nn(
                    U4.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2), qa,
                    chi_max=chi_max)
            else:
                mps.apply_2q_nn(U4.reshape(2, 2, 2, 2), qb, chi_max=chi_max)
            frames.append(mps.copy())
            continue

        # ── 非相邻：SWAP 路由（去程 → 施加 → 回程）──
        # T0 = U4.reshape(2,2,2,2) 的首轴 = MSB = qubits[1] 的物理位。
        # 路由后 LSB 内容在 qb-1、MSB 内容在 qb（qa<qb）或
        # LSB 在 qb+1、MSB 在 qb（qa>qb）→ 由此选择直接/转置。
        if qa < qb:
            for s in range(qa, qb - 1):
                mps.swap(s, chi_max=chi_max)
                n_swaps += 1
            # MSB 内容在 phys_site+1 → T0 首轴 = phys_site+1 → 需转置
            mps.apply_2q_nn(
                U4.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2), qb - 1,
                chi_max=chi_max)
            for s in range(qb - 2, qa - 1, -1):
                mps.swap(s, chi_max=chi_max)
                n_swaps += 1
        else:
            for s in range(qa - 1, qb, -1):
                mps.swap(s, chi_max=chi_max)
                n_swaps += 1
            # MSB 内容在 phys_site → T0 首轴 = phys_site → 直接 reshape
            mps.apply_2q_nn(U4.reshape(2, 2, 2, 2), qb, chi_max=chi_max)
            for s in range(qb + 1, qa):
                mps.swap(s, chi_max=chi_max)
                n_swaps += 1
        frames.append(mps.copy())

    return MPSResult(frames=frames, n_swaps=n_swaps,
                     meta={"chi_max": chi_max, "n_qubits": n_qubits})
