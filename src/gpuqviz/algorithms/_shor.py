"""Shor 周期查找（N=15 演示级）：SWAP 分解的受控模乘 + 经典后处理。

数学基础（tests/cross_validation/test_shor_hhl.py 端到端锁定）：

- N=15、a ∈ {2, 4}：a ∈ ⟨2⟩ = {1,2,4,8}（mod 15 乘法群的 2-子群），
  此时 a·x mod 15 是工作寄存器（4 bit）的**循环移位**，受控模乘分解为
  少量受控 SWAP（Fredkin）——无需基于加法器的模乘线路；
  - 2x mod 15 = 循环左移 1（content i → i+1 mod 4）
    = cswap(2,3)→cswap(1,2)→cswap(0,1)（链向低位走）；
  - 4x mod 15 = 循环左移 2 = cswap(0,2)+cswap(1,3)（对位交换）；
  - 8x mod 15 = 循环左移 3 = cswap(0,1)→cswap(1,2)→cswap(2,3)（链向高位走）；
  - a=7/11/13 不在 ⟨2⟩ 内，需要基于加法器的模乘（超出演示范围）。
- 周期 r：a=2 → r=4；a=4 → r=2。t_bits ≥ 2·log2(N) 时 QPE 峰精确
  落在 s = m·2^t/r（r 整除 2^t → 峰概率恰为 1/r）；
- 经典后处理：峰位 s/2^t 约分分母 → r → gcd(a^{r/2} ± 1, N) → 因子。

a=2 时 r=4：gcd(4-1, 15)=3，gcd(4+1, 15)=5 → 15 = 3 × 5 ✓
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["shor", "shor_period_finding", "continued_fraction_period",
           "factor_from_period"]

# 循环左移 shift 的 SWAP 对（4 bit 工作寄存器，content i → i+shift mod 4）
_ROTATION_SWAPS: dict[int, list[tuple[int, int]]] = {
    1: [(2, 3), (1, 2), (0, 1)],   # 循环左移 1：content 3→2→1→0（链向低位）
    2: [(0, 2), (1, 3)],           # 循环左移 2：对位交换
    3: [(0, 1), (1, 2), (2, 3)],   # 循环左移 3：content 0→1→2→3（链向高位）
}


def _rotation_swap_pairs(shift: int, L: int = 4) -> list[tuple[int, int]]:
    if L != 4:
        raise NotImplementedError("SWAP 分解仅支持 4 bit 工作寄存器（N=15）")
    return _ROTATION_SWAPS.get(shift % L, [])


def _iqft_matrix(bits: int) -> np.ndarray:
    """逆 QFT 矩阵：IQFT[s, x] = e^{-2πi·sx/2^bits}/√2^bits（s/x 为 LSB-first 整数）。

    对相位态 Σ_x e^{+2πi·φx}|x⟩ 给出峰 |⌊φ·2^bits⌉⟩——与 QPE 相位回授
    （受控 e^{i2^j·θ} 作用在计数 qubit j，j 为权重 2^j 的 LSB 位）配套。
    """
    dim = 1 << bits
    s, x = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
    return np.exp(-2j * np.pi * s * x / dim) / np.sqrt(dim)


def shor_period_finding(N: int = 15, a: int = 2, t_bits: int = 6,
                        engine: str = "qiskit"):
    """Shor 周期查找线路：计数寄存器 QPE 峰 → 连分数 → 周期 → 因子。

    N=15、a ∈ {2, 4}（⟨2⟩ 子群，SWAP 可分解）。返回 t_bits 计数
    qubit + 4 工作 qubit 的电路（qiskit）或 Gate 列表（numpy）。

    线路：工作寄存器 |1⟩ → H^⊗t → 受控 a^{2^j} mod N（j = 0..t-1，
    SWAP 分解）→ IQFT^t。a^{2^j} mod 15 的循环移位量 = (j·log2(a)) mod 4；
    a=2 时 shift_j = j mod 4（j≥2 恒 0 → identity）。
    """
    if N != 15:
        raise NotImplementedError("演示级实现仅支持 N=15")
    if a not in (2, 4):
        raise NotImplementedError(
            f"a={a} 不在 ⟨2⟩={{1,2,4,8}} 内；SWAP 分解仅支持 a ∈ {{2, 4}}")
    log2a = 1 if a == 2 else 2
    work = list(range(t_bits, t_bits + 4))  # 工作寄存器 qubit 编号

    def _shift_pairs(j: int) -> list[tuple[int, int]]:
        # a^{2^j} mod 15 的移位量 = log2a · 2^j mod 4（j 是被控的幂次）
        return _ROTATION_SWAPS.get((log2a * (1 << j)) % 4, [])

    if engine == "qiskit":
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate

        qc = QuantumCircuit(t_bits + 4, t_bits)
        qc.x(work[0])  # 工作寄存器 |1⟩
        qc.h(range(t_bits))
        for j in range(t_bits):
            for i, k in _shift_pairs(j):
                qc.cswap(j, work[i], work[k])
        # IQFT：显式矩阵（门分解的 SWAP 位置约定有歧义，矩阵构造即正确）
        qc.append(UnitaryGate(_iqft_matrix(t_bits)), range(t_bits))
        qc.measure(range(t_bits), range(t_bits))
        return qc

    # numpy：框架无关 Gate 列表
    from ..circuits import Gate

    gates: list = []
    gates.append(Gate(name="X", targets=[work[0]]))  # |1⟩
    for j in range(t_bits):
        gates.append(Gate(name="H", targets=[j]))
    for j in range(t_bits):
        shift = (log2a * (1 << j)) % 4
        if shift == 0:
            continue  # a^{2^j} ≡ 1 → identity
        # 受控循环移位：5-qubit UNITARY（1 control + 4 work = 32×32）
        # 矩阵约定：targets[0]=j 为 LSB → index = j_bit + 2·Σ(wi_bit·2^wi)
        dim = 32  # 2^5
        M = np.zeros((dim, dim), dtype=np.complex128)
        for col in range(dim):
            j_bit = col & 1
            work_val = (col >> 1) & 0xF  # 4-bit 工作寄存器值
            if j_bit == 0:
                row = col  # control=0 → identity
            else:
                # 循环左移 shift：work_val → (work_val · 2^shift) mod 15
                new_work = (work_val * (1 << shift)) % 15
                row = 1 | (new_work << 1)
            M[row, col] = 1.0
        gates.append(Gate(name="UNITARY", matrix=M,
                          targets=[j] + work))
    # IQFT：数学直接构造（UNITARY 矩阵，避免 H+CP 分解的约定歧义）
    gates.append(Gate(name="UNITARY", matrix=_iqft_matrix(t_bits),
                      targets=list(range(t_bits))))
    return gates


# --------------------------------------------------------------------------- #
# 经典后处理
# --------------------------------------------------------------------------- #

def shor(a: int = 2, t_bits: int = 6, engine: str = "qiskit"):
    """注册表入口：Shor 周期查找（N=15，a∈{2,4}，SWAP 分解）。

    返回 (t_bits + 4)-qubit 电路；经典后处理见 continued_fraction_period /
    factor_from_period（demo 脚本与交叉验证测试演示完整流程）。
    """
    return shor_period_finding(N=15, a=a, t_bits=t_bits, engine=engine)


def continued_fraction_period(s: int, t_bits: int, N: int) -> int:
    """精确峰 s/2^t 的约分分母 = 周期 r（r | 2^t 的精确峰）。"""
    g = math.gcd(s, 1 << t_bits)
    if g == 0:
        return 1
    return (1 << t_bits) // g


def factor_from_period(N: int, a: int, r: int) -> tuple[int, int] | None:
    """由周期 r 得因子：gcd(a^{r/2} ± 1, N)。

    r 奇或 a^{r/2} ≡ N-1（即 -1 mod N）时返回 None（需换 a 重试）。
    """
    if r % 2 != 0:
        return None
    base = pow(a, r // 2, N)
    if base == N - 1:
        return None
    f1 = math.gcd(base - 1, N)
    f2 = math.gcd(base + 1, N)
    if 1 < f1 < N and 1 < f2 < N:
        return f1, f2
    return None
