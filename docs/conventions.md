# gpuqviz 数值与显示约定规范

> 本文档是 gpuqviz 所有数值输出与视觉呈现的**契约**。任何分析与可视化量
> 必须遵循此处定义，且由 `tests/cross_validation/` 的契约测试锁定。
> 修改约定 = 修改本文档 + 同步修改契约测试，缺一不可合入。

## 1. Qubit 端序：小端序（little-endian）

gpuqviz 全链路遵循 qiskit/pyqpanda 的**小端序**约定：

- qubit 编号 `q = 0 .. n-1`，`q = 0` 是**最低有效位**；
- 态矢量下标 `i` 的二进制展开中，**第 q 位对应 qubit q**：
  `i = Σ_q b_q · 2^q`；
- numpy 实现上，`psi.reshape((2,) * n)` 得到的张量**轴 a 对应 qubit
  `n-1-a`**（轴 0 是最高位）。框架内由 `circuits.py:_as_le` 统一完成
  大端 reshape 到小端逻辑的翻转——这是适配器数值正确性的关键，禁止
  在其他模块绕过它另行 reshape。

**位串显示约定**：基矢位串写作 `q_{n-1} … q_1 q_0`（qiskit 风格，
qubit 0 在最右）。即 `format(index, f"0{n}b")`。

例：n=2 时 index=1（q0=1, q1=0）显示为 `"01"`，
振幅 `psi[1]` 是 `|0₁ 1₀⟩ = |01⟩` 的振幅。

**验证**：`tests/cross_validation/test_conventions.py` 用 qiskit
`Statevector.probabilities_dict` 对拍位串归属。

## 2. 态的存储与全局相位

- 态矢量按**输入原样**存储，不做隐式归一化或全局相位规范
  （全局相位不可观测，强行规范化反而破坏"和上游框架逐位一致"的可核对性）；
- 需要相位无关比较时使用保真度 `F(ψ,φ) = |⟨ψ|φ⟩|²`（对全局相位不变）；
- 归一化仅作为显式操作提供：`Statevector.normalize()`（范数归一，
  不改全局相位）；
- 显示相位 `∠amp = atan2(im, re)`，单位为度，范围 (−180, 180]。

## 3. Bloch 向量

单 qubit q 的 Bloch 向量由**约化密度矩阵**定义：

```
ρ_q = Tr_{rest}(|ψ⟩⟨ψ|)
r = (⟨X⟩, ⟨Y⟩, ⟨Z⟩)，⟨P⟩ = Tr(ρ_q P)
```

- 模长与单 qubit 纯度的换算：`purity(ρ_q) = Tr ρ_q² = (1 + |r|²) / 2`，
  即 `|r| = √(2·Tr ρ_q² − 1)`；
- 因此 `|r| = 1` ⟺ 纯态，`|r| = 0` ⟺ 该 qubit 与其余部分最大纠缠
  （或整体为 I/2）——渲染层的"模长 = 纯度"语义由此而来；
- 轴定义：Z 轴 = 计算基 {|0⟩,|1⟩}，|0⟩ 在 +Z；X 轴 = {|+⟩,|−⟩}；
  Y 轴 = {|+i⟩,|−i⟩}（标准 Bloch 球）。

**验证**：`tests/cross_validation/test_against_qiskit.py` 对拍
`qiskit.quantum_info.SparsePauliOp` 期望值。

## 4. 热图网格排列

`render/heatmap.py:state_to_image` 把态矢量 reshape 成网格：

- `cols = 2^ceil(n/2)`（宽），`rows = 2^floor(n/2)`；
- index `i` → `(row, col) = (i // cols, i % cols)`（行优先）；
- 每格显示内容由 `basis` 参数决定：`"probability"` 显示 |ψ_i|²，
  `"phase"` 显示相位色盘；
- 位串归属遵循 §1（行方向高位在前）。

**验证**：`tests/cross_validation/test_conventions.py` 用已知的
n=2/3 态逐格核对。

## 5. 测量统计（shot 采样）

- 采样器 `analysis.measurement.sample_counts(state, shots, seed)`：
  - **可复现**：同 seed 同电路 ⇒ 逐位相同的 counts；
  - 使用 `numpy.random.Generator(np.random.PCG64(seed))`（稳定且
    跨版本可复现的位生成器）；
  - 输出 `Counts`：`counts[bitstring] -> int`，`Σ counts = shots`；
  - 边际分布 `Counts.marginal(qubits)`：对未选中 qubit 求和，
    返回的位串只含选中 qubit（仍按 §1 顺序，保持相对位置）。
- 采样正确性的统计判据：经验频率与精确概率的逐项偏差，
  在 5σ 二项界内（契约测试用 shots ≥ 10⁵ 验证一次，避免测试过慢）。

**验证**：`tests/cross_validation/test_measurement.py` 对拍
`qiskit_aer.AerSimulator` 的采样分布（卡方一致性）。

## 6. 数值容差

| 场景 | 容差 |
|---|---|
| 精确量（概率/Bloch/熵/保真度）vs qiskit 参考 | ≤ 1e-10（float64） |
| 归一化残差 ‖ψ‖−1 | ≤ 1e-12 |
| 密度矩阵 Hermit 性检查 | ≤ 1e-12 |
| 采样统计验证 | 5σ 二项界 |

GPU（cupy float64）与 CPU（numpy float64）路径对同一输入的精确量
输出必须逐位一致（渲染有超采样差异属显示层，不受此约束）。

## 7. 分析量的公式定义（速查）

| 量 | 定义 | 备注 |
|---|---|---|
| 概率 p_i | \|ψ_i\|²（纯态）/ ρ_ii（混合态） | 归一化到 Σ=1 |
| 纠缠熵 S(ρ_A) | −Tr(ρ_A log₂ ρ_A) | von Neumann，log 以 2 为底，单位 bit |
| 纯度 | Tr ρ² | 纯态 = 1，最大混合 = 1/2^k |
| 线性熵 | 1 − Tr ρ² | 另有归一化变体 d/(d−1)·(1−Tr ρ²) |
| 保真度（纯纯） | \|⟨ψ\|φ⟩\|² | |
| 保真度（纯混） | ⟨ψ\|ρ\|ψ⟩ | |
| 保真度（混混） | (Tr √(√ρ_A ρ_B √ρ_A))² | Uhlmann；与 qiskit StateFidelity 一致 |
| Pauli 期望 | Tr(ρ P) | 位串 "IIXZ"：左起为 q_{n−1}，右端为 q₀ |
| 互信息 I(A:B) | S_A + S_B − S_AB | Phase 1.2（M2） |

## 8. 版本与变更

- 约定变更必须在本文档追加变更记录，并同步更新契约测试；
- 契约测试失败的 PR 不允许合入（无论是否"只是显示问题"——
  显示层的数值就是专家核对的对象）。

| 日期 | 变更 |
|---|---|
| 2026-10-02 | 初版：端序/相位/Bloch/热图/采样/容差/公式定义 |
