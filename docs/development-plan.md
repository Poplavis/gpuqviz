# gpuqviz 双轨开发方案：普通用户 × 专业用户

> 目标：把 gpuqviz 从"态矢量演示工具"升级为**双轨可视化库**——
> 普通用户保持零配置出片体验；专业用户获得**数据准确性可验证**的测量计算、
> 纠缠/噪声分析、以及高阶电路操作支持。

---

## 0. 总体设计原则

### 0.1 一个内核，两个 API 面

```
┌──────────────────── 交互层 ────────────────────┐
│  viewer.js 播放器 / playground / Jupyter / CLI  │
├──────────────────── 渲染层 ────────────────────┤
│  Bloch │ Heatmap │ ★Histogram │ ★EntGraph │    │
│  ★DensityMatrix          （GL + CPU 软光栅 + ★SVG）│
├──────────────── ★分析层（新增） ────────────────┤
│  measurement │ entanglement │ fidelity │        │
│  metrics                  —— 全部带对拍验证 ——   │
├──────────────────── 演化内核 ───────────────────┤
│  Statevector 演化(现有) │ ★DensityMatrix+噪声   │
│  ★参数化绑定 │ ★中途测量/条件门                  │
├──────────────────── 适配层 ────────────────────┤
│  qiskit │ pyqpanda/OriginIR │ 原生 Gate │ ★Aer  │
└────────────────────────────────────────────────┘
  ★ = 新增   现有模块零破坏，全部增量扩展
```

- **普通用户轨道**（现有 API 保持不变）：`render_bloch_video` / `export_html` /
  `gpuqviz demo` / playground——零配置、快、好看。
- **专业用户轨道**（新增）：`gpuqviz.analysis` 分析模块 + `ProVisualizer`
  门面 + 精确数据面板——每个数字可追溯到公式、可对拍、可导出。

### 0.2 数据准确性的四条硬性约定

专业用户能否信任输出，取决于以下四点是否成立且**可验证**：

1. **每个可视化量都有数学定义**：docstring 写明公式与文献出处
   （如纠缠熵 S(ρ_q) = 1 − Tr ρ_q²，互信息 I(A:B) = S_A + S_B − S_AB）。
2. **交叉验证（对拍）**：每个分析量与 `qiskit.quantum_info` 参考实现对比，
   数值容差内必须一致（见 Phase 0 验证套件）。
3. **可复现**：所有随机过程（shot 采样）必须接受 seed，同 seed 同结果。
4. **约定锁死**：小端序（`circuits.py:_as_le` 已实现）、全局相位规范、
   热图基矢排列——用契约测试固定，防止未来改动悄悄破坏。

### 0.3 兼容性承诺

- 现有公开 API（`render_bloch_video` / `render` / `render_frame` /
  `export_html` / `Scene` / `TextOverlay`）签名与行为不变。
- 新能力一律以新模块 + 新可选参数提供，`__init__` 增量导出。
- 版本策略：0.x 系列增量发布；分析 API 稳定后发布 1.0。

---

## Phase 0：准确性地基（约定审计 + 验证套件）

> 定位：专业功能的入场券。没有这一层，后面所有分析数字都不可信。

### P0.1 约定规范文档 `docs/conventions.md`
- 小端序：qubit q ↔ 张量轴 q（qiskit/pyqpanda 均小端），
  `_as_le` 的翻转逻辑写成规范 + 图示。
- 全局相位：态矢量归一化后规定全局相位规范（如首非零振幅取正实数），
  插值与对比前统一规范。
- 热图基矢排列：`state_to_image` 的 reshape 网格中行列与基矢的对应关系。
- Bloch 约定：x/y/z 轴与 Pauli 期望的对应、模长 = 单 qubit 纯度的换算式
  （|r| = √(2·Tr ρ² − 1)）。

### P0.2 交叉验证套件 `tests/cross_validation/`
- 参考实现：`qiskit.quantum_info`（Statevector / DensityMatrix /
  StateFidelity / entropy）。qutip 作为可选第二参考（装了才跑）。
- 对拍范围：Bloch 向量、计算基概率、后续新增的全部熵/互信息/保真度。
- **property-based 测试**（hypothesis）：
  - 酉性：演化每步 ‖ψ‖ = 1；
  - 迹守恒：密度矩阵演化 Tr ρ = 1 且半正定；
  - 熵界：0 ≤ S(ρ_q) ≤ 1，互信息对称、非负；
  - 保真度界：0 ≤ F ≤ 1，F(ψ,ψ) = 1。
- golden 数值测试：固定 seed 的电路 → 固定数值快照（防回归）。

### P0.3 State 统一抽象 `src/gpuqviz/state.py`
- `Statevector` / `DensityMatrix` 两个轻量 dataclass：
  - 统一接口：`.probs()`、`.bloch()`、`.entropy(subsystem)`、`.purity()`；
  - 惰性计算 + cupy 优先（现有 `env.py` 的 xp 分发机制）；
  - 从 qiskit 对象零拷贝/低拷贝构造。
- 现有 `render_bloch_video(states=...)` 等入口继续接受裸 ndarray（兼容），
  内部包成 Statevector。

**验收标准**：`pytest tests/cross_validation` 全绿；
`docs/conventions.md` 覆盖上述四类约定；现有 186 个测试不回归。

---

## Phase 1：专业测量计算模块 `src/gpuqviz/analysis/`（核心）

> 定位：本方案的心脏。所有量返回结构化数据（dataclass），
> 渲染只是数据的视图——这是"表演态矢量"到"呈现量子信息结构"的跨越。

### P1.1 `analysis/measurement.py` — 测量统计
- `sample_counts(state, shots, seed, qubits=None) -> Counts`：
  - 计算基 shot 采样（cupy 批量采样，GPU 优先）；
  - `Counts` dataclass：`counts`（基矢→次数）、`probs`、`shots`、`seed`；
  - 边际分布 `Counts.marginal(qubits)`、条件分布；
  - 可选 qiskit AerSimulator 对拍验证采样分布（卡方检验，容差内一致）。
- 精确模式：`exact_probs(state)` —— 无采样、全精度概率/振幅/相位表。

### P1.2 `analysis/entanglement.py` — 纠缠结构
- `entanglement_entropy(state, qubit) -> float`：
  复用 `evolve.py:117`（bloch_vectors 内部已算 2×2 约化 ρ），
  扩展为任意 qubit 子集的一般约化密度矩阵（n≤子系统用 reshape+partial trace）。
- `mutual_information(state, q_a, q_b) -> float`：两两互信息 I(A:B)。
- `schmidt_coefficients(state, cut) -> ndarray`： Schmidt 分解（关键割）。
- `negativity(state, q_a, q_b) -> float`：部分转置负性（2-qubit 对的
  可分性判据）。
- `entanglement_summary(state) -> EntanglementReport`：一次算全
  （每 qubit 熵 + 全对互信息矩阵 + 最强纠缠对），渲染层直接消费。

### P1.3 `analysis/metrics.py` — 通用度量
- `pauli_expectation(state, pauli_string) -> float`：
  n-qubit Pauli 串期望（"IIXZ" 记法）——专业用户高频操作，
  VQE/哈密顿量项分析的基础。
- `purity(state)` / `linear_entropy(state)`。
- `fidelity(state_a, state_b)`：态保真度；
  `process_fidelity(rho_process, target)`（Phase 3 密度矩阵就绪后）。
- `state_table(state) -> StateTable`：|基矢⟩ / 振幅 / 概率 / 相位的
  精确表格（可导出 CSV/JSON）——专家核对理论的"原始账本"。

### P1.4 输出协议（所有分析量统一）
- 全部返回 frozen dataclass，带 `.to_dict()` / `.to_csv()`；
- docstring 含公式 + 一行自验例子（doctest）；
- 对拍测试与实现同 PR 提交，缺对拍的量不允许合入。

**验收标准**：`analysis` 模块对 qiskit 参考 100% 一致（含 property-based
随机电路 fuzz，≥10⁴ 用例）；每个公开函数有公式 docstring + doctest。

---

## Phase 2：渲染层扩展 — 让分析结果可见

> 定位：分析数据的一等视图。全部走现有 GL/CPU 双后端 + 超采样管线，
> 自动继承文字叠加、H.264/NVENC 编码能力。

### P2.1 `render/histogram.py` — HistogramRenderer
- GL 柱状图：基矢横轴、概率/计数纵轴、可选误差棒（shot 二项统计
  √(p(1−p)/N)）；支持 top-k 截断（大空间只画前 k 项 + "others"）。
- 帧间动画：概率柱插值（线性），配合现有时间轴。

### P2.2 `render/entanglement.py` — EntanglementGraphRenderer
- qubit 为节点（圆环上/网格布局），纠缠为边：
  边宽/透明度 ∝ 互信息或 negativity；节点大小 ∝ 单 qubit 熵。
- 时间轴动画：演化过程中纠缠在 qubit 间"流动"——这是现有 Bloch 视图
  完全缺失的核心叙事。

### P2.3 `render/density.py` — DensityMatrixRenderer
- Hinton 图（方框面积=|ρ_ij|，颜色=符号）/ 实部-虚部分面 / 对角概率条。
- 演化动画：ρ(t) 逐帧绘制（密度矩阵演化就绪后）。

### P2.4 Scene API 扩展
- 新 Track：`HistogramTrack` / `EntanglementTrack` / `DensityMatrixTrack`
  （均接受 states 或 circuit 输入，复用 `scene.regions()` 布局机制）；
- 布局：现有 top/bottom/full 之外增加 `grid` 网格区域（多视图同屏：
  电路 + Bloch + 直方图 + 纠缠图）。

### P2.5 矢量输出（出版级）
- `render_svg(scene, out)`：CPU 软光栅几何层直写 SVG
  （柱图/折线/纠缠图/网格图是矢量图形，天然适合）；
- matplotlib 互操作：`to_matplotlib(fig)` 返回可继续编辑的 fig
  （专家管线里 matplotlib 仍是出版标准）。

### P2.6 LaTeX 标注
- matplotlib mathtext（无 LaTeX 安装依赖）→ 离屏渲染成纹理 →
  SDF 文字管线叠加；用于轴标签、公式注释、图例。

**验收标准**：新渲染器在 GL 与 CPU 后端输出逐像素一致（现有
test_cpu_portability 模式扩展）；SVG 在 Inkscape/浏览器中打开无栅格化痕迹。

---

## Phase 3：高阶电路操作

### P3.1 参数化电路与变分工作流
- `ParameterBinding`：`θ_i` 占位符 + 值绑定；扫描
  `sweep(circuit, bindings, values) -> SweepResult`（每步全部分析量）；
- 可视化：参数-Bloch 轨迹动画、参数-保真度/能量曲线、
  VQE/QAOA 优化轨迹回放（传入历史参数序列即可）。

### P3.2 中途测量与条件门
- 现状：中途 measure/reset 被丢弃并告警（evolve.py:102）；
- 升级：`Gate` 增加 `condition` 字段（经典寄存器条件），
  演化支持分支（确定性经典反馈：测量结果已知时按分支确定性演化）；
- 可视化：分支点标记 + 各分支独立时间轴（teleportation 的经典校正
  也可以画出真实动力学而非门等效）。

### P3.3 噪声与开放系统（专家场景的分水岭）
- `noise.py`：Kraus/Choi 通道库——depolarizing / amplitude_damping /
  phase_damping / dephasing / thermal_relaxation（参数与 Aer/Qiskit 噪声
  模型同名同义）；
- `DensityMatrix` 演化：ρ' = Σ K ρ K†（Kraus 算符张量积展开，cupy 批量）；
- 对拍：与 Aer density_matrix 模式 / qutip mesolve（Markov 近似内）数值一致；
- **渲染天然适配**：去极化/退相干 = Bloch 矢量模长收缩——现有
  "模长 = 纯度"管线（slerp 模长保持修复后）直接正确呈现，零渲染改动；
  配合 DensityMatrixRenderer 可看完整 ρ(t)。
- 理想 vs 含噪对比视图：Scene 双 Track 同屏。

### P3.4 子系统视图（规模桥梁）
- `subsystem(state, qubits)`：任意 qubit 子集的约化态 →
  12+ qubit 电路中选看 2-4 个关键 qubit 的 Bloch/熵/互信息；
- 为 Phase 5 的 MPS 路线预留接口。

**验收标准**：条件门电路与 Aer 条件演化一致；噪声通道与 Aer
density_matrix 对拍一致（≥10⁻⁶ 容差）；teleportation demo 可选
"真实测量分支"模式。

---

## Phase 4：交互层升级

### P4.1 viewer.js 播放器（专业数据面板）
- 状态面板升级：全精度数值（可配置小数位）、hover 单行高亮、
  **导出 CSV/JSON 按钮**（当前帧 state_table）；
- **gate-by-gate 步进**：⏮⏭ 单步按钮（现有 `active_gates` 已把门与
  时间轴对齐，补步进 UI 即可）+ 快捷键；
- 新增 tab：直方图 / 纠缠报告（viewer 内置轻量计算：熵和互信息在
  JS 侧 O(2ⁿ) 可承受，n≤10）；
- payload 扩展：`window.GPUQVIZ_DATA` 增加 `analysis` 段
  （Python 侧预计算的分析量随帧下发）。

### P4.2 playground 升级
- 旋转门（RX/RY/RZ）参数滑杆**实时联动** Bloch 球（拖动即演化）；
- 噪声开关：depolarizing/dephasing 概率滑杆（quantum-sim.js 增加
  简单混合态模拟，n≤4 时 16×16 密度矩阵 JS 可承受）；
- qubit 上限评估（6→8，配合直方图 tab 的 top-k）。

### P4.3 Jupyter 双轨
- `show()` 返回带 `.figure` 的对象：`fig = viz.figure` 给 matplotlib
  管线；`display(viz)` 走现有 HTML 路径；
- ipywidgets：时间轴/视图切换控件（可选依赖）。

### P4.4 `ProVisualizer` 顶层门面（专业轨道 API）
```python
from gpuqviz import ProVisualizer

viz = (ProVisualizer(circuit)
       .with_shots(shots=4096, seed=42)
       .with_noise(depolarizing(0.001))
       .analyze(entanglement=True, pauli=["IZZ", "ZII"])
       .layout("circuit+bloch+histogram"))
viz.export_video("out.mp4")      # 复用现有编码管线
viz.export_svg("fig.svg")
report = viz.report()            # 全部分析量的结构化数据
```

**验收标准**：导出的 HTML 播放器内可复制出与 Python 侧 `report()`
逐位一致的数值；步进模式与 Aer 逐门 Statevector 一致。

---

## Phase 5：规模化

- P5.1 GPU shot 采样：cupy 批量采样已铺基础，补 GPU 直方图归约
  （目标 n≤24、shots≤10⁶ 亚秒级）；
- P5.2 20+ qubit：MPS（矩阵乘积态）近似演化作为可选后端
  （低纠缠电路适用），输出约化分析量喂给现有渲染层；
  接口在 P3.4 subsystem 抽象上扩展；
- P5.3 渲染 LOD：纠缠图/直方图在节点/项数过大时的聚合策略。

---

## 里程碑与工作量

| 里程碑 | 内容 | 规模 | 验收要点 |
|---|---|---|---|
| **M1** | Phase 0 全部 + P1.1/P1.3（measurement + metrics） | L | 对拍套件全绿；直方图 Track 出第一支视频 |
| **M2** | P1.2 纠缠 + P2.2 纠缠图渲染 + P2.1 直方图渲染完善 | M | Bell/GHZ demo 的纠缠流动动画；数值与 qiskit 一致 |
| **M3** | P3.3 噪声 + P0.3/P2.3 密度矩阵链路 | L | 含噪 vs 理想对比视频；与 Aer 对拍 ≤1e-6 |
| **M4** | P3.1/P3.2 参数化+条件门 + P4 全部交互升级 | L | playground 滑杆实时联动；HTML 内导出数值一致 |
| **M5** | P2.5/P2.6 SVG+LaTeX + P5 规模化 | M | 出版级 SVG；24 qubit shot 采样亚秒 |

依赖关系：M1 是一切的地基（无对拍不合入后续分析量）；
M2/M3 可并行；M4 依赖 M1-M2 的数据结构；M5 独立可穿插。

## 测试与质量策略（贯穿全程）

1. **对拍矩阵**：`gpuqviz vs qiskit.quantum_info`（必测），
   `vs qutip`（可选依赖，装了才跑），`vs Aer`（噪声/采样）；
2. **property-based**（hypothesis）：随机酉电路 fuzz，不变量断言
   （酉性/迹/熵界/保真度界）；
3. **golden 数值**：固定 seed 电路 → 固定数值快照；黄金图像测试
   扩展到新渲染器（GL/CPU 双后端逐像素一致）；
4. **回归防线**：现有 186 个测试 + 网站构建（build_site.py）纳入 CI；
5. **文档即验收**：每个分析函数的 docstring 公式 + doctest 是 PR 模板
   检查项。

## 明确不做（本期边界）

- 张量网络完整模拟器（仅 MPS 可选后端）；
- 连续变量/模拟量子系统可视化；
- 云服务/Web 后端——保持纯客户端（静态站点 + 本地 Python）架构。
