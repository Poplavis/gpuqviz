# Changelog

## 未发布（S7 布局/样式系统 + 出版级静态图）

- 样式系统扩展：`STYLES` 新增 `bw`（论文黑白：纯白背景、黑轴、灰球壳、深灰矢量）
  与 `poster`（高对比度深底 + 暖色矢量）预设；`style` 参数支持 str 预设名或 dict，
  新增 `style_overrides` 参数按 merge 语义覆盖任意键（如 `{"vector_color": (1,0,0)}`）
- 布局系统：`render_bloch_video` 新增 `cols`（一行最多几个球，语义对齐 recorder
  的 `num_cols`）与 `figsize`（英寸元组，dpi=100 换算像素，与默认 1920×1080 互斥）；
  多行布局行距自适应，相机距离取行列包络；CPU 后端同步支持 `cols` 网格
- 出版级静态图：`render_frame(circuit|states|scene, t, out, scale, style, cols, figsize)`
  渲染归一化时刻 t∈[0,1] 的单帧 → PNG，`scale` 超采样（2/4×）抗锯齿，
  内部以 scale×分辨率 GL 渲染后 PIL LANCZOS 缩回目标尺寸；等效 300dpi 输出
  方式：`figsize=(英寸,)` 指定像素数 = 英寸 × 100（再乘 scale 做超采样）
- CLI：`gpuqviz frame --scene xxx.json | --states xxx.npz --time 0.5 --scale 2 -o fig.png`
- 依赖：新增 `Pillow>=10.0`（PNG 编码）
- 测试：`tests/test_layout.py`（8 例：预设完整性、覆盖语义、figsize 换算、
  单行/多行/单列布局包围盒不重叠）+ `tests/test_frame_export.py`（7 例：
  PNG 尺寸正确、非纯色、bw 白底、scale 抗锯齿、t 越界报错、style_overrides）
- 示例：`examples/publication_fig.py`（Bell bw 3200×2000 scale=4、8-qubit GHZ
  cols=2 多行、Bell dark 中间时刻）

## 未发布（S6 高层门/复杂电路兼容性）

- 门级模拟器扩展：通用矩阵作用原语 `_apply_matrix`、多控制门构造 `_controlled`、
  U/U1/U2/U3/P/PHASE、受控参数门 CRX/CRY/CRZ/CH/CU、任意控制位 MCX/MCP/Toffoli、
  ISWAP、SDG/TDG
- `adapters.qiskit_to_gates`：qiskit QuantumCircuit → 框架无关 Gate 列表的翻译器，
  优先命名直译 → `to_matrix`/`Operator` 矩阵路径 → `definition` 递归展开 →
  `transpile` 到基础门集兜底（未知指令也能出片）
- 态矢量路径对中途 `measure`/`reset`/`delay` 的处理：自动跳过并提示，
  `measure_all` 自动剔除（不再触发 `Cannot apply instruction with classical bits`）
- ORIGINIR 解析扩展：TOFFOLI/CCX、ISWAP、U1/U2/U3/P、CRZ/CH/CU3
- `encode.nvenc_available` 探测改为隔离子进程执行：Pascal EOL 驱动上
  `nvEncOpenEncodeSessionEx` 触发的原生 access violation 不再杀死宿主进程，
  结果进程内缓存
- 测试：新增 `tests/test_gates_matrix.py`，14 个电路（Bell/GHZ3/QFT/mcx oracle/
  受控参数门全家/StatePreparation/measure/UnitaryGate/global phase/barrier/
  自定义门/嵌套定义/cu+mcp/4-qubit mcx 链）对 qiskit Statevector 末态保真度
  全部 ≥ 1-1e-9；1:1 翻译电路逐层快照逐帧一致
- 示例：`examples/grover_mcx.py`（3-qubit 含 mcx 的两轮 Grover，目标态 |101⟩
  振幅放大至 0.945，出片 `out/grover.mp4`）

## 0.1.0 (2026-09-13)

首个可用版本。

- 零落盘渲染管线：moderngl 离屏 FBO → 进程内 PyAV 编码（h264_nvenc 探测回退 libx264）
- `render_bloch_video` / `render_heatmap_video` / Scene 声明式 API（JSON 持久化 + CLI）
- 布洛赫球（Phong 球壳/轨迹/多 qubit）、概率热图（viridis/inferno）、相位色盘、SDF 中文字幕
- qiskit 电路采样（按深度分层）+ 批量 einsum Bloch 向量 + slerp 关键帧插值（含对跖点）
- NVENC 显存直喂路径（PyNvVideoCodec + cupy RGBA→NV12 kernel），会话探测失败自动回退
- CPU 软光栅后端（numpy，limited 样式），无 OpenGL 环境自动降级
- 规避的兼容性问题：fbo.read() 全零（改读颜色附件）、色表 REPEAT 边缘混色（clamp）、
  矩阵行/列主序、qiskit 小端序 Bloch 索引、freetype SDF 过渡宽度量纲
