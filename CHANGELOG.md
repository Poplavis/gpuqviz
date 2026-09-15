# Changelog

## 未发布（交互式电路图 + Bloch 球联动）

- `export_html(circuit=qc)` / `show(qc)` 自动在播放器顶部绘制 **SVG 量子电路图**，
  与 3D Bloch 球双向联动
  - 浏览器端纯 JavaScript 生成 SVG（无外部电路绘制库依赖），支持单量子门方框、
    受控门（控制点 + ⊕ 目标符）、SWAP（× 符号）、ISWAP（跨行方框）、参数门标签
    （`RX(π/2)` 等角度格式化）、UNITARY（U 方框）
  - 播放时自动高亮当前正在执行的门（橙色 `--accent2`）；点击电路图中的门
    跳转到该门对应时刻，Bloch 球与状态面板同步更新
  - 门-关键帧精确对齐：`circuit` 输入时改用 `evolve_gates` + `sample_snapshots`
    路径采样（与 `sample_circuit` 数值一致），每个关键帧天然对应一个门
  - payload 新增 `circuit` 字段：`{n_qubits, n_ops, gates[], active_gates[],
    gate_times[]}`（`export_html._build_circuit_info`）
  - 优雅降级：states-only 输入 / pyqpanda 翻译失败时隐藏电路面板，其余功能不受
    影响；降级模式（>8MB）保留 `circuit` 字段（体积小）
- `viewer_template.html`：新增 `#circuitPanel` DOM 节点 + 电路图 SVG 样式
  （wire / gate-box / control-dot / target-circle / swap-cross / active 高亮）
- `viewer.js`：新增 `buildCircuitDiagram()` / `updateCircuitHighlight()` /
  `seekToGate()`，`Init` 构建电路图，`renderFrame` 每帧更新高亮
- `jupyter.py`：`show(circuit=qc)` 路径同步使用 `evolve_gates` 采样 +
  `circuit_info` 附加 payload
- 测试：`tests/test_circuit_viewer.py`（18 例：门标签格式化、Bell/GHZ 门元数据、
  active_gates/gate_times 映射、HTML 包含电路图元素、降级保留 circuit、
  viewer.js 函数存在、show() 集成）
- 文档：README 交互式播放器段落增述电路图联动，Roadmap 勾选

## 未发布（S9 Jupyter 交互集成）

- `gpuqviz.show(circuit=None, states=None, scene=None, steps=60, out=None,
  as_video=False, height=520, **kwargs)`：一行代码在 Jupyter notebook 中内嵌
  交互式 3D 播放器（`src/gpuqviz/jupyter.py`）
  - 环境检测：`_is_notebook()` 检查 `IPython.get_ipython()` 是否为 ZMQ 内核
    （notebook / JupyterLab / qtconsole），终端回退写 HTML 文件
  - notebook 内：复用 `export_html` payload 构建，生成自包含 HTML（three.js
    内联，断网可用），通过 `IPython.display.HTML` 以 iframe `srcdoc` 内嵌，
    `height` 可调
  - payload 降级：超 8MB 时自动剥离 `states_re`/`states_im`（只保留 Bloch
    向量），文件从 ~8.7MB 降至 ~0.7MB 并发 `RuntimeWarning`
  - `as_video=True`：先渲染 MP4 再用 `IPython.display.Video` 内嵌
- `viewer.js` 降级兼容：`states_re` 为 null 时 `updatePanel` 跳过概率条更新、
  `renderFrame` 跳过态矢量插值，只显示 Bloch 向量
- 包级入口：`__init__.py` 导出 `show`（惰性 import，未装 IPython 不影响
  `import gpuqviz`）
- 测试：`tests/test_jupyter.py`（11 例：非 notebook 写文件、payload 数值正确、
  降级触发/不触发、`_strip_state_panel`、viewer.js null guard、环境检测、
  nbconvert 执行最小 notebook smoke）
- 文档：README 新增"Jupyter 交互集成"小节（show 示例 + 参数说明），
  `docs/api.md` 增补 `show` 签名与后端/ Jupyter 小节

## 未发布（S8 CPU/无 GL 环境可移植性 + NVENC 修复）

- NVENC 修复：`NvencEncoder` 重写适配 PyNvVideoCodec 2.x API（`FFmpegMuxer`
  完整参数构造 + `MuxVideoPacket` + `Finalize` + `SetUniformPtsIncrement`），
  CAIMemoryView 显存直喂失败（error 8）改用 `usecpuinputbuffer=True` + cupy
  GPU NV12 kernel + `asnumpy` 路径；`nvenc_available()` 探测改隔离子进程避免
  原生崩溃杀宿主。性能：cupy GPU NV12（787 fps）+ NVENC 硬件编码（236 fps）
  端到端 141 fps @1080p，比 libx264 快 3.8×（此前 numpy CPU NV12 路径为瓶颈）
- CPU 软光栅重写（`backends/cpu.py`）：包围盒光栅（不再每帧全屏距离场）+
  numba `@njit(parallel=True, cache=True)` 加速圆盘/圆环/椭圆环/线段/热图LUT/
  相位色盘热路径，numba 缺失时静默回退 numpy。CPU bell 3s@30fps 720p 从
  116.3s 降至 ~6s（19× 加速），远超 ≤20s 目标
- CPU 后端补齐：新增 `SoftRasterHeatmap`（LUT 伪彩，与 GL 路径数值一致）、
  `draw_text`（PIL ImageDraw + ImageFont，支持系统字体回退）、`draw_phase_disc`
  （HSV→RGB 相位色盘 CPU 版）；`render_heatmap_video` 支持 `backend="cpu"`，
  `render_frame` CPU 路径支持 scene（bloch+heatmap+title）
- GL 降级链（`render/context.py`）：`create_gl_context` 先试 GL 3.3 再降 3.2，
  每次降级打印决策日志，全部失败抛 `GLUnavailableError`；新增
  `GLUnavailableError` 异常类型
- 环境变量 `GPUQVIZ_BACKEND`（auto/gl/cpu）强制指定后端，优先级高于
  `detect_backend()` 自动探测和调用方 `backend` 参数默认值；`detect_backend`
  和 `resolve_backend` 尊重环境变量
- `report_env()` 新增"实际渲染路径"列（显示 `GPUQVIZ_BACKEND` 或探测结果）；
  NVENC 探测改用隔离子进程 `nvenc_available()`
- CI：新增 `cpu-portability` job（ubuntu-latest，不装 CUDA，`GPUQVIZ_BACKEND=cpu`
  下 `render_bloch_video` + `render_heatmap_video` 各出 2s 小视频并用 PyAV
  校验帧数，作为可移植性回归门禁）；test job 安装 `cpu-fallback` extras
- 测试：`tests/test_cpu_portability.py`（10 例：环境变量后端选择、CPU 软光栅
  基元、热图 LUT 查表、CPU 热图出片、CPU render_frame PNG、CPU GHZ 多球）
- 基准：`benchmarks/suite.py` 重写，新增 CPU 优化前后对照行；
  `docs/benchmarks.md` 更新

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
