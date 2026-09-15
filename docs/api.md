# gpuqviz API 速览

## 顶层函数（gpuqviz 命名空间）

### `render_bloch_video(circuit=None, states=None, steps=120, fps=60.0, out=..., style="dark", codec="h264", quality=0.9, trail=False, seconds=None, backend="auto")`

布洛赫球动画出片。`circuit`（qiskit QuantumCircuit）与 `states`（态矢量序列）二选一。
`backend`: `"auto"`（探测）| `"gl"` | `"cpu"`（numpy 软光栅，limited）。

### `render_heatmap_video(states=None, circuit=None, steps=120, fps=60.0, out=..., basis="probability", colormap="viridis", codec="h264", quality=0.9, seconds=None)`

态热图动画。`basis`: probability | amplitude | phase | real | imag。

### `render(scene, out=..., codec="h264", quality=0.9, prefer_nvenc=False, states_dir=None)`

渲染 `gpuqviz.scene.Scene`。

### `report_env()`

环境能力报告字符串。S8 起新增"实际渲染路径"列，显示 `GPUQVIZ_BACKEND` 环境变量或探测结果。

### `show(circuit=None, states=None, scene=None, steps=60, out=None, as_video=False, height=520, **kwargs)`

Jupyter 交互集成：一行代码在 notebook 中内嵌 3D 播放器。

- `circuit` / `states` / `scene` 三选一（与 `export_html` / `render_bloch_video` 语义一致）
- **notebook 环境**：`IPython.display.HTML` 以 iframe `srcdoc` 内嵌自包含 HTML
  （three.js 内联，断网可用），`height` 控制播放器高度
- **非 notebook**（终端/脚本）：回退为写 HTML 文件并打印路径
- **payload 降级**：态矢量数据超 8MB 时自动剥离状态面板（只保留 Bloch 向量），
  发出 `RuntimeWarning`，文件从 ~8.7MB 降至 ~0.7MB
- `as_video=True`：先渲染 MP4 再用 `IPython.display.Video` 内嵌
- `kwargs` 透传 `fps` / `duration` / `title` / `colormap` / `style` / `codec` /
  `quality` / `backend` 等参数

惰性 import IPython：未装 IPython 时 `import gpuqviz` 不受影响，调用 `show()` 才报错。

### `export_html(circuit=None, states=None, steps=120, fps=60.0, duration=None, title="量子态演化", out="out/viewer.html", colormap="viridis", embed_three=True, machine=None)`

交互式 3D 播放器单文件 HTML 导出。

- `circuit`（qiskit QuantumCircuit / pyqpanda QProg）与 `states`（态矢量序列）二选一
- `embed_three=True`：内联 three.js（断网可用，文件 ~600KB+）；`False` 用 CDN
- **电路图联动**：`circuit` 输入时自动在播放器顶部绘制 SVG 量子电路图
  - payload 新增 `circuit` 字段：`{n_qubits, n_ops, gates[], active_gates[], gate_times[]}`
  - 播放时高亮当前门；点击门跳转到该门时刻，Bloch 球同步更新
  - states-only / 翻译失败时隐藏电路面板（优雅降级）

## payload 字段（`export_html.build_payload`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `meta` | dict | `title` / `fps` / `duration` / `n_qubits` / `n_keys` / `colormap` / `generated_by` |
| `states_re` | list[list[float]] \| null | 关键帧实部（降级时为 null） |
| `states_im` | list[list[float]] \| null | 关键帧虚部（降级时为 null） |
| `bloch` | list[list[list[float]]] | 关键帧 Bloch 向量 `(n_keys, n_qubits, 3)` |
| `circuit` | dict \| 省略 | 电路图元数据（仅 `circuit` 输入时存在） |

`circuit` 子字段：

| 子字段 | 类型 | 说明 |
|---|---|---|
| `n_qubits` | int | 量子位数 |
| `n_ops` | int | 可见门数（已过滤 BARRIER） |
| `gates` | list[dict] | `{name, targets, controls, params, label, col}` |
| `active_gates` | list[int] | 每个关键帧对应的活跃门索引（长度 = n_keys） |
| `gate_times` | list[float] | 每个门的跳转时间（秒，长度 = n_ops） |

## Scene 模型（gpuqviz.scene）

| 类 | 字段（摘要） |
|---|---|
| `Scene` | width/height/fps/duration/background(`#rrggbb`)/tracks/camera/title |
| `BlochTrack` | states_path(.npz)/start/end/layout(top,bottom,full)/qubit_indices/trail |
| `HeatmapTrack` | states_path/basis/colormap |
| `Camera` | azimuth:(起,止) elevation:(起,止) zoom —— duration 内线性插值 |

方法：`save_json / load_json / validate_states(base_dir) / regions()`。

## 渲染层（gpuqviz.render）

- `GLContext(width, height, fps, device)`：离屏 FBO；`frame_iterator(total, draw_fn)` 逐帧产出 `(t, RGBA ndarray)`。
- `BlochRenderer(gl, style)`：`set_camera(eye, target, up, fov, aspect)` / `draw_static(center, radius)` / `draw_vector(v, center, radius)` / `draw_trail(points, center, radius)`。
- `HeatmapRenderer(gl, colormap)`：`draw(state, rect, basis, colorbar)`。
- `PhaseDiscRenderer(gl)`：`draw(state, rect)`。
- `TextRenderer(gl, font_name)`：`draw(text, position_px, size_px, color)`。
- `compositor.split_view(gl, top_draw, bottom_draw, ratio)`。

## 编码层（gpuqviz.encode）

- `create_encoder(..., prefer_nvenc=False)`：工厂。
- `AvEncoder`：PyAV，h264_nvenc/hevc_nvenc 探测失败回退 libx264/libx265。
- `NvencEncoder`：PyNvVideoCodec 显存直喂（cupy RGBA → NV12 kernel → NVENC → FFmpegMuxer）。
- `nvenc_available()`：真实开一次 64x64 会话的探测。

## 数值层

- `gpuqviz.evolve.sample_circuit(circuit, steps)`：按电路深度均匀采样关键帧态矢量。
- `gpuqviz.evolve.bloch_vectors(states, n_qubits)`：批量 einsum 求 (K, n, 3) Bloch 向量。
- `gpuqviz.interpolate.slerp_keys(keys, out_frames)`：Bloch 向量球面插值（含对跖点处理）。
- `gpuqviz.interpolate.lerp_states(states, out_frames)`：态矢量线性插值 + renormalize。

## 后端（gpuqviz.backends）

`detect_backend()` → `"gl"` | `"cpu"`；尊重 `GPUQVIZ_BACKEND` 环境变量（auto/gl/cpu）。
`resolve_backend(backend)` 把 `"auto"` 解析为实际后端名。
`backends.cpu.SoftRasterContext` / `SoftRasterBloch` / `SoftRasterHeatmap`（S8：numba 加速 + LUT 伪彩）。

## Jupyter 集成（gpuqviz.jupyter）

`show(circuit=None, states=None, scene=None, ...)`：notebook 中一行代码内嵌 3D 播放器，
非 notebook 回退写 HTML 文件。详见顶层函数小节。

## 内置算法库（gpuqviz.algorithms）

12 个经典量子算法电路构建器，每个函数接受 `engine="qiskit"|"pyqpanda"|"numpy"` 参数：

| 函数 | 签名 | 默认 qubit | 说明 |
|---|---|---|---|
| `bell()` | `bell(engine="qiskit")` | 2 | Bell 态 \|Φ+⟩ |
| `ghz()` | `ghz(n=3, engine="qiskit")` | 3 | GHZ 纠缠态 |
| `superposition()` | `superposition(n=3, engine="qiskit")` | 3 | 均匀叠加态 |
| `grover()` | `grover(n=3, marked=0b101, iterations=None, engine="qiskit")` | 3 | Grover 搜索 |
| `qft()` | `qft(n=3, inverse=False, engine="qiskit")` | 3 | 量子傅里叶变换 |
| `phase_estimation()` | `phase_estimation(n_count=3, theta=0.375, engine="qiskit")` | 4 | 量子相位估计 |
| `deutsch_jozsa()` | `deutsch_jozsa(oracle_type="balanced", n=3, engine="qiskit")` | 4 | DJ 算法 |
| `bernstein_vazirani()` | `bernstein_vazirani(secret="101", engine="qiskit")` | 3 | BV 算法 |
| `teleportation()` | `teleportation(engine="qiskit", prepare_state="rx")` | 3 | 量子隐形传态 |
| `superdense()` | `superdense(message="11", engine="qiskit")` | 2 | 超密编码 |
| `simon()` | `simon(s="01", n=None, engine="qiskit")` | 4 | Simon 算法 |
| `quantum_walk()` | `quantum_walk(n=3, steps=3, engine="qiskit")` | 3 | 量子随机游走 |

注册表 API：

- `ALGORITHM_REGISTRY: dict[str, AlgorithmSpec]` — name → spec（含 description/builder/default_n_qubits/category）
- `get_algorithm(name) -> AlgorithmSpec` — 按名获取，不存在抛 KeyError
- `list_algorithms() -> str` — 格式化表格（供 CLI `--list` 输出）

pyqpanda 引擎的算法通过 `<algo>_pyqpanda(qubits, machine, ...)` 函数构建，需传入已分配的 qubit 列表和 QVM 实例。

## CLI demo 命令

```bash
gpuqviz demo --algo <name> [选项]
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--algo` / `-a` | 必填 | 算法名（见 `--list`） |
| `--list` | False | 列出全部算法并退出 |
| `--format` / `-f` | `html` | `html`（交互播放器）/ `mp4`（视频）/ `png`（静态帧） |
| `--engine` / `-e` | `qiskit` | `qiskit` / `pyqpanda` |
| `--n-qubits` / `-n` | 算法默认 | 量子比特数 |
| `--out` / `-o` | `out/<algo>.<ext>` | 输出路径 |
| `--steps` | 120 | 关键帧数 |
| `--fps` | 60.0 | 输出帧率 |
| `--seconds` | steps/30 | 视频时长（秒） |
| `--style` | `dark` | dark/light/bw/poster |
| `--trail` | False | Bloch 球轨迹拖尾 |
| `--time` / `-t` | 0.5 | PNG 归一化时刻 t∈[0,1] |
| `--title` | 算法名 | 标题 |
