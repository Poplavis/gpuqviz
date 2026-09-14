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
