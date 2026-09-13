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

环境能力报告字符串。

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

`detect_backend()` → `"gl"` | `"cpu"`；`backends.cpu.SoftRasterContext / SoftRasterBloch`。
