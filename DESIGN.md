# gpuqviz — GPU 加速量子态演化可视化视频渲染库 · 方案设计

> 目标：替代 `qiskit-state-evolution-record` 的 CPU 逐帧渲染流水线，把 15s 视频的渲染时间从 30+ 分钟压缩到 **1~3 分钟以内**（数量级提升来自：GPU 光栅化 + GPU 上传/合成 + NVENC 硬编码 + 消除逐帧写盘）。

---

## 1. 现状瓶颈分析

典型 CPU 流水线（qiskit-state-evolution-record 类库）的每帧路径：

```
Statevector (CPU) → matplotlib 画布渲染 (~2-10s/帧) → PNG 编码落盘 → ffmpeg 软编码读盘
```

三大瓶颈：

| 瓶颈 | 原因 | 占比（估计） |
|---|---|---|
| matplotlib 逐帧绘制 | CPU 光栅化、每帧重建 Figure、抗锯齿全在 CPU | ~70% |
| 逐帧写盘 PNG | 磁盘 IO + PNG 压缩编码 | ~15% |
| CPU 软编码 + 读盘 | x264/libx264 编码、再次 IO | ~15% |

## 2. 新流水线设计

```
qiskit Circuit/Statevector
   │  (一次性，CPU 上仿真或 GPU 上演化)
   ▼
关键帧态矢量序列 [ψ(t₀), ψ(t₁), ...]  ── GPU 间插值/演化（CuPy）
   │  常驻显存，不落盘
   ▼
GPU 渲染层（ModernGL / VisPy，离屏 FBO）：
   ├─ 布洛赫球（球体网格 + 相机 + 深度测试）
   ├─ 量子态波函数/概率幅热图（纹理 + 片元着色器）
   ├─ 相位色环、坐标轴、标注（线框 + SDF 文字）
   │  关键帧之间由顶点/片段着色器插值，实现任意帧率平滑动画
   ▼
FBO 像素 → NVJPEG/CUDA 直接取回或零拷贝进编码器
   ▼
NVENC 硬编码（PyNvVideoCodec 或 PyAV + h264_nvenc）→ MP4
```

核心原则：
1. **帧不落盘**：渲染帧直接进编码器（pinned memory / CUDA IPC）。
2. **插值在 GPU**：只算 N 个关键帧态矢量，中间帧由着色器插值生成，CPU 仿真次数与输出帧率解耦。
3. **复用 GL 上下文与网格资源**：初始化一次，逐帧只更新 uniform/纹理。

## 3. 技术栈与引用的开源库

| 层 | 选型 | 理由 |
|---|---|---|
| 量子仿真 | `qiskit` + `qiskit-aer`（可选用 `qiskit-aer-gpu`） | 与现有工作流兼容；Aer-GPU 可让态矢量演化也在 GPU |
| GPU 数组/演化 | `cupy`（CUDA 12.x） | 态矢量操作、插值、概率计算全部在 GPU；API 近似 NumPy |
| GPU 渲染 | `moderngl`（离屏渲染，首选） + `moderngl-window`（可选交互预览） | 轻量、无需窗口即可渲染到 FBO；比 VisPy 更底层可控，比 PyOpenGL 省事 |
| 文字/标注 | `freetype-py` + 自制 SDF 字形图集 | GL 里高质量渲染中文/数学标注 |
| 数学网格 | `trimesh`（生成布洛赫球、坐标轴网格，预处理一次） | 生成球体/圆环 obj，运行时上传 VBO |
| 视频编码 | `pynvvideocodec`（NVIDIA 官方 Python 绑定 NVENC，首选）；回退 `PyAV` + `h264_nvenc`/`hevc_nvenc` | 帧数据可留在 GPU 显存直接喂编码器，实现零落盘 |
| CPU 回退路径 | `numpy` + `pygame/SDL` 软渲染 + PyAV x264 | 无 NVIDIA GPU 时仍可用（比旧库快，因无 matplotlib 开销） |
| 配置/CLI | `pydantic` + `typer` | 场景描述用 pydantic 模型，CLI 用 typer |
| 测试 | `pytest` + `pytest-mock`；渲染用像素哈希对比 | 回归测试 |
| 打包 | `pyproject.toml` (hatchling) + 可选 `nvidia-*` wheel 依赖 | pip 一键装 |

明确**不用**的东西：
- ❌ matplotlib（慢的根源；如需静态图保留一个导出接口即可）
- ❌ 逐帧 PNG 中间文件
- ❌ ffmpeg 子进程管道（PyAV/PYNVVIDEOCODEC 进程内编码，省 IO 与序列化）

## 4. 包结构

```
gpuqviz/
├── pyproject.toml
├── README.md
├── src/gpuqviz/
│   ├── __init__.py            # 导出公开 API
│   ├── scene.py               # Scene / Track / 关键帧模型 (pydantic)
│   ├── evolve.py              # 量子演化：qiskit circuit → 关键帧态矢量（CuPy 数组）
│   ├── render/
│   │   ├── context.py         # 隐藏 GL 上下文/设备管理、离屏 FBO 封装
│   │   ├── bloch.py           # 布洛赫球 renderer（多 qubit 网格布局）
│   │   ├── heatmap.py         # 波函数/概率热图 renderer（GPU 纹理）
│   │   ├── phasesphere.py     # 相位色环/幅值柱 renderer
│   │   ├── text.py            # SDF 文字渲染
│   │   └── compositor.py      # 多 renderer 合成到同一帧（framebuffer 混合）
│   ├── interpolate.py         # GPU 关键帧插值（slerp 四元数 / 态矢量线性插值+归一化）
│   ├── encode.py              # NVENC / PyAV 编码器抽象（帧 GPU 内存直入）
│   ├── pipeline.py            # 顶层编排：evolve → render loop → encode
│   ├── preview.py             # moderngl-window 实时预览（可选）
│   └── backends/
│       ├── cuda.py            # CuPy/CUDA 路径
│       └── cpu.py             # numpy 回退路径
├── examples/
│   ├── bell_state_bloch.py
│   └── ghz_heatmap.py
└── tests/
```

## 5. 公开 API 设计

### 5.1 快捷函数（覆盖 90% 用例）

```python
import gpuqviz

# 一行出片：qiskit 电路 → 布洛赫球动画 MP4
gpuqviz.render_bloch_video(
    circuit=qc,                # qiskit.QuantumCircuit；无参演化（按电路层推进）
    steps=120,                 # 电路均匀采样 120 个关键帧
    fps=60,                    # 输出帧率（关键帧间 GPU 插值到 60fps）
    out="bell.mp4",
    style="dark",              # 预设主题
    codec="h264",              # h264 / hevc，NVENC 硬编码
)

# 从 Statevector 序列出片（用户自己演化也行）
gpuqviz.render_heatmap_video(
    states=[psi0, psi1, ...],          # qiskit.quantum_info.Statevector 序列
    fps=30, out="evol.mp4",
)
```

### 5.2 声明式 Scene API（可组合、可持久化为 JSON）

```python
from gpuqviz import Scene, BlochTrack, HeatmapTrack, Camera, render

scene = Scene(
    width=1920, height=1080, fps=60, duration=8.0,   # 秒
    background="#0b0e14",
)
scene.add(BlochTrack(
    states=statevectors,        # 关键帧（qiskit Statevector 或 cupy 数组）
    qubit_indices=[0, 1],       # 多 qubit 各一个球
    trail=True,                 # 显示轨迹尾巴
))
scene.add(HeatmapTrack(
    states=statevectors,
    basis="probability",        # probability / amplitude / phase / real / imag
))
scene.camera = Camera(orbit=(azimuth=(0, 90), elevation=(20, 35)))  # 相机动画
render(scene, "out.mp4", codec="hevc", quality=0.9)
```

### 5.3 底层 API（逐帧回调，完全可控）

```python
from gpuqviz import Renderer, Encoder, GLContext

with GLContext(width=1920, height=1080, device=0) as gl:
    enc = Encoder("out.mp4", fps=60, codec="hevc_nvenc")  # 常驻编码器
    for t, frame_gpu in gl.frame_iterator(total_frames=480):
        # frame_gpu: cupy ndarray (H,W,4) uint8，显存中
        ...  # 自定义绘制，或调用内置 renderer
        enc.write(frame_gpu)          # 显存直入 NVENC，零落盘
    enc.close()
```

### 5.4 与 qiskit 的衔接点

- `evolve.py` 内部用 `qiskit.quantum_info.Statevector(circuit)` + 逐步分解（对电路按层截断得到关键帧），或接受 `qiskit-aer` 的 `save_state` 结果。
- 可选：检测到 `qiskit-aer-gpu` 时把 `AerSimulator(method="statevector", device="GPU")` 用于演化本身。
- 输入协议：任何能转 `numpy/cupy 复数向量 (2^n,)` 的对象都可以（qiskit `Statevector`、`DensityMatrix`、自定义数组），对 qiskit 仅软依赖（extras：`pip install gpuqviz[qiskit]`）。

## 6. 关键技术点

1. **态矢量 → 布洛赫球**：对多 qubit 态求 `ρᵢ = Tr(ψψ† ⊗ 其余)` 得每个 qubit 的 Bloch 向量（CuPy einsum，微秒级）；帧间用球面 slerp 插值。
2. **关键帧 → 任意帧率**：只在上游算 N 个关键帧，输出帧率与仿真解耦；60fps 输出 15s = 900 帧，其中大多数是着色器插值帧，成本可忽略。
3. **零落盘编码**：FBO 用 `glReadPixels` 进 pinned buffer → CuPy 视图 → NVENC 输入。`pynvvideocodec` 支持 CUDA device buffer 直喂；无 NVENC 时回退 PyAV `h264_nvenc`（仍走 GPU 编码）或 libx264（最后手段）。
4. **文字**：freetype 生成字形图集纹理一次，运行时片元着色器采样，支持中文标注。
5. **确定性测试**：GL 渲染对同一驱动/硬件是确定的，用帧哈希做回归；跨硬件则比较结构相似度。

## 7. 依赖文件（pyproject.toml）

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gpuqviz"
version = "0.1.0"
description = "GPU-accelerated quantum state evolution visualization and video rendering"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
dependencies = [
    "numpy>=1.24",
    "cupy-cuda12x>=13.0",          # GPU 数组计算
    "moderngl>=5.10",              # 离屏 GL 渲染
    "pynvvideocodec>=12.2.72.0",   # NVENC 硬编码（NVIDIA 官方包）
    "PyAV>=14.0",                  # 回退/封装编码器
    "freetype-py>=2.4",
    "trimesh>=4.0",
    "pydantic>=2.6",
    "typer>=0.12",
]

[project.optional-dependencies]
qiskit = ["qiskit>=1.0", "qiskit-aer>=0.14"]
qiskit-gpu = ["qiskit>=1.0", "qiskit-aer-gpu>=0.14"]
preview = ["moderngl-window>=2.4"]
cpu-fallback = ["numba>=0.59"]
dev = ["pytest>=8.0", "pytest-cov", "ruff", "mypy"]

[project.scripts]
gpuqviz = "gpuqviz.cli:app"

[tool.ruff]
line-length = 100
```

> 安装：`pip install gpuqviz[qiskit]`；硬件要求：NVIDIA GPU（SM ≥ 6.0，驱动 ≥ 550），CUDA 12.x。无 NVIDIA 卡时自动降级 CPU 后端。

## 8. 性能预算（15s @ 60fps = 900 帧，1920×1080，单卡 RTX 3060 级别）

| 阶段 | 预估 |
|---|---|
| 态演化 + Bloch 向量计算（120 关键帧，CuPy） | < 5 s |
| 布洛赫球光栅化 ~0.5 ms/帧 × 900 | < 1 s |
| 热图纹理更新 ~1 ms/帧 | ~1 s |
| FBO 读回 (pinned) ~2 ms/帧 | ~2 s |
| NVENC hevc 编码 1080p60 | 实时（< 8 s） |
| **合计** | **≈ 10–20 s** |

对比现状 30 分钟，预期 **100× 左右**加速。瓶颈会转移到 FBO 读回，后续可用 CUDA-GL interop（`cudaGraphicsGLRegisterImage`，经 cuda-python）进一步消除拷贝。

## 9. 里程碑

1. **M0（1 周）**：骨架 + GLContext/FBO + 纯色帧 → PyAV 编码出片（打通零落盘管线）。
2. **M1（2 周）**：布洛赫球 renderer + 关键帧 slerp + qiskit 输入，`render_bloch_video` 可用。
3. **M2（1 周）**：接入 PyNvVideoCodec 显存直喂编码；热图/相位 renderer。
4. **M3（1 周）**：Scene 声明式 API、SDF 文字、主题、CLI。
5. **M4（持续）**：CPU 回退后端、CUDA-GL interop 优化、文档与示例、发 PyPI。

## 10. 风险与对策

- **PyNvVideoCodec API 变动/仅部分格式**：Encoder 抽象层隔离，PyAV nvenc 作为等价回退。
- **无 NVIDIA GPU 的用户**：CPU 后端（numba 并行软光栅 + libx264）保证可用性，性能仍远好于 matplotlib 方案。
- **多 qubit 热图维度爆炸**（n>10）：限制热图到 ≤ 10 qubit，提示用约化密度矩阵/局域观测渲染。
- **跨平台（Windows/Linux）GL 离屏**：moderngl 用 EGL/headless 在 Linux 无显示环境可用；Windows 默认 WGL，均无需窗口。
