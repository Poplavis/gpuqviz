# gpuqviz 分步实施计划 & 提示词设计

> 配套设计文档：[DESIGN.md](./DESIGN.md)。本文档把设计拆成 6 个可独立验收的实施步骤（对应里程碑 M0–M4 + 收尾），每步附一段**自包含提示词**，可直接发给 AI 编程助手（Claude/GPT/Cursor 等）或作为开发任务说明使用。
>
> 使用方式：按 S0→S5 顺序，每次投喂一个步骤的提示词；每步完成并验收后再进入下一步。所有提示词都假设工作目录为项目根目录 `E:\Projects\gpuqviz`，且 `DESIGN.md` 在根目录可见（提示词中已要求先阅读它）。

---

## 步骤总览

| 步骤 | 内容 | 验收标准 | 对应里程碑 |
|---|---|---|---|
| S0 | 项目骨架与环境验证 | `pip install -e .[dev]` 成功；GPU 能力自检命令输出环境报告 | M0 前置 |
| S1 | 最小零落盘管线（GLContext + Encoder） | 输出 3 秒纯色渐变 1080p60 MP4，全程无中间文件 | M0 |
| S2 | 布洛赫球渲染器 + qiskit 输入 | `render_bloch_video(qc)` 对 Bell 态出片，球面矢量随电路推进运动 | M1 |
| S3 | NVENC 显存直喂 + 热图/相位渲染器 | 编码路径无 CPU 像素拷贝；热图示例出片 | M2 |
| S4 | Scene 声明式 API + SDF 文字 + CLI | JSON 场景文件 → CLI 渲染出片；含中文标注 | M3 |
| S5 | CPU 回退后端 + interop 优化 + 发布 | 无 N 卡机器可出片；性能基准报告；发 PyPI | M4 |

---

## S0 · 项目骨架与环境验证

### 提示词 S0

```text
请先完整阅读当前目录下的 DESIGN.md（特别是第 3 节技术栈、第 4 节包结构、第 7 节依赖文件），然后搭建 gpuqviz 项目骨架。要求：

1. 创建 DESIGN.md 第 4 节定义的完整目录结构，所有 Python 包目录含空 __init__.py，
   src/gpuqviz/__init__.py 中定义 __version__ = "0.1.0" 并预留公开 API 占位注释。
2. 编写 pyproject.toml，依赖内容照抄 DESIGN.md 第 7 节；使用 hatchling 构建，
   src-layout（[tool.hatch.build.targets.wheel] packages = ["src/gpuqviz"]）。
3. 实现 src/gpuqviz/env.py，提供函数 `gpuqviz.report_env()`：
   - 检测 Python 版本、CUDA 是否可用（尝试 import cupy 并打印版本与 GPU 名称/显存）、
     GL 上下文是否可创建（尝试 moderngl.create_standalone_context()，捕获异常给出原因）、
     NVENC 是否可用（尝试 import pynvvideocodec）、qiskit 是否安装。
   - 输出为对齐的表格文本，每项标注 OK / MISSING / FAILED(+原因)，最后给出
     "推荐后端：cuda" 或 "推荐后端：cpu" 的结论。
4. 在 src/gpuqviz/cli.py 用 typer 建 CLI 入口：`gpuqviz env` 子命令调用 report_env()。
5. 编写最小 pytest 冒烟测试 tests/test_smoke.py：只断言包可导入、__version__ 存在、
   report_env() 能被调用且返回字符串（不因缺 GPU 失败，环境缺失只能降级不能抛异常）。
6. 编写 .gitignore（Python 标准 + 排除 *.mp4、out/）和 README.md 骨架
   （一句话简介 + 安装命令 + env 命令用法）。

验收：在项目根目录执行 `pip install -e .[dev]` 与 `gpuqviz env` 均成功，
env 命令输出环境报告且不抛未捕获异常。所有代码风格符合 ruff 默认规则。
```

---

## S1 · 最小零落盘管线（M0 核心）

### 提示词 S1

```text
请先阅读 DESIGN.md 第 2 节流水线图和第 6 节关键技术点 3。本项目骨架已就绪
（src/gpuqviz 下有 render/、encode.py、pipeline.py 占位）。本步目标：打通"GPU 离屏
渲染帧 → 进程内编码 → MP4，零中间文件"的最小管线，这是整个库的地基。要求：

1. 实现 src/gpuqviz/render/context.py：
   - 类 GLContext(width, height, fps, device=0)：构造时 moderngl.create_standalone_context()，
     创建离屏 FBO（RGBA8）与 depth buffer；实现上下文管理器协议。
   - 方法 `frame_iterator(total_frames)`：生成器，yield (t, readback_frame)，
     t 为帧序号；每帧迭代结束时把 FBO 像素读回为 cupy ndarray（shape=(H,W,4),
     dtype=uint8，BGRA→RGBA 由编码层约定处理，写清注释）。
   - 读回必须使用 PBO/pinned memory 路径（moderngl fbo.read() 到 bytes 后
     cupy.asarray，若 cupy 可用则 cupy.cuda.alloc_pinned_memory 优化）；
     在没有 cupy 的机器上回退 numpy，接口不变。
2. 实现 src/gpuqviz/encode.py：
   - 基类 Encoder(width, height, fps, out_path, codec="h264", quality=0.9)，
     方法 write(frame)，上下文管理器协议负责 close。
   - 首选实现 AvEncoder：用 PyAV 创建输出容器，视频流优先尝试 codec 名
     "h264_nvenc"/"hevc_nvenc"，打开失败则回退 "libx264" 并打印 warning。
     frame 接受 numpy 或 cupy 数组，cupy 时用 cupy.asnumpy() 转回 CPU 再送
     PyAV（本步允许 CPU 拷贝，S3 会消除，留 TODO 注释）。
   - 预留 NvencEncoder 占位类（NotImplementedError），S3 实现。
3. 实现 src/gpuqviz/pipeline.py：
   - 函数 render_frames(draw_fn, total_frames, width, height, fps, out, codec)：
     创建 GLContext 与 Encoder，循环调用 draw_fn(ctx, t) 让调用方画第 t 帧，
     然后读回并 write 给编码器。
   - 提供 demo：render_solid_gradient(out="out/demo.mp4", seconds=3, fps=60)：
     用 GLSL 片元着色器画一个随时间流动的颜色渐变（全 GPU，无 CPU 绘图），
     用于验证管线。
4. examples/minimum_pipeline.py：调用 render_solid_gradient，打印耗时与输出路径。
5. tests/test_pipeline.py（标记为需要 GPU，用 pytest.importorskip("moderngl")）：
   生成 1 秒 320x240 的小视频，断言输出文件存在且用 PyAV 读回的帧数 = fps*1。

验收：运行 python examples/minimum_pipeline.py，3 秒内产出 out/demo.mp4，
ffprobe/PyAV 可读，时长 3s、1080p60，目录中除最终 MP4 外无任何中间文件。
```

---

## S2 · 布洛赫球渲染器 + qiskit 输入（M1）

### 提示词 S2

```text
请先阅读 DESIGN.md 第 2 节、第 5.1/5.4 节 API、第 6 节关键技术点 1/2。零落盘
管线（S1）已完成。本步实现布洛赫球渲染与 qiskit 衔接，让 render_bloch_video
可用。要求：

1. 实现 src/gpuqviz/evolve.py：
   - 函数 `sample_circuit(circuit, steps) -> list[np.ndarray]`：把 qiskit
     QuantumCircuit 按"层"均匀采样 steps 个关键帧，返回每个关键帧的态矢量
     （复数 np.ndarray，长度 2**n）。实现方式：对电路 depth 分段，用
     qiskit.quantum_info.Statevector 逐段演化；对相邻采样点间无门的段直接复用。
   - 函数 `bloch_vectors(states) -> cupy.ndarray (steps, n_qubits, 3)`：对多 qubit
     态求每个 qubit 的 Bloch 向量：ρ_i = Tr_rest(|ψ⟩⟨ψ|)，再取 ⟨X⟩,⟨Y⟩,⟨Z⟩。
     全部用 cupy einsum/tensordot 实现，禁止 Python 层逐帧循环求迹。
   - 所有输入参数遵循鸭子类型：接受 qiskit Statevector 或任何能 .data 出
     复数向量的对象，qiskit 仅在 evolve.py 内 import。
2. 实现预处理器 scripts/gen_bloch_assets.py：用 trimesh 生成球体（64x32 细分）、
   赤道环、坐标轴线段，输出到 src/gpuqviz/assets/bloch.bin（自定义二进制格式，
   首部写顶点/索引数量），提交到仓库，运行时不依赖 trimesh。
3. 实现 src/gpuqviz/render/bloch.py：BlochRenderer 类
   - 初始化时上传网格 VAO、编译 GLSL（顶点/片元着色器写在此文件内，
     含 Phong 光照 + 半透明球壳 + 赤道环 + XYZ 轴）。
   - 方法 draw(bloch_vec: cupy.ndarray shape (3,), position, radius)：
     uniform 更新矢量方向，画箭头从球心指向 Bloch 向量端点（箭头用圆柱+锥
     近似或加粗线段）。
4. 实现 src/gpuqviz/interpolate.py：
   - `slerp_keys(keys: cupy.ndarray, out_frames: int) -> cupy.ndarray`：
     关键帧 Bloch 向量球面插值（角度用 acos(dot)，权重线性），
     态矢量关键帧则线性插值后 renormalize（提供两个函数并写清各自适用场景）。
5. 实现 src/gpuqviz/api.py：
   - render_bloch_video(circuit=None, states=None, steps=120, fps=60, out=...,
     style="dark", codec="h264", trail=False)：circuit 与 states 二选一；
     内部走 evolve → bloch_vectors → slerp → GLContext 循环 → Encoder。
     style 预设（dark/light）先只控制背景色和轴颜色，存为 dict 常量。
   - 多 qubit：把球排成一行，自动相机距离。
6. examples/bell_state_bloch.py：H+CNOT 电路 → out/bell.mp4。
7. tests/test_evolve.py（不需要 GPU）：验证 sample_circuit 对 Bell 电路首帧为
   |00⟩ 末帧为 Bell 态、bloch_vectors 在 CPU 可用时数值正确（Z 分量 ≈ ±1）。

验收：python examples/bell_state_bloch.py 出片 out/bell.mp4；视频内第一个球
从 |0⟩（北极）平滑旋到叠加态（赤道），第二个 qubit 从北极到南极；
总耗时 < 60s。
```

---

## S3 · NVENC 显存直喂 + 热图/相位渲染器（M2）

### 提示词 S3

```text
请先阅读 DESIGN.md 第 2 节、第 5.2 节、第 6 节关键技术点 3、第 8 节性能预算。
S2 已完成。本步两个目标：① 编码路径消除 CPU 像素拷贝；② 补齐热图与相位渲染器。
要求：

1. 在 src/gpuqviz/encode.py 实现 NvencEncoder（替换 S1 占位）：
   - 用 pynvvideocodec 创建 H264/HEVC NVENC 会话，输入 buffer 直接接受
     cupy device ndarray（NV12/YUV420 转换：在 GPU 上用 cupy 实现
     RGBA→YUV420p 的 kernel，或用 PyNvVideoCodec 自带的颜色转换，禁止回 CPU）。
   - 探测失败（无 NVENC、驱动旧、包缺失）时 AvEncoder 自动兜底，写 warning。
   - GLContext.readback 增加选项 return_device_buffer=True：优先用
     cupy 与 GL 的互操作（若 cuda-gl interop 不可用则 pinned memory 路径，
     接口一致），确保返回 cupy 数组。
2. 写性能对照脚本 benchmarks/encode_path.py：同一 10s 1080p60 渐变动画，
   分别用 AvEncoder(libx264)、AvEncoder(nvenc)、NvencEncoder 跑，输出各自
   总耗时与编码耗时 CSV，作为第 8 节预算的验证。
3. 实现 src/gpuqviz/render/heatmap.py：HeatmapRenderer
   - draw(state: cupy.ndarray (2**n,) complex)：
     a) 在 GPU 上把态矢量算成概率/幅值/相位图（cupy），宽取 2**ceil(n/2)，
        高取其余，不足补零；
     b) 上传为 GL 纹理（PBO 路径），片元着色器做 viridis/inferno 伪彩
        （色表烘焙成 1D 纹理，禁止 CPU 查表）；
     c) 画色标条与 qubit 网格线。
   - basis 参数支持 "probability" | "amplitude" | "phase" | "real" | "imag"。
4. 实现 src/gpuqviz/render/phasesphere.py：相位色环渲染器——单位圆盘上
   极角=幅值、色相=相位（片元着色器直接算 HSV→RGB），叠加态矢量指针。
5. 扩展 src/gpuqviz/api.py：
   - render_heatmap_video(states, basis="probability", ...)；
   - render_bloch_video 的 Scene 组合预留：本步允许同时画 1 个 BlochTrack
     和 1 个 HeatmapTrack 的简单上下分屏（src/gpuqviz/render/compositor.py
     提供 split_view(gl, top_renderer, bottom_renderer) 最简实现）。
6. examples/ghz_heatmap.py：GHZ 态逐层演化 → out/ghz.mp4（上 Bloch 下热图）。

验收：benchmarks/encode_path.py 三条路径全部出片；NvencEncoder 路径
（有 N 卡时）总耗时明显低于 libx264；ghz.mp4 中热图随态演化变化、
伪彩正确（|00⟩+|11⟩/√2 的概率图只有两个亮点）。全程无中间文件。
```

---

## S4 · Scene 声明式 API + SDF 文字 + CLI（M3）

### 提示词 S4

```text
请先阅读 DESIGN.md 第 5.2/5.3 节和第 6 节关键技术点 4。S3 已完成。本步把库
打磨成产品级易用形态。要求：

1. 实现 src/gpuqviz/scene.py（pydantic v2 模型）：
   - Scene(width=1920, height=1080, fps=60, duration, background="#0b0e14",
     tracks: list[Track], camera: Camera|None)
   - Track 基类 + BlochTrack(states, qubit_indices, trail) 与
     HeatmapTrack(states, basis)；Track 带 layout 字段（grid 位置/占比，
     默认自动布局）。
   - Camera(orbit: dict[str, tuple], zoom) 支持 azimuth/elevation 随时间线性
     变化；scene.validate() 检查所有 track 的 states 帧数一致。
   - Scene.model_dump_json / model_validate_json 可持久化（states 存为文件
     引用路径而不是内联数据，写清约定：states_path 指向 .npz）。
2. 实现 src/gpuqviz/render/text.py：SDF 文字渲染
   - scripts/gen_font_atlas.py 用 freetype-py 把字体（默认 NotoSansSC，可通过
     参数指定 ttf）烘焙成 SDF 图集 + 元数据 JSON，输出 assets/font_*.bin；
   - TextRenderer.draw(text, position, size, color)：支持中文；图集未包含的
     字符在构建时警告。
3. 实现 src/gpuqviz/render/compositor.py 完整版：按 Track.layout 把多个
   renderer 的输出混合到主 FBO（先各自渲到子 FBO 再 alpha 混合/贴图矩形）。
4. 实现 src/gpuqviz/api.py 的顶层 render(scene: Scene, out, codec, quality)：
   内部统一编排布局→相机→逐帧→编码；快捷函数 render_bloch_video /
   render_heatmap_video 改为 Scene 的语法糖（行为不变，测试回归）。
5. 实现 src/gpuqviz/cli.py 完整版（typer）：
   - `gpuqviz render scene.json -o out.mp4`（JSON 场景文件驱动）
   - `gpuqviz env`（S0 已有）
   - `gpuqviz preview scene.json`：moderngl-window 实时预览（可选依赖缺失时
     提示 pip install gpuqviz[preview]）。
6. examples/scene_demo.py + examples/scene.json：演示 Bloch+Heatmap+中文标题
   "贝尔态演化" 的完整场景。
7. tests/test_scene.py：pydantic 序列化/反序列化往返一致；非法 layout 报错。
   tests/test_text.py：图集生成后 TextRenderer 能渲染已知字符串（可只测元数据）。

验收：`gpuqviz render examples/scene.json -o out/scene.mp4` 一条命令出片，
含中文标注；render_bloch_video 旧用法回归通过。
```

---

## S5 · CPU 回退后端 + interop 优化 + 发布（M4）

### 提示词 S5

```text
请先阅读 DESIGN.md 第 2/3 节（CPU 回退行）、第 8 节、第 10 节风险。S4 已完成。
本步做兼容性收尾与发布。要求：

1. 实现 src/gpuqviz/backends/cpu.py：
   - 无 NVIDIA GPU（cupy/moderngl 任一不可用）时自动启用：态演化与 Bloch
     计算用 numpy + numba（@njit 并行）实现，与 cuda 后端数值一致（容差 1e-6）；
   - 渲染回退：若 moderngl 可用（Intel 核显/软 GL 也行）继续用 GL，仅数组回退
     numpy；若 GL 完全不可用，用纯 numpy 软光栅只支持 BlochTrack 基础样式并
     在文档标注 "limited"。
   - api.py 所有入口自动探测后端，`backend="cuda"|"cpu"` 参数可强制指定。
2. 优化 src/gpuqviz/render/context.py：尝试 CUDA-GL interop（cuda-python 的
   cudaGraphicsGLRegisterImage）把 FBO 注册为 CUDA 资源，读回零拷贝；不可用
   时静默回退 pinned memory，并在 report_env() 中报告互操作状态。
3. benchmarks/suite.py：输出 S2-S5 全场景（bell/ghz/scene, 15s@60fps 1080p）
   在 cuda 与 cpu 后端的耗时对照表（markdown 表格），写入 docs/benchmarks.md。
4. 文档：README.md 补全（安装、30 秒上手、Scene JSON 示例、后端矩阵表、
   常见问题：驱动版本、WDDM TDR、Linux headless EGL）；docs/api.md 生成
   （可用 pdoc 或手写核心 API）。
5. 发布准备：GitHub Actions CI（.github/workflows/ci.yml：ubuntu + windows
   runner，ruff + mypy + pytest，GPU 用例标记 skip）；pyproject.toml 打上
   classifiers；版本号升至 0.1.0 正式版；写 CHANGELOG.md。
6. tests/test_backends.py：同一 Bell 态关键帧在 cuda/cpu 两后端（可强制 numpy
   模拟 cuda 逻辑）下 bloch_vectors 结果一致。

验收：在一台无 N 卡的机器（或 CI）上 `pip install .[cpu-fallback]` 后
render_bloch_video 仍能出片（允许较慢）；15s@60fps Bell 态在 RTX 3060 级
GPU 上 cuda 后端总耗时 ≤ 30s；`python -m build` 产出可用 wheel。
```

---

## 提示词设计原则说明

以上提示词遵循以下设计约定，方便后续自行扩展新步骤时保持一致：

1. **自包含**：每条提示词开头要求先读 DESIGN.md 相关节，并声明"上一步已完成"，独立会话也能执行。
2. **验收前置**：每步末尾有明确可执行的验收命令与量化指标（耗时、帧数、数值容差），避免"做完但不达标"。
3. **单一交付物**：每步只交付一层能力（管线 → 渲染 → 编码 → API → 发布），出问题能精确定位在哪一步。
4. **回退显式化**：所有 GPU 依赖都要求写明不可用时的降级路径，环境探测只降级不抛异常。
5. **禁止项写进提示词**：如"禁止 CPU 查色表""禁止逐帧 Python 循环求迹""无中间文件"，防止 AI 实现时用慢路径蒙混过关。
