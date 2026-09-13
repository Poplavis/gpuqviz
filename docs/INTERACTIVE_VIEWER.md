# gpuqviz 交互式 3D 查看器（Interactive HTML Export）· 方案设计

> 需求：导出一个**可交互的 3D 文件**——单个文件即可在浏览器打开：手动拖动时间轴、选择播放速度、暂停/播放，并在下方实时显示当前量子状态（态矢量振幅、Bloch 向量、概率分布）。

---

## 1. 形态选择：为什么是"单文件 HTML + WebGL"

三个候选形态的对比：

| 形态 | 交互性 | 可分发性 | 复用现有代码 | 结论 |
|---|---|---|---|---|
| **A. 单文件 HTML（three.js/WebGL + 内嵌数据）** | 真 3D（旋转/缩放/平移）+ 播放控制 | ✅ 一个 .html 双击即开，可发微信/邮件 | 需写一份 JS 渲染器 | **选它** |
| B. 桌面交互窗口（moderngl-window 扩展现有 preview） | 真 3D + 播放控制 | ❌ 需要装 gpuqviz 环境 | ✅ 完全复用 | 作为开发期调试工具保留 |
| C. 带控制条的增强视频（mp4 + JS 控制器） | 只有 2D 视角 | ✅ | 高 | 交互性不满足"3D"需求 |

选 A 的关键理由：查看器的**渲染是确定性的小数据回放**（120 个关键帧插值），不需要 qiskit/cupy/GL——浏览器 WebGL 足够，产物是真正"发给谁都能打开"的文件。

## 2. 产物结构

```
out/showcase_viewer.html      ← 单文件，典型体积 1~3 MB
├── three.min.js（内联，离线可开；~600KB）
├── gpuqviz-viewer.js（播放器逻辑，~15KB，本项目产出）
├── 状态数据（base64 Float32 / Uint8，内嵌 <script type="x-gpuqviz-data">）
└── UI（控制栏 + 状态面板，纯 DOM/CSS）
```

**不依赖网络**：three.js 直接内联进文件（作为 vendor 资产随包发布，`gpuqviz/assets/vendor/three.min.js`）。

## 3. 数据格式（内嵌 playback payload）

导出时由 Python 侧一次性计算好，浏览器只做插值和绘制：

```jsonc
{
  "meta": {
    "title": "量子态演化演示", "fps": 60, "duration": 8.0,
    "n_qubits": 2, "n_keys": 120, "colormap": "viridis",
    "generated_by": "gpuqviz 0.1.0"
  },
  "bloch":  { "shape": [120, 2, 3], "data": "<base64 float32>" },   // 关键帧 Bloch 向量
  "states": { "shape": [120, 4],    "data": "<base64 float32[2]>" },// 关键帧态矢量（re/im 交错）
  "cmap":   { "data": "<base64 uint8 256x3>" }                       // 色表 LUT（热图用）
}
```

要点：
- **帧率与数据解耦**：文件里只有 K 个关键帧；播放器按 UI 速度 × fps 消费时间轴，帧间用 JS 做 slerp（Bloch 向量，实现与 Python 端 `interpolate.slerp_keys` 同算法：acos/asin 球面插值 + 对跖点正交轴旋转）和线性插值 + renormalize（态矢量）。
- 态矢量以 **re/im 交错 float32** 存（避免 JS 处理复数库问题），K×2^n×2×4 字节，n≤10 时最多 80KB/百关键帧，base64 后可接受；n>10 的场景导出时提示改用约化观测量。
- 色表直接内嵌 LUT，热图着色在 2D canvas 用 `ImageData` 逐像素查 LUT（每次全量重绘 2^n 像素，n≤10 时 ≤1024 像素，微秒级）。

## 4. 播放器 UI 设计

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│              [3D 视口：布洛赫球 + 轨迹]                │  ← three.js，鼠标拖拽旋转/滚轮缩放
│                                                      │
├──────────────────────────────────────────────────────┤
│ [⏸] [⏵] 速度 [0.25|0.5|1×|2×|4×]  ────●──────── 03.2s / 08.0s │  ← 控制栏
├──────────────────────────────────────────────────────┤
│  当前量子状态  t = 3.20s (关键帧 #48/120)              │
│  │00⟩ ▓▓▓▓▓▓▓▓░░ 0.62  amp=0.787 ∠+0.00°            │  ← 每个基态一行：概率条 + 振幅 + 相位
│  │01⟩ ▓▓░░░░░░░░ 0.18  amp=0.424 ∠−142.3°            │
│  │10⟩ ▓▓▓░░░░░░░ 0.20  amp=0.447 ∠+11.7°             │
│  │11⟩ ░░░░░░░░░░ 0.00                                │
│  q0 Bloch: ( 0.62, −0.31,  0.24)   q1: (…)           │
└──────────────────────────────────────────────────────┘
```

交互细节：
- **播放/暂停**：按钮 + 空格键；**逐帧步进**：←/→ 键（暂停时）；**速度**：0.25×~4× 下拉，改变的是时间轴推进速率，不是丢弃帧（任意速度下插值都平滑）。
- **时间轴 scrubber**：拖动即时响应（拖动中暂停插值，松手恢复）。
- **3D 视口**：OrbitControls（左键旋转、右键平移、滚轮缩放）；相机初始角度取导出时的 Camera 起始姿态。
- **状态面板**：纯 DOM 更新（每帧改 textContent + div 宽度），概率条用 CSS width 动画；复数显示为 `amp ∠相位`。
- 深色主题与渲染主题一致（#0b0e14 背景）。

## 5. 3D 渲染（three.js 侧）与 Python 端的视觉对齐

| 元素 | 实现方式 |
|---|---|
| 球壳 | `SphereGeometry` + 半透明 MeshPhongMaterial（对齐 dark 主题的 sphere_alpha=0.16） |
| 经纬线/赤道 | `SphereGeometry` wireframe 淡化 + `TorusGeometry` 赤道环 |
| XYZ 轴 | `Line` 三根 + 轴端标注 sprite（|0⟩/|1⟩ 标在 z 轴两端） |
| 态矢量 | `ArrowHelper`（青色，端点小球 `SphereGeometry`） |
| 轨迹 | `Line` + `BufferGeometry`，环形缓冲 64 点，随播放追加（与 GL 端一致） |
| 多 qubit | 每球一个 `Group`，横向排布，相机自动距离复用 Python 端公式 |
| 热图（可选第二视图） | 底部状态面板旁的 2D canvas，或 3D 视口内 split（默认给 2D，DOM 更稳） |

## 6. Python 侧 API

```python
# 高层：从 Scene / 电路直接导出
from gpuqviz import export_html

export_html(circuit=qc, out="viewer.html", fps=60, duration=8.0,
            title="量子态演化演示", include_heatmap=True)
export_html(scene=my_scene, out="viewer.html")       # 复用 Scene 的 states_path

# CLI
gpuqviz export examples/scene.json -o viewer.html
gpuqviz export --circuit-qasm bell.qasm -o viewer.html   # 二期
```

实现落点：
```
src/gpuqviz/export_html.py      # payload 组装（复用 evolve/interpolate 的关键帧计算）
src/gpuqviz/assets/viewer/      # viewer.html 模板 + gpuqviz-viewer.js + vendor/three.min.js
src/gpuqviz/cli.py              # export 子命令
tests/test_export_html.py       # payload 数值对齐 Python 端（slerp 一致性）、HTML 结构、three.js 内联存在
```

**数值一致性测试**：JS 的 slerp 不可能在 pytest 里直接跑；策略是 Python 端测试 payload 正确性（对同一关键帧序列，`bloch_vectors`/`slerp_keys` 的输出即为内嵌数据），JS 插值算法用"黄金用例注释对齐"——在 viewer.js 里给出与 `interpolate.py` 相同的测试向量断言（开发时用 node 快速跑一遍）。

## 7. 边界与限制

- **n_qubit ≤ 10**：状态面板行数 2^n、热图像素数都以 2^n 增长，超限时导出报错并建议用约化密度矩阵/局域观测量（与 DESIGN.md 第 10 节一致）。
- **无 GL 依赖**：导出的 HTML 与本机的 GL/NVENC 状态完全无关，任何现代浏览器（Chrome/Edge/Firefox，需 WebGL）可开；`file://` 协议下因无跨域请求（全部内联）可正常工作。
- **离线 vs 在线**：默认内联 three.js（离线可开、单文件）；`embed_vendor=False` 选项可改为 CDN 引用以减小文件体积（联网环境用）。
- **数据隐私**：态矢量明文内嵌在 HTML 里，敏感数据注意分发范围。

## 8. 实施里程碑

1. **V1（核心，~2 天）**：payload 组装 + viewer 模板（3D 球 + 播放/暂停/速度/scrubber + 状态面板），`export_html(circuit|states)`，CLI 子命令，单测。
2. **V1.5（~1 天）**：轨迹、多 qubit 布局、概率条动画、键盘快捷键、色标条。
3. **V2（可选）**：3D 视口内嵌热图平面、Scene 的 camera 轨道回放、导出 Scene 双 track（bloch+heatmap 同屏）、`--circuit-qasm` 输入。

## 9. 与现有代码的关系

- 关键帧计算**完全复用** `evolve.sample_circuit` / `bloch_vectors`（导出前在 Python 端算好，浏览器零量子计算）。
- 插值算法与 `interpolate.py` 保持逐行同构（JS 移植），保证 HTML 播放画面与 mp4 出片一致。
- `moderngl-window` 的 `preview.py` 保留：开发期桌面端交互调试；HTML 导出是面向分享的最终产物。两者共享 Scene 读取逻辑。
