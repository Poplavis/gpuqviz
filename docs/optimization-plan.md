# gpuqviz 功能优化方案

## 一、背景

gpuqviz 当前在**文字叠加**和**分辨率控制**两方面存在功能缺口，本方案针对这两个问题提出具体改进设计。

| 问题域 | 现状 | 目标 |
|--------|------|------|
| 文字叠加 | 仅 `Scene.title` 支持单个硬编码标题 | 支持字幕、水印、多文字元素、自定义位置/字号/颜色/时间范围 |
| 分辨率控制 | `render_bloch_video` 靠 `figsize` 间接指定，`render_heatmap_video` 完全硬编码 | 所有视频/图像输出统一支持 `width`/`height` 像素参数 + 预设 |

---

## 二、文字叠加优化

### 2.1 设计目标

- 支持**水印**：固定在画面四角或居中，半透明，全程显示
- 支持**字幕**：按时间范围显示/隐藏，多段文字
- 支持**多文字元素**：同一画面上放置多个文字（标题 + 水印 + 当前时间轴注释等）
- 支持**自定义样式**：位置、字号、颜色、透明度、对齐方式
- **复用现有底层**：`TextRenderer.draw(text, position, size_px, color)` 已就绪，GL/CPU 双路径完备

### 2.2 数据模型

在 `scene.py` 中新增 `TextOverlay` 模型：

```python
from enum import Enum
from pydantic import BaseModel, Field

class TextPosition(str, Enum):
    """预设位置（基于画面比例的九宫格锚点）"""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"   # 字幕默认位置
    BOTTOM_RIGHT = "bottom_right"

class TextOverlay(BaseModel):
    """可叠加到视频帧上的文字元素。"""
    text: str                           # 文字内容（支持中英文，最长 256 字符）
    position: TextPosition | tuple[float, float] = TextPosition.TOP_LEFT
                                        # 预设锚点 或 自定义像素坐标 (x, y)
    font_size: int = 44                 # 字号（像素）
    color: str = "#ebeef7"             # 颜色（hex 或 rgba）
    opacity: float = 1.0               # 透明度 0~1（水印建议 0.3~0.5）
    start: float = 0.0                 # 显示起始时间（秒），默认从头显示
    end: float | None = None           # 显示结束时间（秒），None = 全程显示
    margin: int = 40                   # 预设位置时距画面边缘的像素间距
    shadow: bool = True                # 文字阴影（提升深色背景上的可读性）

    def is_active(self, t: float) -> bool:
        """当前时间 t（秒）是否在显示范围内。"""
        return self.start <= t and (self.end is None or t <= self.end)

    def resolve_pixels(self, width: int, height: int) -> tuple[int, int]:
        """将预设位置解析为像素坐标。"""
        if isinstance(self.position, tuple):
            return int(self.position[0]), int(self.position[1])
        m = self.margin
        # 粗略估算文字宽度（中文≈font_size, 英文≈font_size*0.6）
        approx_w = len(self.text) * self.font_size * 0.6
        mapping = {
            TextPosition.TOP_LEFT:      (m, m),
            TextPosition.TOP_CENTER:    ((width - approx_w) // 2, m),
            TextPosition.TOP_RIGHT:     (width - int(approx_w) - m, m),
            TextPosition.CENTER:        ((width - approx_w) // 2, (height - self.font_size) // 2),
            TextPosition.BOTTOM_LEFT:   (m, height - self.font_size - m),
            TextPosition.BOTTOM_CENTER: ((width - approx_w) // 2, height - self.font_size - m),
            TextPosition.BOTTOM_RIGHT:  (width - int(approx_w) - m, height - self.font_size - m),
        }
        return mapping[self.position]
```

在 `Scene` 模型中增加 `overlays` 字段：

```python
class Scene(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: float = 60.0
    duration: float = Field(gt=0)
    background: str = "#0b0e14"
    tracks: list[Track] = Field(min_length=1)
    camera: Camera | None = None
    title: str = ""                     # 保留兼容（等价于 overlays 中第一个 TOP_LEFT 元素）
    overlays: list[TextOverlay] = []    # 新增：文字叠加列表
```

### 2.3 渲染集成

#### 2.3.1 GL 后端 (`api.py` render / render_frame)

在每帧绘制完所有 track 之后、编码之前，遍历 `overlays`：

```python
# api.py render() 帧循环内，现有 track 绘制之后
for overlay in scene.overlays:
    if not overlay.is_active(current_time):
        continue
    x, y = overlay.resolve_pixels(W, H)
    r, g, b = _hex_to_rgb01(overlay.color)
    alpha = overlay.opacity
    if overlay.shadow:
        # 先画一层偏移黑色阴影
        gl._text.draw(overlay.text, (x + 2, y + 2), overlay.font_size, (0, 0, 0, alpha * 0.5))
    gl._text.draw(overlay.text, (x, y), overlay.font_size, (r, g, b, alpha))
```

#### 2.3.2 CPU 后端 (`backends/cpu.py`)

对应增加 `SoftRasterContext` 路径的 overlay 绘制：

```python
# cpu.py render_bloch_video_cpu / render_heatmap_video_cpu 帧循环内
for overlay in overlays:
    if not overlay.is_active(t):
        continue
    x, y = overlay.resolve_pixels(width, height)
    r, g, b = _hex_to_rgb01(overlay.color)
    soft.draw_text(overlay.text, (x * scale, y * scale),
                   overlay.font_size * scale, (r, g, b, int(overlay.opacity * 255)))
```

#### 2.3.3 `title` 向后兼容

- 如果 `scene.title` 非空且 `overlays` 为空，自动转换为一个 `TextOverlay(text=scene.title, position=TOP_LEFT, font_size=44)`
- 如果 `overlays` 非空，`title` 被忽略（已显式指定了 overlays）
- 现有用户的 `Scene(title="xxx")` 代码无需修改

### 2.4 高层 API 扩展

给 `render_bloch_video` 和 `render_heatmap_video` 增加 `title` 和 `watermark` 便捷参数：

```python
def render_bloch_video(
    circuit=None, states=None,
    ...,
    title: str | None = None,        # 新增：顶部标题文字
    watermark: str | None = None,    # 新增：右下角水印文字
) -> Path:
```

当 `title` 或 `watermark` 非空时，内部构建对应的 `TextOverlay` 列表传入渲染流程，用户无需构造完整 `Scene` 即可加文字。

### 2.5 使用示例

```python
# 1. 简单标题 + 水印（高层 API）
gpuqviz.render_bloch_video(
    circuit=qc, out="out/grover.mp4",
    title="Grover 搜索算法",
    watermark="gpuqviz · Apache-2.0"
)

# 2. 时间轴字幕（Scene API）
from gpuqviz import Scene, TextOverlay, TextPosition

scene = Scene(
    width=1920, height=1080, fps=60, duration=8.0,
    tracks=[BlochTrack(...)],
    overlays=[
        TextOverlay(text="Step 1: 初始化叠加态", position=TextPosition.BOTTOM_CENTER,
                    font_size=36, start=0.0, end=2.0),
        TextOverlay(text="Step 2: Oracle 相位翻转", position=TextPosition.BOTTOM_CENTER,
                    font_size=36, start=2.0, end=4.0),
        TextOverlay(text="Step 3: Diffuser 振幅放大", position=TextPosition.BOTTOM_CENTER,
                    font_size=36, start=4.0, end=6.0),
        TextOverlay(text="gpuqviz", position=TextPosition.BOTTOM_RIGHT,
                    font_size=20, opacity=0.4, start=0.0, end=None),  # 全程水印
    ]
)
gpuqviz.render(scene, out="out/grover_annotated.mp4")

# 3. JSON 场景文件
# scene.json
{
  "width": 1920, "height": 1080, "fps": 60, "duration": 8.0,
  "tracks": [...],
  "overlays": [
    {"text": "Grover 搜索", "position": "top_left", "font_size": 44},
    {"text": "gpuqviz", "position": "bottom_right", "font_size": 20, "opacity": 0.4}
  ]
}
# CLI: gpuqviz render scene.json -o out.mp4
```

### 2.6 CLI 扩展

```bash
# demo 命令加标题和水印
gpuqviz demo --algo grover --format mp4 --title "Grover 搜索" --watermark "gpuqviz"

# render 命令通过 JSON 文件中的 overlays 字段控制（无需新 CLI flag）
```

### 2.7 实现清单

| 文件 | 改动 |
|------|------|
| `src/gpuqviz/scene.py` | 新增 `TextPosition` 枚举 + `TextOverlay` 模型；`Scene` 增加 `overlays` 字段 |
| `src/gpuqviz/api.py` | `render()` 帧循环中绘制 overlays；`render_bloch_video`/`render_heatmap_video` 增加 `title`/`watermark` 参数；新增 `_hex_to_rgb01` 工具函数 |
| `src/gpuqviz/backends/cpu.py` | CPU 路径对应增加 overlay 绘制 |
| `src/gpuqviz/cli.py` | `demo` 命令增加 `--title`/`--watermark` 选项 |
| `src/gpuqviz/preview.py` | 实时预览中也绘制 overlays（便于调试） |
| `tests/test_overlays.py` | 新增测试：模型序列化、时间范围过滤、位置解析、渲染输出含文字 |

### 2.8 限制与边界

- 单次 `draw` 调用最多 256 字符（`TextRenderer` 限制），超长文字自动截断
- 不支持富文本（粗体/斜体/混排字号），每个 `TextOverlay` 是单一字号单一颜色
- 不支持文字旋转
- 不支持图片水印（仅文字），图片水印需另引入纹理采样路径，留作后续扩展
- `export_html` 的交互式 HTML 播放器不在 3D canvas 上叠加文字（保持 DOM UI 的简洁），overlays 仅作用于 MP4/PNG 渲染输出

---

## 三、分辨率控制优化

### 3.1 设计目标

- 所有视频/图像输出函数统一支持 `width`/`height` 像素参数（直观，无需算英寸）
- 提供常用分辨率预设常量
- `render_heatmap_video` 解除 1920×1080 硬编码
- CLI `demo` 命令支持 `--width`/`--height`
- 修复 CLI `frame --scene` 忽略 `--width`/`--height` 的问题
- 保持 `figsize` 向后兼容

### 3.2 分辨率预设

新增 `src/gpuqviz/presets.py`：

```python
"""常用分辨率预设（像素）。"""

# 16:9 标准
RES_480P  = (854, 480)
RES_720P  = (1280, 720)
RES_1080P = (1920, 1080)
RES_1440P = (2560, 1440)
RES_4K    = (3840, 2160)
RES_8K    = (7680, 4320)

# 1:1 正方形（适合社交媒体）
RES_SQUARE_1080 = (1080, 1080)
RES_SQUARE_2160 = (2160, 2160)

# 9:16 竖屏（适合手机短视频）
RES_VERTICAL_720  = (720, 1280)
RES_VERTICAL_1080 = (1080, 1920)

# 便捷映射
PRESETS = {
    "480p":  RES_480P,
    "720p":  RES_720P,
    "1080p": RES_1080P,
    "1440p": RES_1440P,
    "4k":    RES_4K,
    "8k":    RES_8K,
    "square":    RES_SQUARE_1080,
    "square2k":  RES_SQUARE_2160,
    "vertical":    RES_VERTICAL_720,
    "vertical1080": RES_VERTICAL_1080,
}
```

### 3.3 API 统一改造

#### 3.3.1 `render_bloch_video` — 增加 `width`/`height`/`resolution`

```python
def render_bloch_video(
    circuit=None, states=None,
    ...,
    width: int | None = None,          # 新增：输出像素宽度
    height: int | None = None,         # 新增：输出像素高度
    resolution: str | tuple[int, int] | None = None,  # 新增：预设名或 (w,h) 元组
    figsize: tuple[float, float] | None = None,       # 保留兼容
) -> Path:
    # 解析优先级：resolution > width/height > figsize > 默认 1920x1080
    if resolution is not None:
        if isinstance(resolution, str):
            W, H = PRESETS[resolution.lower()]
        else:
            W, H = resolution
    elif width is not None and height is not None:
        W, H = width, height
    elif figsize is not None:
        W, H = _figsize_to_pixels(figsize, 1920, 1080)
    else:
        W, H = 1920, 1080
```

#### 3.3.2 `render_heatmap_video` — 增加 `width`/`height`/`resolution`

```python
def render_heatmap_video(
    states=None, ...,
    width: int | None = None,          # 新增
    height: int | None = None,         # 新增
    resolution: str | tuple[int, int] | None = None,  # 新增
) -> Path:
    # 同上解析逻辑，替换现有硬编码的 1920, 1080
```

#### 3.3.3 `render_frame` — 增加 `width`/`height`/`resolution`

```python
def render_frame(
    circuit=None, states=None, scene=None, t=0.5,
    ...,
    width: int | None = None,          # 新增
    height: int | None = None,         # 新增
    resolution: str | tuple[int, int] | None = None,  # 新增
    scale: int = 2,
    figsize: tuple[float, float] | None = None,       # 保留兼容
) -> Path:
    # Scene 路径：优先用 scene.width/height（保持现有行为）
    # circuit/states 路径：resolution > width/height > figsize > 默认 1920x1080
```

### 3.4 CLI 扩展

#### 3.4.1 `demo` 命令 — 增加 `--width`/`--height`/`--resolution`

```python
@app.command("demo")
def demo(
    ...,
    width: int = typer.Option(None, "--width", help="输出宽度（像素）"),
    height: int = typer.Option(None, "--height", help="输出高度（像素）"),
    resolution: str = typer.Option(None, "--resolution", help="预设分辨率（480p/720p/1080p/4k）"),
):
```

```bash
# 4K Grover 演示
gpuqviz demo --algo grover --format mp4 --resolution 4k

# 自定义尺寸
gpuqviz demo --algo qft --format mp4 --width 2560 --height 1440

# 竖屏短视频
gpuqviz demo --algo bell --format mp4 --resolution vertical1080
```

#### 3.4.2 `frame` 命令 — 修复 `--scene` 路径

当前 `frame --scene` 忽略 `--width`/`--height`，修复为：如果用户显式指定了 `--width`/`--height`，覆盖 Scene JSON 中的尺寸：

```python
# cli.py frame 命令 --scene 分支
if width is not None:
    scene.width = width
if height is not None:
    scene.height = height
path = _render_frame(scene=scene, t=time, out=out, scale=scale, ...)
```

#### 3.4.3 `render` 命令 — 增加 `--width`/`--height`/`--resolution`

```python
@app.command("render")
def render(
    scene_path: str,
    ...,
    width: int = typer.Option(None, "--width"),
    height: int = typer.Option(None, "--height"),
    resolution: str = typer.Option(None, "--resolution"),
):
    scene = Scene.load_json(scene_path)
    if resolution:
        scene.width, scene.height = PRESETS[resolution.lower()]
    elif width and height:
        scene.width, scene.height = width, height
    ...
```

### 3.5 参数解析优先级

所有函数统一遵循以下优先级（高 → 低）：

```
1. resolution（预设名或元组）     ← 最高优先级，明确指定
2. width + height（像素值对）     ← 直观指定
3. figsize（英寸，dpi=100）       ← 向后兼容
4. Scene.width / Scene.height     ← Scene 路径默认
5. 1920 × 1080                    ← 全局默认
```

冲突时高优先级覆盖低优先级，并发出 `UserWarning` 提示用户。

### 3.6 实现清单

| 文件 | 改动 |
|------|------|
| `src/gpuqviz/presets.py` | 新增：分辨率预设常量 + `PRESETS` 字典 |
| `src/gpuqviz/api.py` | `render_bloch_video`/`render_heatmap_video`/`render_frame` 增加 `width`/`height`/`resolution` 参数；新增 `_resolve_resolution()` 统一解析函数；替换 `render_heatmap_video` 中的硬编码 `1920, 1080` |
| `src/gpuqviz/cli.py` | `demo`/`render`/`frame` 命令增加 `--width`/`--height`/`--resolution`；修复 `frame --scene` 路径忽略尺寸的问题 |
| `src/gpuqviz/__init__.py` | 导出 `RES_1080P`/`RES_4K` 等预设常量 |
| `tests/test_resolution.py` | 新增测试：各函数分辨率参数解析、预设映射、优先级冲突、CLI flag 透传 |

### 3.7 向后兼容性

- `figsize` 参数保留，行为不变
- 所有新参数默认值为 `None`（不指定时走原有默认 1920×1080）
- `Scene.width`/`Scene.height` 默认值不变
- CLI 新增的 `--width`/`--height`/`--resolution` 默认值均为 `None`，不指定时行为与当前一致
- 现有代码无需任何修改即可正常工作

---

## 四、使用示例汇总

```python
import gpuqviz
from gpuqviz import RES_4K, RES_VERTICAL_1080

# ── 4K 布洛赫球动画 + 标题 + 水印 ──
gpuqviz.render_bloch_video(
    circuit=qc, out="out/grover_4k.mp4",
    resolution=RES_4K,                        # 或 resolution="4k"
    title="Grover 搜索算法",
    watermark="gpuqviz · Apache-2.0",
)

# ── 竖屏热图动画（适合手机观看）──
gpuqviz.render_heatmap_video(
    states=states, out="out/heatmap_vertical.mp4",
    resolution="vertical1080",
)

# ── 4K 出版级静态帧 ──
gpuqviz.render_frame(
    circuit=qc, t=0.75, out="out/frame_4k.png",
    resolution=RES_4K, scale=4,
)

# ── Scene API：带时间轴字幕的 1080p 视频 ──
from gpuqviz import Scene, TextOverlay, TextPosition

scene = Scene(
    width=1920, height=1080, fps=60, duration=8.0,
    tracks=[BlochTrack(...)],
    overlays=[
        TextOverlay(text="Step 1: 初始化", position=TextPosition.BOTTOM_CENTER,
                    font_size=36, start=0, end=2),
        TextOverlay(text="Step 2: Oracle", position=TextPosition.BOTTOM_CENTER,
                    font_size=36, start=2, end=4),
        TextOverlay(text="gpuqviz", position=TextPosition.BOTTOM_RIGHT,
                    font_size=20, opacity=0.4),
    ],
)
gpuqviz.render(scene, out="out/annotated.mp4")
```

```bash
# CLI: 4K 带标题的 Grover 演示
gpuqviz demo --algo grover --format mp4 --resolution 4k --title "Grover 搜索"

# CLI: 720p 竖屏 QFT 演示
gpuqviz demo --algo qft --format mp4 --resolution vertical

# CLI: 4K Scene 渲染（覆盖 JSON 中的尺寸）
gpuqviz render scene.json -o out.mp4 --resolution 4k
```

---

## 五、实施计划

### 阶段一：分辨率控制（预计 1~2 天）

1. 新建 `src/gpuqviz/presets.py`
2. 改造 `api.py` 三个函数的分辨率参数 + `_resolve_resolution()` 工具函数
3. 改造 `cli.py` 三个子命令的 `--width`/`--height`/`--resolution`
4. 修复 `frame --scene` 的尺寸忽略 bug
5. 编写测试

### 阶段二：文字叠加（预计 2~3 天）

1. 在 `scene.py` 新增 `TextPosition` + `TextOverlay` 模型
2. 在 `api.py` render/render_frame 帧循环中集成 overlay 绘制
3. 在 `backends/cpu.py` 镜像 CPU 路径
4. 给 `render_bloch_video`/`render_heatmap_video` 加 `title`/`watermark` 便捷参数
5. 在 `preview.py` 预览中支持 overlay
6. CLI `demo` 命令加 `--title`/`--watermark`
7. 编写测试

### 阶段三：文档与示例（预计 0.5 天）

1. 更新 README.md 中 API 参数说明
2. 更新 `docs.html` 网站文档
3. 补充示例代码到 `examples/` 目录

---

## 六、测试要点

| 测试项 | 验证内容 |
|--------|----------|
| 分辨率参数解析 | `resolution="4k"` → (3840, 2160)；`width=1280, height=720` → (1280, 720)；`figsize=(12.8, 7.2)` → (1280, 720) |
| 优先级冲突 | `resolution="4k"` + `figsize=(12.8, 7.2)` → 用 4K，发 UserWarning |
| 向后兼容 | 不传任何新参数时，输出与改造前完全一致（1920×1080） |
| 热图分辨率 | `render_heatmap_video(resolution="720p")` 输出 1280×720 |
| CLI 透传 | `gpuqviz demo --algo bell --resolution 4k` 输出 3840×2160 |
| TextOverlay 模型 | 序列化/反序列化 JSON 正确；时间范围 `is_active()` 过滤正确；`resolve_pixels()` 九宫格位置正确 |
| overlay 渲染 | 输出帧中包含指定文字（通过像素采样或 PIL 文字检测验证） |
| title 向后兼容 | `Scene(title="xxx")` 不传 overlays 时，行为与改造前一致 |
| 水印透明度 | `opacity=0.3` 的水印在深色背景上可见但不喧宾夺主 |
| 字幕时间轴 | `start=2, end=4` 的字幕在第 0~2 秒和第 4~8 秒不显示 |
