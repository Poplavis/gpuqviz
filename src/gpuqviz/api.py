"""公开 API：快捷出片函数（S2: render_bloch_video；S3/S4 继续扩充）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .encode import create_encoder
from .evolve import bloch_vectors
from .interpolate import lerp_states, slerp_keys
from .render import GLContext
from .render.bloch import BlochRenderer
from .render.heatmap import HeatmapRenderer, state_to_image
from .render.text import TextRenderer
from .scene import Scene as _Scene
from .scene import TextOverlay, TextPosition, _hex_to_rgb01, _hex_to_rgba


def _circuit_key_states(circuit, steps: int, machine=None) -> tuple[list[np.ndarray], int]:
    """qiskit / pyqpanda 电路 → (关键帧态矢量列表, n_qubits)。"""
    from .adapters import to_key_states
    from .evolve import _infer_qubits

    key_states = to_key_states(circuit, steps, machine=machine)
    return key_states, _infer_qubits(key_states[0])

# 主题预设（S2 仅控制背景/轴/环/球/矢量颜色）
STYLES: dict[str, dict] = {
    "dark": {
        "background": (0.043, 0.055, 0.078, 1.0),   # #0b0e14
        "axis_color": (0.45, 0.50, 0.58),
        "ring_color": (0.30, 0.36, 0.46),
        "sphere_color": (0.55, 0.65, 0.85),
        "sphere_alpha": 0.16,
        "vector_color": (0.30, 0.85, 0.95),
        "trail_color": (0.95, 0.65, 0.25),
        "trail_alpha": 0.55,
    },
    "light": {
        "background": (0.96, 0.96, 0.97, 1.0),
        "axis_color": (0.25, 0.27, 0.30),
        "ring_color": (0.55, 0.58, 0.62),
        "sphere_color": (0.45, 0.55, 0.75),
        "sphere_alpha": 0.20,
        "vector_color": (0.85, 0.30, 0.25),
        "trail_color": (0.90, 0.55, 0.10),
        "trail_alpha": 0.55,
    },
    # 论文黑白：纯白背景、黑轴、灰球壳、矢量用深灰、轨迹深灰虚线
    "bw": {
        "background": (1.0, 1.0, 1.0, 1.0),
        "axis_color": (0.0, 0.0, 0.0),
        "ring_color": (0.35, 0.35, 0.35),
        "sphere_color": (0.75, 0.75, 0.78),
        "sphere_alpha": 0.12,
        "vector_color": (0.05, 0.05, 0.05),
        "trail_color": (0.40, 0.40, 0.40),
        "trail_alpha": 0.50,
    },
    # 海报/演示：高对比度深底 + 暖色矢量
    "poster": {
        "background": (0.02, 0.02, 0.04, 1.0),
        "axis_color": (0.65, 0.68, 0.75),
        "ring_color": (0.42, 0.46, 0.55),
        "sphere_color": (0.40, 0.55, 0.90),
        "sphere_alpha": 0.18,
        "vector_color": (1.0, 0.78, 0.25),
        "trail_color": (0.95, 0.35, 0.45),
        "trail_alpha": 0.65,
    },
}


def _resolve_style(style, overrides: dict | None = None) -> dict:
    """把 style（str 或 dict）解析为完整 theme dict；overrides 按 merge 语义覆盖。"""
    if isinstance(style, str):
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}; choices: {list(STYLES)}")
        theme = dict(STYLES[style])
    elif isinstance(style, dict):
        # dict 输入：以 dark 为底 merge（保证必填键齐全）
        theme = dict(STYLES["dark"])
        theme.update(style)
    else:
        raise TypeError("style must be a str name or a dict of overrides")
    if overrides:
        theme.update(overrides)
    return theme


def _figsize_to_pixels(figsize: tuple[float, float] | None,
                       width: int, height: int) -> tuple[int, int]:
    """figsize（英寸，dpi=100）→ 像素；与 width/height 互斥。"""
    if figsize is None:
        return width, height
    w_in, h_in = figsize
    return int(round(w_in * 100)), int(round(h_in * 100))


def _resolve_resolution(
    width: int | None,
    height: int | None,
    resolution: str | tuple[int, int] | None,
    figsize: tuple[float, float] | None,
    default: tuple[int, int] = (1920, 1080),
) -> tuple[int, int]:
    """统一解析输出分辨率，优先级：resolution > width/height > figsize > default。

    冲突时高优先级覆盖低优先级，并发出 UserWarning 提示。
    """
    import warnings

    W: int | None = None
    H: int | None = None
    source = ""

    if resolution is not None:
        if isinstance(resolution, str):
            from .presets import resolve_preset
            W, H = resolve_preset(resolution)
        else:
            W, H = int(resolution[0]), int(resolution[1])
        source = "resolution"
    elif width is not None and height is not None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width/height must be positive, got {width}x{height}")
        W, H = int(width), int(height)
        source = "width/height"
    elif figsize is not None:
        if figsize[0] <= 0 or figsize[1] <= 0:
            raise ValueError(f"figsize must be positive inches, got {figsize}")
        W, H = _figsize_to_pixels(figsize, *default)
        source = "figsize"
    else:
        return default

    # 冲突检测：高优先级参数与低优先级参数同时给出时提示
    if source == "resolution":
        conflicts = []
        if width is not None and height is not None:
            conflicts.append(f"width/height={width}x{height}")
        if figsize is not None:
            conflicts.append(f"figsize={figsize}")
        if conflicts:
            warnings.warn(
                f"resolution={resolution!r} overrides {', '.join(conflicts)}",
                UserWarning,
                stacklevel=3,
            )
    elif source == "width/height" and figsize is not None:
        warnings.warn(
            f"width/height={width}x{height} overrides figsize={figsize}",
            UserWarning,
            stacklevel=3,
        )

    return W, H

_SPHERE_SPACING = 3.0
_SPHERE_RADIUS = 1.2


def _grid_layout(n_qubits: int, cols: int | None,
                 width: int, height: int) -> tuple[list[tuple[float, float, float]], float, float, float]:
    """多球布局：返回 (centers_world, spacing, radius, cam_dist)。

    centers 为世界坐标 (x, y, z)；cols 限制一行最多几个球，超出换行。
    """
    if cols is None:
        cols = n_qubits
    cols = max(1, min(cols, n_qubits))
    rows = int(np.ceil(n_qubits / cols))
    spacing = _SPHERE_SPACING
    radius = _SPHERE_RADIUS
    # 多行时行距加大（z 方向）
    row_gap = spacing * 1.2
    centers = []
    for i in range(n_qubits):
        r = i // cols
        c = i % cols
        x = (c - (min(cols, n_qubits - r * cols) - 1) / 2) * spacing
        z = (r - (rows - 1) / 2) * row_gap
        centers.append((x, 0.0, z))
    # 相机距离取行距与列距的包络
    cam_dist = max(5.0, spacing * cols * 1.35, row_gap * rows * 1.6)
    return centers, spacing, radius, cam_dist


def render_bloch_video(circuit=None, states=None, steps: int = 120, fps: float = 60.0,
                       out: str | Path = "out/bloch.mp4", style="dark",
                       codec: str = "h264", quality: float = 0.9,
                       trail: bool = False, seconds: float | None = None,
                       backend: str = "auto", machine=None,
                       cols: int | None = None,
                       figsize: tuple[float, float] | None = None,
                       width: int | None = None,
                       height: int | None = None,
                       resolution: str | tuple[int, int] | None = None,
                       title: str | None = None,
                       watermark: str | None = None,
                       style_overrides: dict | None = None) -> Path:
    """qiskit 电路或态矢量序列 → 布洛赫球动画 MP4（零中间文件）。

    circuit 与 states 二选一。circuit 按 depth 均匀采样 steps 个关键帧，
    输出帧率 fps 由关键帧间 slerp 插值实现；seconds 指定输出时长（默认
    steps / 30 ≈ 每关键帧 33ms）。backend: "auto"|"gl"|"cpu"。

    分辨率参数（优先级 resolution > width/height > figsize > 1920×1080）：
    - resolution：预设名（"480p"/"720p"/"1080p"/"4k" 等）或 (w, h) 元组
    - width/height：像素值
    - figsize：(宽英寸, 高英寸)，dpi=100 换算像素；与 width/height 互斥

    布局参数（S7）：
    - cols：一行最多几个球（None=单行；语义对齐 recorder 的 num_cols）
    - style：str 预设名（dark/light/bw/poster）或 dict；style_overrides
      按 merge 语义覆盖任意键（如 ``{"vector_color": (1,0,0)}``）

    文字叠加（便捷参数，无需构造 Scene）：
    - title：顶部左上角标题文字
    - watermark：右下角半透明水印文字
    """
    if (circuit is None) == (states is None):
        raise ValueError("exactly one of `circuit` or `states` must be provided")

    theme = _resolve_style(style, style_overrides)
    W, H = _resolve_resolution(width, height, resolution, figsize)

    # 构建便捷 overlay 列表
    overlays = _build_convenience_overlays(title, watermark)

    # 1) 演化：电路采样或直接接受态矢量序列
    if circuit is not None:
        key_states, n_qubits = _circuit_key_states(circuit, steps, machine=machine)
    else:
        key_states = [
            s.data if hasattr(s, "data") else np.asarray(s) for s in states
        ]
        key_states = [np.asarray(s).reshape(-1) for s in key_states]
        n_qubits = int(round(np.log2(key_states[0].shape[0])))

    # 2) Bloch 关键帧 + GPU/CPU slerp 插值到输出帧率
    key_bloch = bloch_vectors(key_states, n_qubits=n_qubits)
    out_frames = int(round(seconds * fps)) if seconds else steps * 2
    frames_bloch = slerp_keys(key_bloch, out_frames)  # (F, n, 3)
    if hasattr(frames_bloch, "get"):
        frames_bloch = frames_bloch.get()  # cupy → numpy（渲染端只读小数组）
    total_frames = frames_bloch.shape[0]

    # 3) 后端选择：GL 不可用时走 numpy 软光栅（limited 样式）
    from .backends import resolve_backend

    if backend == "auto":
        backend = resolve_backend("auto")
    if backend == "cpu":
        from .backends.cpu import render_bloch_video_cpu

        return render_bloch_video_cpu(frames_bloch, fps, out, theme,
                                      W, H, n_qubits, codec=codec,
                                      quality=quality, trail=trail, cols=cols,
                                      overlays=overlays, fps_val=fps)

    # 4) 相机与球布局：多 qubit 按 cols 排成网格，相机距离自适应
    centers, spacing, radius, cam_dist = _grid_layout(n_qubits, cols, W, H)
    eye = np.array([cam_dist * 0.35, -cam_dist, cam_dist * 0.55])

    def draw(gl: GLContext, t: int) -> None:
        gl.ctx.clear(*theme["background"])
        renderer = gl._bloch
        renderer.set_camera(eye, aspect=W / H)
        vecs = frames_bloch[t]
        for i in range(n_qubits):
            v = vecs[i]
            if trail:
                trails[i].append(np.asarray(v, dtype=np.float64))
                trails[i] = trails[i][-64:]
                renderer.draw_trail(trails[i], centers[i], radius)
            renderer.draw_vector(v, centers[i], radius)
            renderer.draw_static(centers[i], radius)  # 壳最后画，盖住尾迹出球部分
        # 文字叠加
        _draw_overlays_gl(gl, overlays, W, H, t / fps)

    with GLContext(W, H, fps=fps) as gl:
        gl._bloch = BlochRenderer(gl, theme)
        gl._text = TextRenderer(gl) if overlays else None
        trails: list[list] = [[] for _ in range(n_qubits)]  # 每个 qubit 独立轨迹
        try:
            with create_encoder(W, H, fps, out, codec=codec, quality=quality) as enc:
                for t, frame in gl.frame_iterator(total_frames, draw):
                    enc.write(frame)
        finally:
            gl._bloch.release()
            if gl._text:
                gl._text.release()
    return Path(out)


def _draw_overlays_gl(gl, overlays: list[TextOverlay], W: int, H: int,
                      current_time: float, *,
                      design_w: int | None = None,
                      design_h: int | None = None) -> None:
    """在 GL 上下文中绘制所有处于活跃时间范围的文字叠加。

    W,H 为 GL 缓冲区实际尺寸；design_w,design_h 为逻辑设计尺寸。
    超采样时 design < GL，overlay 坐标与字号在 design 空间解析后按比例放大，
    使降采样后的最终画面与 design 分辨率下直接渲染的效果一致。
    """
    if not overlays:
        return
    dw = design_w or W
    dh = design_h or H
    sx = W / dw
    sy = H / dh
    for ov in overlays:
        if not ov.is_active(current_time):
            continue
        x, y = ov.resolve_pixels(dw, dh)
        x, y = int(x * sx), int(y * sy)
        r, g, b = _hex_to_rgb01(ov.color)
        alpha = ov.opacity
        fs = ov.font_size * sx
        if ov.shadow:
            gl._text.draw(ov.text, (int(x + 2 * sx), int(y + 2 * sy)), fs,
                          (0.0, 0.0, 0.0, alpha * 0.5))
        gl._text.draw(ov.text, (x, y), fs, (r, g, b, alpha))


def _build_convenience_overlays(title: str | None,
                                watermark: str | None) -> list[TextOverlay]:
    """从 title/watermark 便捷参数构建 TextOverlay 列表。"""
    overlays: list[TextOverlay] = []
    if title:
        overlays.append(TextOverlay(
            text=title, position=TextPosition.TOP_LEFT,
            font_size=44, color="#ebeef7"))
    if watermark:
        overlays.append(TextOverlay(
            text=watermark, position=TextPosition.BOTTOM_RIGHT,
            font_size=24, color="#8a94a6", opacity=0.5))
    return overlays


def _draw_overlays_cpu(soft, overlays: list[TextOverlay],
                       sw: int, sh: int, scale: int,
                       current_time: float) -> None:
    """在 CPU SoftRasterContext 中绘制文字叠加（坐标已乘 scale）。"""
    if not overlays:
        return
    # soft 内部分辨率为 sw×sh = W*scale × H*scale
    # overlay.resolve_pixels 返回基于 W×H 的坐标，需乘 scale
    W, H = sw // scale, sh // scale
    for ov in overlays:
        if not ov.is_active(current_time):
            continue
        x, y = ov.resolve_pixels(W, H)
        r, g, b = _hex_to_rgb01(ov.color)
        alpha = ov.opacity
        if ov.shadow:
            soft.draw_text_with_shadow(
                ov.text, (x * scale, y * scale),
                ov.font_size * scale, (r, g, b, alpha),
                shadow_offset=(2 * scale, 2 * scale),
                shadow_alpha=alpha * 0.5)
        else:
            soft.draw_text(ov.text, (x * scale, y * scale),
                           ov.font_size * scale, (r, g, b, alpha))


def render(scene: _Scene, out: str | Path = "out/scene.mp4", codec: str = "h264",
           quality: float = 0.9, prefer_nvenc: bool = False,
           states_dir: str | Path | None = None) -> Path:
    """Scene 声明式场景 → MP4。布局→相机→逐帧→编码统一编排。

    states_dir：track 的 states_path 相对目录（默认 output 文件所在目录）。
    """
    theme = dict(STYLES["dark"])
    theme["background"] = _hex_to_rgba(scene.background)
    base_dir = Path(states_dir) if states_dir else Path(out).parent
    scene.validate_states(base_dir=base_dir)

    # 预加载所有 track 数据（关键帧 → 输出帧插值）
    loaded = []  # (track, kind, frames, region)
    n_qubits = None

    def _interp(states, out_frames):
        """态矢量 → lerp_states；密度矩阵关键帧 → lerp_density（迹归一）。"""
        if states.ndim == 3:
            from .noise import lerp_density
            return lerp_density(states, out_frames)
        return lerp_states(states, out_frames)

    for track, region in scene.regions():
        if track.kind == "bloch_vectors":
            # P5.2 桥梁：预计算 Bloch 向量 (K, n, 3) 直接插值
            bloch_pre = track.load_bloch(base_dir=base_dir)
            out_frames = int(round(scene.duration * scene.fps))
            frames = slerp_keys(bloch_pre, out_frames)
            if hasattr(frames, "get"):
                frames = frames.get()
            loaded.append((track, "bloch", frames, region))  # 复用 bloch 绘制
            continue
        states = track.load_states(base_dir=base_dir)
        n = int(round(np.log2(states.shape[1])))
        n_qubits = n if n_qubits is None else n_qubits
        out_frames = int(round(scene.duration * scene.fps))
        if track.kind == "bloch":
            # 纯态与密度矩阵关键帧统一走 bloch_vectors（后者批量 partial trace）
            frames = slerp_keys(bloch_vectors(list(states), n_qubits=n), out_frames)
            if hasattr(frames, "get"):
                frames = frames.get()
        else:
            frames = _interp(states, out_frames)
        loaded.append((track, track.kind, frames, region))

    W, H, fps = scene.width, scene.height, scene.fps
    total_frames = int(round(scene.duration * fps))

    def region_rect(region: str):
        """返回 (viewport, 满bleed说明)。bloch 用 viewport；heatmap 用像素矩形。"""
        if region == "top":
            return (0, H // 2, W, H - H // 2)
        if region == "bottom":
            return (0, 0, W, H // 2)
        return (0, 0, W, H)

    def draw(gl: GLContext, t: int) -> None:
        gl.ctx.clear(*theme["background"])

        for track, kind, frames, region in loaded:
            vp = region_rect(region)
            t01 = min(max(t / max(total_frames - 1, 1), 0.0), 1.0)

            if kind == "bloch":
                gl.ctx.viewport = vp
                states_np = frames[t]
                idx = track.qubit_indices or list(range(states_np.shape[0]))
                spacing, radius = 3.0, 1.2
                centers = [np.array([(j - (len(idx) - 1) / 2) * spacing, 0.0, 0.0])
                           for j in range(len(idx))]
                cam_dist = max(5.0, spacing * len(idx) * 1.35)
                if scene.camera is not None:
                    eye = scene.camera.eye_at(t01, cam_dist)
                else:
                    eye = np.array([cam_dist * 0.35, -cam_dist, cam_dist * 0.55])
                gl._bloch.set_camera(eye, aspect=vp[2] / vp[3])
                for j, qi in enumerate(idx):
                    if track.trail:
                        gl._trails.setdefault(id(track), []).append(
                            np.asarray(states_np[qi], dtype=np.float64))
                        tr = gl._trails[id(track)][-64:]
                        gl._bloch.draw_trail(tr, centers[j], radius)
                    gl._bloch.draw_vector(states_np[qi], centers[j], radius)
                    gl._bloch.draw_static(centers[j], radius)
                gl.ctx.viewport = (0, 0, W, H)
            elif kind == "histogram":
                from .analysis.measurement import exact_probs
                probs = exact_probs(frames[t])
                x0, y0, rw, rh = vp
                # vp 为 GL 底部原点 → 直方图按图像坐标（顶部原点）绘制
                rect = (x0 + 60, (H - (y0 + rh)) + 30, rw - 120, rh - 50)
                gl._hist.draw(probs, rect, top_k=track.top_k,
                              others=track.show_others)
            elif kind == "entanglement":
                from .analysis.entanglement import entanglement_summary
                report = entanglement_summary(frames[t])
                x0, y0, rw, rh = vp
                rect = (x0 + 40, (H - (y0 + rh)) + 20, rw - 80, rh - 40)
                gl._ent.draw(report, rect,
                             max_edges=track.max_edges or None)
            elif kind == "density":
                x0, y0, rw, rh = vp
                rect = (x0 + 90, (H - (y0 + rh)) + 50, rw - 130, rh - 100)
                gl._dens.draw(frames[t], rect, min_frac=track.min_frac)
            else:
                img_shape = state_to_image(frames[t], basis=track.basis).shape
                ih, iw = img_shape
                x0, y0, rw, rh = vp
                cell = min((rw - 260) / iw, (rh - 100) / ih)
                rect = (x0 + (rw - 160 - cell * iw) / 2, y0 + (rh - cell * ih) / 2,
                        cell * iw, cell * ih)
                gl._heat.draw(frames[t], rect, basis=track.basis)

        # 文字叠加（所有 track 绘制完之后，确保在最上层）
        _draw_overlays_gl(gl, scene.effective_overlays(), W, H, t / fps)

    with GLContext(W, H, fps=fps) as gl:
        gl._bloch = BlochRenderer(gl, theme)
        gl._heat = HeatmapRenderer(gl)
        gl._text = TextRenderer(gl)
        from .render.histogram import HistogramRenderer
        from .render.entanglement import EntanglementGraphRenderer
        from .render.density import DensityMatrixRenderer
        gl._hist = HistogramRenderer(gl)
        gl._ent = EntanglementGraphRenderer(gl)
        gl._dens = DensityMatrixRenderer(gl)
        gl._trails = {}
        try:
            with create_encoder(W, H, fps, out, codec=codec, quality=quality,
                                prefer_nvenc=prefer_nvenc) as enc:
                for t, frame in gl.frame_iterator(total_frames, draw):
                    enc.write(frame)
        finally:
            gl._bloch.release()
            gl._heat.release()
            gl._text.release()
            gl._hist.release()
            gl._ent.release()
            gl._dens.release()
    return Path(out)


def render_heatmap_video(states=None, circuit=None, steps: int = 120, fps: float = 60.0,
                         out: str | Path = "out/heatmap.mp4", basis: str = "probability",
                         colormap: str = "viridis", codec: str = "h264",
                         quality: float = 0.9, seconds: float | None = None,
                         machine=None, backend: str = "auto",
                         width: int | None = None,
                         height: int | None = None,
                         resolution: str | tuple[int, int] | None = None,
                         title: str | None = None,
                         watermark: str | None = None) -> Path:
    """态矢量序列（或电路）→ 概率/幅值/相位热图动画 MP4。

    S8：backend="cpu" 时走纯 numpy 软光栅热图（LUT 伪彩），无需 OpenGL。

    分辨率参数（优先级 resolution > width/height > 1920×1080）：
    - resolution：预设名（"480p"/"720p"/"1080p"/"4k" 等）或 (w, h) 元组
    - width/height：像素值

    文字叠加：title（顶部标题）、watermark（右下角水印）
    """
    if states is None and circuit is not None:
        key_states, _ = _circuit_key_states(circuit, steps, machine=machine)
    elif states is not None:
        key_states = [np.asarray(getattr(s, "data", s)).reshape(-1) for s in states]
    else:
        raise ValueError("provide either `states` or `circuit`")

    from .backends import resolve_backend

    W, H = _resolve_resolution(width, height, resolution, None)
    overlays = _build_convenience_overlays(title, watermark)

    actual = resolve_backend(backend)
    if actual == "cpu":
        from .backends.cpu import render_heatmap_video_cpu

        return render_heatmap_video_cpu(
            key_states, fps=fps, out=out, width=W, height=H,
            basis=basis, colormap=colormap, codec=codec, quality=quality,
            seconds=seconds, steps=steps, overlays=overlays, fps_val=fps,
        )

    frames = lerp_states(key_states, int(round((seconds or steps / 30) * fps)))
    if hasattr(frames, "get"):
        pass  # cupy 数组保留在 GPU，热图计算全程不下行
    total_frames = frames.shape[0]

    def draw(gl: GLContext, t: int) -> None:
        gl.ctx.clear(0.043, 0.055, 0.078, 1.0)
        m = int(round(np.log2(frames.shape[1])))
        # 热图保持网格纵横比，居中放置
        cols = 2 ** int(np.ceil(m / 2))
        rows = max(1, frames.shape[1] // cols)
        area_w, area_h = gl.width - 160, gl.height - 120
        cell = min(area_w / cols, area_h / rows)
        w, h = cell * cols, cell * rows
        rect = ((gl.width - 160 - w) / 2, (gl.height - h) / 2 + 30, w, h)
        gl._heat.draw(frames[t], rect, basis=basis)
        _draw_overlays_gl(gl, overlays, W, H, t / fps)

    with GLContext(W, H, fps=fps) as gl:
        gl._heat = HeatmapRenderer(gl, colormap=colormap)
        gl._text = TextRenderer(gl) if overlays else None
        try:
            with create_encoder(W, H, fps, out, codec=codec, quality=quality) as enc:
                for t, frame in gl.frame_iterator(total_frames, draw):
                    enc.write(frame)
        finally:
            gl._heat.release()
            if gl._text:
                gl._text.release()
    return Path(out)


def render_frame(circuit=None, states=None, scene=None, t: float = 0.5,
                 out: str | Path = "out/frame.png", scale: int = 2,
                 style="dark", cols: int | None = None,
                 figsize: tuple[float, float] | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 resolution: str | tuple[int, int] | None = None,
                 style_overrides: dict | None = None,
                 machine=None,
                 states_dir: str | Path | None = None) -> Path:
    """渲染归一化时刻 t∈[0,1] 的单帧 → PNG（出版级静态图，零中间文件）。

    circuit/states/scene 三选一：scene 时按其布局渲染（含热图/标题），
    circuit/states 渲染布洛赫球网格。scale 为超采样倍数（2 或 4）抗锯齿，
    内部以 scale×分辨率渲染后 PIL LANCZOS 缩回目标尺寸。
    等效 300dpi 输出：像素数 = 英寸 × 300（用 figsize 指定英寸即可）。

    分辨率参数（circuit/states 路径，优先级 resolution > width/height > figsize > 1920×1080）：
    - resolution：预设名或 (w, h) 元组
    - width/height：像素值
    - figsize：(宽英寸, 高英寸)，dpi=100

    scene 路径优先使用 scene.width/scene.height；若显式指定 width/height/resolution，
    则覆盖 Scene 中的尺寸。
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"t must be in [0, 1], got {t}")

    from .backends import resolve_backend
    from PIL import Image

    theme = _resolve_style(style, style_overrides)

    # ---- 确定渲染内容与尺寸 ----
    if scene is not None:
        base_dir = Path(states_dir) if states_dir else Path(out).parent
        scene.validate_states(base_dir=base_dir)
        # scene 路径：显式指定的分辨率参数覆盖 Scene 模型中的尺寸
        if resolution is not None or (width is not None and height is not None):
            ow, oh = _resolve_resolution(width, height, resolution, None,
                                         default=(scene.width, scene.height))
            scene = scene.model_copy(update={"width": ow, "height": oh})
        W, H = scene.width, scene.height
        render_draw = _build_scene_draw(scene, theme, base_dir, t)
    else:
        if (circuit is None) == (states is None):
            raise ValueError("provide exactly one of circuit/states/scene")
        if circuit is not None:
            key_states, n_qubits = _circuit_key_states(circuit, steps=120, machine=machine)
        else:
            key_states = [np.asarray(getattr(s, "data", s)).reshape(-1) for s in states]
            n_qubits = int(round(np.log2(key_states[0].shape[0])))
        W, H = _resolve_resolution(width, height, resolution, figsize)
        out_frames = 120
        frames_bloch = slerp_keys(bloch_vectors(key_states, n_qubits=n_qubits), out_frames)
        if hasattr(frames_bloch, "get"):
            frames_bloch = frames_bloch.get()
        idx = int(round(t * (out_frames - 1)))
        centers, spacing, radius, cam_dist = _grid_layout(n_qubits, cols, W, H)
        eye = np.array([cam_dist * 0.35, -cam_dist, cam_dist * 0.55])

        def render_draw(gl, _t, vw=None, vh=None):
            ow, oh = (vw, vh) if vw is not None else (W, H)
            gl.ctx.clear(*theme["background"])
            gl._bloch.set_camera(eye, aspect=W / H)
            vecs = frames_bloch[idx]
            for i in range(n_qubits):
                gl._bloch.draw_vector(vecs[i], centers[i], radius)
                gl._bloch.draw_static(centers[i], radius)

    # ---- 超采样渲染 + PIL 缩回 + PNG 编码 ----
    sw, sh = W * scale, H * scale
    backend = resolve_backend("auto")
    if backend == "cpu":
        # CPU 软光栅：直接以目标尺寸渲染（scale 抗锯齿由 PIL 下采样实现）
        from .backends.cpu import SoftRasterContext, SoftRasterBloch, SoftRasterHeatmap
        with SoftRasterContext(sw, sh) as soft:
            if scene is not None:
                # S8: CPU 路径支持 scene（bloch + heatmap + overlays）
                soft.clear(theme["background"])
                heat_renderer = None
                total_frames_s = int(round(scene.duration * scene.fps))
                idx_s = int(round(t * (total_frames_s - 1)))
                for track, region in scene.regions():
                    s_base = Path(states_dir) if states_dir else Path(out).parent
                    if track.kind == "bloch_vectors":
                        bloch_pre = track.load_bloch(base_dir=s_base)
                        out_f = int(round(scene.duration * scene.fps))
                        fb = slerp_keys(bloch_pre, out_f)
                        if hasattr(fb, "get"):
                            fb = fb.get()
                        if region == "top":
                            ry0, ry1 = sh // 2, sh
                        elif region == "bottom":
                            ry0, ry1 = 0, sh // 2
                        else:
                            ry0, ry1 = 0, sh
                        r_h = ry1 - ry0
                        qi_list = track.qubit_indices or list(range(fb.shape[1]))
                        r_px_s = int(min(sw / len(qi_list), r_h) * 0.78 * 0.35)
                        for j, qi in enumerate(qi_list):
                            cx = (j + 0.5) * sw / len(qi_list)
                            cy = (ry0 + ry1) / 2
                            bloch_r = SoftRasterBloch(soft, theme)
                            bloch_r.draw(fb[idx_s, qi], (cx, cy), r_px_s)
                        continue
                    track_states = track.load_states(base_dir=s_base)
                    n_s = int(round(np.log2(track_states.shape[1])))
                    out_f = int(round(scene.duration * scene.fps))
                    if track.kind == "bloch":
                        fb = slerp_keys(bloch_vectors(list(track_states), n_qubits=n_s), out_f)
                        if hasattr(fb, "get"):
                            fb = fb.get()
                        qi_list = track.qubit_indices or list(range(fb.shape[1]))
                        # 区域映射到 sw×sh
                        if region == "top":
                            ry0, ry1 = sh // 2, sh
                        elif region == "bottom":
                            ry0, ry1 = 0, sh // 2
                        else:
                            ry0, ry1 = 0, sh
                        r_h = ry1 - ry0
                        sp_px = 3.0
                        r_px_s = int(min(sw / len(qi_list), r_h) * 0.78 * 0.35)
                        for j, qi in enumerate(qi_list):
                            cx = (j + 0.5) * sw / len(qi_list)
                            cy = (ry0 + ry1) / 2
                            bloch_r = SoftRasterBloch(soft, theme)
                            bloch_r.draw(fb[idx_s, qi], (cx, cy), r_px_s)
                    elif track.kind == "histogram":
                        from .analysis.measurement import exact_probs
                        from .render.histogram import draw_histogram_cpu
                        if track_states.ndim == 3:
                            from .noise import lerp_density
                            fh = lerp_density(track_states, out_f)
                        else:
                            fh = lerp_states(track_states, out_f)
                        if hasattr(fh, "get"):
                            fh = fh.get()
                        if region == "top":
                            ry0, rh = sh // 2, sh // 2
                        elif region == "bottom":
                            ry0, rh = 0, sh // 2
                        else:
                            ry0, rh = 0, sh
                        rect = (60 * scale, ry0 + 30 * scale,
                                sw - 120 * scale, rh - 50 * scale)
                        draw_histogram_cpu(soft, exact_probs(fh[idx_s]),
                                           rect, top_k=track.top_k,
                                           others=track.show_others)
                    elif track.kind == "entanglement":
                        from .analysis.entanglement import entanglement_summary
                        from .render.entanglement import draw_entanglement_graph_cpu
                        if track_states.ndim == 3:
                            from .noise import lerp_density
                            fh = lerp_density(track_states, out_f)
                        else:
                            fh = lerp_states(track_states, out_f)
                        if hasattr(fh, "get"):
                            fh = fh.get()
                        if region == "top":
                            ry0, rh = sh // 2, sh // 2
                        elif region == "bottom":
                            ry0, rh = 0, sh // 2
                        else:
                            ry0, rh = 0, sh
                        rect = (40 * scale, ry0 + 20 * scale,
                                sw - 80 * scale, rh - 40 * scale)
                        draw_entanglement_graph_cpu(
                            soft, entanglement_summary(fh[idx_s]), rect,
                            max_edges=track.max_edges or None)
                    elif track.kind == "density":
                        from .render.density import draw_density_cpu
                        if track_states.ndim == 3:
                            from .noise import lerp_density
                            fh = lerp_density(track_states, out_f)
                        else:
                            fh = lerp_states(track_states, out_f)
                        if hasattr(fh, "get"):
                            fh = fh.get()
                        if region == "top":
                            ry0, rh = sh // 2, sh // 2
                        elif region == "bottom":
                            ry0, rh = 0, sh // 2
                        else:
                            ry0, rh = 0, sh
                        rect = (90 * scale, ry0 + 50 * scale,
                                sw - 130 * scale, rh - 100 * scale)
                        draw_density_cpu(soft, fh[idx_s], rect,
                                 min_frac=track.min_frac)
                    else:
                        if heat_renderer is None:
                            heat_renderer = SoftRasterHeatmap(soft, colormap="viridis")
                        from .render.heatmap import state_to_image
                        if track_states.ndim == 3:
                            from .noise import lerp_density
                            fh = lerp_density(track_states, out_f)
                        else:
                            fh = lerp_states(track_states, out_f)
                        if hasattr(fh, "get"):
                            fh = fh.get()
                        img = state_to_image(fh[idx_s], basis=track.basis)
                        if hasattr(img, "get"):
                            img = img.get()
                        ih, iw = img.shape
                        if region == "top":
                            ry0 = sh // 2
                        elif region == "bottom":
                            ry0 = 0
                        else:
                            ry0 = 0
                        cell = min((sw - 260 * scale) / iw, (sh // 2 - 100 * scale) / ih)
                        rw, rh = cell * iw, cell * ih
                        rect = ((sw - 160 * scale - rw) / 2, ry0 + (sh // 2 - rh) / 2 + 30 * scale, rw, rh)
                        heat_renderer.draw(fh[idx_s], rect, basis=track.basis)
                # 文字叠加
                _draw_overlays_cpu(soft, scene.effective_overlays(), sw, sh, scale,
                                   t * scene.duration)
                frame = soft.frame
            else:
                renderer = SoftRasterBloch(soft, theme)
                cell_w = sw / cols if cols else sw / n_qubits
                cell_h = sh
                rows = 1
                if cols and cols < n_qubits:
                    rows = int(np.ceil(n_qubits / cols))
                    cell_w = sw / cols
                    cell_h = sh / rows
                r_px = int(min(cell_w, cell_h) * 0.35 * 0.78)
                vecs = frames_bloch[idx]
                soft.clear(theme["background"])
                for i in range(n_qubits):
                    r = i // (cols or n_qubits)
                    c = i % (cols or n_qubits)
                    cx = (c + 0.5) * cell_w
                    cy = (r + 0.5) * cell_h
                    renderer.draw(vecs[i], (cx, cy), r_px)
                frame = soft.frame
        pil = Image.fromarray(frame, mode="RGBA")
    else:
        with GLContext(sw, sh) as gl:
            gl._bloch = BlochRenderer(gl, theme)
            if scene is not None:
                gl._heat = HeatmapRenderer(gl)
                gl._text = TextRenderer(gl)
                from .render.histogram import HistogramRenderer
                from .render.entanglement import EntanglementGraphRenderer
                from .render.density import DensityMatrixRenderer
                gl._hist = HistogramRenderer(gl)
                gl._ent = EntanglementGraphRenderer(gl)
                gl._dens = DensityMatrixRenderer(gl)
            try:
                render_draw(gl, 0, sw, sh)
                frame = gl.read_frame()
            finally:
                gl._bloch.release()
                if scene is not None:
                    gl._heat.release()
                    gl._text.release()
                    gl._hist.release()
                    gl._ent.release()
                    gl._dens.release()
        if hasattr(frame, "get"):
            frame = frame.get()
        pil = Image.fromarray(np.asarray(frame), mode="RGBA")

    if scale > 1:
        pil = pil.resize((W, H), Image.LANCZOS)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # PNG 不支持 RGBA 的某些查看器兼容性：转 RGB（丢弃全不透明 alpha）
    pil.convert("RGB").save(str(out), format="PNG", optimize=True)
    return Path(out)


def _build_scene_draw(scene, theme, base_dir, t01_target):
    """构造 scene 单帧 draw 闭包（render_frame 用，渲染 t01 对应帧）。"""
    loaded = []

    def _interp(states, out_frames):
        if states.ndim == 3:  # 密度矩阵关键帧 → 迹归一插值
            from .noise import lerp_density
            return lerp_density(states, out_frames)
        return lerp_states(states, out_frames)

    for track, region in scene.regions():
        if track.kind == "bloch_vectors":
            bloch_pre = track.load_bloch(base_dir=base_dir)
            out_frames = int(round(scene.duration * scene.fps))
            frames = slerp_keys(bloch_pre, out_frames)
            if hasattr(frames, "get"):
                frames = frames.get()
            loaded.append((track, "bloch", frames, region))  # 复用 bloch 绘制
            continue
        states = track.load_states(base_dir=base_dir)
        n = int(round(np.log2(states.shape[1])))
        out_frames = int(round(scene.duration * scene.fps))
        if track.kind == "bloch":
            # 纯态与密度矩阵关键帧统一走 bloch_vectors
            frames = slerp_keys(bloch_vectors(list(states), n_qubits=n), out_frames)
            if hasattr(frames, "get"):
                frames = frames.get()
        else:
            frames = _interp(states, out_frames)
        loaded.append((track, track.kind, frames, region))
    # 注意：实际 GL 视口尺寸（sw×sh）在闭包构建时未知，
    # 由 render_draw 闭包参数传入，用于 overlay 坐标缩放。
    W, H = scene.width, scene.height
    total_frames = int(round(scene.duration * scene.fps))
    idx = int(round(t01_target * (total_frames - 1)))

    def region_rect(region):
        if region == "top":
            return (0, H // 2, W, H - H // 2)
        if region == "bottom":
            return (0, 0, W, H // 2)
        return (0, 0, W, H)

    def draw(gl, _t, vw=None, vh=None):
        """vw/vh: 实际 GL 视口尺寸（超采样时 ≠ W×H），overlay 需按此缩放。"""
        ow, oh = (vw, vh) if vw is not None else (W, H)
        s = ow / W  # design → GL 像素统一缩放（scale=1 时 s=1）
        gl.ctx.clear(*theme["background"])
        # 区域/矩形类绘制（histogram/heatmap/overlay）需要全画布视口
        gl.ctx.viewport = (0, 0, ow, oh)
        for track, kind, frames, region in loaded:
            vp = region_rect(region)
            if kind == "bloch":
                # viewport 是 GL 底部原点坐标，须乘 s 才与超采样缓冲对齐
                gl.ctx.viewport = tuple(int(round(v * s)) for v in vp)
                states_np = frames[idx]
                qi_list = track.qubit_indices or list(range(states_np.shape[0]))
                spacing, radius = 3.0, 1.2
                centers = [np.array([(j - (len(qi_list) - 1) / 2) * spacing, 0.0, 0.0])
                           for j in range(len(qi_list))]
                cam_dist = max(5.0, spacing * len(qi_list) * 1.35)
                if scene.camera is not None:
                    eye = scene.camera.eye_at(t01_target, cam_dist)
                else:
                    eye = np.array([cam_dist * 0.35, -cam_dist, cam_dist * 0.55])
                gl._bloch.set_camera(eye, aspect=vp[2] / vp[3])
                for j, qi in enumerate(qi_list):
                    gl._bloch.draw_vector(states_np[qi], centers[j], radius)
                    gl._bloch.draw_static(centers[j], radius)
                gl.ctx.viewport = (0, 0, ow, oh)
            elif kind == "histogram":
                from .analysis.measurement import exact_probs
                probs = exact_probs(frames[idx])
                x0, y0, rw, rh = vp
                # vp 为 GL 底部原点 → 直方图按图像坐标（顶部原点），再 × s
                design_rect = (x0 + 60, (H - (y0 + rh)) + 30, rw - 120, rh - 50)
                rect = tuple(c * s for c in design_rect)
                gl._hist.draw(probs, rect, top_k=track.top_k,
                              others=track.show_others)
            elif kind == "entanglement":
                from .analysis.entanglement import entanglement_summary
                report = entanglement_summary(frames[idx])
                x0, y0, rw, rh = vp
                design_rect = (x0 + 40, (H - (y0 + rh)) + 20, rw - 80, rh - 40)
                rect = tuple(c * s for c in design_rect)
                gl._ent.draw(report, rect,
                             max_edges=track.max_edges or None)
            elif kind == "density":
                x0, y0, rw, rh = vp
                design_rect = (x0 + 90, (H - (y0 + rh)) + 50, rw - 130, rh - 100)
                rect = tuple(c * s for c in design_rect)
                gl._dens.draw(frames[idx], rect, min_frac=track.min_frac)
            else:
                img_shape = state_to_image(frames[idx], basis=track.basis).shape
                ih, iw = img_shape
                x0, y0, rw, rh = vp
                cell = min((rw - 260) / iw, (rh - 100) / ih)
                rect = (x0 + (rw - 160 - cell * iw) / 2, y0 + (rh - cell * ih) / 2,
                        cell * iw, cell * ih)
                gl._heat.draw(frames[idx], tuple(c * s for c in rect),
                              basis=track.basis)
        # 恢复全画布视口，确保文字叠加不被 track viewport 裁切
        gl.ctx.viewport = (0, 0, ow, oh)
        # 文字叠加（GL 尺寸 ow×oh，设计尺寸 W×H → 字号/坐标自动缩放）
        _draw_overlays_gl(gl, scene.effective_overlays(), ow, oh,
                          t01_target * scene.duration,
                          design_w=W, design_h=H)
    return draw
