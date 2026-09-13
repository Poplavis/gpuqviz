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
from .scene import _hex_to_rgba


def _circuit_key_states(circuit, steps: int, machine=None) -> tuple[list[np.ndarray], int]:
    """qiskit / pyqpanda 电路 → (关键帧态矢量列表, n_qubits)。"""
    from .adapters import to_key_states
    from .evolve import _infer_qubits

    key_states = to_key_states(circuit, steps, machine=machine)
    return key_states, _infer_qubits(key_states[0])
from .render.text import TextRenderer
from .scene import Scene as _Scene
from .scene import _hex_to_rgba

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
}

_SPHERE_SPACING = 3.0
_SPHERE_RADIUS = 1.2


def render_bloch_video(circuit=None, states=None, steps: int = 120, fps: float = 60.0,
                       out: str | Path = "out/bloch.mp4", style: str = "dark",
                       codec: str = "h264", quality: float = 0.9,
                       trail: bool = False, seconds: float | None = None,
                       backend: str = "auto", machine=None) -> Path:
    """qiskit 电路或态矢量序列 → 布洛赫球动画 MP4（零中间文件）。

    circuit 与 states 二选一。circuit 按 depth 均匀采样 steps 个关键帧，
    输出帧率 fps 由关键帧间 slerp 插值实现；seconds 指定输出时长（默认
    steps / 30 ≈ 每关键帧 33ms）。backend: "auto"|"gl"|"cpu"。
    """
    if (circuit is None) == (states is None):
        raise ValueError("exactly one of `circuit` or `states` must be provided")

    theme = dict(STYLES[style])

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
    from .backends import detect_backend

    if backend == "auto":
        backend = detect_backend()
    if backend == "cpu":
        from .backends.cpu import render_bloch_video_cpu

        return render_bloch_video_cpu(frames_bloch, fps, out, theme,
                                      1920, 1080, n_qubits, codec=codec,
                                      quality=quality, trail=trail)

    # 4) 相机与球布局：多 qubit 排成一行，相机距离自适应
    spacing, radius = _SPHERE_SPACING, _SPHERE_RADIUS
    centers = [np.array([(i - (n_qubits - 1) / 2) * spacing, 0.0, 0.0]) for i in range(n_qubits)]
    cam_dist = max(4.0, spacing * n_qubits * 1.35)
    eye = np.array([cam_dist * 0.35, -cam_dist, cam_dist * 0.55])

    def draw(gl: GLContext, t: int) -> None:
        gl.ctx.clear(*theme["background"])
        renderer = gl._bloch
        renderer.set_camera(eye)
        vecs = frames_bloch[t]
        for i in range(n_qubits):
            v = vecs[i]
            if trail:
                trails[i].append(np.asarray(v, dtype=np.float64))
                trails[i] = trails[i][-64:]
                renderer.draw_trail(trails[i], centers[i], radius)
            renderer.draw_vector(v, centers[i], radius)
            renderer.draw_static(centers[i], radius)  # 壳最后画，盖住尾迹出球部分

    with GLContext(1920, 1080, fps=fps) as gl:
        gl._bloch = BlochRenderer(gl, theme)
        trails: list[list] = [[] for _ in range(n_qubits)]  # 每个 qubit 独立轨迹
        try:
            with create_encoder(1920, 1080, fps, out, codec=codec, quality=quality) as enc:
                for t, frame in gl.frame_iterator(total_frames, draw):
                    enc.write(frame)
        finally:
            gl._bloch.release()
    return Path(out)


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
    loaded = []  # (track, kind, frames, extra)
    n_qubits = None
    for track, region in scene.regions():
        states = track.load_states(base_dir=base_dir)
        n = int(round(np.log2(states.shape[1])))
        n_qubits = n if n_qubits is None else n_qubits
        out_frames = int(round(scene.duration * scene.fps))
        if track.kind == "bloch":
            frames = slerp_keys(bloch_vectors(list(states), n_qubits=n), out_frames)
            if hasattr(frames, "get"):
                frames = frames.get()
        else:
            frames = lerp_states(states, out_frames)
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
        gl._text.draw(scene.title, (40, 60), 44,
                      (0.92, 0.94, 0.97, 1.0)) if scene.title else None

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
            else:
                img_shape = state_to_image(frames[t], basis=track.basis).shape
                ih, iw = img_shape
                x0, y0, rw, rh = vp
                cell = min((rw - 260) / iw, (rh - 100) / ih)
                rect = (x0 + (rw - 160 - cell * iw) / 2, y0 + (rh - cell * ih) / 2,
                        cell * iw, cell * ih)
                gl._heat.draw(frames[t], rect, basis=track.basis)

    with GLContext(W, H, fps=fps) as gl:
        gl._bloch = BlochRenderer(gl, theme)
        gl._heat = HeatmapRenderer(gl)
        gl._text = TextRenderer(gl)
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
    return Path(out)


def render_heatmap_video(states=None, circuit=None, steps: int = 120, fps: float = 60.0,
                         out: str | Path = "out/heatmap.mp4", basis: str = "probability",
                         colormap: str = "viridis", codec: str = "h264",
                         quality: float = 0.9, seconds: float | None = None,
                         machine=None) -> Path:
    """态矢量序列（或电路）→ 概率/幅值/相位热图动画 MP4。"""
    if states is None and circuit is not None:
        key_states, _ = _circuit_key_states(circuit, steps, machine=machine)
    elif states is not None:
        key_states = [np.asarray(getattr(s, "data", s)).reshape(-1) for s in states]
    else:
        raise ValueError("provide either `states` or `circuit`")

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

    with GLContext(1920, 1080, fps=fps) as gl:
        gl._heat = HeatmapRenderer(gl, colormap=colormap)
        try:
            with create_encoder(1920, 1080, fps, out, codec=codec, quality=quality) as enc:
                for t, frame in gl.frame_iterator(total_frames, draw):
                    enc.write(frame)
        finally:
            gl._heat.release()
    return Path(out)
