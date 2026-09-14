"""CPU 回退后端：纯 numpy 软光栅（无 OpenGL 时的 limited 兜底）。

只支持 BlochTrack 基础样式（正交投影：x→屏幕右，z→屏幕上，忽略 y 深度）。
热图/相位盘/文字/相机动画等能力在 GL 不可用环境下不可用（limited）。
"""

from __future__ import annotations

import numpy as np


class SoftRasterContext:
    """numpy 帧画布。接口与 GLContext 对齐的子集：clear / frame_iterator。"""

    def __init__(self, width: int, height: int, fps: float = 60.0):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frame = np.zeros((self.height, self.width, 4), np.uint8)
        self.ctx = None  # 与 GLContext 接口兼容的占位

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def clear(self, color=(0, 0, 0, 1)):
        rgb = (np.asarray(color[:3]) * 255).astype(np.uint8)
        self.frame[..., :3] = rgb
        self.frame[..., 3] = 255

    def frame_iterator(self, total_frames, draw_fn=None):
        for t in range(total_frames):
            if draw_fn is not None:
                draw_fn(self, t)
            yield t, self.frame

    # -- 软光栅基元 --------------------------------------------------------

    def _dist_field(self, cx, cy):
        """到 (cx, cy) 的像素距离场 (H, W) float32。"""
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        return np.hypot(xs - cx, ys - cy)

    def draw_circle_disk(self, cx, cy, radius, color, alpha):
        """实心圆盘（半透明球壳）。"""
        d = self._dist_field(cx, cy)
        mask = d <= radius
        a = alpha
        self.frame[..., :3][mask] = (
            self.frame[..., :3][mask] * (1 - a) + np.asarray(color) * 255 * a
        ).astype(np.uint8)

    def draw_circle_ring(self, cx, cy, radius, color, thickness=1.5):
        d = self._dist_field(cx, cy)
        mask = np.abs(d - radius) <= thickness
        self.frame[mask, :3] = (np.asarray(color) * 255).astype(np.uint8)

    def draw_ellipse_ring(self, cx, cy, rx, ry, color, thickness=1.0):
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        d = np.hypot((xs - cx) / max(rx, 1e-6), (ys - cy) / max(ry, 1e-6))
        mask = np.abs(d - 1.0) * min(rx, ry) <= thickness
        self.frame[mask, :3] = (np.asarray(color) * 255).astype(np.uint8)

    def draw_segment(self, p0, p1, color, thickness=1.5):
        """线段 p0→p1（像素坐标，y 向下），距离场光栅化。"""
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        pts = np.stack([xs, ys], axis=-1).astype(np.float32)
        d = p1 - p0
        ll = float(d @ d)
        if ll < 1e-9:
            return
        tt = np.clip(((pts - p0) @ d) / ll, 0, 1)[..., None]
        proj = p0 + tt * d
        dist = np.linalg.norm(pts - proj, axis=-1)
        mask = dist <= thickness
        self.frame[mask, :3] = (np.asarray(color) * 255).astype(np.uint8)


class SoftRasterBloch:
    """正交投影布洛赫球（limited 样式：球壳 + 赤道 + 三轴 + 态矢量）。"""

    def __init__(self, soft: SoftRasterContext, style: dict):
        self.soft = soft
        self.style = style

    def draw(self, v, center, radius):
        """center: 像素坐标 (cx, cy)（y 向下）；v: (3,) Bloch 向量（z 向上）。"""
        cx, cy = center
        s = self.soft
        # 球壳与赤道
        s.draw_circle_disk(cx, cy, radius, self.style["sphere_color"][:3],
                           self.style.get("sphere_alpha", 0.16))
        s.draw_ellipse_ring(cx, cy, radius, radius * 0.28,
                            self.style["ring_color"][:3])
        # 三轴（x 右、y 斜向省略深度、z 上）
        axis = np.asarray(self.style["axis_color"][:3])
        s.draw_segment((cx - radius * 1.15, cy), (cx + radius * 1.15, cy), axis)
        s.draw_segment((cx, cy - radius * 1.15), (cx, cy + radius * 1.15), axis)
        # 态矢量：正交投影 (x, -z)
        v = np.asarray(v, float).reshape(3)
        n = np.linalg.norm(v)
        if n > 1e-9:
            v = v / n
            tip = (cx + v[0] * radius, cy - v[2] * radius)
            s.draw_segment((cx, cy), tip, self.style["vector_color"][:3], 2.0)
            s.draw_circle_disk(tip[0], tip[1], 3.5, self.style["vector_color"][:3], 1.0)


def render_bloch_video_cpu(states_bloch, fps, out, theme, width, height,
                           n_qubits, codec="h264", quality=0.9,
                           trail=False, cols=None) -> "object":
    """CPU 软光栅出片：frames_bloch (F, n, 3) → MP4。接口与 GL 路径对齐。

    cols：一行最多几个球（None=单行）。多行布局以正交投影排成网格。
    """
    import time
    from pathlib import Path

    from ..encode import create_encoder

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cols is None:
        cols = n_qubits
    cols = max(1, min(cols, n_qubits))
    rows = int(np.ceil(n_qubits / cols))
    # 网格单元尺寸：取列/行方向的较小者，保证球不重叠
    cell_w = width / cols
    cell_h = height / rows
    spacing_px = int(min(cell_w, cell_h) * 0.78)
    radius_px = int(spacing_px * 0.35)
    total_frames = states_bloch.shape[0]

    trails: list[list] = [[] for _ in range(n_qubits)]
    t0 = time.perf_counter()
    with SoftRasterContext(width, height, fps=fps) as soft:
        renderer = SoftRasterBloch(soft, theme)

        def draw(ctx, t: int):
            ctx.clear(theme["background"])
            vecs = states_bloch[t]
            for i in range(n_qubits):
                r = i // cols
                c = i % cols
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                if trail:
                    trails[i].append(np.asarray(vecs[i], float))
                    for k in range(1, len(trails[i])):
                        a, b = trails[i][k - 1], trails[i][k]
                        renderer.soft.draw_segment(
                            (cx + a[0] * radius_px, cy - a[2] * radius_px),
                            (cx + b[0] * radius_px, cy - b[2] * radius_px),
                            theme["trail_color"][:3], 1.0)
                renderer.draw(vecs[i], (cx, cy), radius_px)

        with create_encoder(width, height, fps, out, codec=codec,
                            quality=quality) as enc:
            for _, frame in soft.frame_iterator(total_frames, draw):
                enc.write(frame)
    print(f"[cpu backend] rendered {total_frames} frames in "
          f"{time.perf_counter() - t0:.1f}s -> {out}")
    return out
