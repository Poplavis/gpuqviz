"""CPU 回退后端：纯 numpy 软光栅（无 OpenGL 时的完整兜底）。

S8 重写：包围盒光栅（不再每帧全屏分配距离场）+ 坐标网格缓存 +
numba @njit 加速热路径（缺失时静默回退 numpy）。新增 HeatmapTrack
（LUT 伪彩）、PIL 文字、PhaseDisc CPU 版，使 CPU 路径从"受限可用"
升级到"好用"，对齐 recorder 的零硬件门槛。
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# -- numba 可选加速 ------------------------------------------------------
# @njit(parallel=True) 用于包围盒内的距离场/混合热路径；缺失时回退 numpy
# 向量化（慢但正确）。import 失败不抛异常，只记一行日志。
try:
    from numba import njit, prange

    _NUMBA_OK = True
except ImportError:  # pragma: no cover
    _NUMBA_OK = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """无 numba 时的 no-op 装饰器：直接返回原函数。"""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap

    def prange(*args):  # type: ignore[misc]
        return range(*args)


if not _NUMBA_OK:
    logger.info("numba not installed; CPU backend falls back to numpy (slower)")


# -- numba 内联热路径 ----------------------------------------------------

if _NUMBA_OK:

    @njit(cache=True)
    def _blend_disk_numba(frame, y0, y1, x0, x1, cx, cy, radius, r, g, b, alpha):
        """实心圆盘 alpha 混合到 frame 的包围盒 [y0:y1, x0:x1]。"""
        inv_r = 1.0 / radius
        for yi in prange(y0, y1):
            dy = yi - cy
            for xi in range(x0, x1):
                dx = xi - cx
                d = (dx * dx + dy * dy) ** 0.5
                if d <= radius:
                    idx = yi * frame.shape[1] + xi
                    frame_flat = frame.reshape(-1, 4)
                    frame_flat[idx, 0] = np.uint8(
                        frame_flat[idx, 0] * (1 - alpha) + r * alpha
                    )
                    frame_flat[idx, 1] = np.uint8(
                        frame_flat[idx, 1] * (1 - alpha) + g * alpha
                    )
                    frame_flat[idx, 2] = np.uint8(
                        frame_flat[idx, 2] * (1 - alpha) + b * alpha
                    )

    @njit(cache=True)
    def _draw_ring_numba(frame, y0, y1, x0, x1, cx, cy, radius, thickness, r, g, b):
        """圆环：|d - radius| <= thickness → 纯色覆盖。"""
        for yi in prange(y0, y1):
            dy = yi - cy
            for xi in range(x0, x1):
                dx = xi - cx
                d = (dx * dx + dy * dy) ** 0.5
                if abs(d - radius) <= thickness:
                    idx = yi * frame.shape[1] + xi
                    frame_flat = frame.reshape(-1, 4)
                    frame_flat[idx, 0] = np.uint8(r)
                    frame_flat[idx, 1] = np.uint8(g)
                    frame_flat[idx, 2] = np.uint8(b)

    @njit(cache=True)
    def _draw_ellipse_ring_numba(
        frame, y0, y1, x0, x1, cx, cy, rx, ry, thickness, r, g, b
    ):
        """椭圆环：归一化距离 ≈ 1 → 纯色。"""
        min_r = min(rx, ry)
        for yi in prange(y0, y1):
            for xi in range(x0, x1):
                dx = (xi - cx) / max(rx, 1e-6)
                dy = (yi - cy) / max(ry, 1e-6)
                d = (dx * dx + dy * dy) ** 0.5
                if abs(d - 1.0) * min_r <= thickness:
                    idx = yi * frame.shape[1] + xi
                    frame_flat = frame.reshape(-1, 4)
                    frame_flat[idx, 0] = np.uint8(r)
                    frame_flat[idx, 1] = np.uint8(g)
                    frame_flat[idx, 2] = np.uint8(b)

    @njit(cache=True)
    def _draw_segment_numba(
        frame, H, W, p0x, p0y, p1x, p1y, thickness, r, g, b
    ):
        """线段光栅化：点到线段距离 <= thickness → 纯色。

        包围盒由调用者算出后传入完整 frame；numba 内部只在 bbox 内迭代。
        """
        dx = p1x - p0x
        dy = p1y - p0y
        ll = dx * dx + dy * dy
        if ll < 1e-9:
            return
        inv_ll = 1.0 / ll
        # 包围盒
        min_x = int(max(0, min(p0x, p1x) - thickness - 1))
        max_x = int(min(W, max(p0x, p1x) + thickness + 1))
        min_y = int(max(0, min(p0y, p1y) - thickness - 1))
        max_y = int(min(H, max(p0y, p1y) + thickness + 1))
        t2 = thickness * thickness
        frame_flat = frame.reshape(-1, 4)
        for yi in prange(min_y, max_y):
            for xi in range(min_x, max_x):
                t = ((xi - p0x) * dx + (yi - p0y) * dy) * inv_ll
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                px = p0x + t * dx
                py = p0y + t * dy
                ddx = xi - px
                ddy = yi - py
                if ddx * ddx + ddy * ddy <= t2:
                    idx = yi * W + xi
                    frame_flat[idx, 0] = np.uint8(r)
                    frame_flat[idx, 1] = np.uint8(g)
                    frame_flat[idx, 2] = np.uint8(b)

    @njit(cache=True)
    def _apply_lut_numba(frame, y0, y1, x0, x1, img_flat, lut):
        """热图伪彩：img 值 ∈ [0,1] → LUT 查表 → 覆盖 frame 的 bbox。"""
        n_cols = x1 - x0
        img_h = y1 - y0
        for yi in prange(img_h):
            for xi in range(n_cols):
                v = img_flat[yi * n_cols + xi]
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                li = int(v * 255)
                fy = yi + y0
                fx = xi + x0
                idx = fy * frame.shape[1] + fx
                frame_flat = frame.reshape(-1, 4)
                frame_flat[idx, 0] = lut[li, 0]
                frame_flat[idx, 1] = lut[li, 1]
                frame_flat[idx, 2] = lut[li, 2]

    @njit(cache=True)
    def _phase_disc_numba(frame, y0, y1, x0, x1, cx, cy, radius, amp, phase):
        """相位色盘 CPU 版：极径=亮度，色相=相位。"""
        for yi in prange(y0, y1):
            dy = (yi - cy) / radius
            for xi in range(x0, x1):
                dx = (xi - cx) / radius
                r = (dx * dx + dy * dy) ** 0.5
                if r > 1.0:
                    continue
                # HSV → RGB（色相=角度，饱和度=1，亮度=r*amp）
                hue = 0.5 + np.arctan2(dy, dx) / 6.28318530718
                s = 1.0
                v = r * amp
                if v < 0.0:
                    v = 0.0
                # 标准 HSV→RGB
                h6 = hue * 6.0
                i = int(h6) % 6
                f = h6 - int(h6)
                p = v * (1.0 - s)
                q = v * (1.0 - f * s)
                t = v * (1.0 - (1.0 - f) * s)
                if i == 0:
                    rr, gg, bb = v, t, p
                elif i == 1:
                    rr, gg, bb = q, v, p
                elif i == 2:
                    rr, gg, bb = p, v, t
                elif i == 3:
                    rr, gg, bb = p, q, v
                elif i == 4:
                    rr, gg, bb = t, p, v
                else:
                    rr, gg, bb = v, p, q
                idx = yi * frame.shape[1] + xi
                frame_flat = frame.reshape(-1, 4)
                frame_flat[idx, 0] = np.uint8(rr * 255)
                frame_flat[idx, 1] = np.uint8(gg * 255)
                frame_flat[idx, 2] = np.uint8(bb * 255)


# -- numpy 回退（无 numba 时） -------------------------------------------

def _bbox(cx, cy, radius, W, H, pad=1):
    """圆的像素包围盒，裁剪到帧边界。"""
    x0 = max(0, int(cx - radius - pad))
    x1 = min(W, int(cx + radius + pad + 1))
    y0 = max(0, int(cy - radius - pad))
    y1 = min(H, int(cy + radius + pad + 1))
    return y0, y1, x0, x1


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

    # -- 软光栅基元（包围盒 + numba/numpy 双路径） ------------------------

    def draw_circle_disk(self, cx, cy, radius, color, alpha):
        """实心圆盘（半透明球壳）。"""
        H, W = self.height, self.width
        y0, y1, x0, x1 = _bbox(cx, cy, radius, W, H)
        if y0 >= y1 or x0 >= x1:
            return
        r, g, b = (np.asarray(color[:3]) * 255).astype(np.float64)
        a = float(alpha)
        if _NUMBA_OK:
            _blend_disk_numba(self.frame, y0, y1, x0, x1, cx, cy, radius, r, g, b, a)
            return
        # numpy 回退
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        d = np.hypot(xs - cx, ys - cy)
        mask = d <= radius
        sl = self.frame[y0:y1, x0:x1, :3]
        sl[mask] = (sl[mask] * (1 - a) + np.array([r, g, b]) * a).astype(np.uint8)

    def draw_circle_ring(self, cx, cy, radius, color, thickness=1.5):
        H, W = self.height, self.width
        y0, y1, x0, x1 = _bbox(cx, cy, radius + thickness, W, H)
        if y0 >= y1 or x0 >= x1:
            return
        r, g, b = (np.asarray(color[:3]) * 255).astype(np.float64)
        if _NUMBA_OK:
            _draw_ring_numba(self.frame, y0, y1, x0, x1, cx, cy, radius, thickness, r, g, b)
            return
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        d = np.hypot(xs - cx, ys - cy)
        mask = np.abs(d - radius) <= thickness
        self.frame[y0:y1, x0:x1][mask, :3] = np.array([r, g, b], np.uint8)

    def draw_ellipse_ring(self, cx, cy, rx, ry, color, thickness=1.0):
        H, W = self.height, self.width
        pad = max(rx, ry) + thickness
        y0, y1, x0, x1 = _bbox(cx, cy, pad, W, H)
        if y0 >= y1 or x0 >= x1:
            return
        r, g, b = (np.asarray(color[:3]) * 255).astype(np.float64)
        if _NUMBA_OK:
            _draw_ellipse_ring_numba(
                self.frame, y0, y1, x0, x1, cx, cy, rx, ry, thickness, r, g, b
            )
            return
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        d = np.hypot((xs - cx) / max(rx, 1e-6), (ys - cy) / max(ry, 1e-6))
        mask = np.abs(d - 1.0) * min(rx, ry) <= thickness
        self.frame[y0:y1, x0:x1][mask, :3] = np.array([r, g, b], np.uint8)

    def draw_segment(self, p0, p1, color, thickness=1.5):
        """线段 p0→p1（像素坐标，y 向下）。"""
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        H, W = self.height, self.width
        r, g, b = (np.asarray(color[:3]) * 255).astype(np.float64)
        if _NUMBA_OK:
            _draw_segment_numba(
                self.frame, H, W,
                float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]),
                float(thickness), r, g, b,
            )
            return
        # numpy 回退（包围盒内算距离）
        dx, dy = p1 - p0
        ll = float(dx * dx + dy * dy)
        if ll < 1e-9:
            return
        min_x = max(0, int(min(p0[0], p1[0]) - thickness - 1))
        max_x = min(W, int(max(p0[0], p1[0]) + thickness + 1))
        min_y = max(0, int(min(p0[1], p1[1]) - thickness - 1))
        max_y = min(H, int(max(p0[1], p1[1]) + thickness + 1))
        if min_x >= max_x or min_y >= max_y:
            return
        ys = np.arange(min_y, max_y)[:, None]
        xs = np.arange(min_x, max_x)[None, :]
        pts = np.stack([xs, ys], axis=-1).astype(np.float32)
        tt = np.clip(((pts - p0) @ (p1 - p0)) / ll, 0, 1)[..., None]
        proj = p0 + tt * (p1 - p0)
        dist = np.linalg.norm(pts - proj, axis=-1)
        mask = dist <= thickness
        self.frame[min_y:max_y, min_x:max_x][mask, :3] = np.array([r, g, b], np.uint8)

    def draw_heatmap(self, img, rect, lut):
        """热图伪彩：img (rows, cols) float32 ∈ [0,1] → LUT 查表 → 写入 rect 区域。

        rect: (x, y, w, h) 像素坐标；img 被 nearest-neighbor 拉伸到 rect 大小。
        """
        x, y, w, h = rect
        x, y, w, h = int(x), int(y), int(w), int(h)
        ih, iw = img.shape
        if w <= 0 or h <= 0:
            return
        # nearest-neighbor 拉伸
        row_idx = (np.arange(h) * ih / h).astype(int).clip(0, ih - 1)
        col_idx = (np.arange(w) * iw / w).astype(int).clip(0, iw - 1)
        stretched = img[row_idx][:, col_idx]  # (h, w)
        x1, y1 = min(x + w, self.width), min(y + h, self.height)
        w_clip, h_clip = x1 - x, y1 - y
        if w_clip <= 0 or h_clip <= 0:
            return
        if _NUMBA_OK:
            _apply_lut_numba(
                self.frame, y, y + h_clip, x, x + w_clip,
                np.ascontiguousarray(stretched[:h_clip, :w_clip], dtype=np.float32).ravel(),
                lut,
            )
            return
        # numpy 回退
        v = np.clip(stretched[:h_clip, :w_clip], 0, 1)
        indices = (v * 255).astype(np.int32).clip(0, 255)
        rgb = lut[indices]  # (h_clip, w_clip, 3)
        self.frame[y : y + h_clip, x : x + w_clip, :3] = rgb

    def draw_text(self, text, position, size_px, color=(1, 1, 1, 1)):
        """PIL ImageDraw 绘制文字（CPU 路径允许直接走 PIL，与 GL SDF 互不影响）。"""
        from PIL import Image, ImageDraw, ImageFont

        W, H = self.width, self.height
        pil = Image.fromarray(self.frame, mode="RGBA")
        draw = ImageDraw.Draw(pil)
        r, g, b = (int(c * 255) for c in color[:3])
        a = int(getattr(color, "__len__", lambda: 1)() and color[3] * 255) if len(color) > 3 else 255

        # 字体：尝试系统字体，失败则用 PIL default
        font = _get_pil_font(size_px)
        # PIL 坐标 y 向下，与我们的 frame 一致
        draw.text((int(position[0]), int(position[1])), text, fill=(r, g, b, a), font=font)
        self.frame = np.array(pil, dtype=np.uint8)

    def draw_phase_disc(self, state, rect):
        """相位色盘 CPU 版：单位圆盘上极径=亮度、色相=相位。"""
        x, y, w, h = rect
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            return
        import numpy as _np

        psi = _np.asarray(state).reshape(-1).astype(_np.complex128)
        if psi.shape[0] != 2:
            raise ValueError("PhaseDisc expects a single-qubit state")
        amp = float(_np.abs(psi[1]))
        amp = min(max(amp, 0.0), 1.0)
        phase = float((_np.angle(psi[1]) + _np.pi) / (2 * _np.pi))

        cx = x + w / 2.0
        cy = y + h / 2.0
        radius = min(w, h) / 2.0
        y0, y1, x0, x1 = _bbox(cx, cy, radius, self.width, self.height)
        if y0 >= y1 or x0 >= x1:
            return
        if _NUMBA_OK:
            _phase_disc_numba(self.frame, y0, y1, x0, x1, cx, cy, radius, amp, phase)
            return
        # numpy 回退
        ys = _np.arange(y0, y1)[:, None]
        xs = _np.arange(x0, x1)[None, :]
        dx = (xs - cx) / radius
        dy = (ys - cy) / radius
        r = _np.hypot(dx, dy)
        mask = r <= 1.0
        hue = (0.5 + _np.arctan2(dy, dx) / (2 * _np.pi)) % 1.0
        # HSV → RGB
        h6 = hue * 6
        i = h6.astype(int) % 6
        f = h6 - h6.astype(int)
        v = _np.clip(r * amp, 0, 1)
        s = 1.0
        p_ = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        rgb = _np.zeros((*r.shape, 3), _np.float32)
        for ci, sel in enumerate([(v, t, p_), (q, v, p_), (p_, v, t), (p_, q, v), (t, p_, v), (v, p_, q)]):
            m = i == ci
            rgb[m, 0] = sel[0][m]
            rgb[m, 1] = sel[1][m]
            rgb[m, 2] = sel[2][m]
        sl = self.frame[y0:y1, x0:x1, :3]
        sl[mask] = (rgb[mask] * 255).astype(_np.uint8)


# -- PIL 字体缓存 --------------------------------------------------------
_pil_font_cache: dict = {}


def _get_pil_font(size_px: float):
    """获取 PIL ImageFont：尝试系统字体，失败回退 default。"""
    key = int(size_px)
    if key in _pil_font_cache:
        return _pil_font_cache[key]
    from PIL import ImageFont

    font = None
    for name in ("arial.ttf", "msyh.ttc", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, key)
            break
        except Exception:  # noqa: BLE001
            continue
    if font is None:
        font = ImageFont.load_default()
    _pil_font_cache[key] = font
    return font


class SoftRasterBloch:
    """正交投影布洛赫球（球壳 + 赤道 + 三轴 + 态矢量）。"""

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
        # 三轴（x 右、z 上）
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


class SoftRasterHeatmap:
    """CPU 热图渲染器：态矢量 → grid → LUT 伪彩。

    与 GL HeatmapRenderer 数值一致（复用 state_to_image + bake_colormap），
    但不依赖 OpenGL 纹理——LUT 烘焙为 256×3 uint8 数组，运行时查表。
    """

    def __init__(self, soft: SoftRasterContext, colormap: str = "viridis"):
        self.soft = soft
        from ..render.heatmap import bake_colormap

        self.lut = bake_colormap(colormap)  # (256, 3) uint8

    def draw(self, state, rect, basis: str = "probability", colorbar: bool = True):
        from ..render.heatmap import state_to_image

        img = state_to_image(state, basis=basis)
        if hasattr(img, "get"):
            img = img.get()
        img = np.ascontiguousarray(img, dtype=np.float32)
        x, y, w, h = rect
        cb_w = 22 if colorbar else 0
        self.soft.draw_heatmap(img, (x, y, w, h), self.lut)
        if colorbar:
            # 色标条：纵向渐变（LUT 倒序，底=0 顶=1）
            cb_h = int(h)
            cb_img = np.linspace(0, 1, cb_h).astype(np.float32)[:, None]
            self.soft.draw_heatmap(cb_img, (x + w + 10, y, cb_w, cb_h), self.lut)

    def release(self):
        pass


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


def render_heatmap_video_cpu(states, fps, out, width, height, basis="probability",
                             colormap="viridis", codec="h264", quality=0.9,
                             seconds=None, steps=120) -> "object":
    """CPU 软光栅热图出片：态矢量序列 → 概率/幅值/相位热图动画 MP4。"""
    import time
    from pathlib import Path

    from ..encode import create_encoder
    from ..interpolate import lerp_states
    from ..render.heatmap import state_to_image

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(round((seconds or steps / 30) * fps))
    frames = lerp_states(states, total_frames)
    if hasattr(frames, "get"):
        frames = frames.get()

    t0 = time.perf_counter()
    with SoftRasterContext(width, height, fps=fps) as soft:
        renderer = SoftRasterHeatmap(soft, colormap=colormap)

        def draw(ctx, t: int):
            ctx.clear((0.043, 0.055, 0.078, 1.0))
            m = int(round(np.log2(frames.shape[1])))
            cols = 2 ** int(np.ceil(m / 2))
            rows = max(1, frames.shape[1] // cols)
            area_w, area_h = width - 160, height - 120
            cell = min(area_w / cols, area_h / rows)
            w, h = cell * cols, cell * rows
            rect = ((width - 160 - w) / 2, (height - h) / 2 + 30, w, h)
            renderer.draw(frames[t], rect, basis=basis)

        with create_encoder(width, height, fps, out, codec=codec,
                            quality=quality) as enc:
            for _, frame in soft.frame_iterator(total_frames, draw):
                enc.write(frame)
    renderer.release()
    print(f"[cpu backend] rendered {total_frames} heatmap frames in "
          f"{time.perf_counter() - t0:.1f}s -> {out}")
    return out
