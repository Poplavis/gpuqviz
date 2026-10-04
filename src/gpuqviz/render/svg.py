"""SVG 矢量画布：与 SoftRasterContext 同签名的绘制原语记录器（P2.5）。

设计：CPU 软光栅渲染器（SoftRasterBloch / SoftRasterHeatmap）与分析几何
绘制函数（draw_histogram_cpu / draw_entanglement_graph_cpu / draw_density_cpu）
均以 ``soft`` 为首参调用上下文原语——SVGContext 实现同一组方法签名，
把每次调用记录为 SVG 矢量元素，从而复用全部现有渲染逻辑直出矢量图。

- 坐标系与 CPU 画布一致：像素坐标，y 向下；
- 颜色为 0-1 float 元组（第 4 位 alpha 可选）；
- 文字用 SVG 原生 <text>（查看器解析系统字体，CJK 正常）；
- draw_heatmap / draw_rgba 为嵌入位图（数据网格/公式与 matplotlib
  imshow 的 SVG 输出同惯例）；
- draw_phase_disc 未使用，暂不支持。

PDF：``save("fig.pdf")`` 经 cairosvg 转换（可选依赖）。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

__all__ = ["SVGContext"]


def _hex_color(color) -> str:
    r, g, b = (int(round(float(c) * 255)) for c in color[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _alpha(color, default: float = 1.0) -> float:
    if len(color) > 3:
        return float(np.clip(color[3], 0.0, 1.0))
    return float(default)


def _rgba_to_png_datauri(rgba: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class SVGContext:
    """矢量 SVG 画布：SoftRasterContext 的 drop-in 矢量替身。"""

    def __init__(self, width: int, height: int, fps: float = 60.0):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.ctx = None  # 与 SoftRasterContext 接口兼容的占位
        self._elements: list[str] = []

    def __enter__(self) -> "SVGContext":
        return self

    def __exit__(self, *exc) -> None:
        pass

    # -- 原语（与 SoftRasterContext 同签名） ------------------------------

    def clear(self, color=(0, 0, 0, 1)):
        hexc = _hex_color(color)
        self._elements.append(
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" '
            f'fill="{hexc}"/>')

    def draw_circle_disk(self, cx, cy, radius, color, alpha):
        hexc, a = _hex_color(color), _alpha(color, alpha)
        self._elements.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{max(radius, 0):.2f}" '
            f'fill="{hexc}" fill-opacity="{a:.3f}"/>')

    def draw_circle_ring(self, cx, cy, radius, color, thickness=1.5):
        hexc, a = _hex_color(color), _alpha(color)
        self._elements.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{max(radius, 0):.2f}" '
            f'fill="none" stroke="{hexc}" stroke-width="{thickness:.2f}" '
            f'stroke-opacity="{a:.3f}"/>')

    def draw_ellipse_ring(self, cx, cy, rx, ry, color, thickness=1.0):
        hexc, a = _hex_color(color), _alpha(color)
        self._elements.append(
            f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{max(rx, 0):.2f}" '
            f'ry="{max(ry, 0):.2f}" fill="none" stroke="{hexc}" '
            f'stroke-width="{thickness:.2f}" stroke-opacity="{a:.3f}"/>')

    def draw_rect_fill(self, x, y, w, h, color, alpha=1.0):
        hexc, a = _hex_color(color), _alpha(color, alpha)
        self._elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" '
            f'height="{max(h, 0):.2f}" fill="{hexc}" fill-opacity="{a:.3f}"/>')

    def draw_segment(self, p0, p1, color, thickness=1.5):
        hexc, a = _hex_color(color), _alpha(color)
        self._elements.append(
            f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" x2="{p1[0]:.2f}" '
            f'y2="{p1[1]:.2f}" stroke="{hexc}" stroke-width="{thickness:.2f}" '
            f'stroke-linecap="round" stroke-opacity="{a:.3f}"/>')

    def draw_text(self, text, position, size_px, color=(1, 1, 1, 1)):
        hexc, a = _hex_color(color), _alpha(color)
        # PIL 的 text y 是升部顶端；SVG text y 是基线 → 近似下移 0.78em
        baseline = float(position[1]) + float(size_px) * 0.78
        self._elements.append(
            f'<text x="{float(position[0]):.2f}" y="{baseline:.2f}" '
            f'font-family="Segoe UI, Microsoft YaHei, sans-serif" '
            f'font-size="{float(size_px):.2f}" fill="{hexc}" '
            f'fill-opacity="{a:.3f}">{_esc(text)}</text>')

    def draw_text_with_shadow(self, text, position, size_px, color,
                              shadow_offset=(2, 2), shadow_alpha=0.5):
        ox, oy = shadow_offset
        self.draw_text(text, (position[0] + ox, position[1] + oy), size_px,
                       (0, 0, 0, shadow_alpha))
        self.draw_text(text, position, size_px, color)

    def draw_heatmap(self, img, rect, lut):
        """LUT 伪彩热图：img (rows, cols) ∈ [0,1] → 查表 → 嵌入位图。

        与 CPU 版一致使用最近邻拉伸（浏览器按 rect 缩放位图）。
        """
        x, y, w, h = (float(v) for v in rect)
        img = np.asarray(img, dtype=np.float64)
        idx = np.clip((img * 255).round().astype(np.int64), 0, 255)
        rgb = np.asarray(lut, dtype=np.uint8)[idx]  # (rows, cols, 3)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        uri = _rgba_to_png_datauri(rgba)
        self._elements.append(
            f'<image x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" '
            f'height="{max(h, 0):.2f}" preserveAspectRatio="none" '
            f'href="{uri}"/>')

    def draw_rgba(self, rgba: np.ndarray, x, y):
        """嵌入 RGBA 位图（LaTeX 标注等预渲染图层）。"""
        rgba = np.asarray(rgba, dtype=np.uint8)
        uri = _rgba_to_png_datauri(rgba)
        self._elements.append(
            f'<image x="{float(x):.2f}" y="{float(y):.2f}" '
            f'width="{rgba.shape[1]}" height="{rgba.shape[0]}" '
            f'href="{uri}"/>')

    def draw_phase_disc(self, state, rect):  # pragma: no cover - 无调用方
        raise NotImplementedError(
            "draw_phase_disc is not supported by SVGContext (unused primitive)")

    # -- 输出 -------------------------------------------------------------- #

    def to_string(self) -> str:
        body = "\n  ".join(self._elements)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n  {body}\n</svg>\n')

    def save(self, out: str | Path) -> Path:
        """写 SVG；.pdf/.png 后缀经 cairosvg 转换（可选依赖）。"""
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        svg_text = self.to_string()
        if out.suffix.lower() == ".svg":
            out.write_text(svg_text, encoding="utf-8")
            return out
        try:
            import cairosvg
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                f"{out.suffix} output requires cairosvg: pip install cairosvg"
            ) from e
        if out.suffix.lower() == ".pdf":
            cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"),
                             write_to=str(out))
        elif out.suffix.lower() == ".png":
            cairosvg.svg2png(bytestring=svg_text.encode("utf-8"),
                             write_to=str(out))
        else:
            raise ValueError(f"unsupported output suffix {out.suffix!r}")
        return out


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))
