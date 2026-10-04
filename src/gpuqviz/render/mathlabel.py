"""LaTeX 标注：matplotlib mathtext → 透明图层（P2.6）。

- ``mathtext_to_rgba(text, fontsize, dpi)``：``$...$`` 数学表达式（及任意
  Unicode 文本）→ 紧裁剪的 (H, W, 4) uint8 透明图层 —— GL 纹理叠加与
  CPU numpy 合成共用；
- 无需系统 LaTeX：matplotlib mathtext 内置排版引擎（可选依赖 matplotlib）。

TextOverlay(latex=True) 的叠加经 _draw_overlays_gl（ImageOverlayRenderer
纹理四边形）与 _draw_overlays_cpu（numpy 合成）分发；SVG 路径经
SVGContext.draw_rgba 嵌入高分辨率位图。
"""

from __future__ import annotations


import numpy as np

__all__ = ["mathtext_to_rgba", "composite_rgba", "ImageOverlayRenderer"]


def _require_matplotlib():
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "LaTeX labels require matplotlib: pip install gpuqviz[plot] "
            "(or pip install matplotlib)"
        ) from e
    matplotlib.use("Agg", force=True)  # headless 安全
    return matplotlib


_CJK_FONT_FAMILY = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                    "PingFang SC", "DejaVu Sans"]


def mathtext_to_rgba(text: str, fontsize: float = 24.0,
                     color=(1.0, 1.0, 1.0), dpi: float = 200.0,
                     font_family: list[str] | None = None) -> np.ndarray:
    """文本（支持 $...$ mathtext）→ 紧裁剪透明 RGBA 图层。

    fontsize 为点值；dpi=200 时输出像素高 ≈ fontsize/72·dpi × 行数。
    color 为 0-1 float 元组。font_family 依次回退（默认含 CJK 字体，
    数学部分始终由 mathtext 排版）。
    """
    _require_matplotlib()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    from matplotlib.figure import Figure

    r, g, b = (float(c) for c in color[:3])
    fp = FontProperties(family=font_family or _CJK_FONT_FAMILY, size=fontsize)

    fig = Figure(figsize=(10, 2), dpi=dpi)
    fig.patch.set_alpha(0.0)  # 透明背景：alpha 通道只承载文字
    canvas = FigureCanvasAgg(fig)
    t = fig.text(0.0, 0.0, text, fontproperties=fp, color=(r, g, b))
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())  # (H, W, 4)，行 0 = 顶部
    # extents = (x0, y0_bottom, x1, y1_top)，显示坐标 y 向上
    # → 行区间 = [H - y1, H - y0]（上下翻转）
    ex = t.get_window_extent(renderer=canvas.get_renderer())
    pad = 2
    col0 = max(int(ex.x0) - pad, 0)
    col1 = min(int(ex.x1) + pad, buf.shape[1])
    row0 = max(buf.shape[0] - int(ex.y1) - pad, 0)
    row1 = min(buf.shape[0] - int(ex.y0) + pad, buf.shape[0])
    return buf[row0:row1, col0:col1].copy()


def composite_rgba(frame: np.ndarray, rgba: np.ndarray,
                   x: float, y: float) -> None:
    """把 RGBA 图层 alpha 混合进软光栅帧（原地）。

    x/y 为图层左上角（像素坐标，可含小数，越界部分裁剪）。
    """
    fh, fw = frame.shape[:2]
    ih, iw = rgba.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = x0 + iw, y0 + ih
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, fw), min(y1, fh)
    if x1 <= x0 or y1 <= y0:
        return
    patch = rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float64)
    sl = frame[y0:y1, x0:x1, :3].astype(np.float64)
    a = patch[:, :, 3:4] / 255.0
    frame[y0:y1, x0:x1, :3] = (sl * (1 - a) + patch[:, :, :3] * a).astype(np.uint8)


class ImageOverlayRenderer:
    """GL 透明图层渲染器：RGBA 纹理四边形（LaTeX 标注的 GL 路径）。

    顶点缓冲布局：[像素矩形坐标 (6 顶点×2f)][UV (6 顶点×2f)]，
    in_pos 绑定 offset 0、in_uv 绑定 offset 48 字节。
    """

    _VS = """
    #version 330
    in vec2 in_pos;      // 像素坐标（y 向下）
    in vec2 in_uv;
    uniform vec2 u_res;
    out vec2 v_uv;
    void main() {
        float cx = in_pos.x / u_res.x * 2.0 - 1.0;
        float cy = 1.0 - in_pos.y / u_res.y * 2.0;
        v_uv = in_uv;
        gl_Position = vec4(cx, cy, 0.0, 1.0);
    }
    """

    _FS = """
    #version 330
    in vec2 v_uv;
    uniform sampler2D u_tex;
    out vec4 frag;
    void main() { frag = texture(u_tex, v_uv); }
    """

    def __init__(self, gl):
        self.gl = gl
        ctx = gl.ctx
        self.prog = gl.program(self._VS, self._FS)
        # 双 buffer：位置动态、UV 静态（同一 quad 恒定）
        self.vbo_pos = ctx.buffer(np.zeros(12, np.float32).tobytes(),
                                  dynamic=True)
        self.vbo_uv = ctx.buffer(np.array(
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], np.float32).tobytes())
        self.vao = ctx.vertex_array(self.prog, [
            (self.vbo_pos, "2f", "in_pos"),
            (self.vbo_uv, "2f", "in_uv"),
        ])
        self._tex = None
        self._tex_key = None

    def draw(self, rgba: np.ndarray, x: float, y: float,
             width: float | None = None, height: float | None = None) -> None:
        """RGBA (H, W, 4) uint8 → (x, y)（图层左上，像素坐标）绘制。"""
        ctx = self.gl.ctx
        rgba = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8))
        ih, iw = rgba.shape[:2]
        w = float(width if width is not None else iw)
        h = float(height if height is not None else ih)
        if self._tex_key != (iw, ih):
            if self._tex is not None:
                self._tex.release()
            self._tex = ctx.texture((iw, ih), 4)
            self._tex_key = (iw, ih)
        self._tex.write(rgba.tobytes())
        self._tex.use(0)
        self.prog["u_tex"].value = 0
        self.prog["u_res"].value = (float(self.gl.width), float(self.gl.height))
        verts = np.array([x, y + h, x + w, y + h, x, y,
                          x, y, x + w, y + h, x + w, y], dtype=np.float32)
        self.vbo_pos.write(verts.tobytes())
        ctx.disable(ctx.DEPTH_TEST)
        self.vao.render(vertices=6)
        ctx.enable(ctx.DEPTH_TEST)

    def release(self):
        if self._tex is not None:
            self._tex.release()
            self._tex = None
        self.vao.release()
        self.vbo_pos.release()
        self.vbo_uv.release()
        self.prog.release()
