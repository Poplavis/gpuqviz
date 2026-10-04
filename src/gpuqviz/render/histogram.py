"""直方图渲染器：测量统计 / 概率分布的柱状图视图。

几何约定（与 docs/conventions.md §1 一致）：
- 柱按基矢 index 升序排列（位串序 q_{n-1}…q_0）；
- 高度为**绝对尺度**：满高 = 概率 1.0，跨帧可比、无逐帧重归一化；
- 只画概率最高的 top_k 项（大空间可读性），其余项不隐藏进 "others"
  （M1 边界：误差棒与 others 聚合在 Phase 2 完善）。

GL 与 CPU 软光栅共享同一几何函数 histogram_bars()，保证双后端
渲染结果一致（test_histogram_track.py 验证）。
"""

from __future__ import annotations

import numpy as np

try:  # cupy 可选
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

_BAR_VS = """
#version 330
in vec2 in_pos;      // 像素坐标（y 向下）
uniform vec2 u_res;  // 画布尺寸
void main() {
    float cx = in_pos.x / u_res.x * 2.0 - 1.0;
    float cy = 1.0 - in_pos.y / u_res.y * 2.0;
    gl_Position = vec4(cx, cy, 0.0, 1.0);
}
"""

_BAR_FS = """
#version 330
uniform vec4 u_color;
out vec4 frag;
void main() { frag = u_color; }
"""

# 视觉参数（GL/CPU 共享，确保一致）
BAR_ALPHA = 0.85
BASELINE_ALPHA = 0.5
LABEL_SIZE_RATIO = 0.42   # 位串标签字号 / 柱宽
LABEL_PAD = 6             # 标签与基线间距（px）


def histogram_bars(probs, rect, top_k: int = 8, n_qubits: int | None = None,
                   others: bool = False):
    """概率分布 → top_k 柱几何（共享布局，双后端一致）。

    rect: (x, y, w, h) 画布内矩形（y 向下）。
    others=True 时把 top-k 之外的概率聚合为一根 "others" 柱
    （P5.3 LOD：大空间可读性；index = -1 作标记）。
    返回 (bars, baseline_y)：
      bars = [(x0, y0, w, h, index, bitstring, p), ...] 按 index 升序
      （others 柱排在末尾）。
    """
    x, y, w, h = map(float, rect)
    if cp is not None and isinstance(probs, cp.ndarray):
        probs = cp.asnumpy(probs)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if n_qubits is None:
        n = int(round(np.log2(p.size)))
    else:
        n = n_qubits

    k = min(top_k, p.size)
    # top-k 概率项 → 按 index 升序显示（位串序，跨帧柱位稳定）
    sel = np.sort(np.argsort(-p)[:k])
    n_bars = len(sel)

    gap = w * 0.02
    bar_w = (w - gap * (n_bars + 1)) / max(n_bars, 1)
    # 标签高度按实际字号推算（与绘制端 fs 规则一致），避免大面积空白
    fs = max(10.0, min(bar_w * LABEL_SIZE_RATIO, 22.0))
    label_h = fs + LABEL_PAD + 4
    plot_h = max(h - label_h, 1.0)
    baseline_y = y + h - label_h

    bars = []
    for j, i in enumerate(sel):
        pi = float(p[i])
        bh = pi * plot_h  # 绝对尺度：满高 = 概率 1.0
        bx = x + gap + j * (bar_w + gap)
        bars.append((bx, baseline_y - bh, bar_w, bh, int(i),
                     format(int(i), f"0{n}b"), pi))

    if others:
        mask = np.ones(p.size, dtype=bool)
        mask[sel] = False
        p_rest = float(p[mask].sum())
        if p_rest > 1e-12:
            bx = x + gap + n_bars * (bar_w + gap)
            bh = p_rest * plot_h
            bars.append((bx, baseline_y - bh, bar_w, bh, -1, "others", p_rest))
    return bars, baseline_y


class HistogramRenderer:
    """GL 柱状图：单色实心柱 + 基线；位串标签由 TextRenderer 叠加。"""

    def __init__(self, gl):
        self.gl = gl
        ctx = gl.ctx
        self.prog = gl.program(_BAR_VS, _BAR_FS)
        # 预分配：64 柱 × 6 顶点 × 2 float
        self._max_verts = 64 * 6
        self.vbo = ctx.buffer(np.zeros(self._max_verts * 2, np.float32).tobytes(),
                              dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])

    def draw(self, probs, rect, top_k: int = 8, color=(0.30, 0.77, 0.95, BAR_ALPHA),
             label_color=(0.55, 0.60, 0.68, 1.0), draw_labels: bool = True,
             others: bool = False) -> None:
        bars, baseline_y = histogram_bars(probs, rect, top_k=top_k,
                                          others=others)
        if not bars:
            return
        x, y, w, h = map(float, rect)
        ctx = self.gl.ctx

        def render_quads(verts, rgba):
            arr = np.asarray(verts, dtype=np.float32)
            # 2D 叠加层不写深度（避免等深丢弃后续同位置文字）
            ctx.disable(ctx.DEPTH_TEST)
            self.prog["u_res"].value = (float(self.gl.width), float(self.gl.height))
            self.vbo.write(arr.tobytes())
            self.prog["u_color"].value = tuple(rgba)
            self.vao.render(vertices=len(verts) // 2)
            ctx.enable(ctx.DEPTH_TEST)

        def quad_into(verts, qx0, qy0, qx1, qy1):
            verts.extend([qx0, qy0, qx1, qy0, qx0, qy1,
                          qx1, qy0, qx1, qy1, qx0, qy1])

        bar_verts: list[float] = []
        for bx, by, bw, bh, *_ in bars:
            if bh > 0.5:  # 忽略不可见的零概率柱
                quad_into(bar_verts, bx, by, bx + bw, by + bh)
        render_quads(bar_verts, color)

        base_verts: list[float] = []
        quad_into(base_verts, x, baseline_y - 1.0, x + w, baseline_y)
        render_quads(base_verts, (label_color[0], label_color[1],
                                  label_color[2], BASELINE_ALPHA))

        if draw_labels and getattr(self.gl, "_text", None) is not None:
            bar_w = bars[0][2]
            fs = max(10.0, min(bar_w * LABEL_SIZE_RATIO, 22.0))
            for bx, by, bw, bh, _i, bs, _p in bars:
                tw = len(bs) * fs * 0.62
                self.gl._text.draw(bs, (bx + (bw - tw) / 2,
                                        baseline_y + LABEL_PAD),
                                   fs, label_color)

    def release(self):
        self.vao.release()
        self.vbo.release()
        self.prog.release()


def draw_histogram_cpu(soft, probs, rect, top_k: int = 8,
                       color=(0.30, 0.77, 0.95, BAR_ALPHA),
                       label_color=(0.55, 0.60, 0.68, 1.0),
                       others: bool = False) -> None:
    """CPU 软光栅直方图（与 GL 共享 histogram_bars 几何）。"""
    bars, baseline_y = histogram_bars(probs, rect, top_k=top_k,
                                      others=others)
    if not bars:
        return
    x, y, w, h = map(float, rect)
    for bx, by, bw, bh, _i, _bs, _p in bars:
        if bh > 0.5:
            soft.draw_rect_fill(bx, by, bw, bh, color[:3], color[3])
    soft.draw_rect_fill(x, baseline_y - 1.0, w, 1.0,
                        label_color[:3], BASELINE_ALPHA)
    bar_w = bars[0][2]
    fs = max(9.0, min(bar_w * LABEL_SIZE_RATIO, 22.0))
    for bx, by, bw, bh, _i, bs, _p in bars:
        tw = len(bs) * fs * 0.62
        soft.draw_text(bs, (bx + (bw - tw) / 2, baseline_y + LABEL_PAD),
                       int(fs), label_color)
