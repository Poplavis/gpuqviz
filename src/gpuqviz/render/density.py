"""密度矩阵 Hinton 图渲染器：ρ 的逐元素可视化。

Hinton 约定：
- 网格 (2^n × 2^n)，行 = row 基矢（高位在前，conventions.md §1），
  列 = col 基矢；
- 方块面积 ∝ |ρ_ij|（按帧内最大 |ρ| 归一，最大元素占满格）；
- 颜色按 Re(ρ_ij) 符号：正 = 蓝（NODE 蓝），负 = 橙；
  纯虚元素按相位象限映射到青/品红（M3 简化：统一蓝/橙按实部符号，
  文档注明局限——相位细分留待 P2.6 LaTeX/色盘迭代）。

GL 与 CPU 软光栅共享 hinton_cells()，保证双后端一致。
"""

from __future__ import annotations

import numpy as np

try:  # cupy 可选
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

# 视觉参数（GL/CPU 共享）
POS_COLOR = (0.30, 0.77, 0.95, 0.9)   # Re ≥ 0：蓝
NEG_COLOR = (0.96, 0.62, 0.25, 0.9)   # Re < 0：橙
GRID_COLOR = (0.55, 0.60, 0.68, 0.35)
LABEL_COLOR = (0.55, 0.60, 0.68, 1.0)
_MAX_LABEL_DIM = 8                     # d ≤ 8 时画位串轴标签


def hinton_cells(rho, rect, min_frac: float = 0.0):
    """密度矩阵 → Hinton 方块几何（共享布局，双后端一致）。

    rect: (x, y, w, h) 画布内矩形（图像坐标，y 向下）。
    min_frac: P5.3 LOD——|ρ_ij| < min_frac·max|ρ| 的方块跳过不画
    （高维密度矩阵的可读性阈值）；0 = 全画。
    返回 (cells, grid_rect)：
      cells = [(x0, y0, size, color, i, j, value), ...]（方块左上角 + 边长）
      grid_rect = (gx, gy, gw, gh) 网格整体矩形
    """
    x, y, w, h = map(float, rect)
    if cp is not None and isinstance(rho, cp.ndarray):
        rho = cp.asnumpy(rho)
    rho = np.asarray(rho, dtype=np.complex128)
    d = rho.shape[0]
    n = int(round(np.log2(d)))
    if 2 ** n != d or rho.ndim != 2 or rho.shape[1] != d:
        raise ValueError(f"rho must be a square 2**n matrix, got {rho.shape}")

    # 网格取矩形内最大正方形，居中
    side = min(w, h)
    gx, gy = x + (w - side) / 2, y + (h - side) / 2
    cell = side / d
    vmax = float(np.abs(rho).max())
    if vmax <= 1e-12:
        vmax = 1.0

    cells = []
    thresh = min_frac * vmax
    for i in range(d):          # 行（row 基矢，高位在前）
        for j in range(d):      # 列（col 基矢）
            v = rho[i, j]
            mag = float(np.abs(v))
            if mag < thresh:    # P5.3 LOD：低于阈值的方块跳过
                continue
            # Hinton：面积 ∝ |v|，最大元素占满格
            size = cell * (mag / vmax) ** 0.5 if mag > 1e-12 else 0.0
            color = POS_COLOR if float(np.real(v)) >= 0 else NEG_COLOR
            cx = gx + (j + 0.5) * cell
            cy = gy + (i + 0.5) * cell
            cells.append((cx - size / 2, cy - size / 2, size, color,
                          i, j, complex(v)))
    return cells, (gx, gy, side, side)


def _bitstring(i: int, n: int) -> str:
    return format(i, f"0{n}b")


class DensityMatrixRenderer:
    """GL Hinton 图：方块（quad shader 复用）+ 网格线 + 位串轴标签。"""

    def __init__(self, gl):
        from .histogram import _BAR_VS, _BAR_FS  # 复用像素坐标着色器
        self.gl = gl
        ctx = gl.ctx
        self.prog = gl.program(_BAR_VS, _BAR_FS)
        self._max_verts = 8192
        self.vbo = ctx.buffer(np.zeros(self._max_verts * 2, np.float32).tobytes(),
                              dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])

    def _render_quads(self, verts, rgba):
        arr = np.asarray(verts, dtype=np.float32)
        if arr.size > self._max_verts * 2:
            arr = arr[: self._max_verts * 2]
        # 2D 叠加层不写深度（等深丢弃后续文字的教训，见 render/entanglement.py）
        ctx = self.gl.ctx
        ctx.disable(ctx.DEPTH_TEST)
        self.prog["u_res"].value = (float(self.gl.width), float(self.gl.height))
        self.vbo.write(arr.tobytes())
        self.prog["u_color"].value = tuple(rgba)
        self.vao.render(vertices=len(arr) // 2)
        ctx.enable(ctx.DEPTH_TEST)

    def draw(self, rho, rect, draw_labels: bool = True,
             min_frac: float = 0.0) -> None:
        cells, (gx, gy, side, _sh) = hinton_cells(rho, rect,
                                                  min_frac=min_frac)
        d = int(round(np.sqrt(len(cells))))
        n = int(round(np.log2(d)))
        cell = side / d

        # 网格底线（横竖各 d+1 条，淡色）
        grid_v: list[float] = []

        def quad_into(v, qx0, qy0, qx1, qy1):
            v.extend([qx0, qy0, qx1, qy0, qx0, qy1,
                      qx1, qy0, qx1, qy1, qx0, qy1])

        for k in range(d + 1):
            quad_into(grid_v, gx, gy + k * cell - 0.5, gx + side, gy + k * cell + 0.5)
            quad_into(grid_v, gx + k * cell - 0.5, gy, gx + k * cell + 0.5, gy + side)
        self._render_quads(grid_v, GRID_COLOR)

        # 按颜色分组渲染方块
        for color in (POS_COLOR, NEG_COLOR):
            v: list[float] = []
            for cx0, cy0, size, c, *_ in cells:
                if size > 0.25 and c == color:
                    quad_into(v, cx0, cy0, cx0 + size, cy0 + size)
            if v:
                self._render_quads(v, color)

        # 位串轴标签（列在上沿、行在左沿），仅小维数
        if draw_labels and d <= _MAX_LABEL_DIM and \
                getattr(self.gl, "_text", None) is not None:
            fs = max(9.0, min(cell * 0.32, 18.0))
            for j in range(d):
                tw = n * fs * 0.62
                self.gl._text.draw(_bitstring(j, n),
                                   (gx + (j + 0.5) * cell - tw / 2,
                                    gy - fs - 4), fs, LABEL_COLOR)
            for i in range(d):
                tw = n * fs * 0.62
                self.gl._text.draw(_bitstring(i, n),
                                   (gx - tw - 6, gy + (i + 0.5) * cell - fs * 0.6),
                                   fs, LABEL_COLOR)

    def release(self):
        self.vao.release()
        self.vbo.release()
        self.prog.release()


def draw_density_cpu(soft, rho, rect, draw_labels: bool = True,
                     min_frac: float = 0.0) -> None:
    """CPU 软光栅 Hinton 图（与 GL 共享 hinton_cells 几何）。"""
    cells, (gx, gy, side, _sh) = hinton_cells(rho, rect, min_frac=min_frac)
    d = int(round(np.sqrt(len(cells))))
    n = int(round(np.log2(d)))
    cell = side / d
    for cx0, cy0, size, color, *_ in cells:
        if size > 0.25:
            soft.draw_rect_fill(cx0, cy0, size, size, color[:3], color[3])
    if draw_labels and d <= _MAX_LABEL_DIM:
        fs = max(8.0, min(cell * 0.32, 18.0))
        for j in range(d):
            tw = n * fs * 0.62
            soft.draw_text(_bitstring(j, n),
                           (gx + (j + 0.5) * cell - tw / 2, gy - fs - 4),
                           int(fs), LABEL_COLOR)
        for i in range(d):
            tw = n * fs * 0.62
            soft.draw_text(_bitstring(i, n),
                           (gx - tw - 6, gy + (i + 0.5) * cell - fs * 0.6),
                           int(fs), LABEL_COLOR)
