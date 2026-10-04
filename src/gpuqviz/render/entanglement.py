"""纠缠图渲染器：qubit 节点 + 纠缠边的力导/环形布局视图。

视觉编码（M2）：
- 节点 = qubit：半径 ∝ 单 qubit 纠缠熵（|0⟩ 小圆 → 最大纠缠大圆）；
- 边 = 两 qubit 纠缠：粗细 ∝ 互信息 I(A:B)（0 截断，上限 2 bit）；
- 节点标签 "q0"…"qn-1" 叠加在圆心。

GL 与 CPU 软光栅共享 entanglement_graph_geometry()，保证双后端一致。
时间轴动画天然呈现"纠缠在 qubit 间流动"——Bloch 视图缺失的核心叙事。
"""

from __future__ import annotations

import numpy as np

try:  # cupy 可选
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

# 视觉参数（GL/CPU 共享）
NODE_COLOR = (0.30, 0.77, 0.95, 0.9)
EDGE_COLOR = (0.62, 0.68, 0.78, 0.8)
LABEL_COLOR = (0.09, 0.11, 0.16, 1.0)   # 深色标签叠在亮节点上
_NODE_SEGMENTS = 28                      # 节点圆盘三角剖分段数

# 两 qubit 约化态互信息上限 = 2 bit（用于粗细归一）
_MI_MAX_BITS = 2.0


def entanglement_graph_geometry(report, rect, max_edges: int | None = None):
    """EntanglementReport → 图几何（共享布局，双后端一致）。

    rect: (x, y, w, h) 画布内矩形（图像坐标，y 向下）。
    max_edges: P5.3 LOD——边数超过上限时按互信息保留最强的若干条
    （O(n²) 边在 n 大时不可读）；None = 不截断。
    返回 (nodes, edges)：
      nodes = [(cx, cy, r, entropy, label), ...]
      edges = [(x0, y0, x1, y1, thickness, weight), ...]
    """
    x, y, w, h = map(float, rect)
    n = report.n_qubits
    cx, cy = x + w / 2, y + h / 2
    rx = max(w * 0.38, 1.0)
    ry = max(h * 0.34, 1.0)

    r_base = min(w, h)
    r_min, r_max = r_base * 0.045, r_base * 0.085

    nodes = []
    for q in range(n):
        ang = -np.pi / 2 + q * 2 * np.pi / max(n, 1)  # 顶部起始，顺时针
        nx = cx + rx * np.cos(ang)
        ny = cy + ry * np.sin(ang)
        s = float(np.clip(report.single_entropy[q], 0.0, 1.0))
        r = r_min + (r_max - r_min) * s
        nodes.append((nx, ny, r, s, f"q{q}"))

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            mi = float(report.mutual_info[i, j])
            if mi < 1e-6:
                continue
            frac = float(np.clip(mi / _MI_MAX_BITS, 0.0, 1.0))
            t = 1.0 + 5.0 * frac
            edges.append((nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1],
                          t, mi))
    if max_edges is not None and len(edges) > max_edges:
        edges.sort(key=lambda e: -e[5])  # 互信息降序，保留最强
        edges = edges[:max_edges]
    return nodes, edges


def _as_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


class EntanglementGraphRenderer:
    """GL 纠缠图：边（细长四边形）+ 节点（三角扇圆盘）+ 文字标签。"""

    def __init__(self, gl):
        from .histogram import _BAR_VS, _BAR_FS  # 复用同一像素坐标着色器
        self.gl = gl
        ctx = gl.ctx
        self.prog = gl.program(_BAR_VS, _BAR_FS)
        # 预分配：16 边×6 顶点 + 16 节点×(28 段×3 顶点) ≈ 1440 顶点
        self._max_verts = 4096
        self.vbo = ctx.buffer(np.zeros(self._max_verts * 2, np.float32).tobytes(),
                              dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])

    def _render_quads(self, verts, rgba):
        arr = np.asarray(verts, dtype=np.float32)
        if arr.size > self._max_verts * 2:
            arr = arr[: self._max_verts * 2]
        # 2D 叠加层不写深度：GLContext 全局开了 DEPTH_TEST，
        # 否则同 z=0 的后绘文字会被等深测试丢弃（标签消失的根因）
        ctx = self.gl.ctx
        ctx.disable(ctx.DEPTH_TEST)
        self.prog["u_res"].value = (float(self.gl.width), float(self.gl.height))
        self.vbo.write(arr.tobytes())
        self.prog["u_color"].value = tuple(rgba)
        self.vao.render(vertices=len(arr) // 2)
        ctx.enable(ctx.DEPTH_TEST)

    def draw(self, report, rect, label_color=LABEL_COLOR,
             max_edges: int | None = None) -> None:
        nodes, edges = entanglement_graph_geometry(report, rect,
                                                   max_edges=max_edges)
        if not nodes:
            return

        edge_verts: list[float] = []
        for x0, y0, x1, y1, t, _mi in edges:
            # 垂直法向加宽的粗线四边形（支持任意方向）
            dx, dy = x1 - x0, y1 - y0
            L = np.hypot(dx, dy)
            if L < 1e-6:
                continue
            nx, ny = -dy / L * (t / 2), dx / L * (t / 2)
            # 四角：a=(x0+n), b=(x1+n), c=(x1-n), d=(x0-n)
            ax, ay = x0 + nx, y0 + ny
            bx, by = x1 + nx, y1 + ny
            cx2, cy2 = x1 - nx, y1 - ny
            dx2, dy2 = x0 - nx, y0 - ny
            edge_verts.extend([ax, ay, bx, by, dx2, dy2,
                               bx, by, cx2, cy2, dx2, dy2])
        if edge_verts:
            self._render_quads(edge_verts, EDGE_COLOR)

        node_verts: list[float] = []
        for nx, ny, r, _s, _label in nodes:
            # 三角扇圆盘
            for k in range(_NODE_SEGMENTS):
                a0 = 2 * np.pi * k / _NODE_SEGMENTS
                a1 = 2 * np.pi * (k + 1) / _NODE_SEGMENTS
                node_verts.extend([nx, ny,
                                   nx + r * np.cos(a0), ny + r * np.sin(a0),
                                   nx + r * np.cos(a1), ny + r * np.sin(a1)])
        self._render_quads(node_verts, NODE_COLOR)

        if getattr(self.gl, "_text", None) is not None:
            fs = max(nodes[0][2] * 0.72, 10.0)
            for nx, ny, r, _s, label in nodes:
                tw = len(label) * fs * 0.62
                self.gl._text.draw(label, (nx - tw / 2, ny - fs * 0.62),
                                   fs, label_color)

    def release(self):
        self.vao.release()
        self.vbo.release()
        self.prog.release()


def draw_entanglement_graph_cpu(soft, report, rect,
                                max_edges: int | None = None) -> None:
    """CPU 软光栅纠缠图（与 GL 共享几何）。"""
    nodes, edges = entanglement_graph_geometry(report, rect,
                                               max_edges=max_edges)
    if not nodes:
        return
    for x0, y0, x1, y1, t, _mi in edges:
        soft.draw_segment((x0, y0), (x1, y1), EDGE_COLOR[:3], t)
    for nx, ny, r, _s, _label in nodes:
        soft.draw_circle_disk(nx, ny, r, NODE_COLOR[:3], NODE_COLOR[3])
    fs = max(nodes[0][2] * 0.72, 9.0)
    for nx, ny, r, _s, label in nodes:
        tw = len(label) * fs * 0.62
        soft.draw_text(label, (nx - tw / 2, ny - fs * 0.62),
                       int(fs), LABEL_COLOR)
