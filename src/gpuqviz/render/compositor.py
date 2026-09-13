"""分屏合成。

坐标系约定：
- 2D 渲染器（HeatmapRenderer 等）：矩形参数永远是【全 FBO 像素空间】，
  绘制时 viewport 必须是全画布 —— 由本模块在调用 bottom_draw 前恢复。
- 3D 渲染器（BlochRenderer）：在子区域内绘制，本模块临时切换 viewport，
  渲染器需用传入的子区域宽高计算纵横比。
"""

from __future__ import annotations


def split_view(gl, top_draw, bottom_draw, ratio: float = 0.5) -> None:
    """上下分屏。

    top_draw(width, top_h) 在子区域 viewport 内绘制（NDC 直接映射子区域）；
    bottom_draw(width, height) 在【全 FBO 像素坐标系】下绘制 2D 内容。
    """
    width, height = gl.width, gl.height
    top_h = int(height * ratio)

    gl.ctx.viewport = (0, height - top_h, width, top_h)
    top_draw(width, top_h)

    gl.ctx.viewport = (0, 0, width, height)
    bottom_draw(width, height)
