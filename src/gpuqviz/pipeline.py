"""顶层管线编排：渲染循环 → 进程内编码 → MP4，零中间文件。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from .encode import create_encoder
from .render import GLContext

# 全屏三角形（顶点着色器免 VBO 技巧）
_VS_SCREEN = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# S1 验收用的演示片元着色器：随时间流动的颜色渐变（全 GPU，无 CPU 绘图）
_FS_GRADIENT = """
#version 330
in vec2 uv;
out vec4 frag;
uniform float u_time;
void main() {
    vec3 c1 = vec3(0.05, 0.08, 0.18);           // 深蓝底
    vec3 c2 = vec3(0.20, 0.75, 0.95);           // 青色高光
    float wave = 0.5 + 0.5 * sin(uv.x * 6.2831 + u_time * 2.0);
    frag = vec4(mix(c1, c2, wave * uv.y + 0.1 * u_time), 1.0);
}
"""


def render_frames(draw_fn: Callable[[GLContext, int], None], total_frames: int,
                  width: int, height: int, fps: float, out: str | Path,
                  codec: str = "h264", quality: float = 0.9,
                  device: int = 0) -> Path:
    """通用渲染循环：draw_fn(ctx, t) 画第 t 帧 → 读回 → write 给编码器。

    帧不落盘：FBO 像素读回后直接进入进程内编码器。
    """
    out = Path(out)
    with GLContext(width, height, fps=fps, device=device) as gl:
        with create_encoder(width, height, fps, out, codec=codec, quality=quality) as enc:
            for t, frame in gl.frame_iterator(total_frames, draw_fn):
                enc.write(frame)
    return out


def render_solid_gradient(out: str | Path = "out/demo.mp4", seconds: float = 3.0,
                          fps: float = 60.0, width: int = 1920, height: int = 1080,
                          codec: str = "h264", quality: float = 0.9) -> Path:
    """S1 验收 demo：GLSL 流动渐变动画 → MP4。"""
    total_frames = int(round(seconds * fps))
    t_start = time.perf_counter()

    def draw(gl: GLContext, t: int) -> None:
        prog = gl.program(_VS_SCREEN, _FS_GRADIENT)
        # moderngl 免 VBO 全屏三角形
        vbo = gl.ctx.buffer(
            np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32).tobytes()
        )
        try:
            vao = gl.ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
            try:
                gl.fbo.use()
                gl.ctx.clear(0.0, 0.0, 0.0, 1.0)
                prog["u_time"].value = t / fps
                vao.render()
            finally:
                vao.release()
        finally:
            vbo.release()

    out = render_frames(draw, total_frames, width, height, fps, out,
                        codec=codec, quality=quality)
    elapsed = time.perf_counter() - t_start
    print(f"rendered {total_frames} frames in {elapsed:.2f}s -> {out}")
    return out
