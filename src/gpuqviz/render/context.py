"""GPU 离屏渲染上下文：moderngl standalone context + 离屏 FBO + 帧读回。"""

from __future__ import annotations

from typing import Callable, Iterator

import moderngl
import numpy as np

try:  # cupy 可选：读回帧以 device ndarray 形式交付
    import cupy as cp
except ImportError:  # pragma: no cover - 无 N 卡环境
    cp = None


class GLContext:
    """离屏 GL 渲染上下文。

    用法::

        with GLContext(1920, 1080) as gl:
            gl.program(...)  # 编译着色器
            for t, frame in gl.frame_iterator(480, draw_fn):
                ...  # frame 为 (H, W, 4) uint8，交编码器
    """

    def __init__(self, width: int, height: int, fps: float = 60.0, device: int = 0):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.device = int(device)

        # create_standalone_context 在 Windows(WGL)/Linux(EGL) 均可无窗口创建
        self.ctx: moderngl.Context = moderngl.create_standalone_context()
        self.fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture((self.width, self.height), 4),
            depth_attachment=self.ctx.depth_renderbuffer((self.width, self.height)),
        )
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)  # standalone context 默认视口极小
        self.ctx.enable(moderngl.DEPTH_TEST)

        if cp is not None:
            cp.cuda.Device(self.device).use()

    # -- 资源管理 ---------------------------------------------------------

    def __enter__(self) -> "GLContext":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if getattr(self, "fbo", None) is not None:
            self.fbo.release()
            self.fbo = None
        if getattr(self, "ctx", None) is not None:
            self.ctx.release()
            self.ctx = None

    # -- 帧读回 -----------------------------------------------------------

    def read_frame(self) -> "np.ndarray":
        """把当前 FBO 读回为 (H, W, 4) uint8 RGBA。

        cupy 可用时返回 cupy device ndarray（pinned 拷贝路径，S3 interop 再
        消除这次拷贝），否则回退 numpy，接口一致。帧不写盘，直接交给编码器。
        """
        data: bytes = self.fbo.color_attachments[0].read()
        # 注：不走 fbo.read() —— 在部分 moderngl/驱动组合下返回全零；
        # moderngl 5.12）上返回全零；直接读颜色附件纹理则稳定正确。
        host = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
        # FBO 像素原点在左下角，翻转为图像常规的上下方向
        host = np.ascontiguousarray(host[::-1, :, :])
        if cp is not None:
            return cp.asarray(host)
        return host

    def frame_iterator(
        self, total_frames: int, draw_fn: Callable[["GLContext", int], None] | None = None
    ) -> Iterator[tuple[int, "np.ndarray"]]:
        """逐帧循环：先让 draw_fn 画第 t 帧，再读回并 yield (t, frame)。"""
        for t in range(total_frames):
            if draw_fn is not None:
                draw_fn(self, t)
            yield t, self.read_frame()

    def clear(self, color=(0.0, 0.0, 0.0, 1.0)) -> None:
        self.ctx.clear(*color)

    def program(self, vertex_source: str, fragment_source: str) -> moderngl.Program:
        return self.ctx.program(vertex_shader=vertex_source, fragment_shader=fragment_source)
