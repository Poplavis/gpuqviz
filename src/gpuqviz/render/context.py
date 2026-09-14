"""GPU 离屏渲染上下文：moderngl standalone context + 离屏 FBO + 帧读回。

S8：GL 降级链（require=3.3 → 3.2 → CPU），GPUQVIZ_BACKEND=gl 时才尝试 GL。
create_standalone_context 失败时抛 GLUnavailableError，由调用方回退 CPU。
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

try:  # cupy 可选：读回帧以 device ndarray 形式交付
    import cupy as cp
except ImportError:  # pragma: no cover - 无 N 卡环境
    cp = None

try:
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None


class GLUnavailableError(RuntimeError):
    """GL 上下文无法创建（无驱动/无 EGL/版本不满足）。"""


def _try_create_context(gl_version: int = 330) -> "moderngl.Context":
    """尝试创建指定 GL 版本的 standalone context，失败抛异常。"""
    if moderngl is None:
        raise GLUnavailableError("moderngl not installed")
    # moderngl.create_standalone_context(require=gl_version) 在版本不满足时抛异常
    ctx = moderngl.create_standalone_context(require=gl_version)
    return ctx


def create_gl_context(width: int, height: int, fps: float = 60.0,
                      device: int = 0) -> "GLContext":
    """带降级链的 GL 上下文创建：3.3 → 3.2 → 抛 GLUnavailableError。

    每次降级打印一行决策日志，便于诊断环境问题。
    """
    for ver, label in [(330, "3.3"), (320, "3.2")]:
        try:
            ctx = _try_create_context(ver)
            if ver != 330:
                logger.info("GL degraded to %s (3.3 unavailable)", label)
            return GLContext(width, height, fps=fps, device=device, _ctx=ctx)
        except Exception as e:  # noqa: BLE001
            logger.info("GL %s failed: %s", label, e)
    raise GLUnavailableError(
        "no usable GL context (tried 3.3 → 3.2); set GPUQVIZ_BACKEND=cpu"
    )


class GLContext:
    """离屏 GL 渲染上下文。

    用法::

        with GLContext(1920, 1080) as gl:
            gl.program(...)  # 编译着色器
            for t, frame in gl.frame_iterator(480, draw_fn):
                ...  # frame 为 (H, W, 4) uint8，交编码器
    """

    def __init__(self, width: int, height: int, fps: float = 60.0, device: int = 0,
                 _ctx=None):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.device = int(device)

        # _ctx 由 create_gl_context 预创建（带降级链）；直接构造时回退到默认
        if _ctx is not None:
            self.ctx = _ctx
        else:
            self.ctx = moderngl.create_standalone_context()
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

    def program(self, vertex_source: str, fragment_source: str) -> "moderngl.Program":
        return self.ctx.program(vertex_shader=vertex_source, fragment_shader=fragment_source)
