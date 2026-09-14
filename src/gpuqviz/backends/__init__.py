"""后端探测与选择：gl（GPU/任意可用 OpenGL 离屏渲染）/ cpu（纯 numpy 软光栅）。

GPUQVIZ_BACKEND 环境变量（auto/gl/cpu）可强制指定后端，优先级高于
detect_backend() 的自动探测和调用方传入的 backend 参数默认值。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_cached: str | None = None


def _env_backend() -> str | None:
    """读 GPUQVIZ_BACKEND 环境变量，返回规范化的后端名或 None。"""
    val = os.environ.get("GPUQVIZ_BACKEND", "").strip().lower()
    if val in ("auto", "gl", "cpu"):
        if val != "auto":
            return val
    return None


def detect_backend(force: bool = False) -> str:
    """探测可用后端。GPUQVIZ_BACKEND=cpu/gl 时强制返回，跳过探测。

    force=True 时清缓存重新探测（环境变量仍优先）。
    """
    global _cached
    env = _env_backend()
    if env is not None:
        if force or _cached != env:
            logger.info("GPUQVIZ_BACKEND=%s → forced backend (skip probe)", env)
        _cached = env
        return _cached
    if _cached is not None and not force:
        return _cached
    try:
        import moderngl

        ctx = moderngl.create_standalone_context()
        ctx.release()
        _cached = "gl"
    except Exception:  # noqa: BLE001
        logger.info("GL standalone context unavailable → CPU backend")
        _cached = "cpu"
    return _cached


def resolve_backend(backend: str = "auto") -> str:
    """把 "auto" 解析为实际后端名（尊重环境变量与探测结果）。"""
    if backend in ("gl", "cpu"):
        return backend
    return detect_backend()
