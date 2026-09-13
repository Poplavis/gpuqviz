"""后端探测与选择：gl（GPU/任意可用 OpenGL 离屏渲染）/ cpu（纯 numpy 软光栅）。"""

from __future__ import annotations

_cached: str | None = None


def detect_backend(force: bool = False) -> str:
    global _cached
    if _cached is not None and not force:
        return _cached
    try:
        import moderngl

        ctx = moderngl.create_standalone_context()
        ctx.release()
        _cached = "gl"
    except Exception:  # noqa: BLE001
        _cached = "cpu"
    return _cached
