"""gpuqviz: GPU-accelerated quantum state evolution visualization and video rendering."""

__version__ = "0.1.0"

from .api import render, render_bloch_video, render_frame, render_heatmap_video  # noqa: E401
from .env import report_env  # noqa: E401
from .export_html import export_html  # noqa: E401


def show(*args, **kwargs):
    """Jupyter 交互集成：notebook 中一行代码内嵌 3D 播放器。

    惰性 import jupyter 模块，避免未装 IPython 时 import gpuqviz 失败。
    """
    from .jupyter import show as _show

    return _show(*args, **kwargs)


__all__ = ["__version__", "report_env", "render_bloch_video", "render_heatmap_video",
           "render", "render_frame", "export_html", "show"]
