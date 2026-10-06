"""gpuqviz: GPU-accelerated quantum state evolution visualization and video rendering."""

__version__ = "0.8.0"

from .api import (RenderConfig, render, render_bloch_video,  # noqa: E401
                  render_frame, render_heatmap_video, render_svg)
from .env import report_env  # noqa: E401
from .export_html import export_html  # noqa: E401
from .presets import (  # noqa: E401
    PRESETS, RES_1080P, RES_1440P, RES_4K, RES_480P, RES_720P, RES_8K,
    RES_SQUARE_1080, RES_SQUARE_2160, RES_VERTICAL_1080, RES_VERTICAL_720,
)
from .scene import TextOverlay, TextPosition  # noqa: E401
from .scene import (Scene, BlochTrack, BlochVectorsTrack,  # noqa: E401
                    DensityMatrixTrack, EntanglementTrack, HeatmapTrack,
                    HistogramTrack, Camera)
from .state import DensityMatrix, Statevector  # noqa: E401

# 分析层子模块（专业轨道：测量统计 / 度量 / 纠缠，不依赖渲染）
from . import analysis  # noqa: E401
from .pro import ProReport, ProVisualizer  # noqa: E401
from .adapters import load_qasm  # noqa: E401

# 算法库子模块（惰性加载，不依赖 qiskit/pyqpanda）
from . import algorithms  # noqa: E401


def show(*args, **kwargs):
    """Jupyter 交互集成：notebook 中一行代码内嵌 3D 播放器。

    惰性 import jupyter 模块，避免未装 IPython 时 import gpuqviz 失败。
    """
    from .jupyter import show as _show

    return _show(*args, **kwargs)


__all__ = ["__version__", "RenderConfig", "report_env", "render_bloch_video", "render_heatmap_video",
           "render", "render_frame", "export_html", "show", "algorithms", "analysis",
           "Statevector", "DensityMatrix",
           "PRESETS", "RES_480P", "RES_720P", "RES_1080P", "RES_1440P",
           "RES_4K", "RES_8K", "RES_SQUARE_1080", "RES_SQUARE_2160",
           "RES_VERTICAL_720", "RES_VERTICAL_1080",
           "TextOverlay", "TextPosition",
           "Scene", "BlochTrack", "HeatmapTrack", "HistogramTrack",
           "EntanglementTrack", "DensityMatrixTrack", "BlochVectorsTrack",
           "Camera", "ProVisualizer", "ProReport", "load_qasm", "render_svg"]
