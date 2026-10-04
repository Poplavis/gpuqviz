"""Jupyter 交互集成：一行代码在 notebook 中内嵌 3D 播放器。

用法::

    from qiskit import QuantumCircuit
    import gpuqviz

    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
    gpuqviz.show(qc)          # notebook 中直接出现交互播放器

非 notebook 环境（终端 / 脚本）自动回退为写 HTML 文件并打印路径。
payload 超 8MB 时自动降级为只内嵌 Bloch 数据（热图状态面板省略），
保持内嵌体积可控。
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np

from .export_html import build_payload, export_html, _build_circuit_info
from .adapters import to_key_states, qiskit_to_gates

logger = logging.getLogger(__name__)

# payload 内嵌体积上限（bytes，约 8MB —— 10 qubit 热图 full states 的经验阈值）
_PAYLOAD_LIMIT = 8 * 1024 * 1024


def _is_notebook() -> bool:
    """检测当前是否运行在 Jupyter notebook / zmqshell 内核中。"""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    # zmqshellInteractiveShell = notebook/qtconsole；TerminalInteractiveShell = 终端
    cls_name = type(ip).__name__
    return "ZMQ" in cls_name or "google" in cls_name.lower()


def _payload_size(payload: dict) -> int:
    """估算 payload 序列化后的字节数。"""
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _strip_state_panel(payload: dict) -> dict:
    """降级：移除 states_re/im（热图状态面板数据），只保留 Bloch 向量。

    Bloch 向量每帧仅 n×3 个 float，体积远小于 2^n 复数态矢量。
    viewer.js 端检测 states_re 为 null 时隐藏状态面板。
    """
    payload = dict(payload)
    payload["states_re"] = None
    payload["states_im"] = None
    payload["meta"]["state_panel_stripped"] = True
    return payload


def _build_html_string(payload: dict, title: str, embed_three: bool = True) -> str:
    """构建自包含 HTML 字符串（复用 export_html 的模板和 assets）。"""
    from importlib import resources

    assets = resources.files("gpuqviz") / "assets"
    template = (assets / "viewer_template.html").read_text(encoding="utf-8")
    viewer_js = (assets / "viewer.js").read_text(encoding="utf-8")

    if embed_three:
        three_src = (assets / "vendor" / "three.min.js").read_text(encoding="utf-8")
        if "</script" in three_src:
            raise RuntimeError("three.min.js contains </script>; cannot inline safely")
        three_block = three_src
    else:
        three_block = (
            'document.write(\'<script src="https://cdn.jsdelivr.net/npm/'
            'three@0.160.0/build/three.min.js"><\\/script>\');'
        )

    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = (template
            .replace("__TITLE__", title)
            .replace("__THREE_JS__", three_block)
            .replace("__VIEWER_JS__", viewer_js)
            .replace("__PAYLOAD__", payload_json))
    return html


def show(circuit=None, states=None, scene=None, steps: int = 60,
         out: str | Path | None = None, as_video: bool = False,
         height: int = 520, **kwargs) -> object | None:
    """在 Jupyter notebook 中内嵌交互式 3D 播放器。

    circuit / states / scene 三选一：
      - circuit: qiskit QuantumCircuit（按 steps 采样关键帧）
      - states: 态矢量序列 list[np.ndarray]
      - scene: gpuqviz.Scene（用其第一个 track 的 states）

    notebook 环境：通过 IPython.display.HTML 以 iframe srcdoc 内嵌自包含 HTML
    （three.js 内联，断网可用），height 可调。

    非 notebook 环境：回退为写 HTML 文件并打印路径（与 CLI export 行为一致）。

    payload 超 8MB 时自动降级：只内嵌 Bloch 数据并警告（状态面板省略）。

    as_video=True：先渲染 MP4 再用 IPython.display.Video 内嵌。

    其余 kwargs 透传给 export_html / render_bloch_video（fps, duration, title,
    colormap, style, codec, quality, backend 等）。
    """
    # ---- 解析输入 → key_states ----
    circuit_info = None
    if scene is not None:
        base_dir = Path(kwargs.get("states_dir", "."))
        scene.validate_states(base_dir=base_dir)
        track0 = scene.tracks[0]
        key_states = track0.load_states(base_dir=base_dir)
        key_states = list(key_states)
        title = scene.title or kwargs.get("title", "量子态演化")
        fps = scene.fps
        duration = scene.duration
    elif circuit is not None:
        machine = kwargs.get("machine")
        # 优先用 evolve_gates + sample_snapshots 路径采样（门-帧精确对齐）
        circuit_info = _build_circuit_info(circuit, steps,
                                           kwargs.get("duration", steps / 30.0),
                                           machine=machine)
        if circuit_info is not None:
            from .circuits import evolve_gates, sample_snapshots
            _, gates = qiskit_to_gates(circuit)
            snapshots = evolve_gates(circuit_info["n_qubits"], gates)
            key_states = sample_snapshots(snapshots, steps)
        else:
            key_states = to_key_states(circuit, steps=steps, machine=machine)
            circuit_info = None
        title = kwargs.get("title", "量子态演化")
        fps = kwargs.get("fps", 60.0)
        duration = kwargs.get("duration", steps / 30.0)
    elif states is not None:
        key_states = [np.asarray(getattr(s, "data", s)).reshape(-1) for s in states]
        title = kwargs.get("title", "量子态演化")
        fps = kwargs.get("fps", 60.0)
        duration = kwargs.get("duration", steps / 30.0)
    else:
        raise ValueError("provide exactly one of circuit/states/scene")

    # ---- as_video 路径：渲染 MP4 + IPython.display.Video ----
    if as_video:
        return _show_video(circuit, states, scene, steps, out, height, kwargs)

    # ---- HTML 内嵌路径 ----
    colormap = kwargs.get("colormap", "viridis")
    payload = build_payload(key_states, fps=fps, duration=duration,
                            title=title, colormap=colormap,
                            circuit_info=circuit_info)

    # payload 降级
    if _payload_size(payload) > _PAYLOAD_LIMIT:
        warnings.warn(
            f"payload exceeds {_PAYLOAD_LIMIT // 1024 // 1024}MB; "
            "stripping state panel (Bloch vectors only)",
            RuntimeWarning, stacklevel=2,
        )
        payload = _strip_state_panel(payload)

    html_str = _build_html_string(payload, title, embed_three=True)

    # 默认落盘路径（.save() 无参时沿用）
    default_out = out if out is not None else Path("out/viewer.html")
    default_out = Path(default_out)
    default_out.parent.mkdir(parents=True, exist_ok=True)
    default_out.write_text(html_str, encoding="utf-8")

    frame_kwargs = {"circuit": circuit, "states": states, "scene": scene,
                    "states_dir": kwargs.get("states_dir")}
    frame_kwargs = {k: v for k, v in frame_kwargs.items() if v is not None}
    handle = ViewerHandle(default_out, html_str, title, height, frame_kwargs)

    if _is_notebook():
        try:
            from IPython.display import HTML, display

            display(HTML(handle._repr_html_()))  # 保持"一行出片即显示"
        except ImportError:
            pass
        return handle
    # 非 notebook：返回 Path 兼容的句柄（文件已写好，打印提示）
    print(f"viewer HTML written to {default_out} (open in browser)")
    return handle


class ViewerHandle(type(Path())):
    """show() 的返回对象（P4.3 Jupyter 双轨）。

    - Path 兼容：非 notebook 下 .exists()/open() 等照常工作；
    - notebook 富显示：作为单元格末表达式自动内嵌播放器（_repr_html_）；
    - ``.figure``：当前帧的 matplotlib Figure（出版管线互操作）；
    - ``.widget()``：ipywidgets 播放控件（需 ipywidgets，可选依赖）；
    - ``.save(out)``：另存 HTML。
    """

    def __new__(cls, path, html_str, title, height, frame_kwargs):
        return super().__new__(cls, str(path))

    def __init__(self, path, html_str, title, height, frame_kwargs):
        self._html_str = html_str
        self._title = title
        self._height = height
        self._frame_kwargs = dict(frame_kwargs)
        self._frame_t = 0.5
        self.fig = None

    # ---- notebook 富显示 ------------------------------------------------ #

    def _repr_html_(self) -> str:
        import html as html_module

        srcdoc = html_module.escape(self._html_str, quote=True)
        return (
            f'<iframe srcdoc="{srcdoc}" width="100%" height="{self._height}" '
            f'style="border:1px solid #232a38;border-radius:8px;" '
            f'allowfullscreen></iframe>'
        )

    def display(self, height: int | None = None):
        """手动内嵌显示（等价于 show() 在 notebook 中的旧行为）。"""
        try:
            from IPython.display import HTML, display
        except ImportError as e:  # pragma: no cover
            raise ImportError("IPython required for display") from e
        h = height or self._height
        display(HTML(self._repr_html_().replace(str(self._height), str(h), 1)))
        return None

    # ---- matplotlib 互操作 ---------------------------------------------- #

    @property
    def figure(self):
        """当前帧的 matplotlib Figure（PNG 帧渲染 → imshow）。

        首次访问后缓存在 ``.fig``；时间点由 ``.frame_t``（0..1 占比）控制。
        """
        if getattr(self, "fig", None) is not None:
            return self.fig
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for .figure; "
                              "install with `pip install matplotlib`") from e

        import tempfile

        from PIL import Image

        from .api import render_frame

        with tempfile.TemporaryDirectory() as td:
            png = render_frame(out=str(Path(td) / "frame.png"),
                               t=self._frame_t, scale=1,
                               **self._frame_kwargs)
            with Image.open(png) as im:
                img = im.copy()  # 释放文件句柄（Windows 临时目录清理需要）

        fig = plt.figure(figsize=(img.width / 100, img.height / 100))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(img)
        plt.close(fig)  # 防止重复显示；fig 对象仍可用
        self.fig = fig
        return fig

    # ---- ipywidgets 控件 ------------------------------------------------- #

    def widget(self, fps: float = 8.0):
        """ipywidgets 播放控件（拖动时间轴逐帧渲染，需 ipywidgets）。"""
        try:
            import ipywidgets as wgt
        except ImportError as e:  # pragma: no cover
            raise ImportError("ipywidgets required for .widget(); "
                              "install with `pip install ipywidgets`") from e
        from IPython.display import display

        duration = float(self._frame_kwargs.get("duration",
                                                self._frame_kwargs.get("seconds",
                                                                       4.0)))
        n_frames = max(2, int(duration * fps))

        img_widget = wgt.Image(format="png")

        def render_to_png(t01: float) -> bytes:
            import io
            import tempfile

            from PIL import Image

            from .api import render_frame

            with tempfile.TemporaryDirectory() as td:
                png = render_frame(out=str(Path(td) / "f.png"), t=t01, scale=1,
                                   **self._frame_kwargs)
                buf = io.BytesIO()
                Image.open(png).save(buf, format="PNG")
                return buf.getvalue()

        def on_value(change):
            img_widget.value = render_to_png(change["new"] / (n_frames - 1))

        play = wgt.Play(value=0, min=0, max=n_frames - 1, interval=1000 / fps,
                        description="播放")
        slider = wgt.IntSlider(value=0, min=0, max=n_frames - 1, step=1,
                               description="t")
        wgt.jslink((play, "value"), (slider, "value"))
        slider.observe(on_value, names="value")
        img_widget.value = render_to_png(0.0)
        box = wgt.VBox([wgt.HBox([play, slider]), img_widget])
        display(box)
        return box

    # ---- 落盘 ------------------------------------------------------------ #

    def save(self, out: str | Path | None = None) -> Path:
        """另存 HTML（默认沿用 show() 时的路径）。"""
        target = Path(out) if out is not None else Path(self)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._html_str, encoding="utf-8")
        print(f"viewer HTML written to {target} (open in browser)")
        return target


def _display_html(payload: dict, title: str, height: int):
    """notebook 内：IPython.display.HTML 以 iframe srcdoc 内嵌。"""
    try:
        from IPython.display import HTML, display
    except ImportError as e:
        raise ImportError(
            "IPython is required for notebook display; "
            "install with `pip install ipython`"
        ) from e

    html_str = _build_html_string(payload, title, embed_three=True)
    # iframe srcdoc 内嵌：避免 notebook 全局 CSS 污染
    import html as html_module

    srcdoc = html_module.escape(html_str, quote=True)
    iframe = (
        f'<iframe srcdoc="{srcdoc}" width="100%" height="{height}" '
        f'style="border:1px solid #232a38;border-radius:8px;" '
        f'allowfullscreen></iframe>'
    )
    display(HTML(iframe))
    return None


def _write_html_file(payload: dict, title: str, out: str | Path | None,
                     embed_three: bool = True) -> Path:
    """非 notebook：写 HTML 文件并打印路径。"""
    if out is None:
        out = "out/viewer.html"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    html_str = _build_html_string(payload, title, embed_three=embed_three)
    out.write_text(html_str, encoding="utf-8")
    print(f"viewer HTML written to {out} (open in browser)")
    return out


def _show_video(circuit, states, scene, steps, out, height, kwargs):
    """as_video=True 路径：渲染 MP4 + IPython.display.Video 内嵌。"""
    try:
        from IPython.display import Video, display
    except ImportError as e:
        raise ImportError(
            "IPython is required for notebook video display; "
            "install with `pip install ipython`"
        ) from e

    if out is None:
        out = "out/show.mp4"
    out = Path(out)

    if scene is not None:
        from .api import render as _render
        _render(scene, out=out, states_dir=kwargs.get("states_dir", "."))
    elif circuit is not None or states is not None:
        from .api import render_bloch_video
        render_bloch_video(
            circuit=circuit, states=states,
            steps=steps,
            fps=kwargs.get("fps", 60.0),
            out=out,
            style=kwargs.get("style", "dark"),
            codec=kwargs.get("codec", "h264"),
            quality=kwargs.get("quality", 0.9),
            seconds=kwargs.get("duration"),
            backend=kwargs.get("backend", "auto"),
        )

    if _is_notebook():
        display(Video(str(out), embed=True, height=height))
        return None
    print(f"video rendered to {out}")
    return out
