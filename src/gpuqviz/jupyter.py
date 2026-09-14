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

from .export_html import build_payload, export_html
from .adapters import to_key_states

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
        key_states = to_key_states(circuit, steps=steps, machine=machine)
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
                            title=title, colormap=colormap)

    # payload 降级
    if _payload_size(payload) > _PAYLOAD_LIMIT:
        warnings.warn(
            f"payload exceeds {_PAYLOAD_LIMIT // 1024 // 1024}MB; "
            "stripping state panel (Bloch vectors only)",
            RuntimeWarning, stacklevel=2,
        )
        payload = _strip_state_panel(payload)

    if _is_notebook():
        if out is not None:
            # 显式指定 out：先写文件，再内嵌显示（双输出）
            _write_html_file(payload, title, out, embed_three=True)
        return _display_html(payload, title, height)
    # 非 notebook：写文件
    return _write_html_file(payload, title, out, embed_three=True)


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
