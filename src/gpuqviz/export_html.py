"""导出交互式 3D 播放器：单文件 HTML（three.js 内联 + 关键帧 payload）。

浏览器端只做插值与绘制（viewer.js 与 interpolate.py 算法同构），
量子计算全部在导出时由 Python 端完成。
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from .adapters import to_key_states
from .evolve import bloch_vectors

# 状态面板/热图复杂度上限（与 DESIGN.md 第 10 节一致）
MAX_QUBITS = 10


def _assets() -> resources.abc.Traversable:
    return resources.files("gpuqviz") / "assets"


def _load_text(ref) -> str:
    return ref.read_text(encoding="utf-8")


def build_payload(states: list[np.ndarray], fps: float, duration: float,
                  title: str, colormap: str = "viridis") -> dict:
    """态矢量关键帧序列 → 播放器 payload（纯 JSON 可序列化 dict）。"""
    arrs = [np.asarray(getattr(s, "data", s)).reshape(-1).astype(np.complex128)
            for s in states]
    dims = {a.shape[0] for a in arrs}
    if len(dims) != 1:
        raise ValueError(f"states must share one dimension, got {dims}")
    dim = dims.pop()
    n_qubits = int(round(np.log2(dim)))
    if 2**n_qubits != dim:
        raise ValueError(f"dimension {dim} is not a power of 2")
    if n_qubits > MAX_QUBITS:
        raise ValueError(f"{n_qubits} qubits > {MAX_QUBITS}: state panel grows as 2^n; "
                         "export reduced observables instead")

    bloch = bloch_vectors(arrs, n_qubits=n_qubits)
    if hasattr(bloch, "get"):
        bloch = bloch.get()

    return {
        "meta": {
            "title": title,
            "fps": float(fps),
            "duration": float(duration),
            "n_qubits": n_qubits,
            "n_keys": len(arrs),
            "colormap": colormap,
            "generated_by": "gpuqviz",
        },
        "states_re": [[float(x.real) for x in a] for a in arrs],
        "states_im": [[float(x.imag) for x in a] for a in arrs],
        "bloch": [[[float(c) for c in vec] for vec in frame] for frame in bloch],
    }


def export_html(circuit=None, states=None, steps: int = 120, fps: float = 60.0,
                duration: float | None = None, title: str = "量子态演化",
                out: str | Path = "out/viewer.html", colormap: str = "viridis",
                embed_three: bool = True, machine=None) -> Path:
    """qiskit 电路或态矢量序列 → 交互式 3D 播放器 HTML（单文件）。

    embed_three=False 时用 CDN 引用（文件更小，但需要联网打开）。
    """
    if (circuit is None) == (states is None):
        raise ValueError("exactly one of `circuit` or `states` must be provided")

    if circuit is not None:
        key_states = to_key_states(circuit, steps=steps, machine=machine)
    else:
        key_states = [np.asarray(getattr(s, "data", s)).reshape(-1) for s in states]
    if duration is None:
        duration = steps / 30.0

    payload = build_payload(key_states, fps, duration, title, colormap)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    assets = _assets()
    template = _load_text(assets / "viewer_template.html")
    viewer_js = _load_text(assets / "viewer.js")

    if embed_three:
        three_src = _load_text(assets / "vendor" / "three.min.js")
        if "</script" in three_src:
            raise RuntimeError("three.min.js contains </script>; cannot inline safely")
        three_block = three_src
    else:
        three_block = ('document.write(\'<script src="https://cdn.jsdelivr.net/npm/'
                       'three@0.160.0/build/three.min.js"><\\/script>\');')

    html = (template
            .replace("__TITLE__", title)
            .replace("__THREE_JS__", three_block)
            .replace("__VIEWER_JS__", viewer_js)
            .replace("__PAYLOAD__", payload_json))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
