"""导出交互式 3D 播放器：单文件 HTML（three.js 内联 + 关键帧 payload）。

浏览器端只做插值与绘制（viewer.js 与 interpolate.py 算法同构），
量子计算全部在导出时由 Python 端完成。
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path

import numpy as np

from .adapters import to_key_states, qiskit_to_gates
from .api import RenderConfig, _apply_config
from .evolve import bloch_vectors

# 状态面板/热图复杂度上限（与 DESIGN.md 第 10 节一致）
MAX_QUBITS = 10


def _assets() -> resources.abc.Traversable:
    return resources.files("gpuqviz") / "assets"


def _load_text(ref) -> str:
    return ref.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 电路图元数据构建（供交互式播放器在浏览器端绘制 SVG 电路图 + 与 Bloch 联动）
# --------------------------------------------------------------------------- #

def _gate_label(gate) -> str:
    """Gate → 浏览器端显示标签（含参数格式化与条件标记）。"""
    name = gate.name.upper()
    if name == "UNITARY":
        return "U"
    if name == "BARRIER":
        return "┊"
    if name == "MEASURE":
        clbit = int(gate.params[0]) if gate.params else 0
        return f"M→c{clbit}"
    if name == "RESET":
        return "RST"
    suffix = ""
    if getattr(gate, "condition", None) is not None:
        suffix = f"·c{gate.condition.clbit}=={gate.condition.value}"
    if gate.params:
        parts = []
        for p in gate.params:
            if abs(p) < 1e-10:
                parts.append("0")
            elif abs(p - math.pi) < 1e-10:
                parts.append("π")
            elif abs(p + math.pi) < 1e-10:
                parts.append("-π")
            elif abs(p - math.pi / 2) < 1e-10:
                parts.append("π/2")
            elif abs(p + math.pi / 2) < 1e-10:
                parts.append("-π/2")
            elif abs(p - 2 * math.pi) < 1e-10:
                parts.append("2π")
            elif abs(p * 4 / math.pi - round(p * 4 / math.pi)) < 1e-6:
                num = round(p * 4 / math.pi)
                if num == 1:
                    parts.append("π/4")
                elif num == -1:
                    parts.append("-π/4")
                else:
                    parts.append(f"{num}π/4")
            else:
                parts.append(f"{p:.2f}")
        return f"{name}({','.join(parts)}){suffix}"
    return name + suffix


def _build_circuit_info(circuit, steps: int, duration: float,
                        machine=None) -> dict | None:
    """从 qiskit/pyqpanda 电路提取门级元数据，供浏览器端绘制 SVG 电路图。

    返回 None 表示无法提取（如纯 states 输入、pyqpanda 翻译失败）——
    此时播放器隐藏电路面板，其余功能不受影响。
    """
    try:
        n_qubits, gates = qiskit_to_gates(circuit)
    except Exception:
        return None

    # 过滤 BARRIER，保留可见门
    visible = [g for g in gates if g.name.upper() != "BARRIER"]
    n_ops = len(visible)
    if n_ops == 0:
        return None

    gate_list = []
    for col, g in enumerate(visible):
        gate_list.append({
            "name": g.name.upper(),
            "targets": list(g.targets),
            "controls": list(g.controls),
            "params": [float(p) for p in g.params],
            "label": _gate_label(g),
            "col": col,
        })

    # 每个关键帧对应的"正在执行"的门索引
    # keyframe i → gate index = min(floor(i/(n_keys-1)*n_ops), n_ops-1)
    n_keys = steps
    active_gates = []
    for i in range(n_keys):
        if n_keys <= 1:
            idx = 0
        else:
            idx = min(int(math.floor(i / (n_keys - 1) * n_ops)), n_ops - 1)
        active_gates.append(idx)

    # 每个门的跳转时间（秒），均匀分布在 [0, duration]
    if n_ops == 1:
        gate_times = [0.0]
    else:
        gate_times = [j / (n_ops - 1) * duration for j in range(n_ops)]
    # 确保首项为 0，末项不超过 duration
    gate_times[0] = 0.0
    if gate_times[-1] > duration:
        gate_times[-1] = duration

    return {
        "n_qubits": n_qubits,
        "n_ops": n_ops,
        "gates": gate_list,
        "active_gates": active_gates,
        "gate_times": gate_times,
    }


def build_payload(states: list[np.ndarray], fps: float, duration: float,
                  title: str, colormap: str = "viridis",
                  circuit_info: dict | None = None) -> dict:
    """态矢量关键帧序列 → 播放器 payload（纯 JSON 可序列化 dict）。

    circuit_info 非空时附加 payload["circuit"]，浏览器端据此绘制
    SVG 电路图并与 Bloch 球联动。
    """
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

    payload = {
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
    if circuit_info is not None:
        payload["circuit"] = circuit_info
    return payload


def export_html(circuit=None, states=None, steps: int | None = None,
                fps: float | None = None,
                duration: float | None = None, title: str | None = None,
                out: str | Path | None = None, colormap: str | None = None,
                embed_three: bool = True, machine=None,
                config: "RenderConfig | None" = None) -> Path:
    """qiskit 电路或态矢量序列 → 交互式 3D 播放器 HTML（单文件）。

    embed_three=False 时用 CDN 引用（文件更小，但需要联网打开）。
    """
    p = _apply_config(
        config,
        defaults={"steps": 120, "fps": 60.0, "title": "量子态演化",
                  "out": "out/viewer.html", "colormap": "viridis"},
        steps=steps, fps=fps, title=title, out=out, colormap=colormap,
    )
    steps, fps, title, out, colormap = (p.steps, p.fps, p.title, p.out,
                                        p.colormap)

    if (circuit is None) == (states is None):
        raise ValueError("exactly one of `circuit` or `states` must be provided")

    circuit_info = None
    if circuit is not None:
        # 优先用 evolve_gates + sample_snapshots 路径采样，
        # 使每个关键帧天然对应一个门（门-帧精确对齐，电路图联动无歧义）
        circuit_info = _build_circuit_info(circuit, steps, duration or 0.0,
                                           machine=machine)
        if circuit_info is not None:
            from .circuits import evolve_gates, sample_snapshots
            _, gates = qiskit_to_gates(circuit)
            snapshots = evolve_gates(circuit_info["n_qubits"], gates)
            key_states = sample_snapshots(snapshots, steps)
        else:
            # 回退到原有采样路径（pyqpanda / 翻译失败）
            key_states = to_key_states(circuit, steps=steps, machine=machine)
    else:
        key_states = [np.asarray(getattr(s, "data", s)).reshape(-1) for s in states]

    if duration is None:
        duration = steps / 30.0
        if circuit_info is not None:
            # 重新计算 gate_times（duration 从 None 推断出来了）
            circuit_info = _build_circuit_info(circuit, steps, duration,
                                               machine=machine)

    payload = build_payload(key_states, fps, duration, title, colormap,
                            circuit_info=circuit_info)
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

    # 渲染元信息（0.8.0 审计 #3）：数据来源与环境随产物留存
    from . import __version__
    from .backends import render_info as _render_info

    meta = {"version": __version__, **_render_info()}
    meta_json = json.dumps(meta, ensure_ascii=False)

    html = (template
            .replace("__TITLE__", title)
            .replace("__THREE_JS__", three_block)
            .replace("__VIEWER_JS__", viewer_js)
            .replace("__PAYLOAD__", payload_json)
            .replace("</head>",
                     '<meta name="gpuqviz-render-info" content=\''
                     + meta_json + '\'>\n</head>'))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
