"""0.8.0 可观测性与统一配置测试（审计 #3/#4/#7/#8）。

覆盖：RenderConfig 合并优先级（显式平铺 > config > 内置默认）、
show() mode 参数、State.provenance、后端降级告警、编码器信息记录。
"""

from pathlib import Path

import numpy as np
import pytest

from gpuqviz import (
    DensityMatrix,
    RenderConfig,
    Statevector,
    export_html,
    render_bloch_video,
    render_frame,
)
from gpuqviz.algorithms import bell


# --------------------------------------------------------------------------- #
# RenderConfig 合并优先级
# --------------------------------------------------------------------------- #

def _frame_with(tmp_path, **kwargs):
    qc = bell()
    kwargs.setdefault("out", tmp_path / "f.png")
    return render_frame(circuit=qc, **kwargs)


def test_config_only_uses_config_values(tmp_path):
    """config 单独提供时其值生效。"""
    cfg = RenderConfig(width=640, height=480)
    out = _frame_with(tmp_path, config=cfg)
    from PIL import Image

    assert Image.open(out).size == (640, 480)


def test_flat_kwargs_override_config(tmp_path):
    """显式平铺参数优先于 config 同名字段。"""
    cfg = RenderConfig(width=640, height=480)
    out = _frame_with(tmp_path, config=cfg, width=320, height=240)
    from PIL import Image

    assert Image.open(out).size == (320, 240)


def test_flat_none_falls_back_to_defaults(tmp_path):
    """平铺传 None 等同于未传——不污染内置默认（回归：合并顺序 bug）。"""
    out = _frame_with(tmp_path, style=None, width=None)  # 全 None
    assert Path(out).exists()


def test_config_fields_fill_unspecified(tmp_path):
    """config 只指定部分字段时，其余字段走入口内置默认。"""
    cfg = RenderConfig(width=640, height=480)  # 未指定 style
    out = _frame_with(tmp_path, config=cfg)  # style 默认 dark 仍生效
    assert Path(out).exists()


def test_render_config_specified():
    cfg = RenderConfig(fps=30, title="t")
    assert cfg.specified() == {"fps": 30, "title": "t"}
    assert RenderConfig().specified() == {}


def test_video_config_precedence(tmp_path):
    """视频入口：config 提供 out/backend，平铺 steps 优先。"""
    qc = bell()
    out = tmp_path / "v.mp4"
    p = render_bloch_video(circuit=qc, steps=6,
                           config=RenderConfig(fps=10, backend="cpu",
                                               out=out))
    assert p == out and p.exists()


# --------------------------------------------------------------------------- #
# show() mode 参数
# --------------------------------------------------------------------------- #

def test_show_mode_validation():
    from gpuqviz import show

    with pytest.raises(ValueError, match="mode"):
        show(circuit=bell(), mode="bogus")


def test_show_mode_figure(tmp_path, monkeypatch):
    """mode="figure" 渲染静态 PNG 并返回 Path。"""
    import gpuqviz.jupyter as jup

    monkeypatch.setattr(jup, "_is_notebook", lambda: False)
    out = tmp_path / "fig.png"
    r = jup.show(circuit=bell(), steps=6, mode="figure", out=out)
    assert Path(r).exists()


# --------------------------------------------------------------------------- #
# State.provenance（审计 #4：近似语义标注）
# --------------------------------------------------------------------------- #

def test_state_provenance_default_exact():
    sv = Statevector.from_int(0, 2)
    assert sv.provenance == "exact"
    rho = DensityMatrix.maximally_mixed(2)
    assert rho.provenance == "exact"
    d = sv.to_dict()
    assert d["provenance"] == "exact"


def test_state_provenance_custom():
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1.0
    sv = Statevector(psi, provenance="mps_chi=4")
    assert sv.provenance == "mps_chi=4"
    assert sv.to_dict()["provenance"] == "mps_chi=4"


def test_mps_provenance_property():
    from gpuqviz.mps import MPS

    mps = MPS.product(3)
    assert mps.provenance.startswith("mps_chi=")


# --------------------------------------------------------------------------- #
# 后端降级可观测性（审计 #3）
# --------------------------------------------------------------------------- #

def test_detect_backend_warns_on_cpu_fallback(monkeypatch):
    """GL 探测失败降级 CPU 时发出 RuntimeWarning。"""
    import gpuqviz.backends as B

    monkeypatch.setattr(B, "_cached", None)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "moderngl":
            raise ImportError("no moderngl")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.warns(RuntimeWarning, match="CPU"):
        backend = B.detect_backend(force=True)
    assert backend == "cpu"


def test_render_info_returns_backend():
    from gpuqviz.backends import render_info

    info = render_info()
    assert info["backend"] in ("gl", "cpu")


def test_last_encoder_info_recorded(tmp_path):
    """create_encoder 记录实际编码器名（NVENC→libx264 回退可见）。"""

    from gpuqviz.encode import create_encoder, last_encoder_info

    out = tmp_path / "enc.mp4"
    with create_encoder(64, 64, 10, out, codec="h264") as enc:
        frame = np.zeros((64, 64, 4), dtype=np.uint8)
        frame[..., 3] = 255
        enc.write(frame)
        info = last_encoder_info()
        assert info is not None
        assert info["requested_codec"] == "h264"
        assert info["encoder"] == enc.encoder_name


def test_export_html_embeds_render_meta(tmp_path):
    """HTML 产物 head 注入 gpuqviz-render-info（版本 + 后端）。"""
    import gpuqviz

    out = export_html(circuit=bell(), steps=6, out=tmp_path / "v.html")
    html = out.read_text(encoding="utf-8")
    assert "gpuqviz-render-info" in html
    assert gpuqviz.__version__ in html


# --------------------------------------------------------------------------- #
# ProReport 可观测字段
# --------------------------------------------------------------------------- #

def test_pro_report_fields():
    from gpuqviz import ProVisualizer

    rep = ProVisualizer(circuit=bell()).report()
    assert rep.provenance == "exact"
    assert rep.backend is None and rep.encoder is None
    d = rep.to_dict()
    assert d["provenance"] == "exact"
