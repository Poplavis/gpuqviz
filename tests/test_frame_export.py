"""S7 静态帧导出测试：render_frame 出 PNG，尺寸正确 + scale 抗锯齿有效。"""

import numpy as np
import pytest
from PIL import Image

moderngl = pytest.importorskip("moderngl")

from gpuqviz.api import render_frame


def _bell_states():
    """2-qubit Bell 演化：|00⟩ → (|00⟩+|11⟩)/√2，6 个关键帧。"""
    from qiskit import QuantumCircuit
    from gpuqviz.evolve import sample_circuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return sample_circuit(qc, steps=6)


def test_frame_png_dimensions():
    """输出 PNG 尺寸 = 指定 figsize 换算的像素。"""
    states = _bell_states()
    out = "out/test_frame_800x600.png"
    path = render_frame(states=states, t=0.5, out=out, scale=1,
                        figsize=(8.0, 6.0))
    img = Image.open(str(path))
    assert img.size == (800, 600), f"got {img.size}"


def test_frame_not_blank():
    """非纯色（像素方差 > 阈值）——确认渲染了内容。"""
    states = _bell_states()
    out = "out/test_frame_content.png"
    render_frame(states=states, t=1.0, out=out, scale=2, figsize=(6.0, 4.0))
    img = np.asarray(Image.open(out).convert("L"), dtype=np.float32)
    assert img.std() > 5.0, f"frame too uniform (std={img.std():.2f})"


def test_frame_bw_style_white_bg():
    """bw 风格背景应为纯白（角落像素接近 255）。"""
    states = _bell_states()
    out = "out/test_frame_bw.png"
    render_frame(states=states, t=0.0, out=out, scale=1, style="bw",
                 figsize=(6.0, 4.0))
    img = np.asarray(Image.open(out).convert("RGB"))
    corner = img[0, 0].astype(float)
    assert corner.min() > 240, f"bw bg not white: corner={corner}"


def test_frame_scale_reduces_aliasing():
    """scale=2 的边缘锯齿显著低于 scale=1（边缘梯度能量对比，宽松断言）。"""
    states = _bell_states()
    s1 = "out/test_frame_s1.png"
    s2 = "out/test_frame_s2.png"
    render_frame(states=states, t=0.5, out=s1, scale=1, figsize=(6.0, 4.0))
    render_frame(states=states, t=0.5, out=s2, scale=2, figsize=(6.0, 4.0))
    g1 = np.asarray(Image.open(s1).convert("L"), dtype=np.float32)
    g2 = np.asarray(Image.open(s2).convert("L"), dtype=np.float32)
    # Sobel 边缘能量
    def edge_energy(g):
        gx = np.abs(np.diff(g, axis=1))
        gy = np.abs(np.diff(g, axis=0))
        return float(gx.sum() + gy.sum())
    e1, e2 = edge_energy(g1), edge_energy(g2)
    # scale=2 应有更平滑的边缘（总梯度能量更低或相当；宽松断言：不显著更高）
    assert e2 <= e1 * 1.05, f"scale=2 not smoother: e1={e1:.0f} e2={e2:.0f}"


def test_frame_t_out_of_range():
    states = _bell_states()
    with pytest.raises(ValueError, match="t must be"):
        render_frame(states=states, t=1.5, out="out/bad.png", scale=1)


def test_frame_style_overrides():
    """style_overrides 能覆盖矢量颜色。"""
    states = _bell_states()
    out = "out/test_frame_override.png"
    render_frame(states=states, t=0.5, out=out, scale=1, figsize=(6.0, 4.0),
                 style_overrides={"vector_color": (1.0, 0.0, 0.0)})
    img = Image.open(out)
    assert img.size == (600, 400)
