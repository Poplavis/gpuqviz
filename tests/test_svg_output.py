"""P2.5/P2.6 测试：SVG 矢量输出 + PDF 转换 + LaTeX 标注。"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gpuqviz.render.histogram import histogram_bars
from gpuqviz.render.svg import SVGContext

NS = "{http://www.w3.org/2000/svg}"


def _parse(svg: SVGContext):
    return ET.fromstring(svg.to_string())


def _bell_states():
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return [np.array([1, 0, 0, 0], dtype=complex), psi]


# --------------------------------------------------------------------------- #
# SVGContext 原语
# --------------------------------------------------------------------------- #

def test_svg_primitives_emit_elements():
    svg = SVGContext(400, 300)
    svg.clear((0.04, 0.055, 0.078, 1.0))
    svg.draw_circle_disk(100, 100, 50, (0.3, 0.77, 0.95), 0.9)
    svg.draw_circle_ring(100, 100, 80, (0.5, 0.5, 0.5), 2.0)
    svg.draw_ellipse_ring(100, 100, 80, 30, (0.5, 0.5, 0.5), 1.0)
    svg.draw_rect_fill(10, 10, 20, 20, (1, 0, 0), 0.5)
    svg.draw_segment((0, 0), (100, 100), (1, 1, 1), 2.0)
    svg.draw_text("q0", (10, 280), 16, (1, 1, 1, 1))
    root = _parse(svg)
    assert root.tag == f"{NS}svg"
    assert root.get("viewBox") == "0 0 400 300"
    tags = [el.tag.replace(NS, "") for el in root.iter() if el.tag != root.tag]
    assert tags.count("circle") == 2   # disk + ring
    assert tags.count("ellipse") == 1
    assert tags.count("rect") == 2     # clear + rect_fill
    assert tags.count("line") == 1
    assert tags.count("text") == 1


def test_svg_text_escapes_xml():
    svg = SVGContext(100, 100)
    svg.draw_text("a<b>&c", (10, 10), 12)
    root = _parse(svg)  # 不抛解析错误 = 转义正确
    assert root.find(f"{NS}text").text == "a<b>&c"


def test_svg_geometry_matches_histogram_bars():
    """直方图 SVG 方块数与共享几何函数一致。"""
    p = np.full(8, 0.125)
    bars, _ = histogram_bars(p, (60, 30, 400, 200), top_k=8)
    svg = SVGContext(500, 300)
    from gpuqviz.render.histogram import draw_histogram_cpu

    draw_histogram_cpu(svg, p, (60, 30, 400, 200), top_k=8)
    root = _parse(svg)
    rects = [el for el in root.iter(f"{NS}rect")
             if float(el.get("width", "0")) > 1
             and el.get("fill") not in (None, "#000000")]
    assert len(rects) >= len(bars)  # 柱 + others（无）/基线


def test_svg_heatmap_embeds_lut_image():
    from gpuqviz.render.heatmap import bake_colormap

    lut = bake_colormap("viridis")
    img = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    svg = SVGContext(200, 200)
    svg.draw_heatmap(img, (10, 10, 180, 180), lut)
    root = _parse(svg)
    images = list(root.iter(f"{NS}image"))
    assert len(images) == 1
    assert images[0].get("href", "").startswith("data:image/png;base64,")


def test_svg_phase_disc_unsupported():
    svg = SVGContext(100, 100)
    with pytest.raises(NotImplementedError, match="phase_disc"):
        svg.draw_phase_disc(np.array([1, 0], dtype=complex), (0, 0, 50, 50))


# --------------------------------------------------------------------------- #
# render_svg 场景
# --------------------------------------------------------------------------- #

def _make_scene(tmp_path):
    from gpuqviz import BlochTrack, HistogramTrack, Scene

    states = []
    for k in range(6):
        t = k / 5
        psi = np.zeros(4, dtype=complex)
        psi[0] = np.cos(t * np.pi / 4)
        psi[3] = np.sin(t * np.pi / 4) * np.exp(1j * t)
        states.append(psi)
    np.savez(tmp_path / "states.npz", states=np.stack(states))
    return Scene(
        width=1280, height=720, fps=30, duration=2.0,
        tracks=[BlochTrack(states_path="states.npz", layout="top"),
                HistogramTrack(states_path="states.npz", top_k=4,
                               show_others=True, layout="bottom")],
    )


def test_render_svg_scene_smoke(tmp_path):
    """Scene（bloch + histogram）→ 合法 SVG，结构完整。"""
    import gpuqviz

    scene = _make_scene(tmp_path)
    out = gpuqviz.render_svg(scene, out=tmp_path / "scene.svg", t=0.8,
                             states_dir=tmp_path)
    assert out.exists() and out.stat().st_size > 1000
    root = ET.parse(out).getroot()
    assert root.tag == f"{NS}svg"
    assert root.get("viewBox") == "0 0 1280 720"
    # 2 个布洛赫球（各 1 球壳圆 + 1 赤道椭圆）+ 4 根概率柱标签
    texts = [el.text for el in root.iter(f"{NS}text")]
    assert any(t in ("00", "01", "10", "11") for t in texts)


def test_render_svg_circuit_bloch_grid(tmp_path):
    """circuit= 输入 → 布洛赫网格 SVG。"""
    import gpuqviz

    qiskit = pytest.importorskip("qiskit")

    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    out = gpuqviz.render_svg(circuit=qc, out=tmp_path / "grid.svg", t=0.6)
    root = ET.parse(out).getroot()
    circles = [el for el in root.iter(f"{NS}circle")]
    assert len(circles) >= 3  # 3 个球的球壳


def test_render_svg_rejects_both_inputs():
    import gpuqviz

    with pytest.raises(ValueError, match="exactly one"):
        gpuqviz.render_svg(circuit=np.zeros(4), states=[np.zeros(4)],
                           out="x.svg")


# --------------------------------------------------------------------------- #
# LaTeX 标注（matplotlib mathtext，可选依赖）
# --------------------------------------------------------------------------- #

matplotlib = pytest.importorskip("matplotlib")


def test_mathtext_to_rgba_shape_and_alpha():
    from gpuqviz.render.mathlabel import mathtext_to_rgba

    rgba = mathtext_to_rgba(r"$\psi = \frac{1}{\sqrt{2}}$", fontsize=30)
    assert rgba.ndim == 3 and rgba.shape[2] == 4
    assert (rgba[:, :, 3] > 0).sum() > 100  # 有内容
    assert rgba.shape[0] < 150  # 紧裁剪：远小于整个 400px 画布
    assert (rgba[0, :, 3] == 0).all()  # 顶边透明（抗锯齿余量 ≤ 数像素）


def test_mathtext_cjk_font_fallback():
    """CJK 文本经字体回退链渲染（无 dummy 方框）。"""
    from gpuqviz.render.mathlabel import mathtext_to_rgba

    rgba = mathtext_to_rgba("量子态", fontsize=24)
    assert rgba.shape[0] > 10 and rgba.shape[1] > 10


def test_composite_rgba_blend():
    from gpuqviz.render.mathlabel import composite_rgba

    frame = np.zeros((50, 50, 4), np.uint8)
    frame[..., :3] = 100
    frame[..., 3] = 255
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = 128  # 半透明白
    composite_rgba(frame, rgba, 5, 5)
    # 混合后中心像素 = (100·0.5 + 255·0.5) ≈ 177
    assert abs(int(frame[10, 10, 0]) - 177) <= 2


def test_overlay_latex_svg_embeds_image(tmp_path):
    """TextOverlay(latex=True) → SVG 嵌入 mathtext 位图。"""
    from gpuqviz import TextOverlay
    from gpuqviz.render.svg import SVGContext

    svg = SVGContext(800, 600)
    svg.clear((0.043, 0.055, 0.078, 1.0))
    from gpuqviz.api import _draw_overlays_cpu

    overlays = [TextOverlay(text=r"$\langle Z \rangle = \cos\theta$",
                            position=(20, 20), font_size=20, latex=True)]
    _draw_overlays_cpu(svg, overlays, 800, 600, 1, 0.0)
    root = _parse(svg)
    assert len(list(root.iter(f"{NS}image"))) == 1


def test_overlay_plain_text_svg_emits_text(tmp_path):
    """非 latex 叠加 → SVG 原生 <text>（矢量文字）。"""
    from gpuqviz import TextOverlay
    from gpuqviz.api import _draw_overlays_cpu
    from gpuqviz.render.svg import SVGContext

    svg = SVGContext(800, 600)
    svg.clear((0, 0, 0, 1))
    overlays = [TextOverlay(text="贝尔态演化", position=(20, 20), font_size=20)]
    _draw_overlays_cpu(svg, overlays, 800, 600, 1, 0.0)
    root = _parse(svg)
    texts = [el.text for el in root.iter(f"{NS}text")]
    assert any("贝尔态演化" in t for t in texts)


# --------------------------------------------------------------------------- #
# PDF（cairosvg 可选依赖）
# --------------------------------------------------------------------------- #

def test_svg_save_pdf(tmp_path):
    pytest.importorskip("cairosvg")
    svg = SVGContext(200, 200)
    svg.clear((0, 0, 0, 1))
    svg.draw_circle_disk(100, 100, 50, (1, 0, 0), 1.0)
    out = svg.save(tmp_path / "fig.pdf")
    assert out.exists() and out.stat().st_size > 500
    # PDF 魔数
    assert out.read_bytes()[:4] == b"%PDF"
