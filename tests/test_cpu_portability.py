"""S8 CPU 可移植性测试：无 GL 环境下完整出片能力。"""

import os

import numpy as np
import pytest

from gpuqviz.backends import detect_backend, resolve_backend
from gpuqviz.backends.cpu import (
    SoftRasterContext,
    SoftRasterBloch,
    SoftRasterHeatmap,
    _NUMBA_OK,
)

qiskit = pytest.importorskip("qiskit")


# -- 后端选择与环境变量 --------------------------------------------------

def test_env_backend_cpu(monkeypatch):
    """GPUQVIZ_BACKEND=cpu 时 detect_backend 强制返回 cpu，跳过 GL 探测。"""
    monkeypatch.setenv("GPUQVIZ_BACKEND", "cpu")
    assert detect_backend(force=True) == "cpu"


def test_env_backend_gl(monkeypatch):
    """GPUQVIZ_BACKEND=gl 时 detect_backend 强制返回 gl（本机有 GL）。"""
    monkeypatch.setenv("GPUQVIZ_BACKEND", "gl")
    assert detect_backend(force=True) == "gl"


def test_env_backend_auto(monkeypatch):
    """GPUQVIZ_BACKEND=auto 时不强制，回退到自动探测。"""
    monkeypatch.setenv("GPUQVIZ_BACKEND", "auto")
    assert detect_backend(force=True) in ("gl", "cpu")


def test_resolve_backend_explicit():
    """resolve_backend("gl"/"cpu") 直接返回，不查环境变量。"""
    assert resolve_backend("gl") == "gl"
    assert resolve_backend("cpu") == "cpu"


# -- CPU 软光栅基元 ------------------------------------------------------

def test_soft_raster_circle_disk():
    """实心圆盘绘制：圆心位置像素被着色。"""
    soft = SoftRasterContext(100, 100)
    soft.clear((0, 0, 0, 1))
    soft.draw_circle_disk(50, 50, 20, (1, 0, 0), 1.0)
    # 圆心应该是红色
    assert soft.frame[50, 50, 0] > 200
    assert soft.frame[50, 50, 1] < 50
    # 圆外应该保持背景色
    assert soft.frame[5, 5, 0] < 50


def test_soft_raster_segment():
    """线段绘制：端点之间被着色。"""
    soft = SoftRasterContext(100, 100)
    soft.clear((0, 0, 0, 1))
    soft.draw_segment((10, 50), (90, 50), (1, 1, 1), 2.0)
    assert soft.frame[50, 50, 0] > 200
    assert soft.frame[10, 10, 0] < 50  # 远离线段


def test_soft_raster_heatmap_lut():
    """热图 LUT 查表：0→LUT[0], 1→LUT[255]。"""
    soft = SoftRasterContext(100, 100)
    soft.clear((0, 0, 0, 1))
    heat = SoftRasterHeatmap(soft, colormap="viridis")
    # 2x1 图：左=0, 右=1
    img = np.array([[0.0, 1.0]], dtype=np.float32)
    heat.draw(img, (0, 0, 100, 50), basis="probability", colorbar=False)
    # 左半应该 ≈ LUT[0]，右半 ≈ LUT[255]
    from gpuqviz.render.heatmap import bake_colormap
    lut = bake_colormap("viridis")
    left = soft.frame[25, 10, :3]
    right = soft.frame[25, 90, :3]
    assert abs(int(left[0]) - int(lut[0, 0])) <= 2
    assert abs(int(right[0]) - int(lut[255, 0])) <= 2


# -- CPU 热图出片 --------------------------------------------------------

def test_render_heatmap_video_cpu(tmp_path):
    """CPU 后端热图出片。"""
    from gpuqviz import render_heatmap_video

    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = render_heatmap_video(circuit=qc, steps=10, fps=10, seconds=1,
                               out=tmp_path / "cpu_heat.mp4", backend="cpu")
    assert out.exists() and out.stat().st_size > 0

    import av

    with av.open(str(out)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == 10  # round(1 * 10)


# -- CPU render_frame PNG ------------------------------------------------

def test_render_frame_cpu_png(tmp_path):
    """CPU 后端 render_frame 出 PNG。"""
    from gpuqviz import render_frame

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    out = render_frame(circuit=qc, t=0.5, scale=1,
                       out=tmp_path / "cpu_frame.png", style="dark")
    assert out.exists() and out.stat().st_size > 1000

    from PIL import Image

    img = Image.open(str(out))
    assert img.size[0] > 0 and img.size[1] > 0


# -- CPU GHZ 多球 --------------------------------------------------------

def test_render_ghz_cpu_cols(tmp_path):
    """CPU 后端 GHZ 3-qubit，cols=1 单列布局。"""
    from gpuqviz import render_bloch_video

    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    out = render_bloch_video(circuit=qc, steps=10, fps=10, seconds=1,
                             out=tmp_path / "cpu_ghz.mp4", backend="cpu", cols=1)
    assert out.exists() and out.stat().st_size > 0

    import av

    with av.open(str(out)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == 10
