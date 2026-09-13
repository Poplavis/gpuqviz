"""S3 测试：NV12 转换、热图计算、NVENC 探测、热图视频端到端。"""

import numpy as np
import pytest

from gpuqviz.encode import nvenc_available
from gpuqviz.render.heatmap import bake_colormap, state_to_image


def test_nvenc_probe_no_exception():
    result = nvenc_available()
    assert isinstance(result, bool)


def test_bake_colormap():
    cmap = bake_colormap("viridis")
    assert cmap.shape == (256, 3) and cmap.dtype == np.uint8
    assert cmap[0][0] == 68 and abs(int(cmap[0][2]) - 84) <= 1  # viridis 起点 #440154
    assert tuple(cmap[-1]) >= (240, 230, 30)      # 尾部亮黄
    with pytest.raises(KeyError):
        bake_colormap("nope")


def test_state_to_image_probability():
    # Bell 态：小端序非零振幅在 index 0 与 3
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    img = state_to_image(psi, basis="probability")
    assert img.shape == (2, 2)
    assert img[0, 0] == pytest.approx(0.5)
    assert img[1, 1] == pytest.approx(0.5)
    assert img[0, 1] == 0 and img[1, 0] == 0


def test_state_to_image_phase_range():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    img = state_to_image(psi, basis="phase")
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_state_to_image_gpu_matches_cpu():
    cp = pytest.importorskip("cupy")
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    cpu = state_to_image(psi, "probability")
    gpu = state_to_image(cp.asarray(psi), "probability")
    assert np.allclose(cpu, gpu.get())


@pytest.mark.skipif(not nvenc_available(), reason="NVENC session unavailable")
def test_nv12_kernel_known_colors():
    from gpuqviz.render.colorconvert import rgba_to_nv12

    img = np.zeros((4, 4, 4), dtype=np.uint8)
    img[..., 0], img[..., 3] = 255, 255  # 纯红
    nv12 = rgba_to_nv12(__import__("cupy").asarray(img))
    y = nv12[:16].get().reshape(4, 4)
    uv = nv12[16:].get().reshape(2, 2, 2)
    assert y[0, 0] == pytest.approx(62, abs=2)      # BT.709 limited: 16+219*0.2126
    assert uv[0, 0, 0] == pytest.approx(102, abs=3)  # Cb
    assert uv[0, 0, 1] == pytest.approx(240, abs=2)  # Cr


def test_heatmap_video_e2e(tmp_path):
    pytest.importorskip("moderngl")
    from gpuqviz import render_heatmap_video

    states = [np.array([1, 0, 0, 0], dtype=complex),
              np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)]
    out = render_heatmap_video(states=states, fps=15, seconds=0.5,
                               out=tmp_path / "hm.mp4")
    assert out.exists() and out.stat().st_size > 0

    import av

    with av.open(str(out)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == 8  # round(0.5 * 15) = 8
