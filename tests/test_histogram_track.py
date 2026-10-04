"""HistogramTrack / histogram_bars 测试（几何契约 + CPU 渲染冒烟）。"""

import numpy as np
import pytest

from gpuqviz.render.histogram import histogram_bars


# --------------------------------------------------------------------------- #
# 几何契约（GL/CPU 共享同一函数，测这里即测两端）
# --------------------------------------------------------------------------- #

def _probs_uniform(dim=8):
    return np.full(dim, 1.0 / dim)


def test_bars_topk_selection_and_order():
    """只画 top-k，且按 index 升序（位串序）排列。"""
    p = np.zeros(8)
    p[5] = 0.5
    p[2] = 0.3
    p[7] = 0.2
    bars, base_y = histogram_bars(p, (0, 0, 800, 400), top_k=3)
    assert [b[4] for b in bars] == [2, 5, 7]  # index 升序
    assert [b[5] for b in bars] == ["010", "101", "111"]  # 位串 q2q1q0
    assert [round(b[6], 10) for b in bars] == [0.3, 0.5, 0.2]


def test_bars_absolute_scale():
    """高度为绝对尺度：bh = p × plot_h，无逐帧归一化（跨帧可比）。"""
    p = _probs_uniform()
    bars, _ = histogram_bars(p, (0, 0, 800, 400), top_k=8)
    heights = [b[3] for b in bars]
    plot_h = max(h for h in heights) / 0.125  # 均匀态 p=0.125
    for b in bars:
        assert abs(b[3] - b[6] * plot_h) < 1e-9


def test_bars_bitstring_format():
    p = np.zeros(16)
    p[10] = 1.0
    bars, _ = histogram_bars(p, (0, 0, 800, 400), top_k=4)
    by_idx = {b[4]: b for b in bars}
    assert 10 in by_idx
    assert by_idx[10][5] == "1010"  # 4 qubit 位串 q3q2q1q0
    # 零概率项补位时按 index 升序
    assert [b[4] for b in bars] == sorted(b[4] for b in bars)


def test_bars_baseline_inside_rect():
    p = _probs_uniform()
    rect = (10, 20, 780, 360)
    bars, base_y = histogram_bars(p, rect, top_k=8)
    x, y, w, h = rect
    assert y < base_y <= y + h
    for bx, by, bw, bh, *_ in bars:
        assert x <= bx and bx + bw <= x + w
        assert by >= y - 1e-9  # 柱顶不越界


def test_bars_topk_larger_than_dim():
    """top_k > 维数时取全部。"""
    p = _probs_uniform(4)
    bars, _ = histogram_bars(p, (0, 0, 800, 400), top_k=99)
    assert len(bars) == 4


# --------------------------------------------------------------------------- #
# Scene 集成
# --------------------------------------------------------------------------- #

def test_histogram_track_json_roundtrip(tmp_path):
    from gpuqviz import HistogramTrack, Scene

    scene = Scene(
        width=640, height=360, fps=30, duration=1.0,
        tracks=[HistogramTrack(states_path="s.npz", top_k=6, layout="bottom")],
    )
    p = tmp_path / "scene.json"
    scene.save_json(p)
    from gpuqviz import Scene as Scene2

    scene2 = Scene2.model_validate_json(p.read_text(encoding="utf-8"))
    assert scene2.tracks[0].kind == "histogram"
    assert scene2.tracks[0].top_k == 6
    assert scene2.tracks[0].layout == "bottom"


def test_cpu_render_frame_histogram_png(tmp_path, monkeypatch):
    """CPU 后端 scene 渲染含 HistogramTrack：柱像素存在且随概率变化。"""
    import gpuqviz
    from gpuqviz import HistogramTrack, Scene

    # 直接 patch 后端缓存（不污染进程级 _cached，teardown 自动恢复）
    import gpuqviz.backends as _backends

    monkeypatch.setattr(_backends, "_cached", "cpu")

    # Bell 态两关键帧：|00> → Bell（0.5/0.5）
    states = np.array([
        [1, 0, 0, 0],
        [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
    ], dtype=complex)
    npz = tmp_path / "states.npz"
    np.savez(npz, states=states)

    scene = Scene(
        width=640, height=360, fps=30, duration=1.0,
        tracks=[HistogramTrack(states_path="states.npz", top_k=4,
                               layout="full")],
    )
    out = tmp_path / "hist.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out, scale=1,
                         states_dir=tmp_path)

    from PIL import Image

    arr = np.array(Image.open(out).convert("RGB"))
    blue = (arr[:, :, 2].astype(int) - arr[:, :, 0].astype(int) > 60)
    assert blue.sum() > 200, "CPU 渲染的直方图应有可见柱像素"
    # Bell 末态：两根 0.5 高的柱 → 蓝色像素分布在左右两半
    cols = np.where(blue.any(axis=0))[0]
    assert cols.min() < arr.shape[1] // 3 and cols.max() > 2 * arr.shape[1] // 3
