"""测试分辨率控制与文字叠加功能。

覆盖：
- presets.py：预设常量、resolve_preset()
- api._resolve_resolution()：优先级 resolution > width/height > figsize > default
- render_bloch_video/render_heatmap_video/render_frame：分辨率参数透传
- scene.TextOverlay/TextPosition：模型序列化、时间范围过滤、位置解析
- Scene.effective_overlays()：title 向后兼容
- render_bloch_video title/watermark 便捷参数
"""

import warnings

import numpy as np
import pytest

from gpuqviz.api import _resolve_resolution
from gpuqviz.presets import (
    PRESETS, RES_1080P, RES_4K, RES_720P, RES_SQUARE_1080,
    RES_VERTICAL_1080, resolve_preset,
)
from gpuqviz.scene import TextOverlay, TextPosition


# ── presets ──────────────────────────────────────────────

def test_preset_constants():
    assert RES_1080P == (1920, 1080)
    assert RES_4K == (3840, 2160)
    assert RES_720P == (1280, 720)
    assert RES_SQUARE_1080 == (1080, 1080)
    assert RES_VERTICAL_1080 == (1080, 1920)


def test_presets_dict_contains_common_names():
    for name in ("480p", "720p", "1080p", "4k", "square", "vertical"):
        assert name in PRESETS


def test_resolve_preset_case_insensitive():
    assert resolve_preset("4K") == (3840, 2160)
    assert resolve_preset("1080P") == (1920, 1080)


def test_resolve_preset_unknown_raises():
    with pytest.raises(ValueError, match="unknown resolution preset"):
        resolve_preset("999p")


# ── _resolve_resolution 优先级 ───────────────────────────

def test_resolution_str_highest_priority():
    W, H = _resolve_resolution(None, None, "4k", None)
    assert (W, H) == (3840, 2160)


def test_resolution_tuple():
    W, H = _resolve_resolution(None, None, (640, 480), None)
    assert (W, H) == (640, 480)


def test_width_height_pair():
    W, H = _resolve_resolution(1280, 720, None, None)
    assert (W, H) == (1280, 720)


def test_figsize_conversion():
    W, H = _resolve_resolution(None, None, None, (12.8, 7.2))
    assert (W, H) == (1280, 720)


def test_default_when_none():
    W, H = _resolve_resolution(None, None, None, None)
    assert (W, H) == (1920, 1080)


def test_resolution_overrides_width_height_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        W, H = _resolve_resolution(800, 600, "4k", None)
        assert (W, H) == (3840, 2160)
        assert len(w) == 1
        assert "overrides" in str(w[0].message)


def test_resolution_overrides_figsize_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        W, H = _resolve_resolution(None, None, "720p", (12.8, 7.2))
        assert (W, H) == (1280, 720)
        assert len(w) == 1


def test_width_height_overrides_figsize_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        W, H = _resolve_resolution(640, 360, None, (12.8, 7.2))
        assert (W, H) == (640, 360)
        assert len(w) == 1


def test_negative_width_raises():
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_resolution(-1, 720, None, None)


def test_negative_figsize_raises():
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_resolution(None, None, None, (-1, 7.2))


# ── TextOverlay 模型 ─────────────────────────────────────

def test_text_overlay_basic():
    ov = TextOverlay(text="Hello", font_size=36)
    assert ov.text == "Hello"
    assert ov.position == TextPosition.TOP_LEFT
    assert ov.font_size == 36
    assert ov.opacity == 1.0
    assert ov.shadow is False


def test_text_overlay_position_from_string():
    ov = TextOverlay(text="X", position="bottom_center")
    assert ov.position == TextPosition.BOTTOM_CENTER


def test_text_overlay_position_tuple():
    ov = TextOverlay(text="X", position=(100, 200))
    x, y = ov.resolve_pixels(1920, 1080)
    assert (x, y) == (100, 200)


def test_text_overlay_is_active_within_range():
    ov = TextOverlay(text="Subtitle", start=2.0, end=4.0)
    assert not ov.is_active(1.9)
    assert ov.is_active(2.0)
    assert ov.is_active(3.0)
    assert ov.is_active(4.0)
    assert not ov.is_active(4.1)


def test_text_overlay_is_active_no_end():
    ov = TextOverlay(text="Watermark", start=0.0, end=None)
    assert ov.is_active(0.0)
    assert ov.is_active(100.0)


def test_text_overlay_resolve_pixels_all_positions():
    for pos in TextPosition:
        ov = TextOverlay(text="Test", position=pos)
        x, y = ov.resolve_pixels(1920, 1080)
        assert 0 <= x <= 1920
        assert 0 <= y <= 1080


def test_text_overlay_json_roundtrip():
    ov = TextOverlay(text="测试字幕", position=TextPosition.BOTTOM_CENTER,
                     font_size=36, start=0, end=2, opacity=0.8)
    js = ov.model_dump_json()
    ov2 = TextOverlay.model_validate_json(js)
    assert ov2.text == "测试字幕"
    assert ov2.position == TextPosition.BOTTOM_CENTER
    assert ov2.font_size == 36
    assert ov2.opacity == 0.8


def test_text_overlay_max_length():
    with pytest.raises(Exception):
        TextOverlay(text="x" * 300)


# ── Scene.effective_overlays ─────────────────────────────

def test_scene_effective_overlays_from_title():
    from gpuqviz.scene import BlochTrack, Scene
    scene = Scene(duration=2.0, title="My Title",
                  tracks=[BlochTrack(states_path="dummy.npz")])
    overlays = scene.effective_overlays()
    assert len(overlays) == 1
    assert overlays[0].text == "My Title"
    assert overlays[0].position == TextPosition.TOP_LEFT


def test_scene_effective_overlays_empty():
    from gpuqviz.scene import BlochTrack, Scene
    scene = Scene(duration=2.0, title="",
                  tracks=[BlochTrack(states_path="dummy.npz")])
    assert scene.effective_overlays() == []


def test_scene_effective_overlays_prefers_explicit():
    from gpuqviz.scene import BlochTrack, Scene
    custom = TextOverlay(text="Custom", position=TextPosition.BOTTOM_CENTER)
    scene = Scene(duration=2.0, title="Ignored Title",
                  tracks=[BlochTrack(states_path="dummy.npz")],
                  overlays=[custom])
    overlays = scene.effective_overlays()
    assert len(overlays) == 1
    assert overlays[0].text == "Custom"


def test_scene_json_with_overlays():
    from gpuqviz.scene import BlochTrack, Scene
    scene = Scene(
        duration=4.0,
        tracks=[BlochTrack(states_path="dummy.npz")],
        overlays=[
            TextOverlay(text="Step 1", position=TextPosition.BOTTOM_CENTER,
                        start=0, end=2),
            TextOverlay(text="gpuqviz", position=TextPosition.BOTTOM_RIGHT,
                        opacity=0.4, font_size=20),
        ],
    )
    js = scene.model_dump_json()
    scene2 = Scene.model_validate_json(js)
    assert len(scene2.overlays) == 2
    assert scene2.overlays[0].text == "Step 1"
    assert scene2.overlays[1].opacity == 0.4


# ── 渲染集成（CPU 后端，不依赖 GL） ─────────────────────

def _make_states(n=4, frames=6):
    states = []
    for i in range(frames):
        s = np.zeros(n, dtype=complex)
        s[0] = np.cos(0.2 * i)
        s[-1] = np.sin(0.2 * i)
        states.append(s)
    return states


def test_render_bloch_video_resolution_preset(tmp_path):
    import av
    from gpuqviz.api import render_bloch_video
    out = tmp_path / "test_720p.mp4"
    p = render_bloch_video(states=_make_states(), out=out,
                           resolution="720p", backend="cpu",
                           steps=6, seconds=0.2)
    assert p.exists()
    container = av.open(str(p))
    for stream in container.streams.video:
        assert stream.width == 1280
        assert stream.height == 720
    container.close()


def test_render_bloch_video_width_height(tmp_path):
    import av
    from gpuqviz.api import render_bloch_video
    out = tmp_path / "test_wh.mp4"
    p = render_bloch_video(states=_make_states(), out=out,
                           width=640, height=360, backend="cpu",
                           steps=6, seconds=0.2)
    assert p.exists()
    container = av.open(str(p))
    for stream in container.streams.video:
        assert stream.width == 640
        assert stream.height == 360
    container.close()


def test_render_heatmap_video_resolution(tmp_path):
    import av
    from gpuqviz.api import render_heatmap_video
    out = tmp_path / "test_heat.mp4"
    p = render_heatmap_video(states=_make_states(), out=out,
                             resolution="720p", backend="cpu",
                             steps=6, seconds=0.2)
    assert p.exists()
    container = av.open(str(p))
    for stream in container.streams.video:
        assert stream.width == 1280
        assert stream.height == 720
    container.close()


def test_render_frame_resolution(tmp_path):
    from PIL import Image
    from gpuqviz.api import render_frame
    out = tmp_path / "test_frame.png"
    p = render_frame(states=_make_states(), out=out,
                     resolution="720p", scale=1)
    assert p.exists()
    img = Image.open(str(p))
    assert img.size == (1280, 720)


def test_render_bloch_video_title_watermark(tmp_path):
    import av
    import numpy as np
    from gpuqviz.api import render_bloch_video
    out = tmp_path / "test_overlay.mp4"
    p = render_bloch_video(states=_make_states(), out=out,
                           title="Test Title", watermark="gpuqviz",
                           backend="cpu", steps=6, seconds=0.2,
                           resolution="720p")
    assert p.exists()
    # 验证帧中包含文字像素（非背景色）
    container = av.open(str(p))
    frame = next(container.decode(video=0))
    arr = frame.to_ndarray(format="rgb24")
    container.close()
    bg = np.array([11, 14, 20])
    # 标题区域（左上角）
    top_left = arr[30:80, 30:300]
    diff = np.abs(top_left.astype(int) - bg).sum(axis=2)
    assert (diff > 50).sum() > 50, "title text not visible"
    # 水印区域（右下角）
    h, w = arr.shape[:2]
    bottom_right = arr[h-60:h-20, w-300:w-20]
    diff_br = np.abs(bottom_right.astype(int) - bg).sum(axis=2)
    assert (diff_br > 30).sum() > 30, "watermark text not visible"


def test_render_bloch_video_backward_compat_figsize(tmp_path):
    import av
    from gpuqviz.api import render_bloch_video
    out = tmp_path / "test_figsize.mp4"
    p = render_bloch_video(states=_make_states(), out=out,
                           figsize=(6.4, 3.6), backend="cpu",
                           steps=6, seconds=0.2)
    assert p.exists()
    container = av.open(str(p))
    for stream in container.streams.video:
        assert stream.width == 640
        assert stream.height == 360
    container.close()


def test_render_bloch_video_default_resolution(tmp_path):
    """不传任何分辨率参数时默认 1920×1080（向后兼容）。"""
    import av
    from gpuqviz.api import render_bloch_video
    out = tmp_path / "test_default.mp4"
    p = render_bloch_video(states=_make_states(), out=out,
                           backend="cpu", steps=6, seconds=0.2)
    assert p.exists()
    container = av.open(str(p))
    for stream in container.streams.video:
        assert stream.width == 1920
        assert stream.height == 1080
    container.close()
