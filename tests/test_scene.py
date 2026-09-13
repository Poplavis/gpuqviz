"""S4 测试：Scene 模型序列化、相机插值、文字图集元数据。"""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from gpuqviz.scene import BlochTrack, Camera, HeatmapTrack, Scene, _hex_to_rgba


def _make_npz(tmp_path):
    states = np.zeros((5, 4), dtype=np.complex64)
    states[0, 0] = 1
    p = tmp_path / "states.npz"
    np.savez(p, states=states)
    return p


def test_scene_roundtrip(tmp_path):
    _make_npz(tmp_path)
    scene = Scene(
        duration=2.0,
        title="测试",
        tracks=[BlochTrack(states_path="states.npz", trail=True, layout="top")],
    )
    js = scene.model_dump_json()
    scene2 = Scene.model_validate_json(js)
    assert scene2.tracks[0].kind == "bloch"
    assert scene2.tracks[0].trail is True
    # 从文件读写
    p = tmp_path / "scene.json"
    scene.save_json(p)
    loaded = Scene.load_json(p)
    assert loaded.title == "测试"
    assert loaded.tracks[0].load_states(base_dir=tmp_path).shape == (5, 4)


def test_scene_rejects_bad_time_window(tmp_path):
    _make_npz(tmp_path)
    with pytest.raises(ValidationError):
        Scene(duration=1.0, tracks=[BlochTrack(states_path="states.npz",
                                               start=0.8, end=0.2)])


def test_scene_states_consistency(tmp_path):
    _make_npz(tmp_path)
    np.savez(tmp_path / "other.npz", states=np.zeros((9, 4), np.complex64))
    scene = Scene(duration=1.0, tracks=[
        BlochTrack(states_path="states.npz"),
        HeatmapTrack(states_path="other.npz"),
    ])
    with pytest.raises(ValueError, match="keyframe counts"):
        scene.validate_states(base_dir=tmp_path)


def test_camera_eye_at():
    cam = Camera(azimuth=(0.0, 90.0), elevation=(0.0, 0.0), zoom=1.0)
    e0 = cam.eye_at(0.0, 10.0)
    e1 = cam.eye_at(1.0, 10.0)
    assert e0[1] < -9.5          # t=0 在 -y 方向
    assert abs(e1[0] - 10.0) < 1e-6  # t=1 在 +x 方向
    assert abs(np.linalg.norm(cam.eye_at(0.5, 10.0)) - 10.0) < 1e-9


def test_hex_to_rgba():
    assert _hex_to_rgba("#0b0e14") == (11 / 255, 14 / 255, 20 / 255, 1.0)
    with pytest.raises(ValueError):
        _hex_to_rgba("#12345")


def test_font_atlas_metadata():
    """图集元数据完整性（生成产物随仓库提交）。"""
    from importlib import resources

    meta = json.loads(
        (resources.files("gpuqviz") / "assets" / "font_msyh_64.json").read_text()
    )
    assert meta["atlas_w"] > 0 and meta["atlas_h"] > 0
    for ch in ["量", "子", "贝", "尔", "A", "9"]:
        g = meta["glyphs"][ch]
        assert g["w"] > 0 and g["advance"] > 0
