"""P5.2 渲染桥梁测试：BlochVectorsTrack（MPS 约化分析量 → 渲染层）。"""

import numpy as np
import pytest

from gpuqviz.circuits import Gate
from gpuqviz.mps import evolve_mps
from gpuqviz.render.histogram import _BAR_VS, _BAR_FS  # noqa: F401  (存在性)


# --------------------------------------------------------------------------- #
# Track 契约
# --------------------------------------------------------------------------- #

def test_bloch_vectors_track_json_roundtrip(tmp_path):
    from gpuqviz import BlochVectorsTrack, Scene

    scene = Scene(width=640, height=360, fps=30, duration=1.0,
                  tracks=[BlochVectorsTrack(states_path="b.npz", trail=True,
                                            layout="full")])
    p = tmp_path / "scene.json"
    scene.save_json(p)
    from gpuqviz import Scene as Scene2

    scene2 = Scene2.model_validate_json(p.read_text(encoding="utf-8"))
    assert scene2.tracks[0].kind == "bloch_vectors"
    assert scene2.tracks[0].trail is True


def test_bloch_vectors_track_load_validation(tmp_path):
    from gpuqviz import BlochVectorsTrack

    good = tmp_path / "good.npz"
    np.savez(good, bloch=np.zeros((5, 3, 3)))
    arr = BlochVectorsTrack(states_path="good.npz").load_bloch(tmp_path)
    assert arr.shape == (5, 3, 3)

    bad = tmp_path / "bad.npz"
    np.savez(bad, bloch=np.zeros((5, 3)))  # 缺最后一维
    with pytest.raises(ValueError, match="K, n, 3"):
        BlochVectorsTrack(states_path="bad.npz").load_bloch(tmp_path)


# --------------------------------------------------------------------------- #
# MPS → 渲染桥梁端到端
# --------------------------------------------------------------------------- #

def test_mps_to_bloch_npz_matches_direct(tmp_path):
    """MPS 约化分析量写 npz → 读回与直接调用一致。"""
    gates = [Gate(name="RY", targets=[q], params=[0.3 * q])
             for q in range(4)]
    gates.append(Gate(name="CX", targets=[2], controls=[1]))
    result = evolve_mps(4, gates)
    bloch = result.bloch_keys()
    assert bloch.shape == (len(result.frames), 4, 3)

    npz = tmp_path / "bloch.npz"
    np.savez(npz, bloch=bloch)
    from gpuqviz import BlochVectorsTrack

    loaded = BlochVectorsTrack(states_path="bloch.npz").load_bloch(tmp_path)
    assert np.allclose(loaded, bloch, atol=1e-12)


def test_cpu_render_bloch_vectors_png(tmp_path, monkeypatch):
    """CPU 后端渲染 BlochVectorsTrack 场景（桥梁端到端冒烟）。"""
    import gpuqviz
    import gpuqviz.backends as _backends
    from gpuqviz import BlochVectorsTrack, Scene

    monkeypatch.setattr(_backends, "_cached", "cpu")

    gates = [Gate(name="RY", targets=[q], params=[0.5 * q])
             for q in range(3)]
    result = evolve_mps(3, gates)
    npz = tmp_path / "bloch.npz"
    np.savez(npz, bloch=result.bloch_keys())

    scene = Scene(width=640, height=360, fps=30, duration=1.0,
                  tracks=[BlochVectorsTrack(states_path="bloch.npz",
                                            layout="full")])
    out = tmp_path / "mps_render.png"
    gpuqviz.render_frame(scene=scene, t=0.8, out=out, scale=1,
                         states_dir=tmp_path)

    from PIL import Image

    arr = np.array(Image.open(out).convert("RGB"))
    bg = np.array([11, 14, 20])
    nonbg = (np.abs(arr.astype(int) - bg).sum(axis=2) > 30).sum()
    assert nonbg > 500, "MPS 桥梁渲染应有可见内容"
