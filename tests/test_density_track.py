"""DensityMatrixTrack / hinton_cells 测试（几何契约 + CPU 冒烟）。"""

import numpy as np
import pytest

from gpuqviz.render.density import NEG_COLOR, POS_COLOR, hinton_cells


def _bell_rho(phase: complex = 1.0) -> np.ndarray:
    """(|00⟩ + phase·|11⟩)/√2 的密度矩阵。"""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[3] = phase / np.sqrt(2)
    return np.outer(psi, psi.conj())


# --------------------------------------------------------------------------- #
# 几何契约（GL/CPU 共享）
# --------------------------------------------------------------------------- #

def test_hinton_grid_geometry():
    rho = _bell_rho()
    cells, grid = hinton_cells(rho, (10, 20, 400, 300))
    assert len(cells) == 16  # 4×4
    gx, gy, side, _ = grid
    assert side == 300.0  # 正方形取 min(w,h)
    assert abs(gx - (10 + (400 - 300) / 2)) < 1e-9  # 水平居中


def test_hinton_area_proportional_to_magnitude():
    """Hinton 约定：方块边长 ∝ √(|ρ_ij|/max|ρ|)，最大元素占满格。"""
    psi = np.zeros(4, dtype=complex)
    psi[0] = np.sqrt(0.75)
    psi[3] = -0.5
    rho = np.outer(psi, psi.conj())  # |ρ00|=0.75（最大），|ρ03|=0.25
    cells, (gx, gy, side, _) = hinton_cells(rho, (0, 0, 400, 400))
    cell = side / 4
    by_ij = {(c[4], c[5]): c for c in cells}
    d00 = by_ij[(0, 0)][2]
    d03 = by_ij[(0, 3)][2]
    off01 = by_ij[(0, 1)][2]
    assert abs(d00 - cell) < 1e-9                    # 最大元素占满格
    assert abs(d03 - cell * np.sqrt(abs(psi[0] * psi[3]) / 0.75)) < 1e-9
    assert off01 == 0.0                              # |ρ_01| = 0 → 不画


def test_hinton_color_by_sign():
    """Re ≥ 0 蓝 / Re < 0 橙。"""
    rho = _bell_rho(phase=1.0)     # ρ_03 = +0.5 实数
    cells, _ = hinton_cells(rho, (0, 0, 400, 400))
    by_ij = {(c[4], c[5]): c for c in cells}
    assert by_ij[(0, 3)][3] == POS_COLOR
    assert by_ij[(0, 0)][3] == POS_COLOR

    # (|00⟩ - |11⟩)/√2：ρ_03 = -0.5
    cells2, _ = hinton_cells(_bell_rho(phase=-1.0), (0, 0, 400, 400))
    by_ij2 = {(c[4], c[5]): c for c in cells2}
    assert by_ij2[(0, 3)][3] == NEG_COLOR
    assert by_ij2[(0, 0)][3] == POS_COLOR  # 对角恒正


def test_hinton_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        hinton_cells(np.zeros((4, 5)), (0, 0, 100, 100))
    with pytest.raises(ValueError, match="square"):
        hinton_cells(np.zeros((5, 5)), (0, 0, 100, 100))


# --------------------------------------------------------------------------- #
# Scene 集成
# --------------------------------------------------------------------------- #

def test_density_track_json_roundtrip(tmp_path):
    from gpuqviz import DensityMatrixTrack, Scene

    scene = Scene(width=640, height=360, fps=30, duration=1.0,
                  tracks=[DensityMatrixTrack(states_path="r.npz",
                                             layout="full")])
    p = tmp_path / "scene.json"
    scene.save_json(p)
    from gpuqviz import Scene as Scene2

    scene2 = Scene2.model_validate_json(p.read_text(encoding="utf-8"))
    assert scene2.tracks[0].kind == "density"


def test_cpu_render_frame_density_png(tmp_path, monkeypatch):
    """CPU 后端渲染 Hinton：正负双色方块均可见。"""
    import gpuqviz
    import gpuqviz.backends as _backends
    from gpuqviz import DensityMatrixTrack, Scene

    monkeypatch.setattr(_backends, "_cached", "cpu")

    # 关键帧：|00⟩ → (|00⟩-|11⟩)/√2（负实非对角）
    k0 = np.diag([1, 0, 0, 0]).astype(complex)
    psi = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    k1 = np.outer(psi, psi.conj())
    npz = tmp_path / "rhos.npz"
    np.savez(npz, states=np.stack([k0, k1]))

    scene = Scene(width=640, height=360, fps=30, duration=1.0,
                  tracks=[DensityMatrixTrack(states_path="rhos.npz",
                                             layout="full")])
    out = tmp_path / "hinton.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out, scale=1,
                         states_dir=tmp_path)

    from PIL import Image

    arr = np.array(Image.open(out).convert("RGB")).astype(int)
    # 蓝方块: B 高 R 低；橙方块: R 高 B 低
    blue = ((arr[:, :, 2] - arr[:, :, 0] > 60)).sum()
    orange = ((arr[:, :, 0] - arr[:, :, 2] > 40) &
              (arr[:, :, 0] > 150)).sum()
    assert blue > 100, "应有蓝色（正）方块"
    assert orange > 50, "应有橙色（负）方块"
