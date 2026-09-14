"""S7 布局系统测试：cols/figsize/style 覆盖语义。"""

import numpy as np
import pytest

from gpuqviz.api import STYLES, _figsize_to_pixels, _grid_layout, _resolve_style


def test_styles_presets_complete():
    """所有预设都含渲染必填键。"""
    required = {"background", "axis_color", "ring_color", "sphere_color",
                "sphere_alpha", "vector_color", "trail_color", "trail_alpha"}
    for name, theme in STYLES.items():
        assert required <= theme.keys(), f"{name} missing keys: {required - theme.keys()}"


def test_resolve_style_string():
    theme = _resolve_style("bw")
    assert theme["background"][:3] == (1.0, 1.0, 1.0)  # 纯白


def test_resolve_style_overrides_merge():
    theme = _resolve_style("dark", {"vector_color": (1.0, 0.0, 0.0)})
    assert theme["vector_color"] == (1.0, 0.0, 0.0)
    # 未覆盖的键保留 dark 预设
    assert theme["background"] == STYLES["dark"]["background"]


def test_resolve_style_dict_input():
    """直接传 dict：以 dark 为底 merge。"""
    theme = _resolve_style({"background": (0.5, 0.5, 0.5, 1.0)})
    assert theme["background"] == (0.5, 0.5, 0.5, 1.0)
    assert "vector_color" in theme  # 来自 dark 底


def test_resolve_style_unknown_name():
    with pytest.raises(ValueError, match="unknown style"):
        _resolve_style("nonexistent")


def test_figsize_to_pixels():
    assert _figsize_to_pixels((8, 6), 1920, 1080) == (800, 600)
    assert _figsize_to_pixels(None, 1920, 1080) == (1920, 1080)
    # 300dpi 等效：10 英寸 → 3000 像素由调用方自行用 figsize=(10, ...) 指定


def test_grid_layout_single_row():
    """cols=None 时单行；centers 在 x 轴上等距分布。"""
    centers, spacing, radius, cam_dist = _grid_layout(3, None, 1920, 1080)
    assert len(centers) == 3
    # 单行：所有 y=0，z 也应为 0（只有一行）
    assert all(abs(c[1]) < 1e-9 for c in centers)
    assert all(abs(c[2]) < 1e-9 for c in centers)
    # 居中：中间球的 x ≈ 0
    assert abs(centers[1][0]) < 1e-9


def test_grid_layout_multi_row():
    """cols=2 四球 → 2 行 2 列，包围盒不重叠。"""
    centers, spacing, radius, cam_dist = _grid_layout(4, 2, 1920, 1080)
    assert len(centers) == 4
    # 两行：z 坐标应有非零值
    z_vals = [c[2] for c in centers]
    assert max(z_vals) - min(z_vals) > 1.0  # 行距 > 1
    # 任意两球间距 ≥ spacing（不重叠）
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            assert d >= spacing - 1e-6, f"spheres {i},{j} overlap: d={d} < spacing={spacing}"


def test_grid_layout_cols_caps_row():
    """cols=1 → 单列（全在 z 轴上）。"""
    centers, spacing, radius, cam_dist = _grid_layout(3, 1, 1920, 1080)
    assert len(centers) == 3
    assert all(abs(c[0]) < 1e-9 for c in centers)  # x=0
    z_vals = [c[2] for c in centers]
    assert max(z_vals) - min(z_vals) > 1.0  # 多行
