"""常用分辨率预设（像素）。

所有预设以 (width, height) 元组表示，供 render_bloch_video /
render_heatmap_video / render_frame 及 CLI --resolution 使用。
"""

from __future__ import annotations

# ── 16:9 标准 ──────────────────────────────────────────────
RES_480P:  tuple[int, int] = (854, 480)
RES_720P:  tuple[int, int] = (1280, 720)
RES_1080P: tuple[int, int] = (1920, 1080)
RES_1440P: tuple[int, int] = (2560, 1440)
RES_4K:    tuple[int, int] = (3840, 2160)
RES_8K:    tuple[int, int] = (7680, 4320)

# ── 1:1 正方形（适合社交媒体缩略图） ────────────────────────
RES_SQUARE_1080: tuple[int, int] = (1080, 1080)
RES_SQUARE_2160: tuple[int, int] = (2160, 2160)

# ── 9:16 竖屏（适合手机短视频） ────────────────────────────
RES_VERTICAL_720:  tuple[int, int] = (720, 1280)
RES_VERTICAL_1080: tuple[int, int] = (1080, 1920)

# ── 便捷映射：字符串名 → (width, height) ───────────────────
PRESETS: dict[str, tuple[int, int]] = {
    "480p":  RES_480P,
    "720p":  RES_720P,
    "1080p": RES_1080P,
    "1440p": RES_1440P,
    "4k":    RES_4K,
    "8k":    RES_8K,
    "square":      RES_SQUARE_1080,
    "square2k":    RES_SQUARE_2160,
    "vertical":      RES_VERTICAL_720,
    "vertical1080":  RES_VERTICAL_1080,
}


def resolve_preset(name: str) -> tuple[int, int]:
    """按名称解析预设分辨率（大小写不敏感）。

    >>> resolve_preset("4K")
    (3840, 2160)
    """
    key = name.lower()
    if key not in PRESETS:
        raise ValueError(
            f"unknown resolution preset {name!r}; "
            f"choices: {sorted(PRESETS)}"
        )
    return PRESETS[key]
