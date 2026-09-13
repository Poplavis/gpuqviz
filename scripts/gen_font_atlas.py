"""字体 SDF 图集烘焙：freetype 渲染字形 → scipy 距离变换生成 SDF → 打包图集。

产物：
- src/gpuqviz/assets/font_<name>_<size>.bin   （SDF 图集，R8）
- src/gpuqviz/assets/font_<name>_<size>.json  （字符度量元数据）

运行::

    python scripts/gen_font_atlas.py --font C:/Windows/Fonts/msyh.ttc \
        --name msyh --size 64 --chars-file scripts/font_chars.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import freetype
from scipy.ndimage import distance_transform_edt

DEFAULT_CHARS = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " ,.:;!?()[]{}+-*/=<>|&^%$#@~_'\"\\·—"
    "量子比特态演化布洛赫球热图叠加纠缠相位振幅概率幅"
    "贝尔态门电路初始终矢量分量实部虚基测测量结果"
    "可视化视频渲染加速导出文件路径输出输入参数默认值"
    "状态矢系统经典计算模拟仿真演示示例场景标题轴标签"
    "一二三四五六七八九十时间频率秒帧率"
)

SDF_RANGE = 16.0  # SDF 有效符号距离范围（源字号像素）


def render_glyph(face, ch: str) -> tuple[np.ndarray, dict]:
    """渲染单字符 → (SDF 数组, 度量 dict)。SDF ∈ [0,1]，0.5 为笔画边缘。"""
    face.load_char(ch, flags=freetype.FT_LOAD_FLAGS["FT_LOAD_RENDER"])
    g = face.glyph
    bmp = g.bitmap
    w, h = bmp.width, bmp.rows
    if w == 0 or h == 0:
        return np.zeros((1, 1), np.uint8), {"empty": True}

    bitmap = np.frombuffer(bytes(bmp.buffer), dtype=np.uint8).reshape(h, w)
    inside = distance_transform_edt(bitmap > 127)
    outside = distance_transform_edt(bitmap <= 127)
    signed = inside - outside  # 内正外负
    sdf = 0.5 + 0.5 * np.clip(signed / SDF_RANGE, -1.0, 1.0)
    return (sdf * 255).astype(np.uint8), {
        "w": w, "h": h,
        "bearing_x": g.bitmap_left,
        "bearing_y": g.bitmap_top,
        "advance": g.advance.x / 64.0,
    }


def bake(font_path: str, name: str, size: int, chars: str, out_dir: Path) -> None:
    face = freetype.Face(font_path)
    face.set_char_size(size * 64)

    cell = size + 16  # 图集单元（留 SDF 扩散边距）
    n = len(chars)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    atlas = np.zeros((rows * cell, cols * cell), np.uint8)
    meta: dict[str, dict] = {}
    missing = []

    for i, ch in enumerate(chars):
        r, c = divmod(i, cols)
        glyph, info = render_glyph(face, ch)
        if info.get("empty"):
            missing.append(ch)
            continue
        gh, gw = glyph.shape
        if gw > cell or gh > cell:
            raise ValueError(f"glyph {ch!r} too large: {gw}x{gh} > cell {cell}")
        ox = c * cell + (cell - gw) // 2
        oy = r * cell + (cell - gh) // 2
        atlas[oy:oy + gh, ox:ox + gw] = glyph
        meta[ch] = {
            "x": ox, "y": oy, "w": gw, "h": gh,
            "bearing_x": info["bearing_x"], "bearing_y": info["bearing_y"],
            "advance": info["advance"], "source_size": size,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"font_{name}_{size}.bin"
    json_path = out_dir / f"font_{name}_{size}.json"
    bin_path.write_bytes(atlas.tobytes())
    json_path.write_text(json.dumps({
        "font": font_path, "name": name, "size": size, "cell": cell,
        "atlas_w": atlas.shape[1], "atlas_h": atlas.shape[0],
        "sdf_range_px": SDF_RANGE, "glyphs": meta,
    }, ensure_ascii=False))
    print(f"atlas {atlas.shape[1]}x{atlas.shape[0]}, glyphs={len(meta)}, "
          f"missing={len(missing)} -> {bin_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default="C:/Windows/Fonts/msyh.ttc")
    ap.add_argument("--name", default="msyh")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--chars", default=DEFAULT_CHARS)
    args = ap.parse_args()
    bake(args.font, args.name, args.size, args.chars,
         Path(__file__).parent.parent / "src" / "gpuqviz" / "assets")
