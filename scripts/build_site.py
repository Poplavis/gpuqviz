#!/usr/bin/env python3
"""gpuqviz 静态站点预生成脚本。

一键生成全部动态内容，执行后 site/ 目录可直接部署::

    python scripts/build_site.py            # 内联 three.js（离线可用，~10MB）
    python scripts/build_site.py --cdn      # CDN 引用 three.js（~1.5MB，需联网）
    python scripts/build_site.py --no-thumbs  # 跳过缩略图（无 GL 环境时）

生成物::

    site/demos/{12个算法}.html     交互式 3D 播放器
    site/images/thumbs/{12个算法}.png  缩略图
    site/assets/gallery-data.js   画廊元数据（window.GPUQVIZ_GALLERY）
    site/assets/viewer.js         从 src/gpuqviz/assets/ 复制
    site/assets/three.min.js      从 vendor 复制
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# 确保能 import gpuqviz（开发模式下脚本从项目根目录运行）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gpuqviz import export_html  # noqa: E402
from gpuqviz.algorithms import ALGORITHM_REGISTRY  # noqa: E402

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #

SITE_DIR = ROOT / "site"
ASSETS_DIR = SITE_DIR / "assets"
DEMOS_DIR = SITE_DIR / "demos"
THUMBS_DIR = SITE_DIR / "images" / "thumbs"

SRC_VIEWER_JS = ROOT / "src" / "gpuqviz" / "assets" / "viewer.js"
SRC_THREE_JS = ROOT / "src" / "gpuqviz" / "assets" / "vendor" / "three.min.js"

# 每个算法的可视化参数（steps/duration 根据电路复杂度调整）
DEMO_PARAMS = {
    "bell":               {"steps": 80,  "duration": 4.0},
    "ghz":                {"steps": 120, "duration": 6.0},
    "superposition":      {"steps": 60,  "duration": 3.0},
    "grover":             {"steps": 200, "duration": 10.0},
    "qft":                {"steps": 150, "duration": 8.0},
    "phase_estimation":   {"steps": 160, "duration": 8.0},
    "deutsch_jozsa":      {"steps": 120, "duration": 6.0},
    "bernstein_vazirani": {"steps": 100, "duration": 5.0},
    "teleportation":      {"steps": 150, "duration": 8.0},
    "superdense":         {"steps": 80,  "duration": 4.0},
    "simon":              {"steps": 140, "duration": 7.0},
    "quantum_walk":       {"steps": 120, "duration": 6.0},
}


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

def _dir_size(path: Path) -> int:
    """递归计算目录总字节。"""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


# --------------------------------------------------------------------------- #
# 1. 复制静态资源
# --------------------------------------------------------------------------- #

def copy_assets() -> None:
    """复制 viewer.js + three.min.js 到 site/assets/。"""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        (SRC_VIEWER_JS, ASSETS_DIR / "viewer.js"),
        (SRC_THREE_JS, ASSETS_DIR / "three.min.js"),
    ]
    for src, dst in targets:
        if not src.exists():
            print(f"  ⚠ 缺失源文件: {src}")
            continue
        shutil.copy2(src, dst)
        print(f"  ✓ 复制 {dst.name:<20} {_fmt_size(dst.stat().st_size)}")


# --------------------------------------------------------------------------- #
# 2. 生成 12 个算法演示 HTML
# --------------------------------------------------------------------------- #

def generate_demos(embed_three: bool = True) -> list[str]:
    """为每个算法生成交互式 HTML 播放器。返回成功生成的算法名列表。"""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    success: list[str] = []

    for name, spec in ALGORITHM_REGISTRY.items():
        params = DEMO_PARAMS.get(name, {"steps": 120, "duration": 6.0})
        out_path = DEMOS_DIR / f"{name}.html"
        title = f"{name} — {spec.description}"

        t0 = time.perf_counter()
        try:
            circuit = spec.builder(engine="qiskit")
            export_html(
                circuit=circuit,
                steps=params["steps"],
                duration=params["duration"],
                title=title,
                out=out_path,
                fps=30,
                embed_three=embed_three,
            )
            elapsed = time.perf_counter() - t0
            size_kb = out_path.stat().st_size / 1024
            print(f"  ✓ {name:<25} {size_kb:>7.0f} KB  {elapsed:.1f}s")
            success.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name:<25} FAILED: {exc}")

    return success


# --------------------------------------------------------------------------- #
# 3. 生成缩略图 PNG
# --------------------------------------------------------------------------- #

def generate_thumbnails(success: list[str]) -> int:
    """为每个算法生成单帧 PNG 缩略图。返回成功数量。"""
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    # 惰性 import render_frame —— 可能需要 GL/CPU 后端
    try:
        from gpuqviz import render_frame
    except ImportError:
        print("  ⚠ 无法 import render_frame，跳过缩略图")
        return 0

    count = 0
    for name in success:
        spec = ALGORITHM_REGISTRY[name]
        out_path = THUMBS_DIR / f"{name}.png"
        t0 = time.perf_counter()
        try:
            circuit = spec.builder(engine="qiskit")
            render_frame(
                circuit=circuit,
                t=0.5,
                out=out_path,
                scale=2,
                style="dark",
                figsize=(6.4, 3.6),  # 16:9 比例
            )
            elapsed = time.perf_counter() - t0
            size_kb = out_path.stat().st_size / 1024
            print(f"  ✓ {name:<25} {size_kb:>5.0f} KB  {elapsed:.1f}s")
            count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ {name:<25} 缩略图失败: {exc}")
            # 留空文件标记，画廊会用渐变占位

    return count


# --------------------------------------------------------------------------- #
# 4. 生成 gallery-data.js
# --------------------------------------------------------------------------- #

def generate_gallery_data(success: list[str], thumb_count: int) -> None:
    """生成画廊元数据 JS 文件。"""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for name in success:
        spec = ALGORITHM_REGISTRY[name]
        thumb_path = THUMBS_DIR / f"{name}.png"
        has_thumb = thumb_path.exists() and thumb_path.stat().st_size > 100
        entries.append({
            "name": name,
            "description": spec.description,
            "category": spec.category,
            "n_qubits": spec.default_n_qubits,
            "demo_url": f"demos/{name}.html",
            "thumb_url": f"images/thumbs/{name}.png" if has_thumb else "",
        })

    # 用 JS 对象字面量输出（避免 JSON 的双引号在 <script> 中问题）
    lines = ["/* gpuqviz 算法画廊元数据 — 由 build_site.py 自动生成 */"]
    lines.append("window.GPUQVIZ_GALLERY = [")
    for e in entries:
        lines.append("  {")
        lines.append(f'    name: {json.dumps(e["name"])},')
        lines.append(f'    description: {json.dumps(e["description"])},')
        lines.append(f'    category: {json.dumps(e["category"])},')
        lines.append(f'    n_qubits: {e["n_qubits"]},')
        lines.append(f'    demo_url: {json.dumps(e["demo_url"])},')
        lines.append(f'    thumb_url: {json.dumps(e["thumb_url"])},')
        lines.append("  },")
    lines.append("];")

    out_path = ASSETS_DIR / "gallery-data.js"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ gallery-data.js  {len(entries)} 个算法")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="gpuqviz 静态站点预生成脚本")
    parser.add_argument(
        "--cdn", action="store_true",
        help="用 CDN 引用 three.js（文件更小，需联网打开）")
    parser.add_argument(
        "--no-thumbs", action="store_true",
        help="跳过缩略图生成（无 GL/CPU 后端环境时使用）")
    args = parser.parse_args()

    print("=" * 60)
    print("  gpuqviz 站点构建")
    print("=" * 60)

    # 1. 复制静态资源
    print("\n[1/4] 复制静态资源 (viewer.js + three.min.js)")
    copy_assets()

    # 2. 生成算法演示 HTML
    embed_three = not args.cdn
    mode_label = "内联 (离线可用)" if embed_three else "CDN (需联网)"
    print(f"\n[2/4] 生成算法演示 HTML（{mode_label}）")
    success = generate_demos(embed_three=embed_three)
    if not success:
        print("  ✗ 无算法演示生成成功，终止")
        return 1

    # 3. 生成缩略图
    thumb_count = 0
    if not args.no_thumbs:
        print(f"\n[3/4] 生成缩略图 PNG")
        thumb_count = generate_thumbnails(success)
    else:
        print("\n[3/4] 跳过缩略图 (--no-thumbs)")

    # 4. 生成 gallery-data.js
    print(f"\n[4/4] 生成画廊元数据")
    generate_gallery_data(success, thumb_count)

    # 汇总
    print("\n" + "=" * 60)
    print("  构建完成")
    print("=" * 60)
    demos_size = _dir_size(DEMOS_DIR)
    thumbs_size = _dir_size(THUMBS_DIR)
    assets_size = _dir_size(ASSETS_DIR)
    total_size = _dir_size(SITE_DIR)
    print(f"  demos/    {len(success):>2} 个 HTML   {_fmt_size(demos_size)}")
    print(f"  thumbs/   {thumb_count:>2} 个 PNG    {_fmt_size(thumbs_size)}")
    print(f"  assets/   generated + static  {_fmt_size(assets_size)}")
    print(f"  ─────────────────────────────────────")
    print(f"  总计                          {_fmt_size(total_size)}")
    print()
    print("  本地预览:  python -m http.server 8080 -d site")
    print("  部署:      将 site/ 目录上传到 GitHub Pages / Vercel / 云服务器")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
