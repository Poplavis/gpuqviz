"""测试案例：标题居中 + 右上角作者标注 + 网址备注。

布局：
  ┌──────────────────────────────────────┐
  │                    Popla             │  ← 右上角作者
  │            量子态演化可视化            │  ← 居中标题
  │          https://poplav.com          │  ← 网址备注（标题下方一行）
  │                                      │
  │            ●  ●  ●  ●                │  ← Bloch 球
  │                                      │
  └──────────────────────────────────────┘
"""

import numpy as np
from pathlib import Path

import gpuqviz
from gpuqviz import TextOverlay, TextPosition


def make_demo_states(n_qubits=2, frames=60):
    """生成一组简单的量子态演化序列（2 qubit Bell 态制备过程）。"""
    states = []
    for i in range(frames):
        t = i / (frames - 1)
        s = np.zeros(2 ** n_qubits, dtype=complex)
        # 从 |00⟩ 演化到 (|00⟩+|11⟩)/√2 的插值
        s[0] = np.cos(t * np.pi / 4)
        s[3] = np.sin(t * np.pi / 4) * np.exp(1j * t * np.pi / 2)
        norm = np.linalg.norm(s)
        if norm > 0:
            s /= norm
        states.append(s)
    return states


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    states = make_demo_states(n_qubits=2, frames=60)

    # ── 文字叠加配置 ──
    overlays = [
        # 1. 居中标题（顶部居中）
        TextOverlay(
            text="量子态演化可视化",
            position=TextPosition.TOP_CENTER,
            font_size=48,
            color="#dedede",
            opacity=1.0,
            shadow=False,
        ),
        # 2. 右上角作者标注
        TextOverlay(
            text="Popla",
            position=TextPosition.TOP_RIGHT,
            font_size=28,
            color="#dedede",
            opacity=0.9,
            shadow=False,
        ),
        # 3. 网址备注（标题下方一行，用自定义像素坐标定位）
        #    1920×1080 画面中，标题 font_size=48 居中在 y≈40，
        #    网址放在 y≈100（标题下方约 60px）
        TextOverlay(
            text="https://poplav.com",
            position=(810, 100),   # 标题下方居中偏左（文字宽约 300px）
            font_size=24,
            color="#FFFFFF",
            opacity=0.8,
            shadow=False,
        ),
    ]

    # ── 渲染 MP4 视频（CPU 后端，不依赖 GPU）──
    out_mp4 = out_dir / "overlay_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render_bloch_video(
        states=states,
        out=out_mp4,
        resolution="1080p",
        steps=60,
        fps=30,
        seconds=2.0,
        backend="cpu",
        trail=True,
        title="量子态演化可视化",      # title 会被 overlays 覆盖
        watermark="Popla · poplav.com",
    )
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    # ── 渲染单帧 PNG 静态图（用 Scene API 精确控制 overlay）──
    # 构造一个只包含 overlay 的"空场景"不太方便（Scene 需要 track），
    # 所以这里用 render_frame + 自定义 draw 的方式不够直接。
    # 更简洁的做法：直接用 render_bloch_video 的 title/watermark 便捷参数
    # 上面的 MP4 已经演示了便捷参数方式。
    #
    # 下面用 Scene API 演示精确的 overlay 控制（需要先保存 states.npz）：

    npz_path = out_dir / "demo_states.npz"
    np.savez(npz_path, states=np.array(states))

    from gpuqviz import Scene, BlochTrack

    scene = Scene(
        width=1920,
        height=1080,
        fps=30,
        duration=2.0,
        background="#0b0e14",
        tracks=[BlochTrack(states_path="demo_states.npz", trail=True)],
        overlays=overlays,   # 精确的 3 个 overlay
    )

    # 保存 Scene JSON（可复现）
    scene_json = out_dir / "overlay_demo_scene.json"
    scene.save_json(scene_json)
    print(f"场景 JSON: {scene_json}")

    # 渲染 Scene → MP4
    out_scene_mp4 = out_dir / "overlay_demo_scene.mp4"
    print(f"渲染 Scene 视频 → {out_scene_mp4}")
    gpuqviz.render(scene, out=out_scene_mp4, states_dir=out_dir)
    print(f"Scene 视频完成: {out_scene_mp4}")

    # 渲染单帧 PNG
    out_png = out_dir / "overlay_demo.png"
    gpuqviz.render_frame(scene=scene, t=0.5, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"静态帧完成: {out_png}")

    print("\n✅ 全部渲染完成！输出文件：")
    print(f"   1. {out_mp4}          — 便捷参数方式（title + watermark）")
    print(f"   2. {out_scene_mp4}  — Scene API 方式（3 个精确 overlay）")
    print(f"   3. {out_png}          — 单帧静态 PNG")


if __name__ == "__main__":
    main()
