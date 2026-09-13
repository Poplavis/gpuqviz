"""编码路径性能对照：AvEncoder(libx264) vs AvEncoder(NVENC优先) vs NvencEncoder。

运行::

    python benchmarks/encode_path.py

输出 out/encode_benchmark.csv 与控制台表格。NVENC 不可用的机器上，
两条 GPU 路径自动回退并在报告中标注。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from gpuqviz.encode import AvEncoder, NvencEncoder, nvenc_available
from gpuqviz.render import GLContext
from gpuqviz.pipeline import _VS_SCREEN, _FS_GRADIENT


def _render_gradient(width, height, total_frames):
    """渲染渐变帧序列（生成器，逐帧产出 GPU 帧数组）。"""
    with GLContext(width, height) as gl:
        prog = gl.program(_VS_SCREEN, _FS_GRADIENT)
        vbo = gl.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32).tobytes())
        vao = gl.ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
        for t in range(total_frames):
            prog["u_time"].value = t / 60.0
            vao.render()
            yield gl.read_frame()
        vbo.release()
        vao.release()


def _bench(name: str, encoder_factory, width, height, seconds, fps):
    total_frames = int(seconds * fps)
    t0 = time.perf_counter()
    enc = encoder_factory()
    n = 0
    try:
        for frame in _render_gradient(width, height, total_frames):
            enc.write(frame)
            n += 1
    finally:
        enc.close()
    elapsed = time.perf_counter() - t0
    return {"path": name, "frames": n, "seconds": round(elapsed, 2),
            "fps_eff": round(n / elapsed, 1)}


def main():
    width, height, fps, seconds = 1920, 1080, 60, 10
    out_csv = Path("out/encode_benchmark.csv")
    out_csv.parent.mkdir(exist_ok=True)

    rows = [
        _bench("AvEncoder(libx264)",
               lambda: AvEncoder(width, height, fps, "out/bench_libx264.mp4",
                                 codec="h264"),
               width, height, seconds, fps),
    ]
    # AvEncoder 默认顺序（先试 h264_nvenc）
    rows.append(_bench("AvEncoder(nvenc-first)",
                       lambda: AvEncoder(width, height, fps, "out/bench_nvenc_av.mp4",
                                         codec="h264"),
                       width, height, seconds, fps))
    # NvencEncoder（显存直喂）
    if nvenc_available():
        rows.append(_bench("NvencEncoder(device)",
                           lambda: NvencEncoder(width, height, fps, "out/bench_pynv.mp4",
                                                codec="h264"),
                           width, height, seconds, fps))
    else:
        rows.append({"path": "NvencEncoder(device)", "frames": 0, "seconds": 0,
                     "fps_eff": "UNAVAILABLE (session open failed)"})

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "frames", "seconds", "fps_eff"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'path':<28}{'frames':>8}{'seconds':>10}{'eff fps':>10}")
    for r in rows:
        print(f"{r['path']:<28}{r['frames']:>8}{str(r['seconds']):>10}{str(r['fps_eff']):>10}")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
