"""S1 验收：最小零落盘管线出片。

运行::

    python examples/minimum_pipeline.py

预期：3 秒内产出 out/demo.mp4（1080p60，流动渐变），目录中无任何中间文件。
"""

import time

from gpuqviz.pipeline import render_solid_gradient

if __name__ == "__main__":
    t0 = time.perf_counter()
    path = render_solid_gradient(out="out/demo.mp4", seconds=3, fps=60)
    print(f"total wall time: {time.perf_counter() - t0:.2f}s, output: {path}")
