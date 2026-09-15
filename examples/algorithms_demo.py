"""批量演示 gpuqviz 内置算法库的全部 12 个算法。

为每个算法生成交互式 HTML 播放器（含 SVG 电路图 + Bloch 球联动），
输出到 out/algorithms_demo/ 目录。

运行::

    python examples/algorithms_demo.py

也可以用 CLI 一行命令单独演示::

    gpuqviz demo --algo grover
    gpuqviz demo --algo qft --format mp4
    gpuqviz demo --list
"""

import time
from pathlib import Path

from gpuqviz import export_html
from gpuqviz.algorithms import ALGORITHM_REGISTRY

OUTPUT_DIR = Path("out/algorithms_demo")

# 每个算法的可视化参数（steps/duration 根据电路复杂度调整）
DEMO_PARAMS = {
    "bell":              {"steps": 80,  "duration": 4.0},
    "ghz":               {"steps": 120, "duration": 6.0},
    "superposition":     {"steps": 60,  "duration": 3.0},
    "grover":            {"steps": 200, "duration": 10.0},
    "qft":               {"steps": 150, "duration": 8.0},
    "phase_estimation":  {"steps": 160, "duration": 8.0},
    "deutsch_jozsa":     {"steps": 120, "duration": 6.0},
    "bernstein_vazirani": {"steps": 100, "duration": 5.0},
    "teleportation":     {"steps": 150, "duration": 8.0},
    "superdense":        {"steps": 80,  "duration": 4.0},
    "simon":             {"steps": 140, "duration": 7.0},
    "quantum_walk":      {"steps": 120, "duration": 6.0},
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR.resolve()}\n")

    success = []
    for name, spec in ALGORITHM_REGISTRY.items():
        params = DEMO_PARAMS.get(name, {"steps": 120, "duration": 6.0})
        out_path = OUTPUT_DIR / f"{name}.html"
        title = f"{name} — {spec.description}"

        t0 = time.perf_counter()
        try:
            circuit = spec.builder(engine="qiskit")
            path = export_html(
                circuit=circuit,
                steps=params["steps"],
                duration=params["duration"],
                title=title,
                out=out_path,
                fps=30,
            )
            elapsed = time.perf_counter() - t0
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {name:<25} {size_kb:>7.0f} KB  {elapsed:.1f}s")
            success.append(name)
        except Exception as e:
            print(f"  ✗ {name:<25} FAILED: {e}")

    print(f"\n完成: {len(success)}/{len(ALGORITHM_REGISTRY)} 个算法")
    print(f"用浏览器打开 {OUTPUT_DIR / (success[0] + '.html')} 查看交互效果")
    print("\nCLI 一行命令演示:")
    print("  gpuqviz demo --list")
    print("  gpuqviz demo --algo grover")
    print("  gpuqviz demo --algo qft --format mp4")


if __name__ == "__main__":
    main()
