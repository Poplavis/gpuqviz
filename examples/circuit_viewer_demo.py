"""交互式电路图 + Bloch 球联动效果演示。

生成 4 个递进复杂度的交互式 HTML 播放器，展示电路图与 Bloch 球的双向联动：
  1. Bell 态（2 qubit, 2 门）—— 最简单的纠缠态
  2. GHZ 态（3 qubit, 3 门）—— 多体纠缠
  3. Grover 算法（3 qubit, 含 CCX）—— 受控门 + 多步演化
  4. 参数门序列（2 qubit, RX/RZ/SWAP）—— 参数门标签格式化

运行::

    python examples/circuit_viewer_demo.py

生成文件在 out/circuit_demo/ 目录下，双击 HTML 即可在浏览器中打开。
交互方式：
  - 播放时观察电路图中橙色高亮门随播放进度移动
  - 点击电路图中任意门 → 跳转到该门对应时刻，Bloch 球同步更新
  - 拖拽 3D 视口旋转视角，滚轮缩放
  - 空格暂停/播放，←/→ 逐帧步进
"""

import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from gpuqviz import export_html

OUT_DIR = Path("out/circuit_demo")


def demo_bell():
    """Bell 态：H(q0) → CX(q0→q1)。"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = OUT_DIR / "bell.html"
    export_html(circuit=qc, steps=80, duration=4.0, title="Bell 态演化",
                out=out, fps=30)
    print(f"  [1/4] Bell 态     → {out}")
    return out


def demo_ghz():
    """GHZ 态：H(q0) → CX(q0→q1) → CX(q1→q2)。"""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    out = OUT_DIR / "ghz.html"
    export_html(circuit=qc, steps=120, duration=6.0, title="GHZ 三量子纠缠",
                out=out, fps=30)
    print(f"  [2/4] GHZ 态      → {out}")
    return out


def demo_grover():
    """Grover 算法（3 qubit）：H 全叠加 → Oracle (CCX) → Diffuser。

    含 CCX（Toffoli）受控门，展示控制点 + ⊕ 目标符号。
    """
    qc = QuantumCircuit(3)
    # 初始化：均匀叠加
    qc.h([0, 1, 2])
    # Oracle：标记 |111⟩（用 CCX + 辅助相位翻转的简化版）
    qc.cz(0, 2)
    qc.ccx(0, 1, 2)
    qc.cz(0, 2)
    # Diffuser：H → X → CCX → X → H
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.ccx(0, 1, 2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])

    out = OUT_DIR / "grover.html"
    export_html(circuit=qc, steps=200, duration=10.0,
                title="Grover 算法（3 qubit）", out=out, fps=30)
    print(f"  [3/4] Grover 算法 → {out}")
    return out


def demo_parametric():
    """参数门序列：RX(π/2) → RZ(π) → SWAP → RX(π/4)。

    展示参数门标签格式化（π/2、π 等）和 SWAP 的 × 符号。
    """
    qc = QuantumCircuit(2)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi, 1)
    qc.swap(0, 1)
    qc.rx(np.pi / 4, 0)
    qc.ry(np.pi / 2, 1)

    out = OUT_DIR / "parametric.html"
    export_html(circuit=qc, steps=100, duration=5.0,
                title="参数门 + SWAP 序列", out=out, fps=30)
    print(f"  [4/4] 参数门序列   → {out}")
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("生成交互式电路图演示 HTML（4 个示例）...\n")
    paths = [
        demo_bell(),
        demo_ghz(),
        demo_grover(),
        demo_parametric(),
    ]
    elapsed = time.perf_counter() - t0
    print(f"\n完成，耗时 {elapsed:.2f}s。生成 {len(paths)} 个文件：")
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p}  ({size_kb:.0f} KB)")
    print("\n在浏览器中打开任一 HTML 文件即可体验交互效果：")
    print("  • 播放时电路图中橙色高亮门随进度移动")
    print("  • 点击电路图中的门可跳转到该时刻，Bloch 球同步更新")
    print("  • 拖拽 3D 视口旋转，滚轮缩放，空格暂停/播放")
