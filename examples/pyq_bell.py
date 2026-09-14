"""pyqpanda 兼容性测试：本源量子电路 → gpuqviz 出片 out/pyq_bell.mp4。

运行::

    python examples/pyq_bell.py

预期：Bell 态电路（H + CNOT + 多层小角度旋转）的布洛赫球动画，
全程零中间文件，与 qiskit 同构电路出片画面一致。
"""

import time

from pyqpanda import CPUQVM, QProg, H, RX, RY, RZ, CNOT

from gpuqviz import render_bloch_video


def build_prog(q):
    """Bell 纠缠 + 多层小角度旋转：让矢量全程保持连续运动。"""
    prog = QProg()
    prog << H(q[0])
    for i in range(6):
        prog << RX(q[0], 0.5) << RY(q[1], 0.4) << RZ(q[0], 0.3)
        if i == 2:
            prog << CNOT(q[0], q[1])   # 中途纠缠
    prog << CNOT(q[0], q[1])
    return prog


def main():
    qm = CPUQVM()
    qm.init_qvm()
    try:
        q = qm.qAlloc_many(2)
        prog = build_prog(q)

        t0 = time.perf_counter()
        render_bloch_video(
            circuit=prog,
            machine=qm,          # pyqpanda 必须传创建 prog 的虚拟机实例
            steps=120,
            fps=60,
            seconds=6.0,
            out="out/pyq_bell.mp4",
            style="dark",
            trail=True,
        )
        print(f"total wall time: {time.perf_counter() - t0:.1f}s")
    finally:
        qm.finalize()


if __name__ == "__main__":
    main()
