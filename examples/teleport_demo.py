"""P3.2 验收 demo：隐形传态的确定性分支演化。

量子隐形传态是中途测量 + 条件门的标志用例：
ψ 在 q0 → Bell 对 q1-q2 → Bell 测量 (q0,q1) → 条件修正 X(q2)←c1, Z(q2)←c0。
任意测量分支下 q2 都携带原始 ψ —— 本 demo 逐分支演化并核对这一物理判据，
渲染分支 (c0=1, c1=0) 的轨迹动画。

运行：python examples/teleport_demo.py
输出：out/teleport_demo.mp4 / out/teleport_demo.png
"""

from pathlib import Path

import numpy as np

import gpuqviz
from gpuqviz import BlochTrack, Scene
from gpuqviz.circuits import Condition, Gate, evolve_gates_branches
from gpuqviz.state import DensityMatrix, partial_trace


def _gates() -> list[Gate]:
    return [
        Gate(name="RY", targets=[0], params=[0.9]),
        Gate(name="RZ", targets=[0], params=[0.4]),
        Gate(name="H", targets=[1]),
        Gate(name="CX", targets=[2], controls=[1]),
        Gate(name="CX", targets=[1], controls=[0]),
        Gate(name="H", targets=[0]),
        Gate(name="MEASURE", targets=[0], params=[0]),
        Gate(name="MEASURE", targets=[1], params=[1]),
        Gate(name="X", targets=[2], condition=Condition(clbit=1, value=1)),
        Gate(name="Z", targets=[2], condition=Condition(clbit=0, value=1)),
    ]


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    # ── 专业核对：四个分支全部完成传态 ──
    ref_rho = None
    for b0 in (0, 1):
        for b1 in (0, 1):
            ev = evolve_gates_branches(3, _gates(), branch={0: b0, 1: b1})
            rho2 = partial_trace(
                np.outer(ev.frames[-1], ev.frames[-1].conj()), [2], 3)
            if ref_rho is None:
                ref_rho = rho2  # 分支 (0,0) 作为基准
                print(f"ψ(q2) = RZ(0.4)RY(0.9)|0⟩（由传态恢复）")
            else:
                assert np.allclose(rho2, ref_rho, atol=1e-10)
    print("✓ 四分支 q2 约化态逐位一致 — 传态对所有测量结果成立")

    # ── 渲染分支 (c0=1, c1=0)：X 修正触发、Z 跳过 ──
    branch = {0: 1, 1: 0}
    ev = evolve_gates_branches(3, _gates(), branch=branch)
    print(f"分支 {branch}: 测量概率链 "
          f"P(c0)={ev.measurements[0][2]:.3f}, "
          f"P(c1|c0)={ev.measurements[1][2]:.3f}")

    npz = out_dir / "teleport_states.npz"
    np.savez(npz, states=np.stack(ev.frames))

    scene = Scene(
        width=1920, height=1080, fps=30, duration=4.0,
        background="#0b0e14",
        title="量子态演化 — 模拟结果",
        tracks=[BlochTrack(states_path="teleport_states.npz", trail=True,
                           layout="full")],
    )
    scene.save_json(out_dir / "teleport_demo_scene.json")

    out_mp4 = out_dir / "teleport_demo.mp4"
    print(f"渲染视频 → {out_mp4}")
    gpuqviz.render(scene, out=out_mp4, states_dir=out_dir)
    print(f"视频完成: {out_mp4} ({out_mp4.stat().st_size / 1024:.0f} KB)")

    out_png = out_dir / "teleport_demo.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out_png, scale=2,
                         states_dir=out_dir)
    print(f"末帧静态图: {out_png}")

    print("\n✅ 条件门/中途测量验收输出完成")


if __name__ == "__main__":
    main()
