"""ProVisualizer：专业轨道的链式门面（docs/development-plan.md P4.4）。

把 参数化/噪声/分析/导出 串成一个可读的调用链::

    viz = (ProVisualizer(circuit)
           .with_shots(shots=4096, seed=42)
           .with_noise(depolarizing, lam=0.05)
           .analyze(pauli=["IZZ", "ZII"], entanglement=True))
    report = viz.report()            # 全部分析量的结构化数据
    viz.export_video("out.mp4")      # 复用现有编码管线
    viz.export_frame("fig.png")

report() 的每个数字都来自 analysis 模块（含对拍锁定），
HTML 导出与 Python 侧数值同源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis.entanglement import EntanglementReport, entanglement_summary
from .analysis.measurement import Counts, sample_counts
from .analysis.metrics import StateTable, fidelity, pauli_expectation, state_table
from .circuits import Gate
from .noise import depolarizing, evolve_density, tensor_channels
from .state import Statevector, as_state

__all__ = ["ProVisualizer", "ProReport"]


@dataclass(frozen=True)
class ProReport:
    """ProVisualizer.report() 的结构化结果。"""

    n_qubits: int
    state_table: StateTable
    counts: Counts | None = None
    entanglement: EntanglementReport | None = None
    pauli: dict[str, float] = field(default_factory=dict)
    fidelity_target: float | None = None
    # 0.8.0 审计 #3/#4：降级链与近似语义可观测
    provenance: str = "exact"          # "exact" / "mps_chi=N" 等
    backend: str | None = None         # 最近一次 export 的渲染后端
    encoder: str | None = None         # 最近一次 export 的实际编码器

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "state": self.state_table.to_dict(min_prob=1e-12),
            "counts": self.counts.to_dict() if self.counts else None,
            "entanglement": self.entanglement.to_dict() if self.entanglement else None,
            "pauli": dict(self.pauli),
            "fidelity_target": self.fidelity_target,
            "provenance": self.provenance,
            "backend": self.backend,
            "encoder": self.encoder,
        }


def _normalize_noise(spec) -> callable:
    """噪声简写 → noise(gate, gi) 回调。

    支持：None / callable(gate, gi) / ("depolarizing", lam)。
    """
    if spec is None:
        return None
    if callable(spec):
        return spec
    if isinstance(spec, tuple) and spec[0] == "depolarizing":
        lam = float(spec[1])

        def noise(gate, gi):
            nq = len(gate.targets) + len(gate.controls)
            return tensor_channels([depolarizing(lam)] * nq)

        return noise
    raise ValueError(f"unsupported noise spec: {spec!r}")


class ProVisualizer:
    """专业轨道门面：电路 → （噪声）演化 → 分析 → 导出。"""

    def __init__(self, circuit=None, template=None, bindings: dict | None = None,
                 gates: list[Gate] | None = None, n_qubits: int | None = None,
                 title: str = "量子态演化 — 模拟结果"):
        # 门来源三选一：qiskit circuit / template+bindings / 现成 Gate 列表
        if sum(x is not None for x in (circuit, template, gates)) != 1:
            raise ValueError("provide exactly one of circuit/template/gates")
        self.title = title
        if circuit is not None:
            from .adapters import qiskit_to_gates

            n_qubits, gates = qiskit_to_gates(circuit)
        elif template is not None:
            gates = template.bind(**(bindings or {}))
            n_qubits = template.n_qubits
        self.n_qubits = int(n_qubits)
        self.gates = list(gates)

        self._shots: int | None = None
        self._seed: int | None = None
        self._noise = None
        self._pauli: list[str] = []
        self._entanglement = False
        self._target = None
        self.backend: str | None = None   # 最近一次 export_video 的渲染后端
        self.encoder: str | None = None   # 最近一次 export_video 的实际编码器

    # -- 链式配置 ----------------------------------------------------------- #

    def with_shots(self, shots: int = 4096, seed: int | None = None) -> "ProVisualizer":
        """启用 shot 采样（report.counts）。"""
        self._shots, self._seed = int(shots), seed
        return self

    def with_noise(self, spec) -> "ProVisualizer":
        """启用含噪演化（密度矩阵路径）。spec 见 _normalize_noise。"""
        self._noise = _normalize_noise(spec)
        return self

    def analyze(self, pauli: list[str] | None = None,
                entanglement: bool = True,
                fidelity_to=None) -> "ProVisualizer":
        """声明 report() 要计算的分析量。"""
        self._pauli = list(pauli or [])
        self._entanglement = bool(entanglement)
        if fidelity_to is not None:
            self._target = np.asarray(
                getattr(fidelity_to, "data", fidelity_to)).reshape(-1)
        return self

    # -- 演化 --------------------------------------------------------------- #

    def keyframes(self) -> list[np.ndarray]:
        """演化关键帧：无噪声 → 纯态；有噪声 → 密度矩阵。"""
        if self._noise is not None:
            return evolve_density(self.n_qubits, self.gates, noise=self._noise)
        from .circuits import evolve_gates

        return evolve_gates(self.n_qubits, self.gates)

    def final_state(self) -> np.ndarray:
        """末帧（纯态 (2^n,) 或密度矩阵 (2^n, 2^n)）。"""
        return self.keyframes()[-1]

    # -- 报告 --------------------------------------------------------------- #

    def report(self) -> ProReport:
        st = as_state(self.final_state())
        table = state_table(st)
        counts = (sample_counts(st, self._shots, seed=self._seed)
                  if self._shots else None)
        ent = entanglement_summary(st) if self._entanglement else None
        pauli = {name: pauli_expectation(st, name) for name in self._pauli}
        f_target = None
        if self._target is not None:
            f_target = fidelity(st, Statevector(self._target))
        provenance = getattr(st, "provenance", "exact")
        return ProReport(n_qubits=self.n_qubits, state_table=table,
                         counts=counts, entanglement=ent, pauli=pauli,
                         fidelity_target=f_target, provenance=provenance)

    # -- 导出 --------------------------------------------------------------- #

    def _save_states(self, out: Path) -> Path:
        """关键帧 → npz（BlochTrack 消费；ρ 关键帧自动走密度渲染路径）。"""
        frames = self.keyframes()
        arr = np.stack([np.asarray(f) for f in frames])
        npz = out.parent / (out.stem + "_states.npz")
        np.savez(npz, states=arr)
        return npz

    def export_video(self, out: str | Path = "out/pro_demo.mp4",
                     duration: float = 4.0, fps: float = 30.0,
                     states_dir: str | Path | None = None,
                     prefer_nvenc: bool = False) -> Path:
        """演化动画视频（理想=纯态；含噪=密度矩阵，渲染自动收缩 Bloch 矢量）。"""
        from .scene import BlochTrack, Scene

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        base_dir = Path(states_dir) if states_dir else out.parent
        npz_rel = self._save_states(out).name
        scene = Scene(width=1920, height=1080, fps=fps, duration=duration,
                      background="#0b0e14", title=self.title,
                      tracks=[BlochTrack(states_path=npz_rel, trail=True,
                                         layout="full")])
        from .api import render
        from .backends import detect_backend
        from .encode import last_encoder_info

        path = render(scene, out=out, states_dir=base_dir,
                      prefer_nvenc=prefer_nvenc)
        enc = last_encoder_info() or {}
        self.backend = detect_backend()
        self.encoder = enc.get("encoder")
        return path

    def export_frame(self, out: str | Path = "out/pro_frame.png",
                     t: float = 1.0, scale: int = 2,
                     states_dir: str | Path | None = None) -> Path:
        """单帧 PNG（超采样）。"""
        from .scene import BlochTrack, Scene

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        base_dir = Path(states_dir) if states_dir else out.parent
        npz_rel = self._save_states(out).name
        scene = Scene(width=1920, height=1080, fps=30, duration=4.0,
                      background="#0b0e14", title=self.title,
                      tracks=[BlochTrack(states_path=npz_rel, layout="full")])
        from .api import render_frame

        return render_frame(scene=scene, t=t, out=out, scale=scale,
                            states_dir=base_dir)
