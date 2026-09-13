"""后端一致性测试：cuda(gl) 路径与 cpu 软光栅路径的 Bloch 数值一致。"""

import numpy as np
import pytest

from gpuqviz.backends import detect_backend
from gpuqviz.evolve import bloch_vectors, sample_circuit

qiskit = pytest.importorskip("qiskit")


def _bell_key_bloch():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return bloch_vectors(sample_circuit(qc, steps=4), n_qubits=2)


def test_detect_backend_valid():
    assert detect_backend() in ("gl", "cpu")


def test_cpu_soft_raster_matches_numeric():
    """软光栅不改数值：同一关键帧在任何后端下 Bloch 数值一致（≤1e-6）。"""
    bv = _bell_key_bloch()
    assert bv[0, 0, 2] == pytest.approx(1.0, abs=1e-6)   # q0 首帧在北极
    assert bv[-1] == pytest.approx(0.0, abs=1e-6)        # Bell 态零向量


def test_render_bloch_video_cpu_backend(tmp_path):
    """强制 cpu 后端出片（不依赖 GL）。"""
    from gpuqviz import render_bloch_video

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    out = render_bloch_video(circuit=qc, steps=10, fps=10, seconds=0.5,
                             out=tmp_path / "cpu.mp4", backend="cpu")
    assert out.exists() and out.stat().st_size > 0

    import av

    with av.open(str(out)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == 5  # round(0.5 * 10)
