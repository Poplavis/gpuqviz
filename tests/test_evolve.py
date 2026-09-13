"""evolve / interpolate 层测试（不需要 GPU 渲染）。"""

import numpy as np
import pytest

from gpuqviz.evolve import bloch_vectors, sample_circuit
from gpuqviz.interpolate import lerp_states, slerp_keys

qiskit = pytest.importorskip("qiskit")


def test_sample_circuit_bell():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    states = sample_circuit(qc, steps=6)

    assert len(states) == 6
    assert states[0].shape == (4,)
    # 首帧 |00>
    assert np.allclose(states[0], [1, 0, 0, 0])
    # 末帧 Bell 态 (|00>+|11>)/sqrt(2)：小端序下非零振幅在 index 0 与 3
    assert np.allclose(np.abs(states[-1]), [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


def test_bloch_vectors_values():
    # |+>⊗|0>（qiskit 小端序：q0 叠加、q1 为 |0>）
    psi = np.array([1, 1, 0, 0]) / np.sqrt(2)
    bv = bloch_vectors([psi])
    assert bv.shape == (1, 2, 3)
    assert np.allclose(bv[0, 0], [1, 0, 0])   # q0: <X>=1
    assert np.allclose(bv[0, 1], [0, 0, 1])   # q1: <Z>=1


def test_bloch_vectors_bell_is_mixed():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    bv = bloch_vectors(sample_circuit(qc, steps=3))
    # Bell 态单 qubit 约化态为最大混合 → Bloch 向量为零
    assert np.allclose(bv[-1], 0.0, atol=1e-12)


def test_slerp_endpoints():
    keys = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    out = slerp_keys(keys, 5)
    assert out.shape == (5, 3)
    assert np.allclose(out[0], [0, 0, 1])
    assert np.allclose(out[-1], [0, 0, -1])
    # 球面插值半径恒为 1
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0)


def test_lerp_states_normalized():
    keys = np.array([[1, 0], [0, 1]], dtype=complex)
    out = lerp_states(keys, 5)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
