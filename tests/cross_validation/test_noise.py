"""噪声通道对拍：qiskit QuantumError 逐通道（1e-10）+ Aer 全电路（1e-6）。"""

import numpy as np
import pytest

from gpuqviz.circuits import Gate
from gpuqviz.noise import (amplitude_damping, apply_channel, apply_matrix_density,
                           dephasing, depolarizing, evolve_density,
                           lerp_density, phase_damping, thermal_relaxation)
from gpuqviz.state import DensityMatrix

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import (  # noqa: E402
    DensityMatrix as QkDM,
    Kraus as QkKraus,
    Statevector as QkSV,
)

TOL = 1e-10
AER_TOL = 1e-6


def _random_rho(seed: int, n: int = 2) -> np.ndarray:
    """随机密度矩阵（酉混合构造，保证 PSD + 迹 1）。"""
    rng = np.random.Generator(np.random.PCG64(seed))
    d = 2 ** n
    rho = np.zeros((d, d), dtype=np.complex128)
    for _ in range(3):
        m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        q, _ = np.linalg.qr(m)
        w = rng.random(d)
        w /= w.sum()
        rho += (q * w) @ q.conj().T
    return rho / 3


def _channel_action_qiskit(ops, rho):
    """qiskit 参考路径：Kraus(ops) 经 DensityMatrix.evolve。"""
    qk = QkKraus(ops)
    return np.asarray(QkDM(rho).evolve(qk).data)


def _assert_channel_matches_qiskit(ops, rho, tol=TOL):
    ours = apply_channel(rho, ops, list(range(int(round(np.log2(rho.shape[0]))))))
    ref = _channel_action_qiskit(ops, rho)
    assert np.allclose(ours, ref, atol=tol)
    # trace-preserving 与 PSD 检查
    dm = DensityMatrix(ours)
    assert dm.is_valid()


# --------------------------------------------------------------------------- #
# 逐通道对拍 qiskit
# --------------------------------------------------------------------------- #

def test_depolarizing_matches_qiskit():
    from qiskit_aer.noise import depolarizing_error

    aer = pytest.importorskip("qiskit_aer")
    rho = _random_rho(1)
    for lam in (0.0, 0.1, 0.5, 0.9):
        ops = depolarizing(lam)
        ref = np.asarray(
            QkDM(rho).evolve(depolarizing_error(lam, 1).to_quantumchannel(), qargs=[0]).data)
        ours = apply_channel(rho, ops, [0])
        assert np.allclose(ours, ref, atol=TOL), f"λ={lam}"


def test_amplitude_damping_matches_qiskit():
    from qiskit_aer.noise import amplitude_damping_error

    rho = _random_rho(2)
    for gamma in (0.0, 0.3, 1.0):
        ops = amplitude_damping(gamma)
        ref = np.asarray(
            QkDM(rho).evolve(amplitude_damping_error(gamma).to_quantumchannel(), qargs=[0]).data)
        assert np.allclose(apply_channel(rho, ops, [0]), ref, atol=TOL), \
            f"γ={gamma}"


def test_phase_damping_matches_qiskit():
    from qiskit_aer.noise import phase_damping_error

    rho = _random_rho(3)
    for gamma in (0.0, 0.4, 1.0):
        ops = phase_damping(gamma)
        ref = np.asarray(
            QkDM(rho).evolve(phase_damping_error(gamma).to_quantumchannel(), qargs=[0]).data)
        assert np.allclose(apply_channel(rho, ops, [0]), ref, atol=TOL), \
            f"γ={gamma}"


def test_dephasing_matches_qiskit():
    from qiskit_aer.noise import pauli_error

    rho = _random_rho(4)
    for p in (0.0, 0.25, 0.5):
        ops = dephasing(p)
        ref_ch = pauli_error([("Z", p), ("I", 1 - p)]).to_quantumchannel()
        ref = np.asarray(QkDM(rho).evolve(ref_ch, qargs=[0]).data)
        assert np.allclose(apply_channel(rho, ops, [0]), ref, atol=TOL), f"p={p}"


def test_thermal_relaxation_matches_qiskit():
    from qiskit_aer.noise import thermal_relaxation_error

    rho = _random_rho(5)
    for t1, t2, t in ((100.0, 50.0, 20.0), (50.0, 90.0, 10.0),
                      (100.0, 100.0, 30.0)):
        ops = thermal_relaxation(t1, t2, t)
        ref = np.asarray(
            QkDM(rho).evolve(thermal_relaxation_error(t1, t2, t).to_quantumchannel(), qargs=[0]).data)
        assert np.allclose(apply_channel(rho, ops, [0]), ref, atol=1e-9), \
            f"T1={t1} T2={t2} t={t}"


def test_depolarizing_bloch_shrinkage():
    """去极化通道的 Bloch 收缩因子 = 1−λ（conventions.md 的物理语义）。"""
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>：Bloch (1,0,0)
    rho = np.outer(psi, psi.conj())
    for lam in (0.25, 0.75):
        out = apply_channel(rho, depolarizing(lam), [0])
        rx = float(np.real(out[0, 1] + out[1, 0]))
        assert abs(rx - (1 - lam)) < TOL


def test_amplitude_damping_decays_to_ground():
    """γ=1 完全弛豫 → 任意态衰到 |0⟩⟨0|。"""
    rho = _random_rho(7, n=1)
    out = apply_channel(rho, amplitude_damping(1.0), [0])
    assert np.allclose(out, np.diag([1, 0]).astype(complex), atol=TOL)


# --------------------------------------------------------------------------- #
# 子集作用 / 多 qubit 通道
# --------------------------------------------------------------------------- #

def test_apply_channel_on_subset():
    """2-qubit 通道只作用于指定 qubit 对（qiskit qargs 对拍）。"""
    rho = _random_rho(8, n=3)
    ops2q = depolarizing(0.3, n_qubits=2)
    ours = apply_channel(rho, ops2q, [0, 2])
    ref = np.asarray(QkDM(rho).evolve(QkKraus(ops2q), qargs=[0, 2]).data)
    assert np.allclose(ours, ref, atol=TOL)
    # 非相邻子集 [0, 2]（LSB-first：Kraus 的 MSB 子系统 → qubit 2）
    ours2 = apply_channel(rho, ops2q, [2, 0])
    ref2 = np.asarray(QkDM(rho).evolve(QkKraus(ops2q), qargs=[2, 0]).data)
    assert np.allclose(ours2, ref2, atol=TOL)


def test_apply_matrix_density_matches_unitary_conjugation():
    rho = _random_rho(9, n=2)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    from qiskit.quantum_info import Operator

    U = Operator(qc).data
    ours = apply_matrix_density(rho, U, [0, 1])
    ref = np.asarray(QkDM(rho).evolve(Operator(qc)).data)
    assert np.allclose(ours, ref, atol=TOL)


# --------------------------------------------------------------------------- #
# 全电路演化对拍 Aer density_matrix + NoiseModel
# --------------------------------------------------------------------------- #

def _build_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.ry(0.7, 2)
    return qc


def test_evolve_density_ideal_matches_qiskit_superop():
    """无噪声时密度演化 = 纯态演化的外积（qiskit Statevector 参考）。"""
    qc = _build_circuit()
    gates = _qiskit_to_gates(qc)
    frames = evolve_density(3, gates)
    sv = QkSV.from_instruction(qc)
    ref_rho = np.outer(np.asarray(sv.data), np.asarray(sv.data).conj())
    assert np.allclose(frames[-1], ref_rho, atol=1e-10)


def _qiskit_to_gates(qc):
    from gpuqviz.adapters import qiskit_to_gates

    n, gates = qiskit_to_gates(qc)
    return gates


def test_evolve_density_noisy_matches_aer():
    """GHZ 电路 + depolarizing 噪声：与 Aer density_matrix 末态对拍 ≤1e-6。"""
    qiskit_aer = pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    lam1, lam2 = 0.02, 0.05
    qc = _build_circuit()

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(lam1, 1), ["h", "ry"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(lam2, 2), ["cx"])

    from qiskit_aer.library import SaveDensityMatrix

    qc_save = qc.copy()
    qc_save.append(SaveDensityMatrix(3), range(3))
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    result = sim.run(qc_save, shots=1).result()
    ref = np.asarray(result.data(0)["density_matrix"])

    gates = _qiskit_to_gates(qc)

    def noise(gate, gi):
        nq = len(gate.targets) + len(gate.controls)
        if nq == 1:
            return depolarizing(lam1)
        if nq == 2:
            return depolarizing(lam2, n_qubits=2)
        return None

    frames = evolve_density(3, gates, noise=noise)
    assert np.allclose(frames[-1], ref, atol=AER_TOL)


def test_evolve_density_amplitude_damping_matches_aer():
    """振幅阻尼全电路对拍 Aer。"""
    qiskit_aer = pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error

    gamma = 0.15
    qc = QuantumCircuit(2)
    from qiskit_aer.library import SaveDensityMatrix

    qc_save = qc.copy()
    qc_save.append(SaveDensityMatrix(2), range(2))
    ad1 = amplitude_damping_error(gamma)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(ad1, ["h"])
    noise_model.add_all_qubit_quantum_error(ad1.tensor(ad1), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    ref = np.asarray(sim.run(qc_save, shots=1).result()
                     .data(0)["density_matrix"])

    from gpuqviz.noise import amplitude_damping as ad
    from gpuqviz.noise import tensor_channels

    def noise(gate, gi):
        nq = len(gate.targets) + len(gate.controls)
        return tensor_channels([ad(gamma)] * nq)

    gates = _qiskit_to_gates(qc)
    frames = evolve_density(2, gates, noise=noise)
    assert np.allclose(frames[-1], ref, atol=AER_TOL)


def test_amplitude_damping_bloch_shrinks_z():
    """物理语义：T1 弛豫把 Bloch 矢量 z 分量拉向 +1（基态）。"""
    gamma = 0.5
    psi = np.array([0, 1], dtype=complex)  # |1>：Bloch (0,0,-1)
    rho = np.outer(psi, psi.conj())
    out = apply_channel(rho, amplitude_damping(gamma), [0])
    rz = float(np.real(out[0, 0] - out[1, 1]))
    assert abs(rz - (-1 + 2 * gamma)) < TOL  # rz: -1 → +1 随 γ 线性


# --------------------------------------------------------------------------- #
# lerp_density
# --------------------------------------------------------------------------- #

def test_lerp_density_trace_preserved():
    rhos = np.stack([np.diag([1, 0, 0, 0]).astype(complex),
                     np.diag([0.5, 0.5, 0, 0]).astype(complex)])
    out = lerp_density(rhos, 11)
    assert out.shape == (11, 4, 4)
    trs = np.real(np.trace(out, axis1=1, axis2=2))
    assert np.allclose(trs, 1.0, atol=1e-12)


def test_evolve_density_keyframe_count():
    gates = [Gate(name="H", targets=[0]), Gate(name="CX", targets=[1], controls=[0])]
    frames = evolve_density(2, gates)
    assert len(frames) == 3  # 初态 + 2 门
    assert all(abs(np.real(np.trace(r)) - 1) < 1e-12 for r in frames)


# --------------------------------------------------------------------------- #
# 密度矩阵关键帧 → 渲染管线
# --------------------------------------------------------------------------- #

def test_bloch_vectors_density_matches_per_frame():
    """bloch_vectors 密度路径 ≡ 逐帧 DensityMatrix.bloch()。"""
    from gpuqviz.evolve import bloch_vectors

    qc = _build_circuit()
    gates = _qiskit_to_gates(qc)
    frames = evolve_density(3, gates)
    batch = bloch_vectors(frames, n_qubits=3)
    assert batch.shape == (len(frames), 3, 3)
    for k, rho in enumerate(frames):
        per_frame = DensityMatrix(rho).bloch()
        assert np.allclose(batch[k], per_frame, atol=1e-10), f"frame {k}"


def test_bloch_vectors_density_physical_bound():
    """物理界：任意混合态的单 qubit Bloch 模长 ≤ 1（回归护栏）。"""
    from gpuqviz.evolve import bloch_vectors
    from gpuqviz.noise import depolarizing as dep

    qc = _build_circuit()
    gates = _qiskit_to_gates(qc)
    frames = evolve_density(3, gates,
                            noise=lambda g, i: dep(0.3, n_qubits=2)
                            if len(g.controls) else dep(0.3))
    batch = bloch_vectors(frames, n_qubits=3)
    assert (np.linalg.norm(batch, axis=2) <= 1.0 + 1e-9).all()


def test_render_frame_density_keyframes_cpu(tmp_path, monkeypatch):
    """CPU 渲染含密度矩阵关键帧的 bloch track（含噪 npz 端到端冒烟）。"""
    import gpuqviz
    from gpuqviz import BlochTrack, Scene
    import gpuqviz.backends as _backends

    monkeypatch.setattr(_backends, "_cached", "cpu")

    qc = _build_circuit()
    gates = _qiskit_to_gates(qc)
    from gpuqviz.noise import depolarizing as dep

    frames = evolve_density(3, gates,
                            noise=lambda g, i: dep(0.15, n_qubits=2)
                            if len(g.controls) else dep(0.15))
    npz = tmp_path / "noisy.npz"
    np.savez(npz, states=np.stack(frames))

    scene = Scene(width=640, height=360, fps=30, duration=1.0,
                  tracks=[BlochTrack(states_path="noisy.npz", layout="full")])
    out = tmp_path / "noisy.png"
    gpuqviz.render_frame(scene=scene, t=0.8, out=out, scale=1,
                         states_dir=tmp_path)
    from PIL import Image

    arr = np.array(Image.open(out).convert("RGB"))
    bg = np.array([11, 14, 20])
    nonbg = (np.abs(arr.astype(int) - bg).sum(axis=2) > 30).sum()
    assert nonbg > 500, "含噪演化渲染应有可见内容"
