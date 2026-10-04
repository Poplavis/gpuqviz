"""P5.2 MPS 后端对拍：精确模式 ≡ 态矢量（1e-10）+ 截断误差有界 + 纠缠谱。"""

import numpy as np
import pytest

from gpuqviz.circuits import Gate
from gpuqviz.mps import evolve_mps
from gpuqviz.state import partial_trace

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector as QkSV  # noqa: E402

TOL = 1e-10
SQ2 = 1 / np.sqrt(2)


def _random_circuit_gates(seed: int, n: int, depth: int) -> list[Gate]:
    """随机 1q + 最近邻 CX 电路（MPS 支持范围内）。"""
    rng = np.random.Generator(np.random.PCG64(seed))
    gates = []
    for _ in range(depth):
        q = int(rng.integers(n))
        gates.append(Gate(name="RY", targets=[q],
                          params=[float(rng.uniform(0, np.pi))]))
        if n > 1:
            t = int(rng.integers(n - 1))
            gates.append(Gate(name="CX", targets=[t + 1], controls=[t]))
    return gates


def _statevector_ref(n: int, gates: list[Gate]) -> np.ndarray:
    from gpuqviz.circuits import evolve_gates

    return evolve_gates(n, gates)[-1]


# --------------------------------------------------------------------------- #
# 精确模式
# --------------------------------------------------------------------------- #

def test_mps_exact_matches_statevector():
    for seed in range(5):
        gates = _random_circuit_gates(seed, 6, 6)
        ours = evolve_mps(6, gates).frames[-1].to_statevector()
        ref = _statevector_ref(6, gates)
        assert np.allclose(ours, ref, atol=TOL), f"seed={seed}"


def test_mps_reduced_rho_matches_partial_trace():
    for seed in range(3):
        gates = _random_circuit_gates(seed + 50, 6, 4)
        frame = evolve_mps(6, gates).frames[-1]
        psi = frame.to_statevector()
        rho_full = np.outer(psi, psi.conj())
        for q in range(6):
            ours = frame.reduced_rho(q)
            ref = partial_trace(rho_full, [q], 6)
            assert np.allclose(ours, ref, atol=TOL), f"seed={seed} q={q}"


def test_mps_ghz_structure():
    """n=10 GHZ：割熵 = 1 bit，单 qubit 约化 = I/2（最大混合）。"""
    gates = [Gate(name="H", targets=[0])]
    for q in range(9):
        gates.append(Gate(name="CX", targets=[q + 1], controls=[q]))
    result = evolve_mps(10, gates)
    frame = result.frames[-1]
    psi = frame.to_statevector()
    assert abs(psi[0]) - SQ2 < 1e-10 and abs(psi[-1]) - SQ2 < 1e-10
    for cut in range(9):
        assert abs(frame.entanglement_entropy(cut) - 1.0) < 1e-9, f"cut={cut}"
    for q in range(10):
        rho = frame.reduced_rho(q)
        assert np.allclose(rho, np.eye(2) / 2, atol=1e-10), f"q={q}"


def test_mps_bloch_keys():
    gates = _random_circuit_gates(7, 5, 3)
    result = evolve_mps(5, gates)
    bloch = result.bloch_keys()
    assert bloch.shape == (len(result.frames), 5, 3)
    assert (np.linalg.norm(bloch, axis=2) <= 1.0 + 1e-9).all()
    # 与纯态直算一致
    from gpuqviz.evolve import bloch_vectors as sv_bloch

    ref = sv_bloch([f.to_statevector() for f in result.frames])
    assert np.allclose(bloch, ref, atol=1e-10)


# --------------------------------------------------------------------------- #
# 截断
# --------------------------------------------------------------------------- #

def test_mps_truncated_error_bounded():
    """χ=2 截断：弱纠缠电路上误差有界且态保持归一。"""
    gates = _random_circuit_gates(13, 8, 8)
    exact = evolve_mps(8, gates).frames[-1].to_statevector()
    truncated = evolve_mps(8, gates, chi_max=2).frames[-1].to_statevector()
    assert abs(np.linalg.norm(truncated) - 1.0) < 1e-10
    err = np.linalg.norm(truncated - exact)
    assert err < 0.5, f"χ=2 truncation error {err:.3f}"


def test_mps_truncated_exact_when_low_entanglement():
    """低纠缠电路（单 qubit 层，无纠缠门）χ=1 也应精确。"""
    gates = [Gate(name="RY", targets=[q], params=[0.3 * q + 0.1])
             for q in range(6)]
    exact = evolve_mps(6, gates).frames[-1].to_statevector()
    trunc = evolve_mps(6, gates, chi_max=1).frames[-1].to_statevector()
    assert np.allclose(trunc, exact, atol=1e-10)


# --------------------------------------------------------------------------- #
# SWAP 路由
# --------------------------------------------------------------------------- #

def test_mps_long_range_cx_routed():
    """非相邻 CX(0, 3) 经 SWAP 路由后与 qiskit 逐位一致。"""
    for ctrl, tgt in [(0, 3), (3, 0), (1, 4)]:
        gates = [Gate(name="H", targets=[0]),
                 Gate(name="RY", targets=[2], params=[0.7]),
                 Gate(name="CX", targets=[tgt], controls=[ctrl])]
        ours = evolve_mps(5, gates).frames[-1].to_statevector()

        qc = QuantumCircuit(5)
        qc.h(0)
        qc.ry(0.7, 2)
        qc.cx(ctrl, tgt)
        ref = np.asarray(QkSV.from_instruction(qc).data)
        assert np.allclose(ours, ref, atol=TOL), f"({ctrl},{tgt})"
        # 路由确实发生了
        assert evolve_mps(5, gates).n_swaps > 0


def test_mps_routing_layout_restored():
    """路由后布局恢复：长-range 门之后再作用 1q 门，位置语义不变。"""
    gates = [Gate(name="X", targets=[0]),
             Gate(name="CX", targets=[3], controls=[0]),
             Gate(name="X", targets=[0])]  # 若布局未恢复，此 X 会打在错误内容上
    ours = evolve_mps(4, gates).frames[-1].to_statevector()
    ref = _statevector_ref(4, gates)
    assert np.allclose(ours, ref, atol=TOL)


def test_mps_rejects_measure():
    gates = [Gate(name="MEASURE", targets=[0], params=[0])]
    with pytest.raises(NotImplementedError, match="measurement"):
        evolve_mps(1, gates)


def test_mps_rejects_3q():
    gates = [Gate(name="CCX", targets=[2], controls=[0, 1])]
    with pytest.raises(NotImplementedError, match="1q/2q"):
        evolve_mps(3, gates)
