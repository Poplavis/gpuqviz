"""内置算法库正确性测试。

对 14 个经典量子算法验证：
1. 电路构建成功（qiskit / numpy 路径）
2. 数值正确性（用 evolve_gates 计算末态，验证关键性质）
3. 注册表完整性
4. CLI 集成
"""

import numpy as np
import pytest

from gpuqviz.algorithms import (
    ALGORITHM_REGISTRY, get_algorithm, list_algorithms,
    bell, ghz, superposition, grover, qft, phase_estimation,
    deutsch_jozsa, bernstein_vazirani, teleportation, superdense,
    simon, quantum_walk,
)
from gpuqviz.circuits import Gate, evolve_gates

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402  (importorskip 守卫)


def _fidelity(a, b):
    return float(abs(np.vdot(a, b)) ** 2)


def _final_state_qiskit(qc):
    """qiskit QuantumCircuit → 末态矢量（经 qiskit_to_gates + evolve_gates）。"""
    from gpuqviz.adapters import qiskit_to_gates
    n, gates = qiskit_to_gates(qc)
    return evolve_gates(n, gates)[-1]


def _final_state_numpy(n, gates):
    """numpy 路径末态。"""
    return evolve_gates(n, gates)[-1]


# --------------------------------------------------------------------------- #
# 注册表测试
# --------------------------------------------------------------------------- #

class TestRegistry:
    def test_registry_has_14_algorithms(self):
        assert len(ALGORITHM_REGISTRY) == 14

    def test_registry_names(self):
        expected = {
            "bell", "ghz", "superposition", "grover", "qft",
            "phase_estimation", "deutsch_jozsa", "bernstein_vazirani",
            "teleportation", "superdense", "simon", "quantum_walk",
            "shor", "hhl",
        }
        assert set(ALGORITHM_REGISTRY.keys()) == expected

    def test_get_algorithm(self):
        spec = get_algorithm("grover")
        assert spec.name == "grover"
        assert spec.builder is grover
        assert spec.default_n_qubits == 3

    def test_get_algorithm_unknown_raises(self):
        with pytest.raises(KeyError):
            get_algorithm("nonexistent")

    def test_list_algorithms_contains_all(self):
        text = list_algorithms()
        for name in ALGORITHM_REGISTRY:
            assert name in text


# --------------------------------------------------------------------------- #
# Bell 态
# --------------------------------------------------------------------------- #

class TestBell:
    def test_qiskit_returns_circuit(self):
        qc = bell(engine="qiskit")
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_numpy_returns_gate_list(self):
        gates = bell(engine="numpy")
        assert isinstance(gates, list)
        assert all(isinstance(g, Gate) for g in gates)
        assert len(gates) == 2

    def test_final_state_is_bell(self):
        qc = bell(engine="qiskit")
        final = _final_state_qiskit(qc)
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        assert _fidelity(final, expected) > 1 - 1e-9

    def test_numpy_matches_qiskit(self):
        qc = bell(engine="qiskit")
        gates = bell(engine="numpy")
        fq = _final_state_qiskit(qc)
        fn = _final_state_numpy(2, gates)
        assert _fidelity(fq, fn) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# GHZ 态
# --------------------------------------------------------------------------- #

class TestGHZ:
    def test_qiskit_3q(self):
        qc = ghz(n=3, engine="qiskit")
        assert qc.num_qubits == 3

    def test_final_state_ghz3(self):
        qc = ghz(n=3, engine="qiskit")
        final = _final_state_qiskit(qc)
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = expected[7] = 1 / np.sqrt(2)
        assert _fidelity(final, expected) > 1 - 1e-9

    def test_numpy_matches_qiskit(self):
        qc = ghz(n=3, engine="qiskit")
        gates = ghz(n=3, engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9

    def test_n_too_small_raises(self):
        with pytest.raises(ValueError):
            ghz(n=1)


# --------------------------------------------------------------------------- #
# 叠加态
# --------------------------------------------------------------------------- #

class TestSuperposition:
    def test_uniform_probability(self):
        qc = superposition(n=3, engine="qiskit")
        final = _final_state_qiskit(qc)
        probs = np.abs(final) ** 2
        assert all(abs(p - 1 / 8) < 1e-9 for p in probs)


# --------------------------------------------------------------------------- #
# Grover
# --------------------------------------------------------------------------- #

class TestGrover:
    def test_qiskit_3q_marked_101(self):
        qc = grover(n=3, marked=0b101, iterations=2, engine="qiskit")
        assert qc.num_qubits == 3
        final = _final_state_qiskit(qc)
        p101 = float(np.abs(final[5]) ** 2)  # 5 = 0b101
        assert p101 > 0.9

    def test_numpy_matches_qiskit(self):
        qc = grover(n=3, marked=0b101, iterations=2, engine="qiskit")
        gates = grover(n=3, marked=0b101, iterations=2, engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9

    def test_default_iterations(self):
        qc = grover(n=3, marked=0b101, engine="qiskit")
        # 3 qubit: floor(pi/4 * sqrt(8)) = floor(2.22) = 2
        assert qc.num_qubits == 3

    def test_marked_out_of_range_raises(self):
        with pytest.raises(ValueError):
            grover(n=3, marked=8)


# --------------------------------------------------------------------------- #
# QFT
# --------------------------------------------------------------------------- #

class TestQFT:
    def test_qiskit_qft3(self):
        qc = qft(n=3, engine="qiskit")
        assert qc.num_qubits == 3

    def test_qft_inverse_is_identity(self):
        """QFT† ∘ QFT 作用到 |0…0⟩ 应保持 |0…0⟩。"""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3)
        qft_circ = qft(n=3, engine="qiskit")
        iqft_circ = qft(n=3, engine="qiskit", inverse=True)
        qc.compose(qft_circ, inplace=True)
        qc.compose(iqft_circ, inplace=True)
        final = _final_state_qiskit(qc)
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = 1.0
        assert _fidelity(final, expected) > 1 - 1e-6

    def test_numpy_matches_qiskit(self):
        qc = qft(n=3, engine="qiskit")
        gates = qft(n=3, engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# 量子相位估计
# --------------------------------------------------------------------------- #

class TestQPE:
    def test_qiskit_qpe(self):
        qc = phase_estimation(n_count=3, theta=0.375, engine="qiskit")
        assert qc.num_qubits == 4  # 3 计数 + 1 本征态

    def test_phase_encoded_correctly(self):
        """theta=0.375=3/8，计数寄存器应高概率测得 011 (=3)。"""
        qc = phase_estimation(n_count=3, theta=0.375, engine="qiskit")
        final = _final_state_qiskit(qc)
        # 计数寄存器是 q0,q1,q2，本征态 q3
        # 3/8 → 二进制 011（q0=1, q1=1, q2=0，小端序）
        # 末态中 |011⟩|1⟩ = index 0b1011 = 11 应高概率
        p_target = float(np.abs(final[0b1011]) ** 2)
        assert p_target > 0.9

    def test_numpy_matches_qiskit(self):
        qc = phase_estimation(n_count=2, theta=0.25, engine="qiskit")
        gates = phase_estimation(n_count=2, theta=0.25, engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# Deutsch-Jozsa
# --------------------------------------------------------------------------- #

class TestDeutschJozsa:
    def test_balanced(self):
        """balanced oracle → 输入寄存器不全为 0（至少有一个 1）。"""
        qc = deutsch_jozsa(oracle_type="balanced", n=3, engine="qiskit")
        assert qc.num_qubits == 4  # 3 输入 + 1 辅助
        final = _final_state_qiskit(qc)
        # balanced → 输入寄存器不全为 |0…0⟩
        n = 3
        prob_all_zeros = 0.0
        for i in range(2**4):
            bits = [(i >> b) & 1 for b in range(4)]
            if all(bits[b] == 0 for b in range(n)):
                prob_all_zeros += float(np.abs(final[i]) ** 2)
        assert prob_all_zeros < 0.1  # balanced 不应全零

    def test_constant(self):
        qc = deutsch_jozsa(oracle_type="constant", n=2, engine="qiskit")
        final = _final_state_qiskit(qc)
        # constant → 输入寄存器全 0
        n = 2
        prob_all_zeros = 0.0
        for i in range(2**3):
            bits = [(i >> b) & 1 for b in range(3)]
            if all(bits[b] == 0 for b in range(n)):
                prob_all_zeros += float(np.abs(final[i]) ** 2)
        assert prob_all_zeros > 0.9

    def test_numpy_matches_qiskit(self):
        qc = deutsch_jozsa(oracle_type="balanced", n=2, engine="qiskit")
        gates = deutsch_jozsa(oracle_type="balanced", n=2, engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# Bernstein-Vazirani
# --------------------------------------------------------------------------- #

class TestBernsteinVazirani:
    def test_recovers_secret(self):
        qc = bernstein_vazirani(secret="101", engine="qiskit")
        assert qc.num_qubits == 4  # 3 输入 + 1 辅助
        final = _final_state_qiskit(qc)
        # 末态输入寄存器 = |101⟩
        # q0=1, q1=0, q2=1 → |101⟩ = 0b101（仅输入寄存器）
        # 辅助 qubit=3，末态索引 = secret_bits | (aux_bit << aux_pos)
        prob_secret = 0.0
        for i in range(2**4):
            bits = [(i >> b) & 1 for b in range(4)]
            if bits[0] == 1 and bits[1] == 0 and bits[2] == 1:
                prob_secret += float(np.abs(final[i]) ** 2)
        assert prob_secret > 0.9

    def test_numpy_matches_qiskit(self):
        qc = bernstein_vazirani(secret="11", engine="qiskit")
        gates = bernstein_vazirani(secret="11", engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# 量子隐形传态
# --------------------------------------------------------------------------- #

class TestTeleportation:
    def test_qiskit_returns_circuit(self):
        qc = teleportation(engine="qiskit")
        assert qc.num_qubits == 3

    def test_q2_gets_state(self):
        """q2 末态的约化密度矩阵应接近 q0 初始态。"""
        from gpuqviz.evolve import bloch_vectors
        qc = teleportation(engine="qiskit", prepare_state="ry")
        final = _final_state_qiskit(qc)
        # q0 初始 RY(π/4)|0⟩ 的 Bloch 向量
        import math
        theta = math.pi / 4
        expected_bloch = [math.sin(theta), 0, math.cos(theta)]
        # 末态 q2 的 Bloch 向量
        bloch = bloch_vectors([final], n_qubits=3)[0]
        q2_bloch = bloch[2]
        # 隐形传态后 q0,q1 可能纠缠，但 q2 应 ≈ 初始 q0 态
        assert abs(q2_bloch[0] - expected_bloch[0]) < 0.1
        assert abs(q2_bloch[1] - expected_bloch[1]) < 0.1
        assert abs(q2_bloch[2] - expected_bloch[2]) < 0.1

    def test_numpy_matches_qiskit(self):
        qc = teleportation(engine="qiskit")
        gates = teleportation(engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# 超密编码
# --------------------------------------------------------------------------- #

class TestSuperdense:
    @pytest.mark.parametrize("message", ["00", "01", "10", "11"])
    def test_message_decoded(self, message):
        qc = superdense(message=message, engine="qiskit")
        assert qc.num_qubits == 2
        final = _final_state_qiskit(qc)
        # Bob 测量后应得 |message⟩
        bit0, bit1 = int(message[0]), int(message[1])
        # 末态应为 |message⟩ = |bit1, bit0⟩（小端序）
        target_idx = bit1 * 2 + bit0
        prob = float(np.abs(final[target_idx]) ** 2)
        assert prob > 0.9

    def test_numpy_matches_qiskit(self):
        qc = superdense(message="11", engine="qiskit")
        gates = superdense(message="11", engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(2, gates)) > 1 - 1e-9

    def test_invalid_message_raises(self):
        with pytest.raises(ValueError):
            superdense(message="10x")


# --------------------------------------------------------------------------- #
# Simon 算法
# --------------------------------------------------------------------------- #

class TestSimon:
    def test_qiskit_simon(self):
        qc = simon(s="01", engine="qiskit")
        assert qc.num_qubits == 4  # 2 输入 + 2 输出

    def test_numpy_matches_qiskit(self):
        qc = simon(s="01", engine="qiskit")
        gates = simon(s="01", engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(4, gates)) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# 量子随机游走
# --------------------------------------------------------------------------- #

class TestQuantumWalk:
    def test_qiskit_walk(self):
        qc = quantum_walk(n=2, steps=2, engine="qiskit")
        assert qc.num_qubits == 3  # 1 硬币 + 2 位置

    def test_numpy_matches_qiskit(self):
        qc = quantum_walk(n=2, steps=1, engine="qiskit")
        gates = quantum_walk(n=2, steps=1, engine="numpy")
        assert _fidelity(_final_state_qiskit(qc), _final_state_numpy(3, gates)) > 1 - 1e-9

    def test_n1_degenerate(self):
        """n=1 时退化为硬币 + CNOT。"""
        qc = quantum_walk(n=1, steps=1, engine="qiskit")
        assert qc.num_qubits == 2


# --------------------------------------------------------------------------- #
# 引擎参数验证
# --------------------------------------------------------------------------- #

class TestEngineValidation:
    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError):
            bell(engine="invalid")

    def test_pyqpanda_not_implemented_raises(self):
        # bell(engine="pyqpanda") 抛 NotImplementedError
        with pytest.raises(NotImplementedError):
            bell(engine="pyqpanda")
