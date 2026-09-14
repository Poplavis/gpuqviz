"""S6 高层门/复杂电路兼容性测试矩阵。

对 ≥12 个电路断言：gpuqviz 门级 numpy 模拟器（qiskit_to_gates →
evolve_gates）与 qiskit Statevector 的逐层快照/末态全局保真度 ≥ 1-1e-9。
"""

import numpy as np
import pytest

from gpuqviz.adapters import qiskit_to_gates
from gpuqviz.circuits import Gate, evolve_gates, sample_snapshots

qiskit = pytest.importorskip("qiskit")


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def _build_matrix():
    """≥12 个电路：覆盖 mcx / 受控参数门 / 复合门递归 / initialize 类 /
    measure / UnitaryGate / global phase / barrier / QFT / 嵌套定义。"""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate, StatePreparation, UnitaryGate

    circs: dict[str, QuantumCircuit] = {}

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    circs["bell"] = qc

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    circs["ghz3"] = qc

    qc = QuantumCircuit(3)
    qc.append(QFTGate(3), range(3))
    circs["qft3"] = qc

    # Grover oracle：mcx 两控制 + H 包装（相位踢回）
    qc = QuantumCircuit(3)
    qc.x(2)
    qc.h(2)
    qc.mcx([0, 1], 2)
    qc.h(2)
    qc.x(2)
    circs["mcx_oracle"] = qc

    # 受控参数门全家
    qc = QuantumCircuit(2)
    qc.crz(0.7, 0, 1)
    qc.crx(-0.4, 1, 0)
    qc.ch(0, 1)
    qc.cp(1.1, 0, 1)
    circs["controlled_family"] = qc

    # StatePreparation（复合门递归 + 任意初态）
    vec = np.array([1, 1j, -1, -1j]) / 2
    qc = QuantumCircuit(2)
    qc.append(StatePreparation(vec), [0, 1])
    circs["state_preparation"] = qc

    # 中途 measure + measure_all（剔除 → 层边界）
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    circs["measure_all"] = qc

    # 任意酉矩阵
    u = np.array([[0, 1, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                 dtype=complex)
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(u), [0, 1])
    circs["unitary_gate"] = qc

    # 全局相位（不影响保真度，验证可忽略）
    qc = QuantumCircuit(1)
    qc.global_phase = 0.3
    qc.rz(0.9, 0)
    qc.sdg(0)
    circs["global_phase"] = qc

    # barrier 分层
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)
    circs["barrier"] = qc

    # 自定义复合门（to_gate + definition 递归）
    inner = QuantumCircuit(2, name="inner")
    inner.ry(0.5, 0)
    inner.cx(0, 1)
    outer = QuantumCircuit(3)
    outer.append(inner.to_gate(), [0, 1])
    outer.cx(1, 2)
    circs["custom_gate"] = outer

    # 嵌套复合门（定义里再套定义）
    lvl1 = QuantumCircuit(1, name="lvl1")
    lvl1.rz(0.3, 0)
    lvl2 = QuantumCircuit(2, name="lvl2")
    lvl2.append(lvl1.to_gate(), [0])
    lvl2.cx(0, 1)
    qc = QuantumCircuit(3)
    qc.append(lvl2.to_gate(), [1, 2])
    qc.h(0)
    circs["nested_gate"] = qc

    # cu / 多控制相位
    qc = QuantumCircuit(3)
    qc.cu(0.3, 0.2, 0.1, 0, 0, 1)
    qc.mcp(0.5, [0, 1], 2)
    circs["cu_mcp"] = qc

    # mcx 深层链（4 qubit，3 控制）
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.mcx([0, 1, 2], 3)
    qc.swap(0, 3)
    circs["mcx_chain"] = qc

    return circs


def _reference_snapshots(circuit):
    """qiskit 逐指令演化参考快照（跳过 barrier/measure）。"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    circuit = circuit.remove_final_measurements(inplace=False)
    n = circuit.num_qubits
    sv = Statevector.from_int(0, 2**n)
    snaps = [np.asarray(sv.data)]
    for inst in circuit.data:
        op = inst.operation
        if op.name in ("barrier", "measure", "reset", "delay"):
            continue
        sub = QuantumCircuit(n)
        sub.append(op, [circuit.find_bit(q).index for q in inst.qubits])
        sv = sv.evolve(sub)
        snaps.append(np.asarray(sv.data))
    return snaps


@pytest.mark.parametrize("name", sorted(_build_matrix().keys()))
def test_fidelity_vs_qiskit(name):
    from qiskit.quantum_info import Statevector

    circuit = _build_matrix()[name]
    n, gates = qiskit_to_gates(circuit)
    assert n == circuit.num_qubits

    ours = evolve_gates(n, gates)
    ref = _reference_snapshots(circuit)

    # 末态保真度
    assert _fidelity(ours[-1], ref[-1]) >= 1 - 1e-9, f"{name}: final fidelity"
    # 翻译为 1:1 指令时逐层快照一致（两边都按非 barrier 对齐）；
    # definition 展开会产生多倍 gate，只对比末态
    n_ours = sum(1 for g in gates if g.name.upper() != "BARRIER")
    if n_ours == len(ref) - 1:
        assert len(ours) == len(ref)
        for i, (a, b) in enumerate(zip(ours, ref)):
            assert _fidelity(a, b) >= 1 - 1e-9, f"{name}: snapshot {i}"
    else:
        assert np.allclose(np.linalg.norm(ours[-1]), 1.0)


def test_sample_snapshots_endpoints():
    """采样后首末帧仍为初态/末态。"""
    circuit = _build_matrix()["bell"]
    n, gates = qiskit_to_gates(circuit)
    snaps = sample_snapshots(evolve_gates(n, gates), 24)
    assert len(snaps) == 24
    assert _fidelity(snaps[0], np.array([1, 0, 0, 0])) >= 1 - 1e-9
    ref = _reference_snapshots(circuit)[-1]
    assert _fidelity(snaps[-1], ref) >= 1 - 1e-9


def test_sample_circuit_with_mid_measure():
    """sample_circuit（qiskit 引擎路径）对含中途 measure 的电路不再崩溃。"""
    from gpuqviz.evolve import sample_circuit

    qc = _build_matrix()["measure_all"]
    states = sample_circuit(qc, steps=6)
    assert len(states) == 6
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-9)


def test_originir_toffoli():
    """ORIGINIR 的 TOFFOLI/ISWAP/U3 扩展解析 + 演化数值正确。"""
    from gpuqviz.circuits import parse_originir

    text = """
QINIT 3
CREG 1
H q[0]
TOFFOLI q[0], q[1], q[2]
U3 q[0],(0.3,0.2,0.1)
ISWAP q[0], q[2]
CRZ q[1], q[2],(0.7)
"""
    n, gates = parse_originir(text)
    assert n == 3
    snaps = evolve_gates(n, gates)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ccx(0, 1, 2)
    qc.u(0.3, 0.2, 0.1, 0)
    qc.iswap(0, 2)
    qc.crz(0.7, 1, 2)
    ref = Operator(qc).data @ np.asarray(Statevector.from_int(0, 8).data)
    assert _fidelity(snaps[-1], ref) >= 1 - 1e-9
