"""适配层测试：ORIGINIR 解析、numpy 演化数值、pyqpanda/qiskit 一致性。"""

import numpy as np
import pytest

from gpuqviz.circuits import evolve_gates, parse_originir, sample_originir


def test_originir_bell():
    n, gates = parse_originir("QINIT 2\nCREG 1\nH q[0]\nCNOT q[0],q[1]\n")
    assert n == 2
    final = evolve_gates(n, gates)[-1]
    assert np.allclose(final, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


def test_originir_rx_sign():
    """RX(pi/2)|0> = (|0> - i|1>)/sqrt2。"""
    _, gates = parse_originir("QINIT 1\nCREG 1\nRX q[0],(1.5707963)\n")
    final = evolve_gates(1, gates)[-1]
    assert final[0] == pytest.approx(1 / np.sqrt(2))
    assert final[1] == pytest.approx(-1j / np.sqrt(2))


def test_originir_swap_and_endianness():
    """X q[1] 得到 q1=1（index 2），SWAP 后 q0=1（index 1）——验证小端序。"""
    _, gates = parse_originir("QINIT 2\nCREG 1\nX q[1]\nSWAP q[0],q[1]\n")
    st = evolve_gates(2, gates)
    assert np.allclose(st[1], [0, 0, 1, 0])  # X q[1]: q1=1
    assert np.allclose(st[2], [0, 1, 0, 0])  # SWAP: q0=1


def test_originir_cz_control_order():
    """CZ q[1],q[0] 与 CZ q[0],q[1] 等价（对称门），且对 |11> 分量相位翻转。"""
    # H q0 制备叠加，X q1 使 |01>/<11> 分量都有：CZ 只翻转 |11> 项
    originir = "QINIT 2\nCREG 1\nH q[0]\nX q[1]\nCZ {order}\n"
    a = evolve_gates(2, parse_originir(originir.format(order="q[0],q[1]"))[1])[-1]
    b = evolve_gates(2, parse_originir(originir.format(order="q[1],q[0]"))[1])[-1]
    assert np.allclose(a, b)
    # 无 CZ 时 |11> 分量为 +1/2，加 CZ 后应为 -1/2（相位翻转可见）
    no_cz = evolve_gates(2, parse_originir(
        "QINIT 2\nCREG 1\nH q[0]\nX q[1]\n")[1])[-1]
    assert a[3] == pytest.approx(-no_cz[3])
    assert a[1] == pytest.approx(no_cz[1])


def test_sample_originir_count_and_norm():
    keys = sample_originir("QINIT 1\nCREG 1\nH q[0]\nT q[0]\n", steps=6)
    assert len(keys) == 6
    for st in keys:
        assert np.linalg.norm(st) == pytest.approx(1.0, abs=1e-9)


def test_qiskit_vs_pyqpanda_fidelity():
    """同一逻辑电路在 qiskit 与 pyqpanda 适配器下保真度 = 1。"""
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("pyqpanda")

    from gpuqviz.adapters import sample_pyqpanda, to_key_states
    from pyqpanda import CPUQVM, QProg, H, RX, CNOT, RY  # noqa: F401

    theta = 0.6
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.ry(0.4, 1)
    qk_states = to_key_states(qc, steps=8)

    qm = CPUQVM()
    qm.init_qvm()
    try:
        q = qm.qAlloc_many(2)
        prog = QProg()
        prog << H(q[0]) << RX(q[0], theta) << CNOT(q[0], q[1]) << RY(q[1], 0.4)
        pq_states = sample_pyqpanda(prog, steps=8, machine=qm)
    finally:
        qm.finalize()

    fidelity = abs(np.vdot(qk_states[-1], pq_states[-1])) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)


def test_to_key_states_auto_detect():
    qiskit = pytest.importorskip("qiskit")
    from gpuqviz.adapters import to_key_states

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    keys = to_key_states(qc, steps=4)
    assert len(keys) == 4 and keys[0].shape == (2,)
    assert np.allclose(keys[-1], [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_export_html_accepts_pyqpanda_prog(tmp_path):
    pytest.importorskip("pyqpanda")
    from gpuqviz import export_html
    from pyqpanda import CPUQVM, QProg, H  # noqa: F401

    qm = CPUQVM()
    qm.init_qvm()
    try:
        q = qm.qAlloc_many(1)
        prog = QProg()
        prog << H(q[0])
        out = export_html(circuit=prog, steps=6, fps=10, duration=0.6,
                          out=tmp_path / "v.html", machine=qm)
    finally:
        qm.finalize()
    assert out.exists() and out.stat().st_size > 100_000
