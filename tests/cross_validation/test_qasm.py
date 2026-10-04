"""P5.2 Roadmap ②：OpenQASM 直接输入对拍（qiskit 参考逐位 1e-10）。"""

import numpy as np
import pytest

from gpuqviz.adapters import load_qasm, resolve_circuit_input, to_key_states

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector as QkSV  # noqa: E402

TOL = 1e-10

QASM_BELL = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
'''


def _final_state(n: int, gates_qasm_source: str) -> np.ndarray:
    """QASM 输入 → 末态。"""
    frames = to_key_states(load_qasm(gates_qasm_source), steps=10)
    return np.asarray(frames[-1])


# --------------------------------------------------------------------------- #
# 文本 / 文件输入
# --------------------------------------------------------------------------- #

def test_qasm2_text_input_matches_qiskit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ref = np.asarray(QkSV.from_instruction(qc).data)
    ours = _final_state(2, QASM_BELL)
    assert np.allclose(ours, ref, atol=TOL)


def test_qasm2_file_input(tmp_path):
    p = tmp_path / "bell.qasm"
    p.write_text(QASM_BELL, encoding="utf-8")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ref = np.asarray(QkSV.from_instruction(qc).data)
    ours = _final_state(2, str(p))
    assert np.allclose(ours, ref, atol=TOL)


def test_resolve_passthrough_non_qasm():
    """qiskit QC / pyqpanda QProg 等对象原样透传。"""
    qc = QuantumCircuit(2)
    assert resolve_circuit_input(qc) is qc


# --------------------------------------------------------------------------- #
# 门覆盖（qasm2 dumps 往返）
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("op_name", ["h", "x", "t", "s"])
def test_qasm2_roundtrip_gate_translation(op_name):
    """qiskit 电路 → qasm2.dumps → load_qasm → 态矢量与原电路一致。"""
    qc = QuantumCircuit(2)
    getattr(qc, op_name)(0)
    qc.cx(0, 1)
    qasm_text = qiskit.qasm2.dumps(qc)
    ours = _final_state(2, qasm_text)
    ref = np.asarray(QkSV.from_instruction(qc).data)
    assert np.allclose(ours, ref, atol=TOL), f"op={op_name}"


def test_qasm2_roundtrip_param_gates():
    qc = QuantumCircuit(3)
    qc.ry(0.7, 0)
    qc.rz(1.2, 1)
    qc.rx(0.4, 2)
    qc.t(1)
    qc.cz(0, 2)
    qasm_text = qiskit.qasm2.dumps(qc)
    ours = _final_state(3, qasm_text)
    ref = np.asarray(QkSV.from_instruction(qc).data)
    assert np.allclose(ours, ref, atol=TOL)


def test_qasm2_final_measurements_stripped():
    """末尾测量按适配器既有约定剔除（不塌缩）。"""
    qasm = QASM_BELL + "measure q -> c;\n"
    ours = _final_state(2, qasm)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ref = np.asarray(QkSV.from_instruction(qc).data)
    assert np.allclose(ours, ref, atol=TOL)


# --------------------------------------------------------------------------- #
# QASM3（可选依赖 qiskit-qasm3-import）
# --------------------------------------------------------------------------- #

def _qasm3_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("qiskit_qasm3_import") is not None


@pytest.mark.skipif(not _qasm3_importable(), reason="qiskit-qasm3-import 未安装")
def test_qasm3_text_input():
    qasm3 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ref = np.asarray(QkSV.from_instruction(qc).data)
    ours = _final_state(2, qasm3)
    assert np.allclose(ours, ref, atol=TOL)


def test_qasm3_missing_dep_hint(monkeypatch):
    """QASM3 输入但可选依赖缺失 → 报带安装提示的 ImportError。"""

    if _qasm3_importable():
        pytest.skip("qiskit-qasm3-import 已安装，无法模拟缺失")
    qasm3 = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nh q[0];\n"
    with pytest.raises(ImportError, match="qiskit-qasm3-import"):
        load_qasm(qasm3)


# --------------------------------------------------------------------------- #
# API / CLI 集成
# --------------------------------------------------------------------------- #

def test_render_frame_accepts_qasm_string(tmp_path, monkeypatch):
    """render_frame(circuit="<OPENQASM...>") 冒烟（CPU 后端）。"""
    import gpuqviz
    import gpuqviz.backends as _backends

    monkeypatch.setattr(_backends, "_cached", "cpu")
    out = tmp_path / "qasm.png"
    gpuqviz.render_frame(circuit=QASM_BELL, t=0.5, out=out, scale=1)
    assert out.exists() and out.stat().st_size > 1000


def test_cli_qasm_command(tmp_path):
    """CLI `gpuqviz qasm` 命令：html / png 两种格式冒烟。"""
    from typer.testing import CliRunner

    from gpuqviz.cli import app

    runner = CliRunner()
    qasm_file = tmp_path / "bell.qasm"
    qasm_file.write_text(QASM_BELL, encoding="utf-8")

    for fmt, ext in (("html", "html"), ("png", "png")):
        out = tmp_path / f"qasm_out.{ext}"
        result = runner.invoke(app, [
            "qasm", str(qasm_file), "--format", fmt, "--out", str(out),
            "--steps", "10", "--seconds", "1",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists() and out.stat().st_size > 1000, f"fmt={fmt}"


def test_cli_qasm_invalid_source():
    from typer.testing import CliRunner

    from gpuqviz.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["qasm", "OPENQASM 2.0; broken syntax ~~"])
    assert result.exit_code == 1
    assert "QASM 解析失败" in result.output
