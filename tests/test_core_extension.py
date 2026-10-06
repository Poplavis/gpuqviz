"""gpuqviz._core C 内核 parity 测试（0.9.0 T3）。

扩展编译缺席（或 GPUQVIZ_DISABLE_CORE=1）时自动跳过；
CI 的 core-parity 矩阵保证两条路径各自全绿。
"""

import os

import numpy as np
import pytest


def _core_available() -> bool:
    if os.environ.get("GPUQVIZ_DISABLE_CORE"):
        return False
    try:
        from gpuqviz import _core  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _core_available(),
                                reason="gpuqviz._core extension not available")

rng = np.random.default_rng(11)


def _rand_unitary(d: int) -> np.ndarray:
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def test_core_import_and_version():
    from gpuqviz import _core

    assert hasattr(_core, "apply_1q_state")
    assert hasattr(_core, "apply_1q_density")


def test_apply_1q_state_matches_numpy():
    """C 内核 vs batched-GEMM 参考：全 target、多规模 1e-12 一致。"""
    from gpuqviz._core import apply_1q_state
    from gpuqviz.circuits import _apply_1q

    for n in (1, 3, 6):
        psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        for q in range(n):
            U = _rand_unitary(2)
            got = apply_1q_state(np.ascontiguousarray(psi, np.complex128), U, q)
            ref = _apply_1q(psi.copy(), U, q)
            assert np.allclose(got, ref, atol=1e-12), (n, q)
        # 纯函数契约：输入不被修改
        snap = psi.copy()
        apply_1q_state(np.ascontiguousarray(psi, np.complex128),
                       _rand_unitary(2), 0)
        assert np.array_equal(psi, snap)


def test_apply_1q_density_matches_einsum():
    """C 内核 vs 显式矩阵参考：全 target、多规模 1e-10 一致。"""
    from gpuqviz._core import apply_1q_density
    from gpuqviz.noise import _CORE_OK, apply_matrix_density

    assert _CORE_OK
    for n in (1, 3, 6):
        R = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
        R = R @ R.conj().T
        R /= np.trace(R).real
        R = np.ascontiguousarray(R, np.complex128)
        for q in range(n):
            U = _rand_unitary(2)
            got = apply_1q_density(R, U, q)
            # 独立参考：显式 kron 矩阵 UρU†
            full = np.kron(np.kron(np.eye(2 ** (n - 1 - q)), U), np.eye(2**q))
            ref2 = full @ R @ full.conj().T
            assert np.allclose(got, ref2, atol=1e-10), (n, q)
            # noise.apply_matrix_density 的 k==1 路径应走同一 C 内核
            ref = apply_matrix_density(R.copy(), U, [q])
            assert np.allclose(got, ref, atol=1e-12), (n, q)


def test_apply_1q_density_pure_function():
    from gpuqviz._core import apply_1q_density

    R = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    R = np.ascontiguousarray(R @ R.conj().T)
    snap = R.copy()
    apply_1q_density(R, _rand_unitary(2), 1)
    assert np.array_equal(R, snap)
