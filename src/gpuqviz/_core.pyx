# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""gpuqviz._core：融合 C 内核（0.9.0 T3）。

编译缺席时自动回退 numpy 路径（circuits/noise 侧 try-import；
GPUQVIZ_DISABLE_CORE=1 可强制禁用，供 CI 对拍两条路径）。

内核（小端序索引：idx 的位 q 即 qubit q）：
- ``apply_1q_state``：单比特门作用到态矢量——无轴重排单遍更新；
- ``apply_1q_density``：单比特门作用到密度矩阵 UρU†——行/列两遍
  strided 更新，消除 numpy einsum 内部的 transpose-reshape 拷贝
  （profiling：reshape 占 evolve_density 时长 ~45%）。

两个内核都返回新建的 numpy.ndarray（纯函数契约：不修改输入）。
"""

import numpy as np

cimport cython

ctypedef double complex dcomplex


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_1q_state(psi not None, dcomplex[:, ::1] U not None, int target):
    """单比特门作用到态矢量，返回新 ndarray。"""
    psi_arr = np.ascontiguousarray(psi, dtype=np.complex128)
    cdef Py_ssize_t dim = psi_arr.shape[0]
    cdef Py_ssize_t lmask = (1 << target) - 1
    cdef Py_ssize_t i, h, l, base
    out_arr = np.empty_like(psi_arr)
    cdef dcomplex[::1] src = psi_arr
    cdef dcomplex[::1] outv = out_arr
    cdef dcomplex u00 = U[0, 0], u01 = U[0, 1], u10 = U[1, 0], u11 = U[1, 1]
    cdef dcomplex x0, x1
    with nogil:
        for i in range(dim >> (target + 1)):
            h = i << (target + 1)
            for l in range(lmask + 1):
                base = h | l
                x0 = src[base]
                x1 = src[base + (<Py_ssize_t>1 << target)]
                outv[base] = u00 * x0 + u01 * x1
                outv[base + (<Py_ssize_t>1 << target)] = u10 * x0 + u11 * x1
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_1q_density(rho not None, dcomplex[:, ::1] U not None, int target):
    """UρU†（单比特门，qubit=target）：行/列两遍 strided 更新，返回新矩阵。

    第一遍作用行（U·ρ），第二遍作用列（·U†，共轭）。每遍 O(4·d) 次乘加、
    零张量拷贝——对比 numpy einsum 路径的 ~4 次 4^n 中间拷贝。
    """
    rho_arr = np.ascontiguousarray(rho, dtype=np.complex128)
    cdef Py_ssize_t d = rho_arr.shape[0]
    cdef Py_ssize_t lmask = (<Py_ssize_t>1 << target) - 1
    cdef Py_ssize_t nblocks = d >> (target + 1)
    cdef Py_ssize_t i, h, l, c, r, base, bit
    out_arr = np.empty_like(rho_arr)
    cdef dcomplex[:, ::1] src = rho_arr
    cdef dcomplex[:, ::1] out = out_arr
    cdef dcomplex u00 = U[0, 0], u01 = U[0, 1], u10 = U[1, 0], u11 = U[1, 1]
    cdef dcomplex v00 = u00.conjugate(), v01 = u01.conjugate()
    cdef dcomplex v10 = u10.conjugate(), v11 = u11.conjugate()
    cdef dcomplex x0, x1
    bit = (<Py_ssize_t>1) << target
    with nogil:
        # 第一遍：行变换 ρ'[a 行块, c] = U[a,i]·ρ[i 行块, c]（列不动）
        for i in range(nblocks):
            h = i << (target + 1)
            for l in range(lmask + 1):
                for c in range(d):
                    base = h | l
                    x0 = src[base, c]
                    x1 = src[base + bit, c]
                    out[base, c] = u00 * x0 + u01 * x1
                    out[base + bit, c] = u10 * x0 + u11 * x1
        # 第二遍：列变换 ρ''[r, b 列块] = Σ_j ρ'[r, j 列块]·conj(U[b,j])（行不动）
        for i in range(nblocks):
            h = i << (target + 1)
            for l in range(lmask + 1):
                for r in range(d):
                    base = h | l
                    x0 = out[r, base]
                    x1 = out[r, base + bit]
                    out[r, base] = x0 * v00 + x1 * v01
                    out[r, base + bit] = x0 * v10 + x1 * v11
    return out_arr
