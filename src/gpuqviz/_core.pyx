# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""gpuqviz._core：融合 C 内核（0.9.0 T3）。

编译缺席时自动回退 numpy 路径（circuits/noise 侧 try-import；
GPUQVIZ_DISABLE_CORE=1 可强制禁用，供 CI 对拍两条路径）。

内核（小端序索引：idx 的位 q 即 qubit q）：
- ``apply_1q_state``：单比特门作用到态矢量——两遍无拷贝更新；
- ``apply_1q_density``：单比特门作用到密度矩阵 UρU†——行/列两遍
  strided 更新，消除 numpy einsum 内部的 transpose-reshape 拷贝
  （profiling：reshape 占 evolve_density 时长 ~45%）。
"""

cimport cython


ctypedef double complex dcomplex


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_1q_state(dcomplex[::1] psi not None,
                   dcomplex[:, ::1] U not None,
                   int target):
    """单比特门作用到态矢量，返回新数组（纯函数契约）。"""
    cdef Py_ssize_t dim = psi.shape[0]
    cdef Py_ssize_t n = 0, d = dim
    while d > 1:
        d >>= 1
        n += 1
    cdef Py_ssize_t q = target
    cdef Py_ssize_t lmask = (1 << q) - 1
    cdef Py_ssize_t i, h, l, base
    cdef dcomplex x0, x1
    out = psi.copy()
    cdef dcomplex[::1] outv = out
    cdef dcomplex u00 = U[0, 0], u01 = U[0, 1], u10 = U[1, 0], u11 = U[1, 1]
    with nogil:
        for i in range(dim >> (q + 1)):
            h = i << (q + 1)
            for l in range(lmask + 1):
                base = h | l
                x0 = outv[base]
                x1 = outv[base + (1 << q)]
                outv[base] = u00 * x0 + u01 * x1
                outv[base + (1 << q)] = u10 * x0 + u11 * x1
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_1q_density(dcomplex[:, ::1] rho not None,
                     dcomplex[:, ::1] U not None,
                     int target):
    """UρU†（单比特门，qubit=target）：行/列两遍 strided 更新，返回新矩阵。

    第一遍作用行（U·ρ），第二遍作用列（·U†，共轭）。每遍 O(4·d) 次乘加、
    零张量拷贝——对比 numpy einsum 路径的 ~4 次 4^n 中间拷贝。
    """
    cdef Py_ssize_t d = rho.shape[0]
    cdef Py_ssize_t n = 0, dd = d
    while dd > 1:
        dd >>= 1
        n += 1
    cdef Py_ssize_t q = target
    cdef Py_ssize_t lmask = (1 << q) - 1
    cdef Py_ssize_t nblocks = d >> (q + 1)
    cdef Py_ssize_t i, h, l, c, r, base_r, base_c
    cdef dcomplex x0, x1
    out_arr = rho.copy()
    cdef dcomplex[:, ::1] out = out_arr
    cdef dcomplex u00 = U[0, 0], u01 = U[0, 1], u10 = U[1, 0], u11 = U[1, 1]
    cdef dcomplex v00 = u00.conjugate(), v01 = u01.conjugate()
    cdef dcomplex v10 = u10.conjugate(), v11 = u11.conjugate()
    cdef dcomplex r0, r1
    with nogil:
        # 行变换：ρ'[a 行块] = U[a,i]·ρ[i 行块]（列不动）
        for i in range(nblocks):
            h = i << (q + 1)
            for l in range(lmask + 1):
                for c in range(d):
                    base_r = h | l
                    x0 = out[base_r, c]
                    x1 = out[base_r + (1 << q), c]
                    out[base_r, c] = u00 * x0 + u01 * x1
                    out[base_r + (1 << q), c] = u10 * x0 + u11 * x1
        # 列变换：ρ''[r, b 列块] = Σ_j ρ'[r, j 列块]·conj(U[b,j])（行不动）
        for i in range(nblocks):
            h = i << (q + 1)
            for l in range(lmask + 1):
                for r in range(d):
                    base_c = h | l
                    x0 = out[r, base_c]
                    x1 = out[r, base_c + (1 << q)]
                    out[r, base_c] = x0 * v00 + x1 * v01
                    out[r, base_c + (1 << q)] = x0 * v10 + x1 * v11
    return out_arr
