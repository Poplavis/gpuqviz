"""GPU 关键帧插值：关键帧数量与输出帧率解耦。

- Bloch 向量序列：球面 slerp（角度线性），保证匀角速度旋转观感。
- 态矢量序列：线性插值后 renormalize（中间态仅用于可视化，不保证物理精确）。
"""

from __future__ import annotations

import numpy as np

try:  # cupy 可选
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


def _xp(a):
    if cp is not None and isinstance(a, cp.ndarray):
        return cp
    return np


def slerp_keys(keys, out_frames: int):
    """关键帧 (K, ..., 3) Bloch 向量球面插值到 out_frames 帧 → (out_frames, ..., 3)。

    保留模长信息：对方向做 slerp（匀角速度），对模长做线性插值，两者相乘。
    当某一端点为零向量（完全混态）时退化为全向量线性插值（自然过零）。
    """
    xp = _xp(keys)
    keys = keys.astype(xp.float64)

    k = keys.shape[0]
    if k < 2:
        return xp.repeat(keys, out_frames, axis=0)

    # 每个输出帧所属的关键帧区间 [i, i+1] 与权重 w
    t = xp.linspace(0.0, 1.0, out_frames)
    pos = t * (k - 1)
    idx = xp.clip(xp.floor(pos).astype(xp.int64), 0, k - 2)
    w = (pos - idx).reshape(-1, *([1] * (keys.ndim - 1)))  # (F,1,...)

    v0 = keys[idx]        # (F,...,3)
    v1 = keys[idx + 1]    # (F,...,3)

    # 模长线性插值（保留混态信息）
    n0 = xp.linalg.norm(v0, axis=-1, keepdims=True)
    n1 = xp.linalg.norm(v1, axis=-1, keepdims=True)
    n_interp = (1 - w) * n0 + w * n1

    # 单位方向
    u0 = v0 / xp.where(n0 > 1e-12, n0, 1.0)
    u1 = v1 / xp.where(n1 > 1e-12, n1, 1.0)

    dot = xp.clip(xp.sum(u0 * u1, axis=-1, keepdims=True), -1.0, 1.0)
    theta = xp.arccos(dot)
    sin_t = xp.sin(theta)
    # sinθ≈0（共线）时退化为线性插值；对跖点（θ≈π）走正交轴旋转路径
    safe = xp.where(sin_t > 1e-8, sin_t, 1.0)
    slerp = (xp.sin((1 - w) * theta) * u0 + xp.sin(w * theta) * u1) / safe
    linear_dir = (1 - w) * u0 + w * u1

    # 任一端点为零向量 → 方向无意义，退化为全向量线性插值
    either_zero = (n0 <= 1e-12) | (n1 <= 1e-12)
    full_linear = (1 - w) * v0 + w * v1

    # 正常情况：方向 slerp × 模长线性插值
    direction = xp.where(sin_t > 1e-8, slerp, linear_dir)
    out = xp.where(either_zero, full_linear, direction * n_interp)

    # 对跖点：为每个对跖帧对选一条与 u0 垂直的轴，绕其匀速旋转 180°
    antipodal = dot < -1.0 + 1e-6
    if xp.any(antipodal):
        ref = xp.zeros_like(u0)
        ref[..., 0] = 1.0
        parallel = xp.abs(u0[..., 0:1]) > 0.9
        axis = xp.where(parallel, xp.roll(ref, 1, axis=-1), ref)
        axis = axis - xp.sum(axis * u0, axis=-1, keepdims=True) * u0
        axis = axis / xp.where(
            xp.linalg.norm(axis, axis=-1, keepdims=True) > 1e-12,
            xp.linalg.norm(axis, axis=-1, keepdims=True), 1.0,
        )
        omega = xp.pi * w
        rot = xp.cos(omega) * u0 + xp.sin(omega) * axis
        out = xp.where(antipodal & ~either_zero, rot * n_interp, out)
    return out


def lerp_states(states, out_frames: int):
    """关键帧态矢量 (K, 2**n) 线性插值 + renormalize → (out_frames, 2**n)。

    适用场景：热图等直接消费态矢量的渲染器。注意插值态只是可视化近似。
    """
    xp = _xp(states)
    states = xp.asarray(states).astype(xp.complex128)
    k = states.shape[0]
    if k < 2:
        return xp.repeat(states, out_frames, axis=0)

    t = xp.linspace(0.0, 1.0, out_frames)
    pos = t * (k - 1)
    idx = xp.clip(xp.floor(pos).astype(xp.int64), 0, k - 2)
    w = (pos - idx).reshape(-1, *([1] * (states.ndim - 1)))

    out = (1 - w) * states[idx] + w * states[idx + 1]
    norms = xp.linalg.norm(out, axis=1, keepdims=True)
    return out / xp.where(norms > 1e-12, norms, 1.0)
