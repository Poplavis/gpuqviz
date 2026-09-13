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

    对最后一维做 slerp；模长信息会丢失（Bloch 向量应为单位向量，先归一化）。
    """
    xp = _xp(keys)
    keys = keys.astype(xp.float64)
    norms = xp.linalg.norm(keys, axis=-1, keepdims=True)
    unit = keys / xp.where(norms > 1e-12, norms, 1.0)

    k = unit.shape[0]
    if k < 2:
        return xp.repeat(unit, out_frames, axis=0)

    # 每个输出帧所属的关键帧区间 [i, i+1] 与权重 w
    t = xp.linspace(0.0, 1.0, out_frames)
    pos = t * (k - 1)
    idx = xp.clip(xp.floor(pos).astype(xp.int64), 0, k - 2)
    w = (pos - idx).reshape(-1, *([1] * (unit.ndim - 1)))  # (F,1,...)

    v0 = unit[idx]        # (F,...,3)
    v1 = unit[idx + 1]    # (F,...,3)
    dot = xp.clip(xp.sum(v0 * v1, axis=-1, keepdims=True), -1.0, 1.0)
    theta = xp.arccos(dot)
    sin_t = xp.sin(theta)
    # sinθ≈0（共线）时退化为线性插值；对跖点（θ≈π）走正交轴旋转路径
    safe = xp.where(sin_t > 1e-8, sin_t, 1.0)
    slerp = (xp.sin((1 - w) * theta) * v0 + xp.sin(w * theta) * v1) / safe
    linear = (1 - w) * v0 + w * v1

    antipodal = dot < -1.0 + 1e-6
    if xp.any(antipodal):
        # 为每个对跖帧对选一条与 v0 垂直的轴，绕其匀速旋转 180°
        ref = xp.zeros_like(v0)
        ref[..., 0] = 1.0
        parallel = xp.abs(v0[..., 0:1]) > 0.9
        axis = xp.where(parallel, xp.roll(ref, 1, axis=-1), ref)
        axis = axis - xp.sum(axis * v0, axis=-1, keepdims=True) * v0
        axis = axis / xp.where(
            xp.linalg.norm(axis, axis=-1, keepdims=True) > 1e-12,
            xp.linalg.norm(axis, axis=-1, keepdims=True), 1.0,
        )
        omega = xp.pi * w
        rot = xp.cos(omega) * v0 + xp.sin(omega) * axis
        out = xp.where(antipodal, rot, xp.where(sin_t > 1e-8, slerp, linear))
    else:
        out = xp.where(sin_t > 1e-8, slerp, linear)
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
