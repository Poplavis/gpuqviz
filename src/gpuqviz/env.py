"""环境能力自检：探测 CUDA / OpenGL / NVENC / qiskit，只降级不抛异常。"""

from __future__ import annotations

import os
import sys
import traceback


def _check(label: str, fn):
    """运行一个探测函数，返回 (状态标记, 详情文本)。任何异常都被捕获为 FAILED。"""
    try:
        detail = fn()
        return "OK", detail
    except ImportError:
        return "MISSING", "not installed"
    except Exception as e:  # noqa: BLE001
        return "FAILED", f"{type(e).__name__}: {e}"


def _check_cupy() -> str:
    import cupy

    name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    free, total = cupy.cuda.runtime.memGetInfo()
    return f"cupy {cupy.__version__}, GPU {name}, VRAM {total >> 20}/{free >> 20} MiB free"


def _check_gl() -> str:
    import moderngl

    ctx = moderngl.create_standalone_context()
    info = ctx.info.get("GL_RENDERER", "?")
    ctx.release()
    return f"moderngl {moderngl.__version__}, renderer {info}"


def _check_nvenc() -> str:
    from .encode import nvenc_available

    if nvenc_available():
        import PyNvVideoCodec
        return f"PyNvVideoCodec {getattr(PyNvVideoCodec, '__version__', 'unknown')} (session OK)"
    return "session unavailable"


def _check_pyav() -> str:
    import av

    return f"PyAV {av.__version__}"


def _check_qiskit() -> str:
    import qiskit

    return f"qiskit {qiskit.__version__}"


def _check_numba() -> str:
    import numba

    return f"numba {numba.__version__}"


def _check_interop() -> str:
    """CUDA-GL interop 探测（S5 优化路径的可用性，未启用时 pinned 路径兜底）。"""
    if sys.platform == "win32":
        # WGL/NV interop 需要共享设备上下文，探测复杂度高：报告"未启用"
        return "available in principle (not enabled; pinned-memory path active)"
    raise ImportError("interop probe only reported on Windows; Linux uses EGL path")


def _actual_render_path() -> str:
    """返回实际将使用的渲染路径（尊重 GPUQVIZ_BACKEND 环境变量与探测结果）。"""
    env = os.environ.get("GPUQVIZ_BACKEND", "").strip().lower()
    if env in ("gl", "cpu"):
        return f"cpu (GPUQVIZ_BACKEND={env})" if env == "cpu" else f"gl (GPUQVIZ_BACKEND={env})"
    from .backends import detect_backend

    return detect_backend()


def report_env() -> str:
    """返回多行环境报告文本。任何一项缺失只影响结论，不抛异常。"""
    checks = [
        ("Python", lambda: sys.version.split()[0]),
        ("numpy", lambda: __import__("numpy").__version__),
        ("CUDA (cupy)", _check_cupy),
        ("OpenGL (moderngl)", _check_gl),
        ("NVENC (pynvvideocodec)", _check_nvenc),
        ("Encoder (PyAV)", _check_pyav),
        ("qiskit", _check_qiskit),
        ("numba (cpu-fallback)", _check_numba),
        ("CUDA-GL interop", _check_interop),
    ]

    rows = []
    statuses = {}
    for label, fn in checks:
        status, detail = _check(label, fn)
        statuses[label] = status
        rows.append(f"  {label:<24} {status:<8} {detail}")

    # 实际渲染路径（S8 新增列）
    try:
        render_path = _actual_render_path()
    except Exception:  # noqa: BLE001
        render_path = "unknown"
    rows.append(f"  {'实际渲染路径':<24} {'':8} {render_path}")

    backend = "cuda" if statuses["CUDA (cupy)"] == "OK" and statuses["OpenGL (moderngl)"] == "OK" else "cpu"

    return "\n".join(
        ["gpuqviz environment report", *rows, "", f"  推荐后端: {backend}"]
    )


if __name__ == "__main__":
    try:
        print(report_env())
    except Exception:  # pragma: no cover - 兜底，保证 CLI 永不崩溃
        traceback.print_exc()
        sys.exit(1)
