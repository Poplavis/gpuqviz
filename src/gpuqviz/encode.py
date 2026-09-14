"""编码器抽象：帧数据（numpy/cupy）直接 write()，全程无中间文件。

S1 阶段 AvEncoder 在进程内完成 RGBA→YUV420p 转换（FFmpeg swscale，C 实现，
速度可接受）；cupy 输入需 cupy.asnumpy() 拷回 CPU —— S3 将以 NvencEncoder
消除这次拷贝。
"""

from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

try:  # cupy 可选
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

logger = logging.getLogger(__name__)

# codec 逻辑名 → 候选编码器（GPU 优先，软编码兜底）
_CODEC_CANDIDATES = {
    "h264": ["h264_nvenc", "libx264"],
    "hevc": ["hevc_nvenc", "libx265"],
}


def _probe_encoder(name: str) -> bool:
    """真开一次 64x64 编码会话探测可用性。

    add_stream 成功不代表能打开（如 FFmpeg 能找到 nvenc 符号但会话初始化
    失败），失败发生在懒加载的 avcodec_open2 阶段，因此必须实际编码一帧。
    """
    try:
        ctx = av.CodecContext.create(name, "w")
        ctx.width, ctx.height, ctx.pix_fmt = 64, 64, "yuv420p"
        ctx.time_base = Fraction(1, 30)
        frame = av.VideoFrame.from_ndarray(
            np.zeros((64, 64, 4), dtype=np.uint8), format="rgba"
        ).reformat(format="yuv420p")
        frame.pts = 0
        list(ctx.encode(frame))
        list(ctx.encode())
        return True
    except Exception:  # noqa: BLE001 - 任何异常都视为该编码器不可用
        return False


class Encoder:
    """编码器基类。子类实现 _open / write / close，支持上下文管理器。"""

    def __init__(self, width: int, height: int, fps: float, out_path: str | Path,
                 codec: str = "h264", quality: float = 0.9):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.out_path = Path(out_path)
        self.codec = codec
        self.quality = float(quality)
        self._open()

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def write(self, frame) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError


class AvEncoder(Encoder):
    """PyAV 进程内编码。GPU 编码器探测失败自动回退软编码。"""

    def _open(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.out_path), mode="w")

        candidates = _CODEC_CANDIDATES.get(self.codec, [self.codec, "libx264"])
        chosen = None
        for name in candidates:
            if _probe_encoder(name):
                chosen = name
                break
        if chosen is None:
            self.container.close()
            raise RuntimeError(f"no available encoder for codec {self.codec!r}: tried {candidates}")
        if chosen != candidates[0]:
            warnings.warn(
                f"GPU encoder {candidates[0]!r} unavailable, falling back to {chosen!r}",
                RuntimeWarning,
            )
        logger.info("AvEncoder using codec %s", chosen)

        self.stream = self.container.add_stream(chosen, rate=int(round(self.fps)))
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        # quality ∈ (0,1] → CRF/CQ；libx 系用 crf，NVENC 系用 cq（不支持 crf 选项）
        q = int(round((1.0 - min(max(self.quality, 0.05), 1.0)) * 51))
        if chosen.startswith("libx"):
            self.stream.codec_context.options = {"crf": str(q)}
        elif "nvenc" in chosen:
            self.stream.codec_context.options = {"cq": str(q), "rc": "vbr"}
        self._frame_index = 0

    def write(self, frame) -> None:
        if cp is not None and isinstance(frame, cp.ndarray):
            frame = cp.asnumpy(frame)  # TODO(S3): NvencEncoder 将保持显存直入
        arr = np.ascontiguousarray(frame)
        if arr.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame shape {arr.shape[:2]} != ({self.height}, {self.width})")
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(f"expected RGBA (H,W,4), got {arr.shape}")

        vframe = av.VideoFrame.from_ndarray(arr, format="rgba")
        vframe = vframe.reformat(format="yuv420p")
        vframe.pts = self._frame_index
        self._frame_index += 1
        for packet in self.stream.encode(vframe):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


class NvencEncoder(Encoder):
    """NVENC 硬件编码：cupy RGBA → GPU NV12 (kernel) → cupy.asnumpy → NVENC → MP4。

    会话在部分机器上不可用（驱动限制/无 NVENC 硬件），create_encoder 会先
    探测并在失败时自动回退 AvEncoder。

    输入路径：cupy RGBA → cupy kernel 转 NV12 → asnumpy 拷回 CPU →
    PyNvVideoCodec(usecpuinputbuffer=True) 喂给 NVENC。显存直喂
    (CAIMemoryView) 在部分驱动上报 error 8（"incorrect usage of CPU input
    buffer"），故统一走 CPU buffer 路径——颜色转换仍在 GPU 完成，只有最终
    NV12 一次 GPU→CPU 拷贝。
    """

    def _open(self) -> None:
        try:
            import PyNvVideoCodec as nvc
        except ImportError as e:
            raise RuntimeError("PyNvVideoCodec not installed") from e

        self._nvc = nvc
        codec = {"h264": "h264", "hevc": "hevc"}.get(self.codec, self.codec)
        # quality ∈ (0,1] → CQ（NVENC 不支持 crf 选项）
        q = int(round((1.0 - min(max(self.quality, 0.05), 1.0)) * 51))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._enc = nvc.CreateEncoder(
            self.width, self.height, "NV12", usecpuinputbuffer=True,
            codec=codec, rc="vbr", cq=str(q), fps=str(int(round(self.fps))),
        )
        self._frame_index = 0
        # 预创建 muxer（PyNvVideoCodec 2.x 的 FFmpegMuxer 需要完整参数）
        self._muxer = nvc.FFmpegMuxer(
            str(self.out_path), nvc.MEDIA_FORMAT.MP4, codec,
            self.width, self.height,
            fps_num=int(round(self.fps)), fps_den=1,
        )
        # 原始 YUV 编码（无输入容器）→ 需要告知 muxer 每帧的 tick 增量，
        # 否则 Finalize 时 source PTS 为空，输出 duration=0
        self._muxer.SetUniformPtsIncrement(int(round(90000 / self.fps)))

    def write(self, frame) -> None:
        from .render.colorconvert import rgba_to_nv12

        # cupy 显存帧 → GPU NV12 → CPU 拷回 → NVENC
        if cp is not None and isinstance(frame, cp.ndarray):
            nv12_gpu = rgba_to_nv12(frame)
            nv12 = cp.asnumpy(nv12_gpu)
        elif isinstance(frame, np.ndarray):
            # numpy RGBA：在 CPU 上做 NV12 转换（简单 BT.709 limited range）
            nv12 = _rgba_to_nv12_numpy(frame)
        else:
            raise TypeError(f"NvencEncoder expects numpy or cupy RGBA, got {type(frame)}")
        # Encode() 通常延迟输出（B-frame reorder），多数帧返回空；包在 EndEncode 时统一吐出
        for pkt in self._enc.Encode(nv12):
            self._muxer.MuxVideoPacket(pkt["data"], pkt["picture_type"],
                                       pkt["timestamp"])
        self._frame_index += 1

    def close(self) -> None:
        for pkt in self._enc.EndEncode():
            self._muxer.MuxVideoPacket(pkt["data"], pkt["picture_type"],
                                       pkt["timestamp"])
        if getattr(self, "_muxer", None) is not None:
            self._muxer.Finalize()
            self._muxer = None


# NVENC 探测在隔离子进程执行：部分驱动（如 Pascal EOL 后的 R580+）会在
# nvEncOpenEncodeSessionEx 阶段触发原生层崩溃（access violation），
# Python 层 try/except 接不住，会直接杀死宿主进程。
_NVENC_PROBE_SNIPPET = (
    "import PyNvVideoCodec as nvc\n"
    "enc = nvc.CreateEncoder(64, 64, 'NV12', usecpuinputbuffer=True, codec='h264')\n"
    "enc.EndEncode()\n"
)

_NVENC_CACHE: bool | None = None


def _rgba_to_nv12_numpy(rgba: np.ndarray) -> np.ndarray:
    """numpy RGBA (H, W, 4) uint8 → NV12 flat uint8（BT.709 limited range）。

    NvencEncoder 在无 cupy 时的 CPU 后备路径，与 colorconvert.py 的 GPU
    kernel 数值一致。
    """
    h, w = rgba.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"even dimensions required, got {w}x{h}")
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    y709 = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    Y = np.clip(16 + 219 * y709, 0, 255).astype(np.uint8)
    # 2x2 下采样色度
    Cb = 128 + 224 * ((rgb[:, :, 2] - y709) / 1.8556)
    Cr = 128 + 224 * ((rgb[:, :, 0] - y709) / 1.5748)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)
    # NV12: Y plane (h*w) + interleaved UV plane (h/2 * w, [U V U V ...])
    nv12 = np.empty(h * w * 3 // 2, dtype=np.uint8)
    nv12[:h * w] = Y.flatten()
    uv = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    uv[:, :, 0] = Cb[0::2, 0::2]
    uv[:, :, 1] = Cr[0::2, 0::2]
    nv12[h * w:] = uv.flatten()
    return nv12


def nvenc_available() -> bool:
    """探测 NVENC 会话能否真正打开（部分驱动上 nvEncOpenEncodeSessionEx 失败）。

    在子进程中执行探测：正常退出 → 可用；非零退出（含原生崩溃、ImportError、
    设备不支持）→ 不可用。结果进程内缓存。
    """
    global _NVENC_CACHE
    if _NVENC_CACHE is not None:
        return _NVENC_CACHE
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run(
            [sys.executable, "-c", _NVENC_PROBE_SNIPPET],
            capture_output=True, timeout=30, creationflags=flags,
        )
        _NVENC_CACHE = r.returncode == 0
    except Exception:  # noqa: BLE001 - 探测永远不抛
        _NVENC_CACHE = False
    if not _NVENC_CACHE:
        logger.info("NVENC probe failed (session unavailable or driver blocked)")
    return _NVENC_CACHE


def create_encoder(width: int, height: int, fps: float, out_path, codec: str = "h264",
                   quality: float = 0.9, prefer_nvenc: bool = False) -> Encoder:
    """工厂：prefer_nvenc=True 且探测通过时走显存直喂 NvencEncoder，
    否则 AvEncoder（其内部同样优先 GPU 编码、软编码兜底）。"""
    if prefer_nvenc and nvenc_available():
        logger.info("using NvencEncoder (device-memory path)")
        return NvencEncoder(width, height, fps, out_path, codec=codec, quality=quality)
    return AvEncoder(width, height, fps, out_path, codec=codec, quality=quality)
