"""编码器抽象：帧数据（numpy/cupy）直接 write()，全程无中间文件。

S1 阶段 AvEncoder 在进程内完成 RGBA→YUV420p 转换（FFmpeg swscale，C 实现，
速度可接受）；cupy 输入需 cupy.asnumpy() 拷回 CPU —— S3 将以 NvencEncoder
消除这次拷贝。
"""

from __future__ import annotations

import logging
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
    """NVENC 显存直喂编码：cupy RGBA → GPU NV12 (kernel) → NVENC → MP4。

    会话在部分机器上不可用（驱动限制/无 NVENC 硬件），create_encoder 会先
    探测并在失败时自动回退 AvEncoder。
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
            self.width, self.height, "NV12", usecpuinputbuffer=False,
            codec=codec, rc="vbr", cq=str(q), fps=str(int(round(self.fps))),
        )
        self._frame_index = 0

    def write(self, frame) -> None:
        from .render.colorconvert import rgba_to_nv12

        if cp is None or not isinstance(frame, cp.ndarray):
            raise TypeError("NvencEncoder requires cupy RGBA frames (GPU-resident)")
        nv12 = rgba_to_nv12(frame)
        view = self._nvc.CAIMemoryView(
            [int(nv12.size)], [1], "|u1", int(nv12.data.ptr), 1, True
        )
        for pkt in self._enc.Encode(view):
            self._mux_packet(pkt, keyframe=False)
        self._frame_index += 1

    def _mux_packet(self, data: bytes, keyframe: bool) -> None:
        if getattr(self, "_muxer", None) is None:
            self._muxer = self._nvc.FFmpegMuxer(str(self.out_path), "mp4")
        self._muxer.Write(bytes(data), keyframe)

    def close(self) -> None:
        for pkt in self._enc.EndEncode():
            self._mux_packet(pkt, keyframe=False)
        if getattr(self, "_muxer", None) is not None:
            self._muxer.Close()
            self._muxer = None


def nvenc_available() -> bool:
    """探测 NVENC 会话能否真正打开（部分驱动上 nvEncOpenEncodeSessionEx 失败）。"""
    try:
        import PyNvVideoCodec as nvc

        enc = nvc.CreateEncoder(64, 64, "NV12", usecpuinputbuffer=False, codec="h264")
        enc.EndEncode()
        return True
    except Exception:  # noqa: BLE001
        return False


def create_encoder(width: int, height: int, fps: float, out_path, codec: str = "h264",
                   quality: float = 0.9, prefer_nvenc: bool = False) -> Encoder:
    """工厂：prefer_nvenc=True 且探测通过时走显存直喂 NvencEncoder，
    否则 AvEncoder（其内部同样优先 GPU 编码、软编码兜底）。"""
    if prefer_nvenc and nvenc_available():
        logger.info("using NvencEncoder (device-memory path)")
        return NvencEncoder(width, height, fps, out_path, codec=codec, quality=quality)
    return AvEncoder(width, height, fps, out_path, codec=codec, quality=quality)
