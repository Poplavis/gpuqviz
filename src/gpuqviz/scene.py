"""Scene 声明式场景：pydantic 模型，可 JSON 持久化（态矢量存 .npz 文件引用）。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


def _hex_to_rgba(hex_color: str) -> tuple[float, float, float, float]:
    """'#0b0e14' → (r, g, b, 1.0)，数值 ∈ [0,1]。"""
    s = hex_color.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected #rrggbb, got {hex_color!r}")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """'#ebeef7' → (r, g, b)，数值 ∈ [0,1]。供 TextRenderer / PIL 使用。"""
    s = hex_color.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected #rrggbb, got {hex_color!r}")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class TextPosition(str, Enum):
    """预设位置（基于画面比例的九宫格锚点）。"""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class TextOverlay(BaseModel):
    """可叠加到视频帧上的文字元素（字幕/水印/标题）。

    复用底层 TextRenderer.draw(text, position, size_px, color)（GL）
    或 SoftRasterContext.draw_text（CPU），支持中英文。
    """
    text: str = Field(max_length=256)
    position: TextPosition | tuple[float, float] = TextPosition.TOP_LEFT
    font_size: int = Field(default=44, gt=0)
    color: str = "#ebeef7"
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    start: float = 0.0
    end: float | None = None
    margin: int = 40
    shadow: bool = False

    def is_active(self, t: float) -> bool:
        """当前时间 t（秒）是否在显示范围内。"""
        return self.start <= t and (self.end is None or t <= self.end)

    def resolve_pixels(self, width: int, height: int) -> tuple[int, int]:
        """将预设位置解析为像素坐标 (x, y)，y 向下（图像坐标）。"""
        if isinstance(self.position, tuple):
            return int(self.position[0]), int(self.position[1])
        m = self.margin
        approx_w = len(self.text) * self.font_size * 0.6
        fs = self.font_size
        mapping = {
            TextPosition.TOP_LEFT:      (m, m),
            TextPosition.TOP_CENTER:    (int((width - approx_w) / 2), m),
            TextPosition.TOP_RIGHT:     (int(width - approx_w - m), m),
            TextPosition.CENTER:        (int((width - approx_w) / 2),
                                         int((height - fs) / 2)),
            TextPosition.BOTTOM_LEFT:   (m, height - fs - m),
            TextPosition.BOTTOM_CENTER: (int((width - approx_w) / 2),
                                         height - fs - m),
            TextPosition.BOTTOM_RIGHT:  (int(width - approx_w - m),
                                         height - fs - m),
        }
        return mapping[self.position]


class Camera(BaseModel):
    """轨道相机动画：azimuth/elevation 在 duration 内线性插值（单位：度）。"""

    azimuth: tuple[float, float] = (0.0, 0.0)
    elevation: tuple[float, float] = (25.0, 25.0)
    zoom: float = 1.0  # >1 拉近

    def eye_at(self, t01: float, distance: float) -> np.ndarray:
        az, el = [
            a + (b - a) * t01 for a, b in ((self.azimuth), (self.elevation))
        ]
        az_r, el_r = np.radians(az), np.radians(el)
        d = distance / max(self.zoom, 1e-3)
        return np.array([
            d * np.cos(el_r) * np.sin(az_r),
            -d * np.cos(el_r) * np.cos(az_r),
            d * np.sin(el_r),
        ])


class TrackBase(BaseModel):
    """态矢量序列引用：states_path 指向 .npz（键 'states'，形状 (K, 2**n) complex）。"""

    states_path: str
    start: float = 0.0  # 时间轴占比 [start, end] ∈ (0,1]
    end: float = 1.0
    layout: str | None = None  # None=auto；或 "top"/"bottom"/"full"

    def load_states(self, base_dir: Path | None = None) -> np.ndarray:
        p = Path(self.states_path)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        data = np.load(p)
        return data["states"].astype(np.complex128)


class BlochTrack(TrackBase):
    kind: Literal["bloch"] = "bloch"
    qubit_indices: list[int] | None = None  # None = 全部 qubit
    trail: bool = False


class HeatmapTrack(TrackBase):
    kind: Literal["heatmap"] = "heatmap"
    basis: Literal["probability", "amplitude", "phase", "real", "imag"] = "probability"
    colormap: str = "viridis"


class HistogramTrack(TrackBase):
    """测量统计 / 概率分布柱状图（states_path 存态矢量，逐帧取 |ψ|²）。"""

    kind: Literal["histogram"] = "histogram"
    top_k: int = Field(default=8, ge=1)  # 只画概率最高的 top_k 项
    show_others: bool = False            # P5.3：其余项聚合为一根 others 柱


class EntanglementTrack(TrackBase):
    """纠缠图：节点=qubit（半径∝纠缠熵），边=互信息加粗。"""

    kind: Literal["entanglement"] = "entanglement"
    max_edges: int = Field(default=0, ge=0)  # P5.3：0 = 不截断


class DensityMatrixTrack(TrackBase):
    """Hinton 图：states_path 存密度矩阵关键帧 (K, 2**n, 2**n)。"""

    kind: Literal["density"] = "density"
    min_frac: float = Field(default=0.0, ge=0.0, le=1.0)  # P5.3 幅值阈值


class BlochVectorsTrack(TrackBase):
    """预计算 Bloch 向量轨道（P5.2 渲染桥梁）。

    states_path 指向 .npz（键 'bloch'，形状 (K, n, 3)）——
    MPS 等大规模后端的约化分析量直接喂给渲染层，
    免去全态矢量的存在（20+ qubit 下态矢量不可显式表示）。
    """

    kind: Literal["bloch_vectors"] = "bloch_vectors"
    qubit_indices: list[int] | None = None
    trail: bool = False

    def load_bloch(self, base_dir: Path | None = None) -> np.ndarray:
        p = Path(self.states_path)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        data = np.load(p)
        arr = data["bloch"].astype(np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"bloch npz must be (K, n, 3), got {arr.shape}")
        return arr

    def load_states(self, base_dir: Path | None = None) -> np.ndarray:
        """validate_states 等通用路径的兼容入口（返回 (K, n, 3) Bloch 数组）。"""
        return self.load_bloch(base_dir=base_dir)


Track = Annotated[BlochTrack | HeatmapTrack | HistogramTrack | EntanglementTrack
                  | DensityMatrixTrack | BlochVectorsTrack,
                  Field(discriminator="kind")]


class Scene(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: float = 60.0
    duration: float = Field(gt=0)  # 秒
    background: str = "#0b0e14"
    tracks: list[Track] = Field(min_length=1)
    camera: Camera | None = None
    title: str = ""  # 顶部标题（SDF 文字）；overlays 为空时自动转为 overlay
    overlays: list[TextOverlay] = []  # 文字叠加列表（字幕/水印/标题）

    @model_validator(mode="after")
    def _check_tracks(self):
        for t in self.tracks:
            if not (0.0 <= t.start < t.end <= 1.0):
                raise ValueError(f"track time window invalid: {t.start}..{t.end}")
        return self

    def effective_overlays(self) -> list[TextOverlay]:
        """返回实际要渲染的 overlay 列表。

        - overlays 非空时直接返回
        - overlays 为空但 title 非空时，自动构造一个 TOP_LEFT 标题 overlay
        """
        if self.overlays:
            return self.overlays
        if self.title:
            return [TextOverlay(text=self.title, position=TextPosition.TOP_LEFT,
                                font_size=44)]
        return []

    def validate_states(self, base_dir: Path | None = None) -> None:
        """所有 track 的关键帧数一致性检查（states_path 相对 base_dir 解析）。"""
        counts = {t.load_states(base_dir=base_dir).shape[0] for t in self.tracks}
        if len(counts) > 1:
            raise ValueError(f"tracks have different keyframe counts: {counts}")

    # -- 持久化 ------------------------------------------------------------

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "Scene":
        return cls.model_validate_json(Path(path).read_text())

    # -- 布局 ---------------------------------------------------------------

    def regions(self) -> list[tuple[Track, str]]:
        """给每个 track 分配区域标签。auto：1 个 → full；多个 → 依序上下分。"""
        out = []
        n = len(self.tracks)
        auto = ["full"] if n == 1 else ["top", "bottom"][:2] + (["bottom"] * max(0, n - 2))
        for t in self.tracks:
            out.append((t, t.layout or auto.pop(0)))
        return out
