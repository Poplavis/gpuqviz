"""Scene 声明式场景：pydantic 模型，可 JSON 持久化（态矢量存 .npz 文件引用）。"""

from __future__ import annotations

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


Track = Annotated[BlochTrack | HeatmapTrack, Field(discriminator="kind")]


class Scene(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: float = 60.0
    duration: float = Field(gt=0)  # 秒
    background: str = "#0b0e14"
    tracks: list[Track] = Field(min_length=1)
    camera: Camera | None = None
    title: str = ""  # 顶部标题（SDF 文字）

    @model_validator(mode="after")
    def _check_tracks(self):
        for t in self.tracks:
            if not (0.0 <= t.start < t.end <= 1.0):
                raise ValueError(f"track time window invalid: {t.start}..{t.end}")
        return self

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
