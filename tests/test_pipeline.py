"""S1 管线测试：小尺寸出片并读回帧数校验（需要 GL，缺依赖自动跳过）。"""

from pathlib import Path

import pytest

pytest.importorskip("moderngl")
pytest.importorskip("av")

from gpuqviz.pipeline import render_solid_gradient  # noqa: E402


def test_minimum_pipeline(tmp_path: Path):
    out = tmp_path / "mini.mp4"
    render_solid_gradient(out=str(out), seconds=1.0, fps=30, width=320, height=240)

    assert out.exists() and out.stat().st_size > 0

    import av

    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        assert stream.duration is not None
    assert frames == 30  # fps * seconds
