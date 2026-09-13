"""gpuqviz 命令行入口（typer）。"""

from pathlib import Path

import numpy as np
import typer

from . import __version__
from .env import report_env

app = typer.Typer(help="gpuqviz: GPU-accelerated quantum visualization")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"gpuqviz {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """gpuqviz command line interface."""


@app.command()
def env() -> None:
    """打印环境能力报告（CUDA / OpenGL / NVENC / qiskit）。"""
    typer.echo(report_env())


@app.command()
def render(scene_file: Path = typer.Argument(..., exists=True, readable=True,
                                            help="Scene JSON 文件"),
           out: Path = typer.Option("out/scene.mp4", "--out", "-o"),
           codec: str = typer.Option("h264", "--codec"),
           quality: float = typer.Option(0.9, "--quality", min=0.05, max=1.0),
           nvenc: bool = typer.Option(True, "--nvenc/--no-nvenc")) -> None:
    """渲染 Scene JSON → MP4（states_path 相对于 JSON 文件所在目录解析）。"""
    from .api import render as _render
    from .scene import Scene

    scene = Scene.load_json(scene_file)
    path = _render(scene, out=out, codec=codec, quality=quality,
                   prefer_nvenc=nvenc, states_dir=scene_file.parent)
    typer.echo(f"rendered -> {path}")


@app.command()
def preview(scene_file: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """实时预览 Scene（需要 gpuqviz[preview] 扩展）。"""
    try:
        import moderngl_window  # noqa: F401
    except ImportError:
        typer.echo("preview 需要 moderngl-window：pip install gpuqviz[preview]")
        raise typer.Exit(1)
    from .preview import run_preview

    run_preview(str(scene_file))


@app.command()
def export(scene_file: Path = typer.Argument(None, exists=True, readable=True,
                                            help="Scene JSON 文件"),
           states_npz: Path = typer.Option(None, "--states", exists=True,
                                           help="态矢量 .npz（与 Scene JSON 二选一）"),
           out: Path = typer.Option("out/viewer.html", "--out", "-o"),
           title: str = typer.Option("量子态演化", "--title"),
           fps: float = typer.Option(60.0, "--fps"),
           steps: int = typer.Option(120, "--steps"),
           duration: float = typer.Option(None, "--duration",
                                          help="秒；默认 steps/30"),
           online: bool = typer.Option(False, "--online",
                                       help="three.js 走 CDN（文件更小，需联网）")) -> None:
    """导出交互式 3D 播放器 HTML（播放/暂停/倍速/时间轴 + 当前量子状态面板）。"""
    from .export_html import export_html as _export

    if scene_file is not None:
        from .scene import Scene

        scene = Scene.load_json(scene_file)
        states = scene.tracks[0].load_states(base_dir=scene_file.parent)
        path = _export(states=states, fps=scene.fps, duration=scene.duration,
                       title=scene.title or title, out=out, embed_three=not online)
    elif states_npz is not None:
        states = np.load(states_npz)["states"]
        path = _export(states=list(states), fps=fps, duration=duration or steps / 30.0,
                       title=title, out=out, embed_three=not online)
    else:
        typer.echo("需要提供 Scene JSON 或 --states xxx.npz")
        raise typer.Exit(1)
    typer.echo(f"exported -> {path}")


if __name__ == "__main__":
    app()
