"""gpuqviz 命令行入口（typer）。"""

from pathlib import Path

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


if __name__ == "__main__":
    app()
