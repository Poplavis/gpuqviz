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
           nvenc: bool = typer.Option(True, "--nvenc/--no-nvenc"),
           width: int = typer.Option(None, "--width", help="覆盖 Scene 中的宽度（像素）"),
           height: int = typer.Option(None, "--height", help="覆盖 Scene 中的高度（像素）"),
           resolution: str = typer.Option(None, "--resolution",
                                          help="预设分辨率（480p/720p/1080p/4k/square/vertical 等）")) -> None:
    """渲染 Scene JSON → MP4（states_path 相对于 JSON 文件所在目录解析）。"""
    from .api import render as _render
    from .scene import Scene

    scene = Scene.load_json(scene_file)
    # 显式分辨率参数覆盖 Scene 模型中的尺寸
    if resolution is not None:
        from .presets import resolve_preset
        scene.width, scene.height = resolve_preset(resolution)
    elif width is not None and height is not None:
        scene.width, scene.height = width, height
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


@app.command()
def frame(scene_file: Path = typer.Option(None, "--scene", exists=True, readable=True,
                                          help="Scene JSON 文件（与 --states 二选一）"),
          states_npz: Path = typer.Option(None, "--states", exists=True,
                                          help="态矢量 .npz"),
          out: Path = typer.Option("out/frame.png", "--out", "-o"),
          time: float = typer.Option(0.5, "--time", "-t", min=0.0, max=1.0,
                                     help="归一化时刻 t∈[0,1]"),
          scale: int = typer.Option(2, "--scale", min=1, max=8,
                                    help="超采样倍数（抗锯齿，2 或 4）"),
          style: str = typer.Option("dark", "--style",
                                    help="dark/light/bw/poster 或自定义键值"),
          cols: int = typer.Option(None, "--cols", min=1,
                                   help="一行最多几个球（仅 Bloch 网格）"),
          width: int = typer.Option(1920, "--width", help="输出宽度（像素）"),
          height: int = typer.Option(1080, "--height", help="输出高度（像素）"),
          resolution: str = typer.Option(None, "--resolution",
                                         help="预设分辨率（480p/720p/1080p/4k/square/vertical 等）")) -> None:
    """渲染单帧静态 PNG（出版级，超采样抗锯齿）。"""
    from .api import render_frame as _render_frame

    if scene_file is not None:
        from .scene import Scene

        scene = Scene.load_json(scene_file)
        # 显式分辨率参数覆盖 Scene 模型中的尺寸
        if resolution is not None:
            from .presets import resolve_preset
            scene.width, scene.height = resolve_preset(resolution)
        elif width != 1920 or height != 1080:
            scene.width, scene.height = width, height
        path = _render_frame(scene=scene, t=time, out=out, scale=scale,
                             style=style, states_dir=scene_file.parent)
    elif states_npz is not None:
        states = list(np.load(states_npz)["states"])
        path = _render_frame(states=states, t=time, out=out, scale=scale,
                             style=style, cols=cols,
                             width=width, height=height, resolution=resolution)
    else:
        typer.echo("需要提供 --scene xxx.json 或 --states xxx.npz")
        raise typer.Exit(1)
    typer.echo(f"frame -> {path}")



@app.command()
def qasm(
    source: str = typer.Argument(...,
                                 help="OpenQASM 文件路径（.qasm/.qasm3），或以 OPENQASM 开头的内联文本"),
    fmt: str = typer.Option("html", "--format", "-f",
                            help="输出格式：html（交互播放器）/ mp4（视频）/ png（静态帧）"),
    out: Path = typer.Option(None, "--out", "-o",
                             help="输出路径（默认 out/qasm.<ext>）"),
    steps: int = typer.Option(120, "--steps", help="关键帧数"),
    fps: float = typer.Option(30.0, "--fps"),
    seconds: float = typer.Option(None, "--seconds", help="视频时长（秒）"),
    time: float = typer.Option(0.5, "--time", "-t", min=0.0, max=1.0,
                               help="PNG 静态帧归一化时刻"),
    title: str = typer.Option("OpenQASM 电路", "--title"),
    trail: bool = typer.Option(False, "--trail", help="Bloch 球轨迹拖尾"),
) -> None:
    """OpenQASM 2/3 电路直接可视化（文件路径或内联文本）。

    
    示例::

        gpuqviz qasm bell.qasm                          # 交互播放器
        gpuqviz qasm bell.qasm --format mp4 -o out.mp4  # 视频
        gpuqviz qasm bell.qasm --format png -t 0.8      # 静态帧
        gpuqviz qasm "OPENQASM 2.0; include \"qelib1.inc\";
        qreg q[2]; h q[0]; cx q[0],q[1];"               # 内联文本
    """
    from .adapters import load_qasm

    fmt = fmt.lower()
    if fmt not in ("html", "mp4", "png"):
        typer.echo(f"不支持的格式 {fmt!r}；可选 html / mp4 / png")
        raise typer.Exit(1)
    if out is None:
        out = Path(f"out/qasm.{fmt}")

    typer.echo(f"解析 OpenQASM（{('文件: ' + source) if Path(source).exists() else '内联文本'}）")
    try:
        qc = load_qasm(source)
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"QASM 解析失败：{e}")
        raise typer.Exit(1)
    typer.echo(f"电路：{qc.num_qubits} qubit, {len(qc.data)} 指令")

    if fmt == "html":
        from .export_html import export_html as _export

        path = _export(circuit=qc, out=out, steps=steps, fps=fps,
                       duration=seconds or steps / 30.0, title=title)
    elif fmt == "mp4":
        from .api import render_bloch_video as _render

        path = _render(circuit=qc, steps=steps, fps=fps,
                       seconds=seconds or steps / 30.0, out=out,
                       title=title, trail=trail)
    else:  # png
        from .api import render_frame as _frame

        path = _frame(circuit=qc, t=time, out=out, scale=2)
    typer.echo(f"qasm -> {path}")


@app.command()
def demo(
    algo: str = typer.Option(None, "--algo", "-a",
                             help="算法名（bell/ghz/grover/qft/phase_estimation/"
                                  "deutsch_jozsa/bernstein_vazirani/teleportation/"
                                  "superdense/simon/quantum_walk/superposition）"),
    list_algos: bool = typer.Option(False, "--list",
                                    help="列出所有可用算法并退出"),
    fmt: str = typer.Option("html", "--format", "-f",
                            help="输出格式：html（交互播放器）/ mp4（视频）/ png（静态帧）"),
    engine: str = typer.Option("qiskit", "--engine", "-e",
                               help="模拟引擎：qiskit / pyqpanda"),
    n_qubits: int = typer.Option(None, "--n-qubits", "-n",
                                 help="量子比特数（部分算法可调，默认取算法默认值）"),
    out: Path = typer.Option(None, "--out", "-o",
                             help="输出路径（默认 out/<algo>.<ext>）"),
    steps: int = typer.Option(120, "--steps",
                              help="关键帧数"),
    fps: float = typer.Option(60.0, "--fps"),
    seconds: float = typer.Option(None, "--seconds",
                                  help="视频时长（秒），默认 steps/30"),
    style: str = typer.Option("dark", "--style",
                              help="dark/light/bw/poster"),
    trail: bool = typer.Option(False, "--trail",
                               help="Bloch 球轨迹拖尾"),
    time: float = typer.Option(0.5, "--time", "-t", min=0.0, max=1.0,
                               help="PNG 静态帧归一化时刻 t∈[0,1]"),
    title: str = typer.Option(None, "--title",
                              help="播放器/视频标题（默认算法名）"),
    watermark: str = typer.Option(None, "--watermark",
                                  help="右下角水印文字（仅 mp4/png）"),
    width: int = typer.Option(None, "--width", help="输出宽度（像素）"),
    height: int = typer.Option(None, "--height", help="输出高度（像素）"),
    resolution: str = typer.Option(None, "--resolution",
                                   help="预设分辨率（480p/720p/1080p/4k/square/vertical 等）"),
) -> None:
    """一行命令演示内置量子算法可视化。

    \b
    示例::

        gpuqviz demo --list
        gpuqviz demo --algo grover
        gpuqviz demo --algo qft --format mp4 --engine qiskit
        gpuqviz demo --algo bell --format png --time 0.5
        gpuqviz demo --algo ghz --n-qubits 4 --steps 180 --trail
    """
    from .algorithms import ALGORITHM_REGISTRY, list_algorithms as _la

    # --list：打印算法列表并退出
    if list_algos:
        typer.echo(_la())
        raise typer.Exit()

    if algo is None:
        typer.echo("需要指定 --algo <name>，或用 --list 查看可用算法")
        raise typer.Exit(1)

    if algo not in ALGORITHM_REGISTRY:
        typer.echo(f"未知算法 {algo!r}。可用算法：")
        typer.echo(_la())
        raise typer.Exit(1)

    spec = ALGORITHM_REGISTRY[algo]
    fmt = fmt.lower()
    if fmt not in ("html", "mp4", "png"):
        typer.echo(f"不支持的格式 {fmt!r}；可选 html / mp4 / png")
        raise typer.Exit(1)

    # 确定输出路径
    if out is None:
        ext = {"html": "html", "mp4": "mp4", "png": "png"}[fmt]
        out = Path(f"out/{algo}.{ext}")
    if title is None:
        title = algo

    # ---- 构建电路 ----
    builder_kwargs: dict = {}
    if n_qubits is not None:
        # 按算法签名传参
        if algo == "bell":
            builder_kwargs = {}  # Bell 固定 2 qubit
        elif algo == "superdense":
            builder_kwargs = {}  # 固定 2 qubit
        elif algo == "teleportation":
            builder_kwargs = {}  # 固定 3 qubit
        elif algo in ("ghz", "superposition", "qft"):
            builder_kwargs = {"n": n_qubits}
        elif algo == "grover":
            builder_kwargs = {"n": n_qubits}
        elif algo == "phase_estimation":
            builder_kwargs = {"n_count": n_qubits - 1} if n_qubits > 1 else {}
        elif algo == "deutsch_jozsa":
            builder_kwargs = {"n": n_qubits - 1} if n_qubits > 1 else {}
        elif algo == "bernstein_vazirani":
            # secret 字符串长度 = n_qubits - 1（减去辅助 qubit）
            builder_kwargs = {"secret": "1" * (n_qubits - 1)} if n_qubits > 1 else {}
        elif algo == "simon":
            builder_kwargs = {"s": "01" * ((n_qubits // 2) or 1)}
        elif algo == "quantum_walk":
            builder_kwargs = {"n": max(1, n_qubits - 1)}
        elif algo == "shor":
            # 计数寄存器 = n_qubits - 4（工作寄存器固定 4 bit）
            builder_kwargs = {"t_bits": max(2, n_qubits - 4)}
        elif algo == "hhl":
            # clock 精度 = n_qubits - 3（2 input + 1 ancilla 固定）
            builder_kwargs = {"clock_bits": max(2, n_qubits - 3)}

    # ---- qiskit 引擎 ----
    if engine == "qiskit":
        try:
            import qiskit  # noqa: F401
        except ImportError:
            typer.echo("qiskit 未安装：pip install gpuqviz[qiskit]")
            raise typer.Exit(1)

        circuit = spec.builder(engine="qiskit", **builder_kwargs)

        if fmt == "html":
            from .export_html import export_html as _export
            path = _export(circuit=circuit, steps=steps, fps=fps,
                          duration=seconds, title=title, out=out)
        elif fmt == "mp4":
            from .api import render_bloch_video as _render
            path = _render(circuit=circuit, steps=steps, fps=fps,
                          out=out, style=style, trail=trail, seconds=seconds,
                          width=width, height=height, resolution=resolution,
                          title=title, watermark=watermark)
        else:  # png
            from .api import render_frame as _render_frame
            path = _render_frame(circuit=circuit, t=time, out=out,
                                scale=2, style=style,
                                width=width, height=height, resolution=resolution)
        typer.echo(f"demo [{algo}] -> {path}")

    # ---- pyqpanda 引擎 ----
    elif engine == "pyqpanda":
        try:
            import pyqpanda  # noqa: F401
        except ImportError:
            typer.echo("pyqpanda 未安装：pip install gpuqviz[pyqpanda]")
            raise typer.Exit(1)

        # pyqpanda 需要 machine 和 qubits
        from pyqpanda import CPUQVM
        from .algorithms import (
            bell_pyqpanda, ghz_pyqpanda, superposition_pyqpanda,
            grover_pyqpanda, teleportation_pyqpanda, superdense_pyqpanda,
        )

        qm = CPUQVM()
        qm.init_qvm()
        try:
            # 确定总 qubit 数
            n_total = n_qubits if n_qubits else spec.default_n_qubits
            q = qm.qAlloc_many(n_total)

            # 构建电路（仅支持有 pyqpanda 实现的算法）
            pyq_builders = {
                "bell": bell_pyqpanda,
                "ghz": ghz_pyqpanda,
                "superposition": superposition_pyqpanda,
                "grover": grover_pyqpanda,
                "teleportation": teleportation_pyqpanda,
                "superdense": superdense_pyqpanda,
            }
            if algo not in pyq_builders:
                typer.echo(
                    f"算法 {algo!r} 暂不支持 pyqpanda 引擎"
                    f"（支持：{sorted(pyq_builders)}）"
                )
                raise typer.Exit(1)

            if algo == "grover":
                marked = builder_kwargs.get("marked", 0b101)
                iters = builder_kwargs.get("iterations")
                circuit = pyq_builders[algo](q, qm, marked=marked, iterations=iters)
            else:
                circuit = pyq_builders[algo](q, qm)

            if fmt == "html":
                from .export_html import export_html as _export
                path = _export(circuit=circuit, steps=steps, fps=fps,
                              duration=seconds, title=title, out=out,
                              machine=qm)
            elif fmt == "mp4":
                from .api import render_bloch_video as _render
                path = _render(circuit=circuit, steps=steps, fps=fps,
                              out=out, style=style, trail=trail,
                              seconds=seconds, machine=qm,
                              width=width, height=height, resolution=resolution,
                              title=title, watermark=watermark)
            else:  # png
                from .api import render_frame as _render_frame
                path = _render_frame(circuit=circuit, t=time, out=out,
                                    scale=2, style=style, machine=qm,
                                    width=width, height=height, resolution=resolution)
            typer.echo(f"demo [{algo}] (pyqpanda) -> {path}")
        finally:
            qm.finalize()

    else:
        typer.echo(f"不支持的引擎 {engine!r}；可选 qiskit / pyqpanda")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
