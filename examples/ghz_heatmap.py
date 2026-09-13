"""S3 验收：GHZ 态逐层演化 → 上布洛赫球 / 下热图分屏 out/ghz.mp4。

运行::

    python examples/ghz_heatmap.py
"""

import time

import numpy as np
from qiskit import QuantumCircuit

from gpuqviz.encode import create_encoder
from gpuqviz.evolve import bloch_vectors, sample_circuit
from gpuqviz.interpolate import lerp_states, slerp_keys
from gpuqviz.render import GLContext
from gpuqviz.render.bloch import BlochRenderer
from gpuqviz.render.compositor import split_view
from gpuqviz.render.heatmap import HeatmapRenderer, state_to_image
from gpuqviz.api import STYLES

if __name__ == "__main__":
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    fps, seconds = 60, 6.0
    key_states = sample_circuit(qc, steps=120)
    frames = lerp_states(key_states, int(seconds * fps))          # 态矢量（热图用）
    frames_bloch = slerp_keys(bloch_vectors(key_states), int(seconds * fps))  # Bloch（球用）
    if hasattr(frames_bloch, "get"):
        frames_bloch = frames_bloch.get()
    n_qubits = 3
    theme = dict(STYLES["dark"])

    t0 = time.perf_counter()
    with GLContext(1920, 1080, fps=fps) as gl:
        bloch = BlochRenderer(gl, theme)
        heat = HeatmapRenderer(gl)
        try:
            spacing, radius = 3.0, 1.2
            centers = [np.array([(i - (n_qubits - 1) / 2) * spacing, 0.0, 0.0])
                       for i in range(n_qubits)]
            cam_dist = max(5.0, spacing * n_qubits * 0.9)

            def draw(gl: GLContext, t: int) -> None:
                gl.ctx.clear(*theme["background"])
                vecs = frames[t]  # 态矢量（热图用）

                def top(w, h):
                    bloch.set_camera(np.array([cam_dist * 0.35, -cam_dist, cam_dist * 0.55]),
                                     aspect=w / h)
                    for i in range(n_qubits):
                        bloch.draw_vector(frames_bloch[t][i], centers[i], radius)
                        bloch.draw_static(centers[i], radius)

                def bottom(w, h):
                    # 全 FBO 像素坐标：矩形放在下半区中央
                    img = state_to_image(vecs, basis="probability")
                    ih, iw = img.shape
                    cell = min((w - 220) / iw, (h * 0.5 - 80) / ih)
                    rect = ((w - 160 - cell * iw) / 2,
                            (h * 0.5 - cell * ih) / 2,  # y=0 在 FBO 底部
                            cell * iw, cell * ih)
                    heat.draw(vecs, rect, basis="probability")

                split_view(gl, top, bottom, ratio=0.5)

            with create_encoder(1920, 1080, fps, "out/ghz.mp4",
                                codec="h264", quality=0.9, prefer_nvenc=True) as enc:
                for t, frame in gl.frame_iterator(int(seconds * fps), draw):
                    enc.write(frame)
        finally:
            bloch.release()
            heat.release()

    print(f"total wall time: {time.perf_counter() - t0:.2f}s -> out/ghz.mp4")
