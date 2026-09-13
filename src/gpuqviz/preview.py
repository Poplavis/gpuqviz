"""moderngl-window 实时预览（可选依赖 gpuqviz[preview]）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run_preview(scene_path: str) -> None:  # pragma: no cover - 需要窗口环境
    """打开窗口实时预览 Scene；空格暂停，ESC 退出。"""
    import moderngl
    import moderngl_window as mglw

    from .api import STYLES
    from .evolve import bloch_vectors
    from .interpolate import lerp_states, slerp_keys
    from .render.bloch import BlochRenderer
    from .render.heatmap import HeatmapRenderer, state_to_image
    from .render.text import TextRenderer
    from .scene import Scene, _hex_to_rgba

    scene = Scene.load_json(scene_path)
    base_dir = Path(scene_path).parent
    theme = dict(STYLES["dark"])
    theme["background"] = _hex_to_rgba(scene.background)

    loaded = []
    for track, region in scene.regions():
        states = track.load_states(base_dir=base_dir)
        n = int(round(np.log2(states.shape[1])))
        out_frames = int(round(scene.duration * scene.fps))
        if track.kind == "bloch":
            frames = slerp_keys(bloch_vectors(list(states), n_qubits=n), out_frames)
            if hasattr(frames, "get"):
                frames = frames.get()
        else:
            frames = lerp_states(states, out_frames)
        loaded.append((track, track.kind, frames, region))

    class PreviewWindow(mglw.WindowConfig):
        gl_version = (3, 3)
        title = "gpuqviz preview"
        aspect_ratio = None
        resizable = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.frame = 0
            self.paused = False
            self.bloch = BlochRenderer(_MGLGL(self.ctx, self.wnd.buffer_size), theme)
            self.heat = HeatmapRenderer(_MGLGL(self.ctx, self.wnd.buffer_size), )
            self.text = TextRenderer(_MGLGL(self.ctx, self.wnd.buffer_size))
            self._sync_size()

        def _sync_size(self):
            for r in (self.bloch, self.heat, self.text):
                r.gl.width, r.gl.height = self.wnd.buffer_size

        def on_resize(self, width, height):
            self._sync_size()

        def key_event(self, key, action, modifiers):
            keys = self.wnd.keys
            if action == keys.ACTION_PRESS:
                if key == keys.SPACE:
                    self.paused = not self.paused
                elif key == keys.ESCAPE:
                    self.wnd.close()

        def render(self, time, frame_time):
            if not self.paused:
                self.frame = (self.frame + 1) % int(scene.duration * scene.fps)
            t = self.frame
            W, H = self.wnd.buffer_size
            self.ctx.clear(*theme["background"])
            if scene.title:
                self.text.draw(scene.title, (40, 30), 44, (0.92, 0.94, 0.97, 1.0))
            t01 = t / max(int(scene.duration * scene.fps) - 1, 1)
            for track, kind, frames, region in loaded:
                if region == "top":
                    top_h = H // 2
                    self.ctx.viewport = (0, H - top_h, W, top_h)
                    self._draw_bloch(track, frames, t, t01, W, top_h)
                    self.ctx.viewport = (0, 0, W, H)
                else:
                    rect = (40, 40, W - 300, H // 2 - 80)
                    self.heat.draw(frames[t], rect, basis=getattr(track, "basis", "probability"))

        def _draw_bloch(self, track, frames, t, t01, w, h):
            states_np = frames[t]
            idx = track.qubit_indices or list(range(states_np.shape[0]))
            spacing, radius = 3.0, 1.2
            centers = [np.array([(j - (len(idx) - 1) / 2) * spacing, 0.0, 0.0])
                       for j in range(len(idx))]
            cam_dist = max(5.0, spacing * len(idx) * 0.9)
            cam = scene.camera
            if cam is not None:
                eye = cam.eye_at(t01, cam_dist)
            else:
                eye = np.array([cam_dist * 0.35, -cam_dist, cam_dist * 0.55])
            self.bloch.set_camera(eye, aspect=w / h)
            for j, qi in enumerate(idx):
                self.bloch.draw_vector(states_np[qi], centers[j], radius)
                self.bloch.draw_static(centers[j], radius)

    class _MGLGL:
        """适配器：让渲染器复用 moderngl-window 的 ctx 与 buffer 尺寸。"""

        def __init__(self, ctx, size):
            self.ctx = ctx
            self.width, self.height = size

        def program(self, vs, fs):
            return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    PreviewWindow.run()
