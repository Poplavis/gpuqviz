"""量子态热图渲染器：态矢量 → GPU 网格化图像 → 伪彩纹理 → 像素空间贴图。

色表（viridis/inferno）在初始化时烘焙为 256 级 1D 纹理，运行时由片元着色器
采样，CPU 不参与逐帧查表。
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

_QUAD_VS = """
#version 330
in vec2 in_pos;          // 单位方形 (0,0)-(1,1)
uniform vec4 u_clip;     // clip 空间 (x0, y0, w, h)
out vec2 v_uv;
void main() {
    vec2 p = u_clip.xy + in_pos * u_clip.zw;
    v_uv = vec2(in_pos.x, 1.0 - in_pos.y);  // 纹理第 0 行画在矩形顶部
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_HEAT_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;    // 数据图（R 通道 ∈ [0,1]）
uniform sampler2D u_cmap;   // 256x1 色表
uniform int u_mode;         // 0=数据过色表  1=色标条渐变  2=灰度
out vec4 frag;
void main() {
    if (u_mode == 1) {
        frag = texture(u_cmap, vec2(clamp(v_uv.y, 0.0, 1.0), 0.5));
    } else if (u_mode == 2) {
        float v = texture(u_tex, v_uv).r;
        frag = vec4(v, v, v, 1.0);
    } else {
        float v = texture(u_tex, v_uv).r;
        frag = texture(u_cmap, vec2(clamp(v, 0.0, 1.0), 0.5));
    }
}
"""

# viridis/inferno 关键色（公开色表的采样点）
_VIRIDIS_STOPS = np.array([
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
])
_INFERNO_STOPS = np.array([
    [0.001, 0.000, 0.014], [0.056, 0.022, 0.210], [0.199, 0.020, 0.437],
    [0.365, 0.047, 0.460], [0.521, 0.130, 0.420], [0.663, 0.216, 0.354],
    [0.793, 0.320, 0.264], [0.901, 0.448, 0.149], [0.968, 0.617, 0.047],
    [0.988, 0.799, 0.229], [0.988, 0.998, 0.645],
])


def bake_colormap(name: str = "viridis", levels: int = 256) -> np.ndarray:
    """色表烘焙：关键色线性插值成 (levels, 3) uint8（初始化时一次，不逐帧）。"""
    stops = {"viridis": _VIRIDIS_STOPS, "inferno": _INFERNO_STOPS}[name]
    t = np.linspace(0.0, 1.0, len(stops))
    x = np.linspace(0.0, 1.0, levels)
    out = np.stack([np.interp(x, t, stops[:, i]) for i in range(3)], axis=-1)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def state_to_image(state, basis: str = "probability"):
    """态矢量 (2**n,) → 网格化显示图 (rows, cols) float32 ∈ [0,1]。

    cols = 2**ceil(n/2)（宽），rows = 其余；计算在 GPU（cupy 输入时）。
    """
    xp = cp if (cp is not None and isinstance(state, cp.ndarray)) else np
    psi = xp.asarray(state).reshape(-1).astype(xp.complex128)
    dim = psi.shape[0]
    n = int(round(np.log2(dim)))
    if 2**n != dim:
        raise ValueError(f"dimension {dim} is not a power of 2")

    if basis == "probability":
        img = xp.abs(psi) ** 2
    elif basis == "amplitude":
        m = xp.abs(psi).max()
        img = xp.abs(psi) / xp.where(m > 1e-12, m, 1.0)
    elif basis == "phase":
        img = (xp.angle(psi) + np.pi) / (2 * np.pi)
    elif basis == "real":
        m = xp.abs(xp.real(psi)).max()
        img = xp.real(psi) / xp.where(m > 1e-12, m, 1.0) * 0.5 + 0.5
    elif basis == "imag":
        m = xp.abs(xp.imag(psi)).max()
        img = xp.imag(psi) / xp.where(m > 1e-12, m, 1.0) * 0.5 + 0.5
    else:
        raise ValueError(f"unknown basis {basis!r}")

    cols = 2 ** int(np.ceil(n / 2))
    rows = max(1, dim // cols)
    img = img[: rows * cols].reshape(rows, cols)
    return img.astype(xp.float32)


class HeatmapRenderer:
    """在当前 FBO 的指定矩形内绘制态热图 + 色标条。"""

    def __init__(self, gl, colormap: str = "viridis"):
        self.gl = gl
        ctx = gl.ctx
        self.prog = gl.program(_QUAD_VS, _HEAT_FS)
        quad = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=np.float32)
        self.vbo = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])

        cmap = bake_colormap(colormap)
        self.cmap_tex = ctx.texture((len(cmap), 1), 3, cmap.tobytes())
        self.cmap_tex.filter = (ctx.LINEAR, ctx.LINEAR)
        # 默认 REPEAT 会让边缘 LINEAR 混到另一端（紫+黄→棕），必须钳边
        self.cmap_tex.repeat_x = False
        self.cmap_tex.repeat_y = False
        self.cmap_tex.use(1)
        self.prog["u_cmap"].value = 1

        self._tex = None
        self._tex_size = (0, 0)

    def _upload_texture(self, img):
        ctx = self.gl.ctx
        h, w = img.shape
        if self._tex_size != (w, h):
            if self._tex is not None:
                self._tex.release()
            self._tex = ctx.texture((w, h), 1, dtype="f4")
            self._tex_size = (w, h)
        data = img.get() if (cp is not None and isinstance(img, cp.ndarray)) else img
        self._tex.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())
        self._tex.filter = (ctx.NEAREST, ctx.NEAREST)  # 态网格保持锐利
        self._tex.repeat_x = False
        self._tex.repeat_y = False
        # unit 2 专用于数据纹理（unit 0 留给其他渲染器的默认绑定，unit 1 = 色表）
        self._tex.use(2)

    def _draw(self, rect, mode: int):
        x, y, w, h = rect
        W, H = self.gl.width, self.gl.height
        self.prog["u_clip"].value = (
            (x / W) * 2 - 1, (y / H) * 2 - 1, (w / W) * 2, (h / H) * 2
        )
        self.prog["u_mode"].value = mode
        self.prog["u_tex"].value = 2
        self.prog["u_cmap"].value = 1
        self.vao.render()

    def draw(self, state, rect, basis: str = "probability", colorbar: bool = True):
        """state: (2**n,) 复数；rect: 像素空间 (x, y, w, h)，y=0 为 FBO 底部。"""
        img = state_to_image(state, basis)
        self._upload_texture(img)

        x, y, w, h = rect
        cb_w = 22 if colorbar else 0
        self._draw((x, y, w, h), mode=0)
        if colorbar:
            self._draw((x + w + 10, y, cb_w, h), mode=1)

    def release(self):
        for obj in (self.vao, self.vbo, self.cmap_tex):
            obj.release()
        if self._tex is not None:
            self._tex.release()
