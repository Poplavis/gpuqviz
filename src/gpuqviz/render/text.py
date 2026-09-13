"""SDF 文字渲染：图集采样 + smoothstep 边缘，支持中文，运行时不依赖 freetype。"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

_FONT_VS = """
#version 330
in vec2 in_pos;      // clip 空间
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FONT_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_atlas;
uniform vec4 u_color;
uniform float u_range;   // 期望的屏幕像素级过渡半宽（如 1.5px）
uniform vec2 u_texel;    // 1/atlas_w, 1/atlas_h
uniform float u_span;    // SDF 编码跨度：0.5 对应 ±u_span/2 源像素
out vec4 frag;
void main() {
    float d = texture(u_atlas, v_uv).r - 0.5;         // 以笔画边缘为 0，∈[-0.5,0.5]
    float atlas_px_per_screen = length(vec2(dFdx(v_uv.x), dFdy(v_uv.y))) / u_texel.x;
    float w = u_range * atlas_px_per_screen / u_span; // 换算回 SDF 编码单位
    float alpha = smoothstep(-w, w, d);
    frag = vec4(u_color.rgb, u_color.a * alpha);
}
"""

_DEFAULT_FONTS = ["msyh", "arial"]


class TextRenderer:
    """在当前 FBO 上以像素坐标绘制 SDF 文字（初始化加载图集，无 freetype 依赖）。"""

    def __init__(self, gl, font_name: str | None = None):
        self.gl = gl
        atlas_bin, meta = self._load_atlas(font_name)
        self.meta = meta
        self.atlas_w = meta["atlas_w"]
        self.atlas_h = meta["atlas_h"]

        ctx = gl.ctx
        self.prog = gl.program(_FONT_VS, _FONT_FS)
        # 预分配：最多 256 字符 × 6 顶点 × 4 float；draw 用 vertices= 截断
        self._max_chars = 256
        self.vbo = ctx.buffer(np.zeros(self._max_chars * 24, np.float32).tobytes(),
                              dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")])

        self.atlas_tex = ctx.texture((self.atlas_w, self.atlas_h), 1, atlas_bin)
        self.atlas_tex.repeat_x = self.atlas_tex.repeat_y = False
        self.atlas_tex.filter = (ctx.LINEAR, ctx.LINEAR)
        self.atlas_tex.use(0)
        self.prog["u_atlas"].value = 0
        self.prog["u_texel"].value = (1.0 / self.atlas_w, 1.0 / self.atlas_h)
        self.prog["u_span"].value = float(meta["sdf_range_px"]) * 2.0  # 0..1 编码的像素跨度

        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA

    @staticmethod
    def _load_atlas(font_name: str | None):
        assets = resources.files("gpuqviz") / "assets"
        names = [font_name] if font_name else _DEFAULT_FONTS
        for name in names:
            bin_ref = assets / f"font_{name}_64.bin"
            json_ref = assets / f"font_{name}_64.json"
            try:
                return bin_ref.read_bytes(), json.loads(json_ref.read_text())
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"font atlas not found for {names}; run scripts/gen_font_atlas.py")

    def draw(self, text: str, position, size_px: float, color=(1, 1, 1, 1)) -> None:
        """position: 图像坐标（左上原点，y 向下），size_px 为字形高。"""
        W, H = self.gl.width, self.gl.height
        scale = size_px / self.meta["size"]
        verts: list[float] = []
        pen_x, pen_y = float(position[0]), float(position[1])

        for ch in text:
            g = self.meta["glyphs"].get(ch)
            if g is None or g.get("empty"):
                pen_x += size_px * 0.5  # 未知字符按半宽空格
                continue
            gw, gh = g["w"], g["h"]
            x0 = pen_x + g["bearing_x"] * scale
            y0 = pen_y - g["bearing_y"] * scale  # 图像坐标向下
            x1, y1 = x0 + gw * scale, y0 + gh * scale
            u0, v0 = g["x"] / self.atlas_w, g["y"] / self.atlas_h
            u1, v1 = (g["x"] + gw) / self.atlas_w, (g["y"] + gh) / self.atlas_h

            # clip 坐标：图像 y 向下 → clip y 向上翻转
            cx0, cy0 = x0 / W * 2 - 1, 1 - y0 / H * 2
            cx1, cy1 = x1 / W * 2 - 1, 1 - y1 / H * 2
            # 两三角形
            verts += [cx0, cy1, u0, v1, cx1, cy1, u1, v1, cx0, cy0, u0, v0,
                      cx1, cy1, u1, v1, cx1, cy0, u1, v0, cx0, cy0, u0, v0]
            pen_x += g["advance"] * scale

        if not verts:
            return
        nverts = len(verts) // 4
        if nverts > self._max_chars * 6:
            verts = verts[: self._max_chars * 24]
            nverts = self._max_chars * 6
        # 重绑图集：纹理单元是全局状态，其他渲染器（如热图）会占用 unit 0
        self.atlas_tex.use(0)
        self.prog["u_atlas"].value = 0
        self.prog["u_texel"].value = (1.0 / self.atlas_w, 1.0 / self.atlas_h)
        self.prog["u_span"].value = float(self.meta["sdf_range_px"]) * 2.0
        self.vbo.write(np.asarray(verts, dtype=np.float32).tobytes())
        self.prog["u_color"].value = color
        # 过渡半宽约 1.5 源像素 → 任意输出字号下都锐利
        self.prog["u_range"].value = 1.5
        # 其他渲染器（如 BlochRenderer.draw_static）会关闭 BLEND，文字必须自管
        ctx = self.gl.ctx
        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA
        self.vao.render(vertices=len(verts) // 4)

    def release(self):
        self.vao.release()
        self.vbo.release()
        self.atlas_tex.release()
        self.gl.ctx.disable(self.gl.ctx.BLEND)
