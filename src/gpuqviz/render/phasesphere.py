"""相位色盘渲染器：单位圆盘上极径=幅值、色相=相位（片元着色器直算 HSV→RGB）。"""

from __future__ import annotations

import numpy as np

_DISC_VS = """
#version 330
in vec2 in_pos;
uniform vec4 u_clip;
out vec2 v_local;   // 以盘心为原点的单位坐标 [-1,1]
void main() {
    vec2 p = u_clip.xy + (in_pos * 2.0 - 1.0) * u_clip.zw;
    v_local = in_pos * 2.0 - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_DISC_FS = """
#version 330
in vec2 v_local;
uniform vec2 u_vector;   // (振幅, 相位)，振幅∈[0,1]
out vec4 frag;
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(vec3(1.0), clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    float r = length(v_local);
    if (r > 1.0) discard;
    float amp = clamp(u_vector.x, 0.0, 1.0);
    // 圆盘：色相 = 相位，亮度 = 归一化振幅
    float hue = atan(v_local.y, v_local.x) / 6.28318530718 + 0.5;
    vec3 rgb = hsv2rgb(vec3(hue, 1.0, r * amp));
    // 矢量指针：相位方向、长度 = 振幅（宽度按像素近似）
    vec2 dir = vec2(cos(u_vector.y * 6.28318530718 - 3.14159),
                    sin(u_vector.y * 6.28318530718 - 3.14159));
    float cross = abs(v_local.x * dir.y - v_local.y * dir.x);
    float along = dot(v_local, dir);
    if (along > 0.0 && along < amp && cross < 0.02) {
        rgb = vec3(1.0);  // 白色指针
    }
    // 外圈
    if (r > 0.97) rgb = vec3(0.5);
    frag = vec4(rgb, 1.0);
}
"""


class PhaseDiscRenderer:
    """在当前 FBO 的指定矩形内绘制相位色盘。"""

    def __init__(self, gl):
        self.gl = gl
        self.prog = gl.program(_DISC_VS, _DISC_FS)
        quad = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=np.float32)
        self.vbo = gl.ctx.buffer(quad.tobytes())
        self.vao = gl.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])

    def draw(self, state, rect):
        """state: (2,) 单 qubit 态矢量；rect: 像素空间 (x, y, w, h)。"""
        amp, phase = self._amp_phase(state)
        x, y, w, h = rect
        W, H = self.gl.width, self.gl.height
        self.prog["u_clip"].value = (
            (x / W) * 2 - 1, (y / H) * 2 - 1, (w / W) * 2, (h / H) * 2
        )
        self.prog["u_vector"].value = (amp, phase)
        self.vao.render()

    @staticmethod
    def _amp_phase(state):
        import cupy as cp

        xp = cp if isinstance(state, cp.ndarray) else np
        psi = xp.asarray(state).reshape(-1).astype(xp.complex128)
        if psi.shape[0] != 2:
            raise ValueError("PhaseDiscRenderer expects a single-qubit state")
        amp = float(xp.abs(psi[1]))  # |1> 分量的振幅作为指针长度
        phase = float((xp.angle(psi[1]) + np.pi) / (2 * np.pi))
        return amp, phase

    def release(self):
        self.vao.release()
        self.vbo.release()
