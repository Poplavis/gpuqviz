"""布洛赫球渲染器：GLSL 离屏绘制（球壳 + 赤道环 + 坐标轴 + 态矢量箭头 + 轨迹）。

网格几何用 numpy 程序化生成（UV 球体，仅此一种简单网格，不值得引入 trimesh
预处理步骤 —— 与 DESIGN.md 第 3 节的差异已在交付说明中记录）。
坐标约定与 Bloch 向量一致：z 轴为北极（|0⟩）。
"""

from __future__ import annotations

import numpy as np
import moderngl

try:  # cupy 可选
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

_VS = """
#version 330
in vec3 in_pos;
in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal;  // 模型旋转（本渲染器模型矩阵只有平移+均匀缩放，可用）
out vec3 v_normal;
out vec3 v_world;
void main() {
    vec4 world = u_model * vec4(in_pos, 1.0);
    v_world = world.xyz;
    v_normal = u_normal * in_normal;
    gl_Position = u_mvp * world;
}
"""

_FS = """
#version 330
in vec3 v_normal;
in vec3 v_world;
uniform vec4 u_color;
uniform float u_lit;   // 1.0=球面光照, 0.0=纯色线/点
out vec4 frag;
void main() {
    if (u_lit > 0.5) {
        vec3 n = normalize(v_normal);
        // 双面光照：背面（看到球壳内侧）法线翻转
        if (!gl_FrontFacing) n = -n;
        vec3 l = normalize(vec3(0.4, 0.5, 0.8));
        float diff = max(dot(n, l), 0.0);
        float rim = pow(1.0 - abs(dot(n, vec3(0.0, 0.0, 1.0))), 2.0);
        vec3 rgb = u_color.rgb * (0.35 + 0.65 * diff) + vec3(0.10, 0.16, 0.22) * rim;
        frag = vec4(rgb, u_color.a);
    } else {
        frag = u_color;
    }
}
"""


def _unit_sphere(segments: int = 48, rings: int = 24):
    """UV 球（z 轴为极轴），返回 (positions, normals, indices)。"""
    phi = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    theta = np.linspace(0.0, np.pi, rings + 1)
    ph, th = np.meshgrid(phi, theta, indexing="ij")
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (S*(R+1), 3)
    normals = pos.copy()
    idx = []
    for i in range(segments):
        for j in range(rings):
            a = i * (rings + 1) + j
            b = ((i + 1) % segments) * (rings + 1) + j
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return pos.astype(np.float32), normals.astype(np.float32), np.array(idx, dtype=np.uint32)


def _look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    # 行赋值：R 的三行是 right/up/-forward
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _perspective(fov_y: float, aspect: float, near: float, far: float):
    f = 1.0 / np.tan(fov_y / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


class BlochRenderer:
    """绘制一个或多个布洛赫球。初始化上传网格，逐帧仅更新 uniform。"""

    def __init__(self, gl, style: dict):
        self.gl = gl
        self.style = style
        ctx = gl.ctx
        self.prog = gl.program(_VS, _FS)
        pos, nor, idx = _unit_sphere()
        self.sphere_vbo = ctx.buffer(pos.tobytes())
        self.sphere_nbo = ctx.buffer(nor.tobytes())
        self.sphere_ibo = ctx.buffer(idx.tobytes())
        self.sphere_vao = ctx.vertex_array(
            self.prog,
            [(self.sphere_vbo, "3f", "in_pos"), (self.sphere_nbo, "3f", "in_normal")],
            index_buffer=self.sphere_ibo,
        )

        # 赤道环（xy 平面）与坐标轴线
        t = np.linspace(0.0, 2 * np.pi, 128, endpoint=False)
        ring = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1).astype(np.float32)
        axes = np.array(
            [
                [-1.2, 0, 0], [1.2, 0, 0],
                [0, -1.2, 0], [0, 1.2, 0],
                [0, 0, -1.2], [0, 0, 1.2],
            ], dtype=np.float32,
        )
        self.ring_vao = self._line_vao(ring, mode="LINE_LOOP")
        self.axes_vao = self._line_vao(axes, mode="LINES")

        # 态矢量箭头（球心→端点，2 点线段，逐帧更新）与轨迹（最多 64 点）
        self.arrow_vbo = ctx.buffer(np.zeros(2 * 3, dtype=np.float32).tobytes(), dynamic=True)
        self.arrow_vao = self._line_vao_from_vbo(self.arrow_vbo, 2, "LINE_STRIP")
        self.trail_vbo = ctx.buffer(np.zeros(64 * 3, dtype=np.float32).tobytes(), dynamic=True)
        self.trail_vao = self._line_vao_from_vbo(self.trail_vbo, 64, "LINE_STRIP")
        self._arrow = np.zeros((2, 3), dtype=np.float32)

        self._mvp = np.eye(4)
        self._view = np.eye(4)
        self._proj = np.eye(4)

    def _line_vao(self, points: np.ndarray, mode: str):
        vbo = self.gl.ctx.buffer(points.tobytes())
        return self._line_vao_from_vbo(vbo, len(points), mode)

    def _line_vao_from_vbo(self, vbo, n: int, mode: str):
        vao = self.gl.ctx.vertex_array(self.prog, [(vbo, "3f", "in_pos")])
        return (vao, getattr(moderngl, mode), n)

    def _render_lines(self, line_obj) -> None:
        vao, mode, n = line_obj
        vao.render(mode=mode, vertices=n)

    # -- 相机 --------------------------------------------------------------

    def set_camera(self, eye: np.ndarray, target: np.ndarray = np.zeros(3),
                   up=np.array([0.0, 0.0, 1.0]), fov: float = 45.0,
                   aspect: float | None = None) -> None:
        if aspect is None:
            aspect = self.gl.width / self.gl.height
        self._view = _look_at(np.asarray(eye, dtype=np.float64),
                              np.asarray(target, dtype=np.float64),
                              np.asarray(up, dtype=np.float64))
        self._proj = _perspective(np.radians(fov), aspect, 0.1, 100.0)
        self._mvp = self._proj @ self._view

    # -- 绘制 ---------------------------------------------------------------

    def _set_common(self, color, alpha, lit):
        p = self.prog
        # numpy 行主序 → GLSL 列主序：上传前必须转置
        p["u_mvp"].write(self._mvp.T.astype(np.float32).tobytes())
        p["u_color"].value = (*color, alpha)
        p["u_lit"].value = lit

    def _model_matrix(self, center, radius):
        m = np.eye(4)
        m[:3, :3] = np.eye(3) * radius
        m[:3, 3] = center
        return m

    def _write_model(self, center, radius):
        m = self._model_matrix(center, radius).astype(np.float32)
        self.prog["u_model"].write(m.T.tobytes())
        self.prog["u_normal"].write((np.eye(3) * 1.0).astype(np.float32).tobytes())

    def draw_static(self, center, radius) -> None:
        """画坐标轴 + 赤道环 + 半透明球壳（不含态矢量）。"""
        ctx = self.gl.ctx
        self._write_model(center, radius)
        # 不透明部分
        self._set_common(self.style["axis_color"], 1.0, 0.0)
        self._render_lines(self.axes_vao)
        self._set_common(self.style["ring_color"], 1.0, 0.0)
        self._render_lines(self.ring_vao)
        # 半透明球壳（关深度写入，最后画）
        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA
        ctx.depth_mask = False
        self._set_common(self.style["sphere_color"], self.style["sphere_alpha"], 1.0)
        self.sphere_vao.render()
        ctx.depth_mask = True
        ctx.disable(ctx.BLEND)

    def draw_vector(self, v, center, radius) -> None:
        """画态矢量箭头（球心→Bloch 向量端点）+ 端点小球标记。"""
        v = np.asarray(v, dtype=np.float64).reshape(3)
        n = np.linalg.norm(v)
        if n > 1e-9:
            v = v / n
            self._arrow[0] = center
            self._arrow[1] = center + v * radius
            self.arrow_vbo.write(self._arrow.astype(np.float32).tobytes())
            self._set_common(self.style["vector_color"], 1.0, 0.0)
            # 箭头顶点已是世界坐标，用单位模型矩阵（勿二次变换）
            self._write_model(np.zeros(3), 1.0)
            self._render_lines(self.arrow_vao)
            # 端点标记小球（复用球网格，缩放到 0.07）
            self._set_common(self.style["vector_color"], 1.0, 1.0)
            self._write_model(self._arrow[1].astype(np.float64), radius * 0.07)
            self.sphere_vao.render()

    def draw_trail(self, points, center, radius) -> None:
        """画轨迹尾巴：points 为单位球坐标点列 (N,3)，绕 center 缩放平移。"""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            return
        world = pts * radius + center
        self.trail_vbo.write(world.astype(np.float32).tobytes())
        self._set_common(self.style["trail_color"], self.style.get("trail_alpha", 0.6), 0.0)
        self._write_model(np.zeros(3), 1.0)
        vao, mode, _ = self.trail_vao
        vao.render(mode=mode, vertices=len(pts))

    def release(self) -> None:
        for obj in (self.sphere_vao, self.ring_vao[0], self.axes_vao[0],
                    self.arrow_vao[0], self.trail_vao[0],
                    self.sphere_vbo, self.sphere_nbo, self.sphere_ibo,
                    self.arrow_vbo, self.trail_vbo):
            obj.release()
