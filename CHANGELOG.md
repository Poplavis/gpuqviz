# Changelog

## 0.1.0 (2026-09-13)

首个可用版本。

- 零落盘渲染管线：moderngl 离屏 FBO → 进程内 PyAV 编码（h264_nvenc 探测回退 libx264）
- `render_bloch_video` / `render_heatmap_video` / Scene 声明式 API（JSON 持久化 + CLI）
- 布洛赫球（Phong 球壳/轨迹/多 qubit）、概率热图（viridis/inferno）、相位色盘、SDF 中文字幕
- qiskit 电路采样（按深度分层）+ 批量 einsum Bloch 向量 + slerp 关键帧插值（含对跖点）
- NVENC 显存直喂路径（PyNvVideoCodec + cupy RGBA→NV12 kernel），会话探测失败自动回退
- CPU 软光栅后端（numpy，limited 样式），无 OpenGL 环境自动降级
- 规避的兼容性问题：fbo.read() 全零（改读颜色附件）、色表 REPEAT 边缘混色（clamp）、
  矩阵行/列主序、qiskit 小端序 Bloch 索引、freetype SDF 过渡宽度量纲
