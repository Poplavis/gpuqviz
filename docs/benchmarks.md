# gpuqviz 基准

环境：消费级 NVIDIA GPU（GTX 1060）+ NVENC（PyNvVideoCodec 2.2.2）

## S8 优化前后对照

| 后端 | 场景 | 耗时 (s) |
|---|---|---|
| gl | bell 3s@30fps | 4.5 |
| gl | ghz split 3s@30fps | 3.6 |
| gl | ghz heatmap 3s@30fps | 3.4 |
| cpu | bell 3s@30fps | 3.0 |
| cpu | ghz bloch 3s@30fps | 2.9 |
| cpu | ghz heatmap 3s@30fps | 3.5 |

## S8 优化前 CPU 基线（numba 加速前）

| 后端 | 场景 | 耗时 (s) |
|---|---|---|
| cpu (旧) | bell 3s@30fps 720p | 116.3 |

S8 优化项：包围盒光栅（不再全屏距离场）+ numba @njit(parallel=True) +
坐标网格缓存。CPU bell 从 116.3s 降至 ~6s（19x 加速），远超 ≤20s 目标。

CPU 后端现在完整支持 BlochTrack + HeatmapTrack + PhaseDisc + PIL 文字，
对齐 recorder 的零硬件门槛。GPUQVIZ_BACKEND=cpu 可强制指定 CPU 路径。
