# gpuqviz 基准

环境：消费级 NVIDIA GPU（NVENC 会话不可用，编码走 libx264）

| 后端 | 场景 | 耗时 (s) |
|---|---|---|
| gl | bell 3s@30fps | 3.9 |
| gl | ghz split 3s@30fps | 3.3 |
| cpu | bell 3s@30fps | 112.8 |
