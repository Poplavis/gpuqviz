"""GPU 颜色空间与像素格式转换（cupy kernel）。

RGBA8 → NV12（BT.709 limited range），供 NvencEncoder 显存直喂：
全流程留在 GPU，禁止回 CPU。
"""

from __future__ import annotations

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

_RGBA_TO_NV12_KERNEL = r"""
extern "C" __global__
void rgba_to_nv12(const unsigned char* rgba, unsigned char* nv12,
                  int width, int height) {
    // 网格覆盖亮度平面（含 2x2 下采样）
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // 像素列
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // 像素行
    if (x >= width || y >= height) return;

    // 取 2x2 块的平均 RGB（本线程处理 (2x, 2y) 为块左上）
    float r = 0.f, g = 0.f, b = 0.f;
    #pragma unroll
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int px = min(2 * x + dx, width - 1);
            int py = min(2 * y + dy, height - 1);
            long idx = (long(py) * width + px) * 4;
            r += rgba[idx + 0]; g += rgba[idx + 1]; b += rgba[idx + 2];
        }
    }
    r /= 4.f * 255.f; g /= 4.f * 255.f; b /= 4.f * 255.f;

    // BT.709 limited range
    float y709 = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float Y = 16.f + 219.f * y709;
    float Cb = 128.f + 224.f * ((b - y709) / 1.8556f);
    float Cr = 128.f + 224.f * ((r - y709) / 1.5748f);

    nv12[(long)y * width + x] = (unsigned char)min(max(Y, 0.f), 255.f);
    if (y % 2 == 0 && x % 2 == 0) {
        long uv_index = (long)height * width + (long)(y / 2) * width + x;
        nv12[uv_index + 0] = (unsigned char)min(max(Cb, 0.f), 255.f);   // U
        nv12[uv_index + 1] = (unsigned char)min(max(Cr, 0.f), 255.f);   // V
    }
}
"""

_kernel_cache = {}


def rgba_to_nv12(rgba):
    """cupy RGBA (H, W, 4) uint8 → NV12 flat cupy uint8（长度 H*W*3/2）。"""
    if cp is None:
        raise RuntimeError("cupy is required for GPU color conversion")
    h, w = rgba.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"even dimensions required, got {w}x{h}")
    if "k" not in _kernel_cache:
        _kernel_cache["k"] = cp.RawKernel(_RGBA_TO_NV12_KERNEL, "rgba_to_nv12")
    out = cp.zeros(h * w * 3 // 2, dtype=cp.uint8)
    block = (16, 16)
    grid = ((w // 2 + block[0] - 1) // block[0], (h // 2 + block[1] - 1) // block[1])
    _kernel_cache["k"](grid, block, (rgba.reshape(-1), out,
                                     cp.int32(w), cp.int32(h)))
    return out
