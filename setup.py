"""扩展模块构建（0.9.0 T3：Cython C 内核）。

- 默认把 src/gpuqviz/_core.pyx 编译为 gpuqviz._core 扩展；
- 无 Cython 或显式 GPUQVIZ_DISABLE_CYTHON=1 时跳过（纯 Python 轮子）；
- 运行时缺扩展自动回退 numpy 路径（gpuqviz.noise / gpuqviz.circuits）。
"""

import os

from setuptools import Extension, setup

ext_modules = []
if not os.environ.get("GPUQVIZ_DISABLE_CYTHON"):
    try:
        from Cython.Build import cythonize

        ext_modules = cythonize(
            [Extension("gpuqviz._core", ["src/gpuqviz/_core.pyx"])],
            language_level=3,
        )
    except ImportError:
        ext_modules = []  # 构建Env无 Cython：纯 Python 安装

setup(ext_modules=ext_modules)
