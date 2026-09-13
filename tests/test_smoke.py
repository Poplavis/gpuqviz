"""冒烟测试：包可导入、版本存在、环境报告可生成（缺 GPU 不允许抛异常）。"""

import gpuqviz


def test_version():
    assert gpuqviz.__version__


def test_report_env_returns_string():
    text = gpuqviz.report_env()
    assert isinstance(text, str)
    assert "Python" in text
    assert "推荐后端" in text
