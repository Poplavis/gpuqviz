"""P5.3 渲染 LOD 测试：others 聚合 / 纠缠图边数上限 / Hinton 幅值阈值。"""

import numpy as np

from gpuqviz.analysis.entanglement import EntanglementReport
from gpuqviz.render.density import hinton_cells
from gpuqviz.render.entanglement import entanglement_graph_geometry
from gpuqviz.render.histogram import histogram_bars


# --------------------------------------------------------------------------- #
# 直方图 others 聚合
# --------------------------------------------------------------------------- #

def test_histogram_others_aggregation():
    """others 柱：概率 = 1 − Σ top-k，index = -1，标签 others，排在末尾。"""
    p = np.full(16, 0.04)
    p[:6] = 0.1            # top-6 共 0.6，其余 10 项共 0.4（Σ = 1）
    bars, _ = histogram_bars(p, (0, 0, 800, 400), top_k=6, others=True)
    assert len(bars) == 7
    last = bars[-1]
    assert last[4] == -1 and last[5] == "others"
    assert abs(last[6] - 0.4) < 1e-9
    # others 柱在 x 方向位于最后一根主柱之后
    assert last[0] > bars[-2][0]


def test_histogram_others_skipped_when_rest_negligible():
    """其余概率 ≈ 0 时不画 others 柱。"""
    p = np.zeros(8)
    p[:] = 0.125  # top-8 = 全部
    bars, _ = histogram_bars(p, (0, 0, 800, 400), top_k=8, others=True)
    assert all(b[5] != "others" for b in bars)


def test_histogram_others_default_off():
    p = np.zeros(16)
    p[:6] = 0.1
    bars, _ = histogram_bars(p, (0, 0, 800, 400), top_k=6)
    assert len(bars) == 6  # 默认不聚合


# --------------------------------------------------------------------------- #
# 纠缠图边数上限
# --------------------------------------------------------------------------- #

def _dense_mi_report(n: int) -> EntanglementReport:
    rng = np.random.Generator(np.random.PCG64(3))
    mi = np.triu(rng.random((n, n)) * 2.0, k=1)
    mi = mi + mi.T
    return EntanglementReport(
        n_qubits=n,
        single_entropy=rng.random(n),
        mutual_info=mi,
        negativity=np.zeros((n, n)))


def test_entanglement_edge_cap_keeps_strongest():
    """边数超上限时按互信息保留最强者。"""
    n = 6  # 15 对
    report = _dense_mi_report(n)
    nodes, edges_all = entanglement_graph_geometry(report, (0, 0, 600, 400))
    assert len(edges_all) == 15
    nodes5, edges5 = entanglement_graph_geometry(report, (0, 0, 600, 400),
                                                 max_edges=5)
    assert len(edges5) == 5
    ref_mis = sorted((e[5] for e in edges_all), reverse=True)[:5]
    got_mis = sorted(e[5] for e in edges5)
    assert np.allclose(got_mis, sorted(ref_mis), atol=1e-12)


def test_entanglement_edge_cap_none_keeps_all():
    report = _dense_mi_report(6)
    _, edges = entanglement_graph_geometry(report, (0, 0, 600, 400),
                                           max_edges=None)
    assert len(edges) == 15


# --------------------------------------------------------------------------- #
# Hinton 幅值阈值
# --------------------------------------------------------------------------- #

def test_hinton_min_frac_skips_small_cells():
    """|ρ_ij| < min_frac·max|ρ| 的方块被跳过。"""
    rho = np.diag([0.8, 0.15, 0.04, 0.01]).astype(complex)
    cells_all, _ = hinton_cells(rho, (0, 0, 400, 400))
    assert len(cells_all) == 16
    cells_10, _ = hinton_cells(rho, (0, 0, 400, 400), min_frac=0.1)
    # max|ρ| = 0.8，阈值 0.08 → 非对角（全零）与 0.04/0.01 均被跳过
    kept_i = {c[4] for c in cells_10}
    assert len(cells_10) == 2
    assert 0 in kept_i and 1 in kept_i
    cells_50, _ = hinton_cells(rho, (0, 0, 400, 400), min_frac=0.5)
    # 阈值 0.4 → 只剩 ρ00
    assert len(cells_50) == 1


def test_hinton_min_frac_zero_keeps_all():
    rho = np.diag([0.8, 0.15, 0.04, 0.01]).astype(complex)
    cells, _ = hinton_cells(rho, (0, 0, 400, 400), min_frac=0.0)
    assert len(cells) == 16
