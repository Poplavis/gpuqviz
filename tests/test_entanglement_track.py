"""EntanglementTrack / entanglement_graph_geometry 测试（几何契约 + CPU 冒烟）。"""

import numpy as np

from gpuqviz.analysis.entanglement import EntanglementReport, entanglement_summary
from gpuqviz.render.entanglement import entanglement_graph_geometry
from gpuqviz.state import Statevector


# --------------------------------------------------------------------------- #
# 几何契约（GL/CPU 共享）
# --------------------------------------------------------------------------- #

def _ghz_report(n=3) -> EntanglementReport:
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    return entanglement_summary(Statevector(psi))


def test_geometry_nodes_on_ring():
    rep = _ghz_report()
    rect = (40, 20, 560, 320)
    nodes, edges = entanglement_graph_geometry(rep, rect)
    assert len(nodes) == 3
    x, y, w, h = rect
    _cx, _cy = x + w / 2, y + h / 2
    for nx, ny, r, _s, label in nodes:
        assert x <= nx <= x + w and y <= ny <= y + h
        assert label.startswith("q")
    # 全纠缠 GHZ：所有节点半径 = r_max
    r_max = min(w, h) * 0.085
    for _, _, r, _, _ in nodes:
        assert abs(r - r_max) < 1e-9


def test_geometry_edges_track_mutual_info():
    rep = _ghz_report()
    nodes, edges = entanglement_graph_geometry(rep, (0, 0, 600, 400))
    # GHZ: 3 对全纠缠 → 3 条边，粗细一致
    assert len(edges) == 3
    for x0, y0, x1, y1, t, mi in edges:
        assert abs(mi - 1.0) < 1e-9
        assert abs(t - (1.0 + 5.0 * 0.5)) < 1e-9  # MI=1 → 一半上限

    # 乘积态：无边
    rep0 = entanglement_summary(Statevector(np.array([1, 0, 0, 0], dtype=complex)))
    _, edges0 = entanglement_graph_geometry(rep0, (0, 0, 600, 400))
    assert edges0 == []


def test_geometry_edge_ends_at_node_centers():
    rep = _ghz_report()
    nodes, edges = entanglement_graph_geometry(rep, (0, 0, 600, 400))
    centers = {(round(n[0], 6), round(n[1], 6)) for n in nodes}
    for x0, y0, x1, y1, *_ in edges:
        assert (round(x0, 6), round(y0, 6)) in centers
        assert (round(x1, 6), round(y1, 6)) in centers


# --------------------------------------------------------------------------- #
# Scene 集成
# --------------------------------------------------------------------------- #

def test_entanglement_track_json_roundtrip(tmp_path):
    from gpuqviz import EntanglementTrack, Scene

    scene = Scene(
        width=640, height=360, fps=30, duration=1.0,
        tracks=[EntanglementTrack(states_path="s.npz", layout="bottom")],
    )
    p = tmp_path / "scene.json"
    scene.save_json(p)
    from gpuqviz import Scene as Scene2

    scene2 = Scene2.model_validate_json(p.read_text(encoding="utf-8"))
    assert scene2.tracks[0].kind == "entanglement"


def test_cpu_render_frame_entanglement_png(tmp_path, monkeypatch):
    """CPU 后端 scene 渲染含 EntanglementTrack：节点像素存在。"""
    import gpuqviz
    from gpuqviz import EntanglementTrack, Scene

    # 直接 patch 后端缓存（不污染进程级 _cached，teardown 自动恢复）
    import gpuqviz.backends as _backends

    monkeypatch.setattr(_backends, "_cached", "cpu")

    states = np.array([
        [1, 0, 0, 0],
        [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
    ], dtype=complex)
    npz = tmp_path / "states.npz"
    np.savez(npz, states=states)

    scene = Scene(
        width=640, height=360, fps=30, duration=1.0,
        tracks=[EntanglementTrack(states_path="states.npz", layout="full")],
    )
    out = tmp_path / "ent.png"
    gpuqviz.render_frame(scene=scene, t=1.0, out=out, scale=1,
                         states_dir=tmp_path)

    from PIL import Image

    arr = np.array(Image.open(out).convert("RGB"))
    blue = (arr[:, :, 2].astype(int) - arr[:, :, 0].astype(int) > 60)
    assert blue.sum() > 100, "CPU 渲染的纠缠图应有可见节点"
