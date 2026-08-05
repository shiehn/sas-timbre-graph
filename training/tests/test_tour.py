"""The tour builder's pure core: spread, path selection, artifact shape.

Everything here runs without Surge. The rendering stages (screening, edge
validation, the composed sweep) are covered by the shipped-artifact
integration test, which needs a real host.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.morph import load_graph, params_at, save_graph
from timbre_graph_lab.tour import (
    TOUR_VERSION,
    Candidate,
    choose_lens,
    delta_to,
    farthest_point_sample,
    role_entry,
    select_tour,
    shared_basis,
)
from timbre_graph_lab.tour import assemble as tour_assemble


def _cand(
    pid: str,
    x: list[float],
    z: list[float],
    sigma: float = 0.01,
    category: str = "Basses",
    names: tuple[str, ...] = ("p0", "p1"),
    subcategory: str = "Basses",
) -> Candidate:
    return Candidate(
        preset_id=pid, name=f"name-{pid}", category=category,
        subcategory=subcategory, path=f"/tmp/{pid}.fxp", sigma=sigma,
        raw={n: v for n, v in zip(names, x)},
        z0=np.array(z, dtype=float),
        x=np.array(x, dtype=float),
    )


# --------------------------------------------------------------------------
# the delta frame
# --------------------------------------------------------------------------


def test_delta_to_lands_exactly_on_the_target():
    base = {"p0": 0.25, "p1": 0.75}
    d = delta_to(np.array([0.6, 0.1]), base, ["p0", "p1"])
    assert d == pytest.approx({"p0": 0.35, "p1": -0.65})


def test_delta_to_skips_names_the_live_host_does_not_expose():
    """Matches the app host's lenient relative mode: unknown names are
    skipped, never defaulted to a value nobody measured."""
    d = delta_to(np.array([0.5, 0.9]), {"p0": 0.2}, ["p0", "p_absent"])
    assert d == pytest.approx({"p0": 0.3})


# --------------------------------------------------------------------------
# the basis is what every anchor shares
# --------------------------------------------------------------------------


def test_basis_is_the_intersection_not_the_union():
    """Surge names oscillator params by oscillator type, so the exposed set
    differs per patch. Only parameters every anchor has can be shipped."""
    a = _cand("a", [0.2, 0.4], [1.0], names=("p0", "p1"))
    b = _cand("b", [0.9, 0.9], [2.0], names=("p0", "p_exotic"))
    names = shared_basis([a, b], ["p0", "p1", "p_exotic"])
    assert names == ["p0"]
    assert a.x == pytest.approx([0.2]) and b.x == pytest.approx([0.9])


def test_basis_follows_allow_list_order_not_dict_order():
    a = _cand("a", [0.1, 0.2, 0.3], [1.0], names=("p2", "p0", "p1"))
    b = _cand("b", [0.4, 0.5, 0.6], [2.0], names=("p2", "p0", "p1"))
    assert shared_basis([a, b], ["p0", "p1", "p2"]) == ["p0", "p1", "p2"]
    assert a.x == pytest.approx([0.2, 0.3, 0.1])


# --------------------------------------------------------------------------
# lens choice
# --------------------------------------------------------------------------


def test_choose_lens_prefers_a_role_appropriate_category():
    cands = [
        _cand("a", [0.1], [1.0], sigma=0.001, category="Keys"),
        _cand("b", [0.2], [2.0], sigma=0.02, category="Basses"),
    ]
    # the Keys patch is steadier, but bass tours must sound like basses
    assert choose_lens(cands, "bass") == 1


def test_choose_lens_takes_the_steadiest_within_the_preferred_category():
    cands = [
        _cand("a", [0.1], [1.0], sigma=0.03, category="Basses"),
        _cand("b", [0.2], [2.0], sigma=0.004, category="Basses"),
    ]
    assert choose_lens(cands, "bass") == 1


# --------------------------------------------------------------------------
# farthest-point sampling
# --------------------------------------------------------------------------


def test_fps_returns_everything_when_the_pool_is_small():
    zs = np.array([[0.0], [1.0], [2.0]])
    assert farthest_point_sample(zs, 0, list("abc"), 5) == [0, 1, 2]


def test_fps_picks_the_spread_not_the_cluster():
    # four rows bunched at 0 and one far away: the far one must be kept
    zs = np.array([[0.0], [0.01], [0.02], [0.03], [10.0]])
    kept = farthest_point_sample(zs, 0, list("abcde"), 2)
    assert 4 in kept


def test_fps_always_keeps_the_lens_and_is_deterministic():
    zs = np.array([[0.0], [1.0], [2.0], [3.0]])
    ids = list("abcd")
    kept = farthest_point_sample(zs, 2, ids, 3)
    assert 2 in kept, "the lens supplies the structure — it cannot be dropped"
    assert kept == farthest_point_sample(zs, 2, ids, 3)


# --------------------------------------------------------------------------
# tour selection
# --------------------------------------------------------------------------


def test_select_tour_walks_every_node_of_a_simple_chain():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    path, info = select_tour(4, edges, list("abcd"), 4)
    assert path == [0, 1, 2, 3]
    assert info["shipped"] == 4
    assert info["isolated_nodes"] == 0


def test_select_tour_prefers_the_heavier_route_at_a_fork():
    # 0-1 then a fork: 1-2 carries the travel, 1-3 does not
    edges = [(0, 1, 1.0), (1, 2, 9.0), (1, 3, 0.1)]
    path, _ = select_tour(4, edges, list("abcd"), 3)
    assert path == [0, 1, 2]
    assert 3 not in path


def test_select_tour_takes_the_largest_component_and_counts_isolated_nodes():
    # component {0,1,2} beats component {3,4}; node 5 has no valid edge at all
    edges = [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 5.0)]
    path, info = select_tour(6, edges, list("abcdef"), 20)
    assert sorted(path) == [0, 1, 2]
    assert info["component_size"] == 3
    assert info["isolated_nodes"] == 1
    assert info["shipped"] == 3 and info["wanted"] == 20


def test_select_tour_reports_a_shortfall_rather_than_inventing_anchors():
    path, info = select_tour(3, [(0, 1, 1.0), (1, 2, 1.0)], list("abc"), 20)
    assert len(path) == 3
    assert info["wanted"] == 20 and info["shipped"] == 3


def test_select_tour_keeps_the_heaviest_window_when_the_path_is_too_long():
    # a chain of 6 where the last three hops carry all the travel
    edges = [(0, 1, 0.1), (1, 2, 0.1), (2, 3, 5.0), (3, 4, 5.0), (4, 5, 5.0)]
    path, info = select_tour(6, edges, list("abcdef"), 4)
    assert path == [2, 3, 4, 5]
    assert info["shipped"] == 4


def test_select_tour_on_an_empty_graph_declines():
    path, info = select_tour(3, [], list("abc"), 20)
    assert path == []
    assert info["shipped"] == 0 and info["isolated_nodes"] == 3


def test_select_tour_is_deterministic_under_reordered_input():
    edges = [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 1.0), (0, 3, 0.5)]
    ids = list("abcd")
    a, _ = select_tour(4, edges, ids, 4)
    b, _ = select_tour(4, list(reversed(edges)), ids, 4)
    assert a == b


# --------------------------------------------------------------------------
# the lens must start the tour: it is the preset the panel actually loads
# --------------------------------------------------------------------------


def test_select_tour_starts_on_the_lens_when_one_is_given():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    path, _ = select_tour(4, edges, list("abcd"), 4, start=2)
    assert path[0] == 2


def test_select_tour_truncates_from_the_lens_outward():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 9.0)]
    path, info = select_tour(5, edges, list("abcde"), 3, start=0)
    # the heaviest hop is at the far end, but dropping the lens is not an option
    assert path[0] == 0 and len(path) == 3
    assert info["shipped"] == 3


def test_select_tour_declines_when_the_lens_has_no_valid_edge():
    # the lens (node 3) is isolated; the rest form a usable component
    edges = [(0, 1, 1.0), (1, 2, 1.0)]
    path, info = select_tour(4, edges, list("abcd"), 4, start=3)
    assert path == []
    assert info["shipped"] == 0


# --------------------------------------------------------------------------
# artifact shape
# --------------------------------------------------------------------------


def test_role_entry_spaces_control_points_over_the_whole_dial():
    cands = [_cand("a", [0.1], [1.0]), _cand("b", [0.5], [2.0]), _cand("c", [0.9], [3.0])]
    entry = role_entry(cands, [0, 1, 2], ["p0"])
    assert entry["control_points"][0] == 0.0
    assert entry["control_points"][-1] == 1.0
    assert len(entry["control_points"]) == len(entry["anchors"]) == 3
    assert entry["declined"] is False


def test_role_entry_declines_a_tour_that_cannot_travel():
    entry = role_entry([_cand("a", [0.1], [1.0])], [0], ["p0"])
    assert entry["declined"] is True


def test_tour_artifact_round_trips_and_interpolates_exactly_at_anchors(tmp_path):
    cands = [_cand("a", [0.0, 1.0], [1.0]), _cand("b", [0.5, 0.5], [2.0]),
             _cand("c", [1.0, 0.0], [3.0])]
    entry = role_entry(cands, [0, 1, 2], ["p0", "p1"])
    entry["role"] = "bass"
    graph = {
        "version": TOUR_VERSION,
        "roles": {"bass": entry}, "quality": {},
    }
    path = save_graph(graph, tmp_path / "tour.json")
    back = load_graph(path)
    assert back["version"] == TOUR_VERSION

    # `params_at` reads control_points off the graph root, so a tour role
    # supplies its own — the runtime contract the panel mirrors.
    role_graph = {"control_points": entry["control_points"], "roles": {"bass": entry}}
    for i, c in enumerate(entry["control_points"]):
        assert params_at(role_graph, "bass", c) == pytest.approx(cands[i].x)
    mid = params_at(role_graph, "bass", 0.25)
    assert mid == pytest.approx([0.25, 0.75])


# --------------------------------------------------------------------------
# the whole pipeline, on a fake synth (no Surge, no rendering)
# --------------------------------------------------------------------------


class FakeWorker:
    """A deterministic stand-in synth.

    Its "sound" is a linear function of the parameters, so travel in
    descriptor space is predictable and the orchestration can be tested
    without Surge. `bad` presets fail to load; `noisy` ones wander enough to
    be screened out.
    """

    def __init__(self, presets: dict[str, dict[str, float]],
                 bad: set[str] | None = None, noisy: set[str] | None = None):
        self.presets = presets
        self.bad = bad or set()
        self.noisy = noisy or set()
        self.loaded: str | None = None
        self._live: dict[str, float] = {}

    def load_preset(self, path, freeze: bool = True) -> bool:
        key = str(path)
        if key in self.bad or key not in self.presets:
            return False
        self.loaded = key
        self._live = dict(self.presets[key])
        return True

    @property
    def baseline_raw(self) -> dict[str, float]:
        return dict(self.presets[self.loaded])

    def apply_delta(self, delta: dict[str, float]) -> dict[str, float]:
        base = self.presets[self.loaded]
        self._live = {
            n: float(np.clip(base[n] + d, 0.0, 1.0))
            for n, d in delta.items() if n in base
        }
        return self._live

    def restore_baseline(self) -> None:
        self._live = dict(self.presets[self.loaded])

    def render(self, probe, settle: bool = True) -> np.ndarray:
        # A real waveform, not a constant: qc_audio rejects flat-topped audio,
        # and a constant array is entirely "pinned at peak".
        t = np.linspace(0, 1, 2048, endpoint=False)
        return 0.4 * np.sin(2 * np.pi * 220 * t) * (1 + sum(self._live.values()) * 1e-6)

    def render_descriptors(self, probe, k: int = 1) -> np.ndarray:
        # 20 descriptors spread over the parameter values, so two different
        # parameter vectors are two different sounds
        vals = [self._live.get(n, 0.0) for n in sorted(self.presets[self.loaded])]
        z = np.zeros(20)
        for i, v in enumerate(vals):
            z[i % 20] += v * 50.0
        return z

    def noise_floor(self, probe, k: int = 4, avg: int = 1) -> np.ndarray:
        return np.full(20, 5.0 if self.loaded in self.noisy else 0.0)


def _fake_pool(n: int, n_params: int = 4) -> dict[str, dict[str, float]]:
    presets = {}
    for i in range(n):
        # spread the pool out so neighbours are genuinely different
        presets[f"/p{i}.fxp"] = {
            f"a_p{j}": ((i * 7 + j * 13) % 11) / 10.0 for j in range(n_params)
        }
    return presets


def _run_pipeline(monkeypatch, tmp_path, presets, bad=None, noisy=None,
                  anchors=5, cap=50, keep=10, k=3):
    from timbre_graph_lab import tour as tour_mod

    entries = [
        {"preset_id": p.strip("/.fxp"), "name": p, "category": "Basses",
         "path": p, "roles": ["bass"]}
        for p in presets
    ]
    monkeypatch.setattr(tour_mod, "load_manifest", lambda cfg: {"entries": entries})
    cfg = LabConfig(workspace=tmp_path)
    worker = FakeWorker(presets, bad=bad, noisy=noisy)
    allowed = sorted({n for v in presets.values() for n in v})
    return tour_mod.build_role_tour(
        cfg, worker, "bass", allowed, anchors, cap, keep, k, 2
    )


def test_pipeline_builds_a_tour_end_to_end(monkeypatch, tmp_path):
    entry, quality = _run_pipeline(monkeypatch, tmp_path, _fake_pool(12))

    assert entry["role"] == "bass"
    assert entry["declined"] is False
    assert len(entry["anchors"]) == len(entry["snapshots"]) == len(entry["control_points"])
    assert entry["control_points"][0] == 0.0
    assert entry["control_points"][-1] == 1.0
    assert len(entry["param_names"]) == 4
    for snap in entry["snapshots"]:
        assert len(snap) == 4
    # every stop is a different preset — the whole point of a tour
    ids = [a["preset_id"] for a in entry["anchors"]]
    assert len(set(ids)) == len(ids)
    assert quality["n_passed"] == 12


def test_pipeline_drops_presets_that_will_not_load(monkeypatch, tmp_path):
    presets = _fake_pool(10)
    _, quality = _run_pipeline(
        monkeypatch, tmp_path, presets, bad={"/p3.fxp", "/p7.fxp"}
    )
    assert quality["n_passed"] == 8
    assert quality["n_pool"] == 10


def test_pipeline_drops_presets_whose_own_renders_wander(monkeypatch, tmp_path):
    presets = _fake_pool(10)
    _, quality = _run_pipeline(
        monkeypatch, tmp_path, presets, noisy={"/p2.fxp", "/p5.fxp"}
    )
    assert quality["n_passed"] == 8


def test_pipeline_declines_rather_than_shipping_a_tour_of_one(monkeypatch, tmp_path):
    entry, quality = _run_pipeline(monkeypatch, tmp_path, _fake_pool(1))
    assert entry["declined"] is True
    assert entry["anchors"] == []
    assert quality["shortfall"]["shipped"] == 0


def test_pipeline_reports_a_shortfall_instead_of_padding(monkeypatch, tmp_path):
    # ask for 20 anchors from a pool that cannot supply them
    entry, quality = _run_pipeline(monkeypatch, tmp_path, _fake_pool(6), anchors=20)
    assert len(entry["anchors"]) < 20
    assert quality["shortfall"] is not None
    assert quality["shortfall"]["wanted"] == 20


def test_pipeline_starts_the_tour_on_the_lens(monkeypatch, tmp_path):
    entry, quality = _run_pipeline(monkeypatch, tmp_path, _fake_pool(12))
    # the lens is the preset the panel loads; it must be dial position 0
    assert entry["anchors"][0]["preset_id"] == quality["lens"]["preset_id"]


def test_pipeline_saves_after_every_role(monkeypatch, tmp_path):
    """Six roles is most of an hour of rendering; a crash on the last one must
    not discard the first five."""
    from timbre_graph_lab import tour as tour_mod

    presets = _fake_pool(8)
    entries = [
        {"preset_id": p.strip("/.fxp"), "name": p, "category": "Basses",
         "path": p, "roles": ["bass", "pad"]}
        for p in presets
    ]
    monkeypatch.setattr(tour_mod, "load_manifest", lambda cfg: {"entries": entries})
    monkeypatch.setattr(tour_mod, "RenderWorker", lambda cfg: FakeWorker(presets))
    allowed = sorted({n for v in presets.values() for n in v})
    policy = tmp_path / "policy"
    policy.mkdir(parents=True)
    (policy / "policy-v2.json").write_text(json.dumps({"allowed": allowed}))

    out = tmp_path / "tour.json"
    seen: list[int] = []
    real_save = tour_mod.__dict__.get("assemble")

    def spy(role_entries, quality):
        seen.append(len(role_entries))
        return real_save(role_entries, quality)

    monkeypatch.setattr(tour_mod, "assemble", spy)
    tour_mod.build_tour_graph(
        LabConfig(workspace=tmp_path), roles=["bass", "pad"],
        anchors=4, cap=20, keep=6, k=3, n_interior=2, save_to=out,
    )

    # written once per role, not only at the end
    assert seen[:2] == [1, 2]
    written = json.loads(out.read_text())
    assert set(written["roles"]) == {"bass", "pad"}
    assert written["version"] == TOUR_VERSION


def test_assemble_summarises_only_the_roles_it_has():
    entry = role_entry([_cand("a", [0.1], [1.0]), _cand("b", [0.9], [2.0])],
                       [0, 1], ["p0"])
    entry["role"] = "bass"
    g = tour_assemble({"bass": entry}, {"bass": {}})
    assert g["quality"]["_summary"]["anchors_by_role"] == {"bass": 2}
    assert g["quality"]["_summary"]["roles_declined"] == []


# --------------------------------------------------------------------------
# the greedy trap: what shipped 3 anchors out of a 31-node component
# --------------------------------------------------------------------------


def _spur_graph(core: int = 31, spurs: int = 3) -> tuple[int, list, list[str]]:
    """A long chain of modest hops with a few FAT dead-end spurs hanging off
    its start — the shape a kNN descriptor graph actually takes, where the
    biggest jumps lead to the sparse edge of the space."""
    edges = [(i, i + 1, 1.0) for i in range(core - 1)]
    for s in range(spurs):
        edges.append((s + 1, core + s, 9.0))
    ids = [f"p{i:03d}" for i in range(core + spurs)]
    return core + spurs, edges, ids


def test_select_tour_does_not_strand_itself_on_a_fat_dead_end():
    """Taking the heaviest hop first walked into a spur and ended after 3
    anchors, out of a 31-node connected component (measured on the real pad
    and kick roles, 2026-08-03)."""
    n, edges, ids = _spur_graph()
    path, info = select_tour(n, edges, ids, 20, start=0)
    assert info["shipped"] == 20
    assert len(set(path)) == 20


def test_select_tour_still_takes_a_dead_end_when_it_is_the_only_move():
    # 0-1 then only a spur: refusing dead ends must not mean refusing to move
    path, _ = select_tour(3, [(0, 1, 1.0), (1, 2, 5.0)], list("abc"), 20, start=0)
    assert path == [0, 1, 2]


def test_select_tour_prefers_travel_among_equally_safe_steps():
    # two onward-equivalent branches; the heavier one wins
    edges = [(0, 1, 1.0), (0, 2, 9.0), (1, 3, 1.0), (2, 4, 1.0)]
    path, _ = select_tour(5, edges, list("abcde"), 3, start=0)
    assert path[:2] == [0, 2]
