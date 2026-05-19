"""Pure-classical triage layer (`polygram.analysis`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polygram import Cancellation, Dictionary, Feature, load_toy_sae
from polygram.analysis import (
    FLOOR_BLOCK,
    KNOB_SELECTION_GUIDANCE,
    SEPARATION_EDGE_FORMULA,
    SHARING_EDGE_FORMULA,
    SUITABILITY_FORMULA,
    FeatureEdge,
    FeatureGraph,
    PairPrediction,
    TriagePrediction,
    build_separation_graph,
    build_sharing_graph,
    encoding_suitability_score,
    feature_sensitivity,
    predict_cancellation_depth,
    render_feature_graph_section,
    render_report,
)
from polygram.cli import main as cli_main
from polygram.sae_import import SelectionReport

FIXTURE = Path("tests/fixtures/toy_sae.json")


@pytest.fixture
def records():
    return load_toy_sae(FIXTURE)


def test_predicted_floor_matches_cancellation_for_target_pair(records):
    feature_ids = [0, 1, 4, 5]  # 2 mammals + 2 birds in toy fixture
    prediction = predict_cancellation_depth(records, feature_ids)

    a_name = prediction.dictionary.features[0].name
    b_name = prediction.dictionary.features[2].name  # cross-cluster pair
    pair = next(
        p for p in prediction.pairs
        if {p.feature_a, p.feature_b} == {a_name, b_name}
    )

    canc = Cancellation(
        dictionary=prediction.dictionary,
        target_pair=(a_name, b_name),
    )
    assert abs(canc.structural_floor() - pair.structural_floor) < 1e-9


def test_per_feature_sensitivity_is_mean_abs_v_over_pairs(records):
    feature_ids = [0, 1, 4, 5]
    prediction = predict_cancellation_depth(records, feature_ids)

    name = prediction.dictionary.features[0].name
    pairs_with = [
        p for p in prediction.pairs
        if name in (p.feature_a, p.feature_b)
    ]
    expected = float(np.mean([abs(p.V) for p in pairs_with]))
    assert abs(prediction.feature_sensitivity[name] - expected) < 1e-12


def test_encoding_suitability_score_in_unit_interval(records):
    score = encoding_suitability_score(records, [0, 1, 4, 5])
    assert 0.0 <= score <= 1.0


def test_pair_decomposition_obeys_M_plus_V_identity(records):
    prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
    for p in prediction.pairs:
        assert abs(p.M + p.V - p.current_overlap) < 1e-9
        assert abs(p.M - p.V - p.m_pi) < 1e-9
        assert abs(p.structural_floor - (p.M - abs(p.V))) < 1e-9
        assert p.cancellation_gap >= -1e-12


def test_feature_sensitivity_helper_matches_full_prediction(records):
    full = predict_cancellation_depth(records, [0, 1, 4, 5])
    helper = feature_sensitivity(records, [0, 1, 4, 5])
    assert helper == full.feature_sensitivity


def test_render_report_contains_required_sections(records):
    prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
    text = render_report(
        prediction, sae_path=str(FIXTURE), feature_ids=[0, 1, 4, 5]
    )
    assert "# Polygram analysis" in text
    assert "## Caveats" in text
    assert "## Pair predictions" in text
    assert "## Per-feature sensitivity" in text
    assert "## Choosing knobs" in text
    assert "## Encoding suitability" in text
    assert "rung-1" in text  # caveats mention the assumption
    assert SUITABILITY_FORMULA.splitlines()[0] in text


def test_render_report_quotes_knob_selection_guidance(records):
    prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
    text = render_report(
        prediction, sae_path=str(FIXTURE), feature_ids=[0, 1, 4, 5]
    )
    assert KNOB_SELECTION_GUIDANCE in text
    assert "cluster-shatterer" in text


def test_knob_selection_guidance_constant_exposed():
    assert isinstance(KNOB_SELECTION_GUIDANCE, str)
    assert KNOB_SELECTION_GUIDANCE  # non-empty
    for required in (
        "cluster-shatterer",
        "Structural floor",
        "Rz",
        "<cluster>.phi",
    ):
        assert required in KNOB_SELECTION_GUIDANCE, (
            f"missing required substring: {required!r}"
        )


def test_predict_refuses_oversized_subset(records):
    # Triage operates on flat dictionaries only and pins
    # `clustered=False` internally; oversized subsets must raise.
    with pytest.raises(ValueError, match="caps a Dictionary"):
        predict_cancellation_depth(records, list(range(9)))


def test_cli_analyze_writes_report(tmp_path: Path):
    out = tmp_path / "report.md"
    rc = cli_main(
        [
            "analyze",
            str(FIXTURE),
            "--features",
            "0,1,4,5",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text()
    assert "# Polygram analysis" in text
    assert "## Pair predictions" in text


def test_cli_analyze_rejects_bad_features(tmp_path: Path):
    out = tmp_path / "report.md"
    with pytest.raises(SystemExit, match="comma-separated ints"):
        cli_main(
            [
                "analyze",
                str(FIXTURE),
                "--features",
                "1,2,abc",
                "--output",
                str(out),
            ]
        )


def test_cli_analyze_missing_path(tmp_path: Path):
    with pytest.raises(SystemExit, match="SAE file not found"):
        cli_main(
            [
                "analyze",
                str(tmp_path / "ghost.json"),
                "--features",
                "0,1",
                "--output",
                str(tmp_path / "r.md"),
            ]
        )


def _write_diverse_sibling_fixture(path: Path) -> list[int]:
    """Write a 4-feature toy-SAE JSON whose two cluster siblings have
    deliberately diverse projection vectors. Without `assign_gamma`,
    rung-1's identical-β-per-cluster collapses within-cluster overlap
    to 1.0; with `assign_gamma`, per-cluster PCA gives each sibling a
    distinct γ and the overlap drops below 1.

    Returns the list of feature ids written.
    """
    payload = {
        "schema_version": 1,
        "description": "diverse-sibling fixture",
        "features": [
            {
                "feature_id": 0,
                "name": "a0",
                "label": "alpha/a0",
                "projection": [1.0, 0.0, 0.0, 0.0, 0.05, -0.02, 0.03, -0.04],
            },
            {
                "feature_id": 1,
                "name": "a1",
                "label": "alpha/a1",
                "projection": [0.0, 1.0, 0.0, 0.0, -0.05, 0.04, -0.03, 0.02],
            },
            {
                "feature_id": 2,
                "name": "b0",
                "label": "beta/b0",
                "projection": [0.0, 0.0, 1.0, 0.0, 0.02, -0.05, 0.04, -0.03],
            },
            {
                "feature_id": 3,
                "name": "b1",
                "label": "beta/b1",
                "projection": [0.0, 0.0, 0.0, 1.0, -0.02, 0.05, -0.04, 0.03],
            },
        ],
    }
    path.write_text(json.dumps(payload))
    return [0, 1, 2, 3]


class TestAnalyzeAssignGamma:
    def test_no_flag_collapses_within_cluster_overlap_to_one(
        self, tmp_path: Path
    ):
        fix = tmp_path / "diverse.json"
        ids = _write_diverse_sibling_fixture(fix)
        out = tmp_path / "report_no_gamma.md"
        rc = cli_main(
            [
                "analyze",
                str(fix),
                "--features",
                ",".join(str(i) for i in ids),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        text = out.read_text()
        # Within-cluster pairs (a0↔a1, b0↔b1) collapse to 1.0000.
        assert "| a0 ↔ a1 | intra | 1.0000 |" in text
        assert "| b0 ↔ b1 | intra | 1.0000 |" in text

    def test_flag_breaks_within_cluster_collapse(self, tmp_path: Path):
        fix = tmp_path / "diverse.json"
        ids = _write_diverse_sibling_fixture(fix)
        out = tmp_path / "report_gamma.md"
        rc = cli_main(
            [
                "analyze",
                str(fix),
                "--features",
                ",".join(str(i) for i in ids),
                "--output",
                str(out),
                "--assign-gamma",
            ]
        )
        assert rc == 0
        text = out.read_text()
        # Within-cluster pairs no longer collapse to 1.0 — γ-PCA
        # differentiates the siblings.
        assert "| a0 ↔ a1 | intra | 1.0000 |" not in text
        assert "| b0 ↔ b1 | intra | 1.0000 |" not in text

    def test_n_clusters_forwarded(self, tmp_path: Path):
        # Strip labels so k-means is the cluster path; ask for 3
        # clusters and confirm the report names kmeans.
        fix = tmp_path / "diverse.json"
        ids = _write_diverse_sibling_fixture(fix)
        # Drop labels in-place so `from_sae_lens` falls back to k-means.
        data = json.loads(fix.read_text())
        for entry in data["features"]:
            entry.pop("label", None)
        fix.write_text(json.dumps(data))
        out = tmp_path / "report_kmeans.md"
        rc = cli_main(
            [
                "analyze",
                str(fix),
                "--features",
                ",".join(str(i) for i in ids),
                "--output",
                str(out),
                "--n-clusters",
                "3",
            ]
        )
        assert rc == 0
        assert "Cluster method: `kmeans`" in out.read_text()

    def test_n_clusters_zero_rejected(self, tmp_path: Path):
        fix = tmp_path / "diverse.json"
        _write_diverse_sibling_fixture(fix)
        with pytest.raises(SystemExit):
            cli_main(
                [
                    "analyze",
                    str(fix),
                    "--features",
                    "0,1,2,3",
                    "--output",
                    str(tmp_path / "r.md"),
                    "--n-clusters",
                    "0",
                ]
            )


# ---------------------------------------------------------------------------
# Sharing / separation graph artifacts
# ---------------------------------------------------------------------------


def _synthetic_prediction(
    pairs: list[PairPrediction],
    feature_names: list[str],
    cluster_per_feature: dict[str, str],
) -> TriagePrediction:
    """Build a TriagePrediction without re-running the analytic Gram —
    handy for hand-tuned edge-weight scenarios."""
    features = [
        Feature(name=n, cluster=cluster_per_feature[n], beta=0.0)
        for n in feature_names
    ]
    hierarchy: dict[str, list[str]] = {}
    for n in feature_names:
        hierarchy.setdefault(cluster_per_feature[n], []).append(n)
    dictionary = Dictionary(
        name="Synthetic", features=features, hierarchy=hierarchy
    )
    report = SelectionReport(
        n_input_features=len(features),
        n_selected=len(features),
        cluster_assignments={n: cluster_per_feature[n] for n in feature_names},
        cluster_method="user",
        beta_variance_explained=1.0,
        reconstruction_error={n: 0.0 for n in feature_names},
        tier_preservation=None,
    )
    sensitivity = {n: 0.0 for n in feature_names}
    return TriagePrediction(
        dictionary=dictionary,
        selection_report=report,
        pairs=pairs,
        feature_sensitivity=sensitivity,
        encoding_suitability_score=0.0,
    )


def _pair(
    a: str,
    b: str,
    *,
    ca: str,
    cb: str,
    current: float,
    floor: float,
) -> PairPrediction:
    gap = current - floor
    big_v = 0.5 * (current - floor)
    big_m = 0.5 * (current + floor)
    return PairPrediction(
        feature_a=a,
        feature_b=b,
        cluster_a=ca,
        cluster_b=cb,
        current_overlap=current,
        m_pi=floor,
        M=big_m,
        V=big_v,
        structural_floor=floor,
        cancellation_gap=gap,
    )


class TestSharingGraph:
    def test_edges_respect_threshold(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_sharing_graph(prediction, threshold=0.6)
        for e in graph.edges:
            assert e.weight >= 0.6 - 1e-12

    def test_weights_in_unit_interval(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_sharing_graph(
            prediction, threshold=0.0, allow_cross_cluster=True
        )
        for e in graph.edges:
            assert 0.0 <= e.weight <= 1.0

    def test_cross_cluster_gated_by_flag(self):
        # Cross-cluster pair with low floor + high gap so the cluster
        # gate is the only thing blocking it.
        cross = _pair("a", "x", ca="c1", cb="c2", current=0.4, floor=0.05)
        within = _pair("a", "b", ca="c1", cb="c1", current=0.4, floor=0.05)
        prediction = _synthetic_prediction(
            pairs=[cross, within],
            feature_names=["a", "b", "x"],
            cluster_per_feature={"a": "c1", "b": "c1", "x": "c2"},
        )
        gated = build_sharing_graph(
            prediction, threshold=0.0, allow_cross_cluster=False
        )
        for e in gated.edges:
            assert not e.is_cross_cluster

        opened = build_sharing_graph(
            prediction, threshold=0.0, allow_cross_cluster=True
        )
        assert any(e.is_cross_cluster for e in opened.edges)

    def test_high_floor_blocks_edge(self):
        # Floor above FLOOR_BLOCK with a tiny gap — would otherwise pass
        # threshold=0 but the floor gate must zero the weight.
        high = _pair("x", "y", ca="c1", cb="c1", current=0.95, floor=0.7)
        low = _pair("p", "q", ca="c1", cb="c1", current=0.30, floor=0.05)
        prediction = _synthetic_prediction(
            pairs=[high, low],
            feature_names=["x", "y", "p", "q"],
            cluster_per_feature={"x": "c1", "y": "c1", "p": "c1", "q": "c1"},
        )
        graph = build_sharing_graph(prediction, threshold=0.0)
        names = {(e.source, e.target) for e in graph.edges}
        assert ("x", "y") not in names
        assert ("p", "q") in names
        assert FLOOR_BLOCK == 0.5

    def test_clusters_are_connected_components(self):
        # Edges: a–b kept, b–c kept, d–e kept, f isolated.
        # Expected components: {a,b,c}, {d,e}, {f}.
        ab = _pair("a", "b", ca="g1", cb="g1", current=0.40, floor=0.04)
        bc = _pair("b", "c", ca="g1", cb="g1", current=0.40, floor=0.04)
        de = _pair("d", "e", ca="g2", cb="g2", current=0.40, floor=0.04)
        # Force these below threshold:
        af = _pair("a", "f", ca="g1", cb="g3", current=0.10, floor=0.09)
        prediction = _synthetic_prediction(
            pairs=[ab, bc, de, af],
            feature_names=["a", "b", "c", "d", "e", "f"],
            cluster_per_feature={
                "a": "g1", "b": "g1", "c": "g1",
                "d": "g2", "e": "g2",
                "f": "g3",
            },
        )
        graph = build_sharing_graph(
            prediction, threshold=0.5, allow_cross_cluster=True
        )
        # All nodes appear exactly once across components.
        flat = [n for c in graph.clusters for n in c]
        assert sorted(flat) == sorted(graph.nodes)
        assert len(flat) == len(set(flat))
        component_sets = [set(c) for c in graph.clusters]
        assert {"a", "b", "c"} in component_sets
        assert {"d", "e"} in component_sets
        assert {"f"} in component_sets

    def test_kind_and_formula_are_sharing(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_sharing_graph(prediction)
        assert graph.kind == "sharing"
        assert graph.metadata["kind"] == "sharing"
        assert graph.metadata["formula"] == SHARING_EDGE_FORMULA
        assert graph.metadata["threshold"] == 0.5
        assert graph.metadata["allow_cross_cluster"] is False
        assert graph.metadata["total_features"] == 4


class TestSeparationGraph:
    def test_weights_equal_floor_on_kept_pairs(self):
        # Two cross-cluster pairs above threshold; weight should equal
        # the structural floor (clipped).
        cross_hi = _pair("a", "x", ca="c1", cb="c2", current=0.6, floor=0.5)
        cross_lo = _pair("a", "y", ca="c1", cb="c2", current=0.3, floor=0.25)
        # Below-threshold cross-cluster pair (floor < 0.2):
        cross_drop = _pair("b", "y", ca="c1", cb="c2", current=0.2, floor=0.1)
        prediction = _synthetic_prediction(
            pairs=[cross_hi, cross_lo, cross_drop],
            feature_names=["a", "b", "x", "y"],
            cluster_per_feature={
                "a": "c1", "b": "c1", "x": "c2", "y": "c2",
            },
        )
        graph = build_separation_graph(prediction, threshold=0.2)
        kept = {(e.source, e.target): e for e in graph.edges}
        assert ("a", "x") in kept
        assert ("a", "y") in kept
        assert ("b", "y") not in kept
        for src, tgt, expected_floor in [("a", "x", 0.5), ("a", "y", 0.25)]:
            edge = kept[(src, tgt)]
            assert abs(edge.weight - min(expected_floor, 1.0)) < 1e-12
            assert abs(edge.floor - expected_floor) < 1e-12

    def test_within_cluster_gated_by_flag(self):
        # Within-cluster pair with high floor — gated by include flag.
        within = _pair("a", "b", ca="c1", cb="c1", current=0.95, floor=0.6)
        # Provide a cross-cluster anchor so feature x doesn't drop out.
        cross = _pair("a", "x", ca="c1", cb="c2", current=0.4, floor=0.3)
        prediction = _synthetic_prediction(
            pairs=[within, cross],
            feature_names=["a", "b", "x"],
            cluster_per_feature={"a": "c1", "b": "c1", "x": "c2"},
        )
        gated = build_separation_graph(
            prediction, threshold=0.2, include_within_cluster=False
        )
        for e in gated.edges:
            assert e.is_cross_cluster

        opened = build_separation_graph(
            prediction, threshold=0.2, include_within_cluster=True
        )
        kept = {(e.source, e.target) for e in opened.edges}
        assert ("a", "b") in kept

    def test_clusters_are_connected_components(self):
        # Cross-cluster floor edges form a chain a–x–c, plus d alone.
        ax = _pair("a", "x", ca="c1", cb="c2", current=0.6, floor=0.5)
        xc = _pair("x", "c", ca="c2", cb="c3", current=0.5, floor=0.4)
        # Below threshold:
        dx = _pair("d", "x", ca="c4", cb="c2", current=0.2, floor=0.05)
        prediction = _synthetic_prediction(
            pairs=[ax, xc, dx],
            feature_names=["a", "x", "c", "d"],
            cluster_per_feature={
                "a": "c1", "x": "c2", "c": "c3", "d": "c4",
            },
        )
        graph = build_separation_graph(prediction, threshold=0.2)
        component_sets = [set(c) for c in graph.clusters]
        assert {"a", "x", "c"} in component_sets
        assert {"d"} in component_sets
        flat = [n for c in graph.clusters for n in c]
        assert sorted(flat) == ["a", "c", "d", "x"]

    def test_kind_and_formula_are_separation(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_separation_graph(prediction)
        assert graph.kind == "separation"
        assert graph.metadata["kind"] == "separation"
        assert graph.metadata["formula"] == SEPARATION_EDGE_FORMULA
        assert graph.metadata["threshold"] == 0.2
        assert graph.metadata["include_within_cluster"] is False
        assert graph.metadata["total_features"] == 4


class TestFeatureGraphSerialization:
    def test_to_json_round_trips(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_sharing_graph(
            prediction, threshold=0.0, allow_cross_cluster=True
        )
        parsed = json.loads(graph.to_json())
        assert parsed["kind"] == graph.kind
        assert parsed["nodes"] == list(graph.nodes)
        assert len(parsed["edges"]) == len(graph.edges)
        for raw, edge in zip(parsed["edges"], graph.edges):
            for fld in (
                "source",
                "target",
                "weight",
                "floor",
                "gap",
                "is_cross_cluster",
                "reason",
            ):
                assert fld in raw
            assert raw["source"] == edge.source
            assert raw["target"] == edge.target
            assert raw["reason"] == edge.reason
        assert parsed["metadata"]["formula"] == SHARING_EDGE_FORMULA
        assert parsed["clusters"] == [list(c) for c in graph.clusters]

    def test_to_json_byte_identical(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_separation_graph(prediction, threshold=0.05)
        a = graph.to_json()
        b = graph.to_json()
        assert a == b
        # Independent build of an "equal" graph also serializes the
        # same — the deterministic ordering is what we promise.
        again = build_separation_graph(prediction, threshold=0.05)
        assert again.to_json() == a


class TestRenderFeatureGraphSection:
    def test_sharing_section_headings_and_formula(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_sharing_graph(prediction, threshold=0.0)
        text = render_feature_graph_section(graph)
        assert "## Sharing graph" in text
        assert "## Separation graph" not in text
        assert SHARING_EDGE_FORMULA in text
        assert "### Edges" in text
        assert "### Components" in text
        assert "### Formula" in text

    def test_separation_section_headings_and_formula(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_separation_graph(prediction, threshold=0.0)
        text = render_feature_graph_section(graph)
        assert "## Separation graph" in text
        assert "## Sharing graph" not in text
        assert SEPARATION_EDGE_FORMULA in text


class TestSharingGraphEdgeOrdering:
    def test_edges_sorted_by_descending_weight(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_sharing_graph(
            prediction, threshold=0.0, allow_cross_cluster=True
        )
        weights = [e.weight for e in graph.edges]
        assert weights == sorted(weights, reverse=True)


class TestFeatureGraphBuilderInvariants:
    def test_nodes_order_matches_dictionary(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_sharing_graph(prediction)
        assert list(graph.nodes) == [
            f.name for f in prediction.dictionary.features
        ]

    def test_singletons_appear_as_size_one_components(self, records):
        prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
        graph = build_separation_graph(prediction, threshold=0.95)
        # threshold high enough that no edges survive — every feature
        # should still appear as a singleton component.
        flat = [n for c in graph.clusters for n in c]
        assert sorted(flat) == sorted(graph.nodes)
        assert all(len(c) == 1 for c in graph.clusters)

    def test_edges_and_feature_edge_dataclass_fields(self):
        e = FeatureEdge(
            source="a",
            target="b",
            weight=0.42,
            floor=0.1,
            gap=0.3,
            is_cross_cluster=True,
            reason="phase_headroom",
        )
        # Frozen — assignment must fail.
        with pytest.raises(Exception):
            e.weight = 0.5
        assert isinstance(e, FeatureEdge)
        empty = FeatureGraph(
            kind="sharing", nodes=("a",), edges=(), clusters=(("a",),)
        )
        assert empty.metadata == {}
        assert empty.to_json()
