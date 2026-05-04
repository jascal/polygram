"""BatchExperiment / BatchResults / BatchRun coverage."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from polygram import (
    BatchExperiment,
    BatchResults,
    BatchRun,
    Dictionary,
    Feature,
    HEA_Rung2,
    MPSRung1,
)
from polygram.analysis import (
    build_separation_graph,
    triage_dictionary,
)
from polygram.analysis.feature_graph import FeatureEdge, FeatureGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _animals_mps() -> Dictionary:
    return Dictionary(
        name="AnimalsMps",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.5),
            Feature("dog_beagle", "dogs", beta=-0.5),
            Feature("bird_hawk", "birds", beta=0.5),
            Feature("bird_sparrow", "birds", beta=0.5),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=MPSRung1(),
    )


def _animals_hea() -> Dictionary:
    return Dictionary(
        name="AnimalsHea",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.50, alpha=0.05, gamma=0.02),
            Feature("dog_beagle", "dogs", beta=-0.48, alpha=0.04, gamma=0.03),
            Feature("bird_hawk", "birds", beta=0.50, alpha=-0.04, gamma=0.02),
            Feature("bird_sparrow", "birds", beta=0.52, alpha=-0.03, gamma=0.01),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=HEA_Rung2(depth=2),
    )


def _separation_graph(dictionary: Dictionary) -> FeatureGraph:
    prediction = triage_dictionary(dictionary)
    return build_separation_graph(
        prediction, threshold=0.0, include_within_cluster=True
    )


def _fast_kwargs() -> dict[str, object]:
    return {
        "tolerance": 0.0,
        "optimize": {"method": "grid", "max_steps": 4},
    }


# ---------------------------------------------------------------------------
# BatchRun
# ---------------------------------------------------------------------------


class TestBatchRun:
    def test_frozen(self):
        run = BatchRun(
            source="a",
            target="b",
            predicted_floor=0.1,
            predicted_gap=0.2,
            current_overlap=0.4,
            achieved_overlap=0.2,
            cancellation_efficiency=1.0,
            best_knobs={"a.phi": 0.0},
            tier_separation_after=None,
            artifact_subpath=None,
        )
        with pytest.raises(FrozenInstanceError):
            run.source = "z"  # type: ignore[misc]

    def test_efficiency_zero_when_predicted_gap_zero(self):
        # The BatchExperiment runner is responsible for emitting a
        # zero-efficiency BatchRun when predicted_gap == 0; this test
        # documents the contract by constructing the row directly.
        run = BatchRun(
            source="a",
            target="b",
            predicted_floor=0.4,
            predicted_gap=0.0,
            current_overlap=0.4,
            achieved_overlap=0.4,
            cancellation_efficiency=0.0,
            best_knobs={},
            tier_separation_after=None,
            artifact_subpath=None,
        )
        assert run.cancellation_efficiency == 0.0


# ---------------------------------------------------------------------------
# BatchResults
# ---------------------------------------------------------------------------


class TestBatchResults:
    def test_round_trip_preserves_fields(self, tmp_path: Path):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            knobs="cluster_shared",
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        path = tmp_path / "br.json"
        result.to_json(path)
        round_tripped = BatchResults.from_json(path)
        assert round_tripped == result
        # source_graph survives byte-for-byte:
        assert round_tripped.source_graph.to_json() == graph.to_json()

    def test_to_json_byte_identical_across_calls(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        a = result.to_json()
        b = result.to_json()
        assert a == b

    def test_to_json_is_deterministic_dict(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=1,
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        parsed = json.loads(result.to_json())
        assert set(parsed.keys()) == {
            "source_graph", "dictionary_name", "knobs", "created_at", "runs"
        }
        assert parsed["dictionary_name"] == d.name
        # source_graph is a nested object, not a string:
        assert isinstance(parsed["source_graph"], dict)
        assert parsed["source_graph"]["kind"] in ("sharing", "separation")

    def test_source_graph_preserved_verbatim(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        assert result.source_graph.to_json() == graph.to_json()


# ---------------------------------------------------------------------------
# BatchExperiment
# ---------------------------------------------------------------------------


class TestBatchExperiment:
    def test_pairs_match_input_graph_top_k_in_order(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=3,
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        assert len(result.runs) == min(3, len(graph.edges))
        for run, edge in zip(result.runs, graph.edges[: len(result.runs)]):
            assert run.source == edge.source
            assert run.target == edge.target
            assert run.predicted_floor == pytest.approx(edge.floor, abs=1e-5)
            assert run.predicted_gap == pytest.approx(edge.gap, abs=1e-5)

    def test_top_k_clamped_to_edge_count(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        n_edges = len(graph.edges)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=16,
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        assert len(result.runs) == n_edges

    def test_top_k_above_cap_rejected(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        with pytest.raises(ValueError, match="16"):
            BatchExperiment(
                feature_graph=graph, dictionary=d, top_k=17,
            )

    def test_top_k_below_one_rejected(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        with pytest.raises(ValueError):
            BatchExperiment(
                feature_graph=graph, dictionary=d, top_k=0,
            )

    def test_unknown_knobs_rejected(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        with pytest.raises(ValueError, match="cluster_shared"):
            BatchExperiment(
                feature_graph=graph, dictionary=d, knobs="bogus",
            )

    def test_dictionary_missing_graph_node_rejected(self):
        d = _animals_mps()
        synthetic_graph = FeatureGraph(
            kind="sharing",
            nodes=("dog_poodle", "ghost"),
            edges=(
                FeatureEdge(
                    source="dog_poodle",
                    target="ghost",
                    weight=0.7,
                    floor=0.1,
                    gap=0.4,
                    is_cross_cluster=True,
                    reason="phase_headroom",
                ),
            ),
            clusters=(("dog_poodle", "ghost"),),
        )
        with pytest.raises(ValueError, match="ghost"):
            BatchExperiment(
                feature_graph=synthetic_graph, dictionary=d,
            )

    def test_cluster_shared_knob_paths_on_mps(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            knobs="cluster_shared",
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        for run in result.runs:
            for path in run.best_knobs:
                assert path.endswith(".phi")
                # leading identifier must be a cluster name on MPS
                lead = path.split(".", 1)[0]
                assert lead in d.hierarchy

    def test_per_feature_knob_paths_on_mps(self):
        d = _animals_mps()
        # Hand-built graph guarantees a per-feature within-cluster pair
        # survives independent of the Animals fixture's exact triage
        # numbers.
        graph = FeatureGraph(
            kind="sharing",
            nodes=tuple(f.name for f in d.features),
            edges=(
                FeatureEdge(
                    source="dog_poodle",
                    target="dog_beagle",
                    weight=0.8,
                    floor=0.05,
                    gap=0.4,
                    is_cross_cluster=False,
                    reason="phase_headroom",
                ),
                FeatureEdge(
                    source="bird_hawk",
                    target="bird_sparrow",
                    weight=0.7,
                    floor=0.05,
                    gap=0.3,
                    is_cross_cluster=False,
                    reason="phase_headroom",
                ),
            ),
            clusters=(
                ("dog_poodle", "dog_beagle"),
                ("bird_hawk", "bird_sparrow"),
            ),
        )
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            knobs="per_feature",
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        feature_names = {f.name for f in d.features}
        for run in result.runs:
            assert len(run.best_knobs) == 2
            for path in run.best_knobs:
                lead = path.split(".", 1)[0]
                assert lead in feature_names

    def test_artifact_subdirs_written_when_output_dir_set(self, tmp_path: Path):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            output_dir=tmp_path,
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        for run in result.runs:
            sub = tmp_path / f"{run.source}_x_{run.target}"
            assert sub.is_dir()
            # Standard Cancellation bundle: machine + summary + trajectory.
            files = {p.name for p in sub.iterdir()}
            assert any(f.endswith(".q.orca.md") for f in files)
            assert any(f.endswith("_summary.md") for f in files)
            assert any(f.endswith("_trajectory.csv") for f in files)
            assert run.artifact_subpath == f"{run.source}_x_{run.target}"
        # Aggregated batch_results.json at the top level.
        assert (tmp_path / "batch_results.json").is_file()

    def test_hea_dictionary_produces_theta_knobs(self):
        d = _animals_hea()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            knobs="cluster_shared",
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        for run in result.runs:
            for path in run.best_knobs:
                assert ".theta[" in path
                lead = path.split(".", 1)[0]
                assert lead in d.hierarchy

    def test_run_no_output_dir_leaves_artifact_subpath_none(self):
        d = _animals_mps()
        graph = _separation_graph(d)
        result = BatchExperiment(
            feature_graph=graph,
            dictionary=d,
            top_k=2,
            cancellation_kwargs=_fast_kwargs(),
        ).run()
        for run in result.runs:
            assert run.artifact_subpath is None
