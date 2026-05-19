"""`EpochReport.to_json(...)` ↔ `from_json(...)` round-trip."""

from __future__ import annotations

from polygram.compression.epoch_report import (
    SCHEMA_VERSION,
    EpochIteration,
    EpochReport,
    Panel,
)


def _hand_built_report() -> EpochReport:
    iter0_panels = (
        Panel(panel_id=0, anchor=12999, feature_ids=(12999, 19398, 4192,
                                                     23625, 8371, 2287, 68, 13737),
              cosines_to_anchor=(0.95, 0.92, 0.89, 0.55, 0.50, 0.48, 0.41)),
        Panel(panel_id=1, anchor=4192, feature_ids=(4192, 5000, 5001, 5002,
                                                    5003, 5004, 5005, 5006),
              cosines_to_anchor=(0.93, 0.91, 0.88, 0.85, 0.83, 0.81, 0.79)),
    )
    iter1_panels = (
        Panel(panel_id=2, anchor=12999, feature_ids=(12999, 5000, 5001, 5002,
                                                     5003, 5004, 5005, 5006),
              cosines_to_anchor=(0.96, 0.94, 0.91, 0.88, 0.86, 0.84, 0.82)),
    )
    return EpochReport(
        schema_version=SCHEMA_VERSION,
        source_checkpoint="/path/to/source.safetensors",
        source_checkpoint_sha256="a" * 64,
        output_checkpoint="/path/to/epoch.safetensors",
        output_checkpoint_sha256="b" * 64,
        convergence_reason="stable_clusters",
        n_features_zeroed_total=11,
        n_features_input=16,
        redundancy_ratio=11 / 16,
        n_panels_total=3,
        coverage_achieved=0.954,
        wall_seconds=6420.7,
        iterations=(
            EpochIteration(
                iteration=0, panels=iter0_panels,
                validation_report_paths=("a.json", "b.json"),
                confirmed_pair_count=14, clusters_compressed=2,
                features_zeroed_this_iteration=(19398, 4192, 23625, 2287,
                                                 8371, 13737),
                cross_entropy_delta=0.001234,
                convergence_state="continuing",
            ),
            EpochIteration(
                iteration=1, panels=iter1_panels,
                validation_report_paths=("c.json",),
                confirmed_pair_count=10, clusters_compressed=1,
                features_zeroed_this_iteration=(5000, 5001, 5002, 5003, 5004),
                cross_entropy_delta=0.001789,
                convergence_state="stable_clusters",
            ),
        ),
    )


class TestRoundTrip:
    def test_string_round_trip(self):
        r = _hand_built_report()
        rt = EpochReport.from_json(r.to_json())
        assert rt == r

    def test_path_round_trip(self, tmp_path):
        r = _hand_built_report()
        out = tmp_path / "report.json"
        r.to_json(out)
        rt = EpochReport.from_json(out)
        assert rt == r

    def test_required_keys_present(self):
        import json
        r = _hand_built_report()
        payload = json.loads(r.to_json())
        for key in (
            "schema_version", "source_checkpoint",
            "source_checkpoint_sha256", "output_checkpoint",
            "output_checkpoint_sha256", "convergence_reason",
            "n_features_zeroed_total", "n_features_input",
            "redundancy_ratio", "n_panels_total",
            "coverage_achieved", "wall_seconds", "iterations",
        ):
            assert key in payload

    def test_iteration_panels_round_trip(self):
        r = _hand_built_report()
        rt = EpochReport.from_json(r.to_json())
        assert rt.iterations[0].panels[0].feature_ids == r.iterations[0].panels[0].feature_ids
        assert rt.iterations[0].panels[0].anchor == r.iterations[0].panels[0].anchor


class TestRedundancyRatio:
    def test_ratio_matches_division(self):
        """Task 3.1 — `redundancy_ratio == n_features_zeroed_total /
        n_features_input` on a fresh report."""
        r = _hand_built_report()
        assert abs(
            r.redundancy_ratio
            - r.n_features_zeroed_total / r.n_features_input
        ) < 1e-12

    def test_ratio_round_trips_through_json(self):
        """Task 3.2 — to_json → from_json preserves the new field."""
        r = _hand_built_report()
        rt = EpochReport.from_json(r.to_json())
        assert rt.redundancy_ratio == r.redundancy_ratio
        assert rt.n_features_input == r.n_features_input

    def test_legacy_payload_loads_without_new_fields(self):
        """Task 3.3 — legacy v1 payload (no redundancy_ratio,
        no n_features_input) loads without crashing. Ratio is
        either computed from any available divisor or set to the
        0.0 sentinel (NOT nan)."""
        import json
        import math
        legacy_payload = {
            "schema_version": 1,
            "source_checkpoint": "/path/source.safetensors",
            "source_checkpoint_sha256": "a" * 64,
            "output_checkpoint": "/path/out.safetensors",
            "output_checkpoint_sha256": "b" * 64,
            "convergence_reason": "stable_clusters",
            "n_features_zeroed_total": 5,
            "n_panels_total": 1,
            "coverage_achieved": 0.5,
            "wall_seconds": 100.0,
            "iterations": [],
        }
        rt = EpochReport.from_json(json.dumps(legacy_payload))
        assert rt.n_features_input == 0
        # Divisor missing → 0.0 sentinel, NOT nan.
        assert rt.redundancy_ratio == 0.0
        assert not math.isnan(rt.redundancy_ratio)

    def test_zero_n_features_input_sentinel_is_zero_not_nan(self):
        """Defensive: when n_features_input=0 (degenerate),
        redundancy_ratio is 0.0, not nan. Downstream consumers
        consistently misread nan as a real measurement."""
        import math
        r = EpochReport(
            schema_version=SCHEMA_VERSION,
            source_checkpoint="/s",
            source_checkpoint_sha256="0" * 64,
            output_checkpoint="/o",
            output_checkpoint_sha256="1" * 64,
            convergence_reason="max_iterations",
            n_features_zeroed_total=0,
            n_features_input=0,
            redundancy_ratio=0.0,
            n_panels_total=0,
            coverage_achieved=0.0,
            wall_seconds=0.0,
            iterations=(),
        )
        assert r.redundancy_ratio == 0.0
        assert not math.isnan(r.redundancy_ratio)
