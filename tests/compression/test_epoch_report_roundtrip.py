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
            "n_features_zeroed_total", "n_panels_total",
            "coverage_achieved", "wall_seconds", "iterations",
        ):
            assert key in payload

    def test_iteration_panels_round_trip(self):
        r = _hand_built_report()
        rt = EpochReport.from_json(r.to_json())
        assert rt.iterations[0].panels[0].feature_ids == r.iterations[0].panels[0].feature_ids
        assert rt.iterations[0].panels[0].anchor == r.iterations[0].panels[0].anchor
