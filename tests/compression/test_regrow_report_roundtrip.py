"""`RegrowReport.to_json(...)` ↔ `from_json(...)` round-trip."""

from __future__ import annotations

from polygram.compression.regrow_report import (
    SCHEMA_VERSION,
    RegrowPlan,
    RegrowReport,
    SlotPopulation,
)


def _hand_built_report() -> RegrowReport:
    plan = RegrowPlan(
        strategy="residual_kmeans",
        n_residual_tokens=654,
        zeroed_input=(2, 5, 9, 13),
        feature_ids=tuple(range(16)),
        slots=(
            SlotPopulation(feature_id=2, cluster_size=87,
                           decoder_norm=1.0, encoder_norm=1.0),
            SlotPopulation(feature_id=5, cluster_size=134,
                           decoder_norm=1.0, encoder_norm=1.0),
            SlotPopulation(feature_id=9, cluster_size=0,
                           decoder_norm=0.0, encoder_norm=0.0),
            SlotPopulation(feature_id=13, cluster_size=433,
                           decoder_norm=1.0, encoder_norm=1.0),
        ),
    )
    return RegrowReport(
        schema_version=SCHEMA_VERSION,
        source_checkpoint="/path/to/source.safetensors",
        source_checkpoint_sha256="a" * 64,
        output_checkpoint="/path/to/regrown.safetensors",
        output_checkpoint_sha256="b" * 64,
        strategy="residual_kmeans",
        plan=plan,
        n_slots_repopulated=3,
        n_slots_left_zero=1,
        strategy_params={"seed": 0, "n_init": 4},
        provenance={
            "compression_report_source_sha256": "c" * 64,
            "compression_report_output_sha256": "d" * 64,
            "compression_report_dictionary_name": "UpstreamDict",
        },
    )


class TestRoundTrip:
    def test_string_round_trip(self):
        r = _hand_built_report()
        rt = RegrowReport.from_json(r.to_json())
        assert rt == r

    def test_path_round_trip(self, tmp_path):
        r = _hand_built_report()
        out = tmp_path / "report.json"
        r.to_json(out)
        rt = RegrowReport.from_json(out)
        assert rt == r

    def test_required_keys_present(self):
        import json
        r = _hand_built_report()
        payload = json.loads(r.to_json())
        for key in (
            "schema_version",
            "source_checkpoint",
            "source_checkpoint_sha256",
            "output_checkpoint",
            "output_checkpoint_sha256",
            "strategy",
            "n_slots_repopulated",
            "n_slots_left_zero",
            "feature_ids",
            "plan",
            "strategy_params",
            "provenance",
        ):
            assert key in payload, f"missing key {key!r}"

    def test_empty_provenance_round_trip(self):
        r = _hand_built_report()
        # Build a new instance with empty provenance — frozen dataclass
        # replace would also work, but constructing directly avoids the
        # unused-import lint and reads more clearly.
        r_empty = RegrowReport(
            schema_version=r.schema_version,
            source_checkpoint=r.source_checkpoint,
            source_checkpoint_sha256=r.source_checkpoint_sha256,
            output_checkpoint=r.output_checkpoint,
            output_checkpoint_sha256=r.output_checkpoint_sha256,
            strategy=r.strategy,
            plan=r.plan,
            n_slots_repopulated=r.n_slots_repopulated,
            n_slots_left_zero=r.n_slots_left_zero,
            strategy_params=r.strategy_params,
            provenance={},
        )
        rt = RegrowReport.from_json(r_empty.to_json())
        assert rt == r_empty
        assert rt.provenance == {}
