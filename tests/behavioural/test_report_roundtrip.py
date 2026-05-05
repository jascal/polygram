"""`ValidationReport.from_json(to_json(r)) == r` round-trip.

Builds a hand-rolled fixture report (no model, no SAE, no torch) that
exercises NaN preservation across pair fields, summary fields, and
empty buckets.
"""

from __future__ import annotations

import math
from pathlib import Path

from polygram import (
    BucketStats,
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)


def _make_fixture_report() -> ValidationReport:
    pairs = (
        CandidatePair(
            i=12999,
            j=19398,
            polygram_overlap=0.9939,
            decoder_overlap=0.3639,
            jaccard=0.5072,
            pearson_activation=0.9893,
            kl_ablate_i=0.2364,
            kl_ablate_j=0.1293,
            kl_ratio_paired=1.8296,
            kl_log_ratio_abs=0.6041,
            n_fires_i=311,
            n_fires_j=215,
            n_both_fire=177,
            n_either_fire=349,
            gate_pass=True,
        ),
        CandidatePair(
            i=68,
            j=2287,
            polygram_overlap=0.05,
            decoder_overlap=0.02,
            jaccard=float("nan"),       # non-firing pair → NaN
            pearson_activation=float("nan"),
            kl_ablate_i=float("nan"),
            kl_ablate_j=float("nan"),
            kl_ratio_paired=float("nan"),
            kl_log_ratio_abs=float("nan"),
            n_fires_i=0,
            n_fires_j=2,
            n_both_fire=0,
            n_either_fire=2,
            gate_pass=False,
        ),
    )
    summary = ValidationSummary(
        spearman_polygram_jaccard=0.6371,
        spearman_decoder_jaccard=-0.0542,
        spearman_polygram_log_kl_abs=float("nan"),
        pearson_polygram_jaccard=0.6943,
        pearson_decoder_jaccard=-0.0753,
        buckets={
            "low_overlap": BucketStats(
                polygram_range="\u2264 0.4",
                n_pairs=0,
                jaccard_mean=float("nan"),
                jaccard_ci_95=(float("nan"), float("nan")),
            ),
            "mid_overlap": BucketStats(
                polygram_range="(0.4, 0.7)",
                n_pairs=16,
                jaccard_mean=0.1445,
                jaccard_ci_95=(0.0963, 0.1924),
            ),
            "high_overlap": BucketStats(
                polygram_range="\u2265 0.7",
                n_pairs=12,
                jaccard_mean=0.6209,
                jaccard_ci_95=(0.4267, 0.8228),
            ),
        },
        outcome="high_spearman_loop_unblocked",
    )
    return ValidationReport(
        schema_version=1,
        dictionary_name="ScaleupBlocks10",
        model_name="gpt2",
        layer=10,
        n_prompts=12,
        n_tokens=654,
        polygram_overlap_threshold=0.7,
        jaccard_threshold=0.30,
        min_firing_rate=0.01,
        min_both_fire=5,
        feature_ids=(12999, 19398, 68, 2287),
        pairs=pairs,
        summary=summary,
        confirmed=((12999, 19398),),
    )


class TestRoundTrip:
    def test_from_json_to_json_is_identity_in_memory(self):
        r = _make_fixture_report()
        json_text = r.to_json()
        r2 = ValidationReport.from_json(json_text)
        assert r2 == r

    def test_from_json_to_json_through_disk(self, tmp_path: Path):
        r = _make_fixture_report()
        path = tmp_path / "report.json"
        r.to_json(path)
        assert path.is_file()
        r2 = ValidationReport.from_json(path)
        assert r2 == r

    def test_pairs_are_sorted_in_serialized_form(self):
        # Hand-built with j-then-i ordering; serializer should sort.
        r = _make_fixture_report()
        # Re-build with reversed pair order.
        reversed_pairs = tuple(reversed(r.pairs))
        r2 = ValidationReport(
            schema_version=r.schema_version,
            dictionary_name=r.dictionary_name,
            model_name=r.model_name,
            layer=r.layer,
            n_prompts=r.n_prompts,
            n_tokens=r.n_tokens,
            polygram_overlap_threshold=r.polygram_overlap_threshold,
            jaccard_threshold=r.jaccard_threshold,
            min_firing_rate=r.min_firing_rate,
            min_both_fire=r.min_both_fire,
            feature_ids=r.feature_ids,
            pairs=reversed_pairs,
            summary=r.summary,
            confirmed=r.confirmed,
        )
        # Both serializations must be identical (deterministic order).
        assert r.to_json() == r2.to_json()

    def test_csv_emission_has_15_columns(self, tmp_path: Path):
        r = _make_fixture_report()
        csv_path = r.to_csv(tmp_path / "pairs.csv")
        rows = csv_path.read_text().strip().splitlines()
        header = rows[0].split(",")
        assert len(header) == 15
        assert header[-1] == "gate_pass"
        # First 14 columns match the §4.4 scaleup CSV exactly.
        assert header == [
            "i",
            "j",
            "polygram_overlap",
            "decoder_overlap",
            "jaccard",
            "pearson_activation",
            "n_fires_i",
            "n_fires_j",
            "n_both_fire",
            "n_either_fire",
            "kl_ablate_i_on_both_fire",
            "kl_ablate_j_on_both_fire",
            "kl_ratio_i_over_j",
            "kl_log_ratio_abs",
            "gate_pass",
        ]

    def test_csv_gate_pass_serializes_as_true_false(self, tmp_path: Path):
        r = _make_fixture_report()
        csv_path = r.to_csv(tmp_path / "pairs.csv")
        rows = csv_path.read_text().strip().splitlines()
        # header + 2 data rows.
        assert len(rows) == 3
        gate_columns = [row.split(",")[-1] for row in rows[1:]]
        assert set(gate_columns) <= {"true", "false"}

    def test_nan_floats_serialize_as_null(self):
        r = _make_fixture_report()
        text = r.to_json()
        # NaN should not appear as `NaN` (which is invalid JSON) — it
        # must round-trip via `null`.
        assert "NaN" not in text
        assert "null" in text  # the second pair's NaN fields

    def test_nan_compares_equal_for_round_trip(self):
        # NaN-aware equality: round-tripping a NaN pair field must
        # produce a report that compares equal to the original.
        r = _make_fixture_report()
        r2 = ValidationReport.from_json(r.to_json())
        # Spot-check the NaN field on the non-firing pair.
        original_nan_pair = r.pairs[1]
        roundtrip_nan_pair = next(
            p for p in r2.pairs if (p.i, p.j) == (original_nan_pair.i, original_nan_pair.j)
        )
        assert math.isnan(original_nan_pair.jaccard)
        assert math.isnan(roundtrip_nan_pair.jaccard)
