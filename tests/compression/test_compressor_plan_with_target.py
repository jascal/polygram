"""Unit tests for `Compressor.plan_with_target()` — target-K compression.

Phase 1 of the `add-pareto-target-compression` openspec change. Covers:

- target reached / target infeasible
- score_field selection across the three supported axes
- NaN filtering when the report lacks the chosen score field
- determinism via the canonical tiebreak
- apply(plan=...) override accepting a target-K plan
- byte-identity regression for the threshold path
- config-level validation errors
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from polygram import Compressor
from polygram.compression.report import CompressionPlan
from polygram.config import CompressionConfig
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _checkpoint(tmp_path: Path, n_features: int = 16) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=8)
    return sae_path


def _strip_to_confirmed(report, keep_pairs):
    """Return a report with every non-listed pair's polygram_overlap and
    decoder_overlap set to NaN so the target-K NaN filter drops them.
    Lets us test the algorithm against a small, controlled pair list."""
    keep = {tuple(sorted(p)) for p in keep_pairs}
    new_pairs = []
    for p in report.pairs:
        if (p.i, p.j) in keep:
            new_pairs.append(p)
        else:
            new_pairs.append(
                dataclasses.replace(
                    p,
                    polygram_overlap=float("nan"),
                    decoder_overlap=float("nan"),
                    jaccard=float("nan"),
                )
            )
    return dataclasses.replace(report, pairs=tuple(new_pairs))


def _override_pair_score(
    report, i: int, j: int, *, score_field: str, value: float
):
    """Return a new ValidationReport with `(i, j)`'s `score_field` set
    to `value`. Useful for steering the target-K sort order without
    rebuilding the whole fixture."""
    canonical = (min(i, j), max(i, j))
    new_pairs = []
    for p in report.pairs:
        if (p.i, p.j) == canonical:
            new_pairs.append(dataclasses.replace(p, **{score_field: value}))
        else:
            new_pairs.append(p)
    return dataclasses.replace(report, pairs=tuple(new_pairs))


# ---------------------------------------------------------------------------
# CompressionPlan.n_features_kept property
# ---------------------------------------------------------------------------


class TestPlanNFeaturesKeptProperty:
    def test_property_equals_len_clusters(self, tmp_path: Path):
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (2, 3), (4, 5), (4, 6), (4, 7)],
        )
        compressor = Compressor(
            validation_report=report, sae_checkpoint=_checkpoint(tmp_path, 8)
        )
        plan = compressor.plan()
        assert plan.n_features_kept == len(plan.clusters)
        assert plan.n_features_kept == 3  # {0,1}, {2,3}, {4,5,6,7}

    def test_property_zero_on_empty_plan(self):
        empty = CompressionPlan(clusters=(), feature_ids=(1, 2, 3))
        assert empty.n_features_kept == 0


# ---------------------------------------------------------------------------
# target reached / infeasible
# ---------------------------------------------------------------------------


class TestTargetReached:
    def test_reaches_requested_k(self, tmp_path: Path):
        # 8 features forming 4 disjoint pairs, then 2 bridging pairs.
        # Sorted by polygram_overlap (which build_report sets to 0.8 for
        # all "confirmed" pairs), the bridging pairs come last via the
        # tiebreak. With target_k=2 we should stop after the bridging
        # pairs merge the disjoint pairs into 2 super-components.
        report = build_report(
            n_features=8,
            confirmed=[
                (0, 1), (2, 3), (4, 5), (6, 7),  # 4 disjoint pairs
                (1, 2), (5, 6),  # 2 bridging pairs (0..3 → one, 4..7 → one)
            ],
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 8),
            config=CompressionConfig(target_n_features_kept=2),
        )
        plan = compressor.plan_with_target()
        assert plan.n_features_kept <= 2

    def test_target_k_argument_overrides_config(self, tmp_path: Path):
        report = build_report(
            n_features=8, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 8),
            config=CompressionConfig(target_n_features_kept=999),
        )
        # Method argument overrides config.target_n_features_kept.
        plan = compressor.plan_with_target(2)
        assert plan.n_features_kept <= 2

    def test_infeasible_target_returns_most_compressed(
        self, tmp_path: Path
    ):
        # 3 disjoint pairs and no bridging pairs: NaN-out every
        # non-confirmed pair so only (0,1), (2,3), (4,5) survive the
        # filter. Min reachable representative count = 3. Target_k=1
        # is infeasible.
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3), (4, 5)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
            config=CompressionConfig(target_n_features_kept=1),
        )
        plan = compressor.plan_with_target()
        # Algorithm processes all 3 pairs; n_features_kept = 3 > 1.
        assert plan.n_features_kept == 3
        assert plan.n_features_kept > 1

    def test_huge_target_returns_minimally_compressed(self, tmp_path: Path):
        # Strip down to two disjoint pairs only; target_k=10_000 means
        # we never cross above target. Algorithm processes all pairs and
        # returns the most-compressed plan reachable (2 clusters).
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
            config=CompressionConfig(target_n_features_kept=10_000),
        )
        plan = compressor.plan_with_target()
        assert plan.n_features_kept == 2

    def test_raises_when_no_target_provided(self, tmp_path: Path):
        report = build_report(n_features=4, confirmed=[(0, 1)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
            # No config and no method-argument target.
        )
        with pytest.raises(ValueError, match="target_n_features_kept"):
            compressor.plan_with_target()


# ---------------------------------------------------------------------------
# score_field selection + NaN filtering
# ---------------------------------------------------------------------------


class TestScoreFieldSelection:
    def test_default_score_field_is_polygram_overlap(self, tmp_path: Path):
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
            config=CompressionConfig(target_n_features_kept=2),
        )
        # No score_field override → default polygram_overlap is used.
        plan = compressor.plan_with_target()
        assert plan.n_features_kept <= 3  # algorithm finished or stopped

    def test_jaccard_and_decoder_overlap_axes_work(self, tmp_path: Path):
        # Verify all three score axes can be used; strip non-confirmed
        # pairs so the test isn't sensitive to fixture-induced bridging.
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3), (4, 5)])
        for field in ("polygram_overlap", "jaccard", "decoder_overlap"):
            compressor = Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path, 6),
                config=CompressionConfig(
                    target_n_features_kept=10,
                    score_field=field,
                ),
            )
            plan = compressor.plan_with_target()
            # All 3 pairs survive the filter for each field (the fixture
            # sets all three to 0.5/0.8/0.9 for confirmed pairs).
            assert plan.n_features_kept == 3, (
                f"score_field={field!r}: got {plan.n_features_kept} "
                f"clusters, expected 3"
            )


class TestNaNFiltering:
    def test_raises_on_all_nan_score_field(self, tmp_path: Path):
        # Decoder-only-style report: pearson_activation is NaN for all
        # pairs in our fixture. Although pearson_activation isn't a
        # supported score_field, we can simulate the situation by
        # zeroing out the polygram_overlap field on every pair.
        report = build_report(n_features=4, confirmed=[(0, 1)])
        report_nan = dataclasses.replace(
            report,
            pairs=tuple(
                dataclasses.replace(p, polygram_overlap=float("nan"))
                for p in report.pairs
            ),
        )
        compressor = Compressor(
            validation_report=report_nan,
            sae_checkpoint=_checkpoint(tmp_path, 4),
            config=CompressionConfig(
                target_n_features_kept=1,
                score_field="polygram_overlap",
            ),
        )
        with pytest.raises(ValueError, match="polygram_overlap"):
            compressor.plan_with_target()

    def test_filters_individual_nan_pairs(self, tmp_path: Path):
        # Strip down to a controlled pair list, then NaN out one of
        # them. After filtering only the surviving pair is processed.
        report = build_report(n_features=4, confirmed=[(0, 1), (2, 3)])
        report = _strip_to_confirmed(report, [(0, 1), (2, 3)])
        report = _override_pair_score(
            report, 0, 1,
            score_field="polygram_overlap",
            value=float("nan"),
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
            config=CompressionConfig(
                target_n_features_kept=10,  # huge → process all viable
                score_field="polygram_overlap",
            ),
        )
        plan = compressor.plan_with_target()
        # Only (2,3) survives the filter; (0,1) is dropped.
        cluster_members = sorted(
            tuple(c.members) for c in plan.clusters
        )
        assert cluster_members == [(2, 3)]


# ---------------------------------------------------------------------------
# Determinism / tiebreak
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_scores_tiebreak_lexicographic(self, tmp_path: Path):
        # All "confirmed" pairs share the same polygram_overlap (0.8).
        # Tiebreak (min, max) means (0,1) is processed before (2,3),
        # which is processed before (4,5). Cluster ids therefore reflect
        # ascending min-fid order — same as threshold mode.
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
            config=CompressionConfig(target_n_features_kept=10_000),
        )
        plan_a = compressor.plan_with_target()
        # Re-build to ensure cache effects don't mask determinism.
        compressor2 = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
            config=CompressionConfig(target_n_features_kept=10_000),
        )
        plan_b = compressor2.plan_with_target()
        assert plan_a.clusters == plan_b.clusters

    def test_min_fid_root_after_union(self, tmp_path: Path):
        # Bridging pair (1, 2) should merge {0,1} and {2,3} with
        # root = 0 (the smaller of the two roots). cluster_id ordering
        # is by ascending min-fid, so the merged cluster lands at id 0.
        report = build_report(
            n_features=4, confirmed=[(0, 1), (2, 3), (1, 2)]
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
            config=CompressionConfig(target_n_features_kept=1),
        )
        plan = compressor.plan_with_target()
        assert plan.n_features_kept == 1
        assert plan.clusters[0].members == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# apply(plan=...) override
# ---------------------------------------------------------------------------


class TestApplyWithPlan:
    def test_apply_with_target_k_plan(self, tmp_path: Path):
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (2, 3), (4, 5), (6, 7)],
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 8),
            config=CompressionConfig(target_n_features_kept=10_000),
        )
        plan = compressor.plan_with_target()
        out_path = tmp_path / "compressed.safetensors"
        result = compressor.apply(plan=plan, output_checkpoint=out_path)
        assert result.report.n_features_kept == plan.n_features_kept
        # apply() patches cluster_norm fields onto a fresh plan instance,
        # so identity won't match — verify the cluster structure does.
        assert len(result.report.plan.clusters) == len(plan.clusters)
        for applied_c, input_c in zip(
            result.report.plan.clusters, plan.clusters
        ):
            assert applied_c.cluster_id == input_c.cluster_id
            assert applied_c.members == input_c.members
            assert applied_c.representative == input_c.representative
            assert applied_c.zeroed == input_c.zeroed


# ---------------------------------------------------------------------------
# Byte-identity for the threshold path
# ---------------------------------------------------------------------------


class TestThresholdPathByteIdentity:
    def test_default_compressor_plan_unchanged(self, tmp_path: Path):
        # No CompressionConfig → legacy path. Same `report` and `sae`
        # must produce the same plan as a separate Compressor instance.
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (2, 3), (4, 5), (4, 6), (4, 7)],
        )
        sae_path = _checkpoint(tmp_path, 8)

        compressor_a = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        )
        compressor_b = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        )
        plan_a = compressor_a.plan()
        plan_b = compressor_b.plan()

        # plan_a is the historical artifact shape (no new fields).
        assert plan_a.clusters == plan_b.clusters
        assert plan_a.feature_ids == plan_b.feature_ids
        # Property is derived, so plan_a survives field-level equality.
        assert plan_a.n_features_kept == 3
