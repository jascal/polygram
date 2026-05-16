"""Unit tests for `Compressor.plan_pareto()` + `ParetoReport`.

Phase 2 of the `add-pareto-target-compression` openspec change. Covers:

- Nested plans across multiple K (every cluster at higher K is a
  subset of some cluster at lower K).
- Single-target equivalence between `plan_pareto([K])` and
  `plan_with_target(K)`.
- Sort-once invariant: `_sort_pairs_by_score` is invoked exactly once
  per `plan_pareto` call, regardless of `len(targets)`.
- `to_json` / `from_json` round-trip via `_cluster_to_dict` /
  `_cluster_from_dict`.
- Per-K `reached_target` correctly reflects whether the resulting
  plan satisfies `plan.n_features_kept <= target_k`.
- Empty targets raises `ValueError`.
- Targets are deduplicated and returned in descending order.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from polygram import Compressor, ParetoOutcome, ParetoReport
from polygram.config import CompressionConfig
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _checkpoint(tmp_path: Path, n_features: int = 16) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=8)
    return sae_path


def _strip_to_confirmed(report, keep_pairs):
    """Mirror of the helper in test_compressor_plan_with_target.py:
    NaN out every non-listed pair so the target-K NaN filter drops
    them and the algorithm sees only the curated pair list."""
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


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestPlanParetoBasic:
    def test_returns_pareto_report_per_target(self, tmp_path: Path):
        report = build_report(
            n_features=8,
            confirmed=[
                (0, 1), (2, 3), (4, 5), (6, 7),
                (1, 2), (5, 6),
            ],
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 8),
            config=CompressionConfig(score_field="polygram_overlap"),
        )
        pr = compressor.plan_pareto([2, 1])
        assert isinstance(pr, ParetoReport)
        assert pr.targets == (2, 1)
        assert len(pr.outcomes) == 2
        for outcome in pr.outcomes:
            assert isinstance(outcome, ParetoOutcome)

    def test_single_target_matches_plan_with_target(self, tmp_path: Path):
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (2, 3), (4, 5), (6, 7)],
        )
        sae = _checkpoint(tmp_path, 8)
        compressor_pareto = Compressor(
            validation_report=report,
            sae_checkpoint=sae,
            config=CompressionConfig(target_n_features_kept=4),
        )
        compressor_single = Compressor(
            validation_report=report,
            sae_checkpoint=sae,
            config=CompressionConfig(target_n_features_kept=4),
        )
        pr = compressor_pareto.plan_pareto([4])
        plan = compressor_single.plan_with_target()
        assert pr.outcomes[0].plan.clusters == plan.clusters
        assert pr.outcomes[0].plan.feature_ids == plan.feature_ids

    def test_dedupe_and_sort_descending(self, tmp_path: Path):
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3), (4, 5)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
        )
        pr = compressor.plan_pareto([1, 3, 2, 1, 2])
        assert pr.targets == (3, 2, 1)
        assert len(pr.outcomes) == 3
        # outcomes parallel targets
        assert [o.target_k for o in pr.outcomes] == [3, 2, 1]

    def test_empty_targets_raises(self, tmp_path: Path):
        report = build_report(n_features=4, confirmed=[(0, 1)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
        )
        with pytest.raises(ValueError, match=r"empty"):
            compressor.plan_pareto([])

    def test_none_targets_raises(self, tmp_path: Path):
        report = build_report(n_features=4, confirmed=[(0, 1)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
        )
        with pytest.raises(ValueError, match=r"None"):
            compressor.plan_pareto(None)  # type: ignore[arg-type]

    def test_non_positive_target_raises(self, tmp_path: Path):
        report = build_report(n_features=4, confirmed=[(0, 1)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
        )
        with pytest.raises(ValueError, match=r"integer >= 1"):
            compressor.plan_pareto([0, 2])
        with pytest.raises(ValueError, match=r"integer >= 1"):
            compressor.plan_pareto([-1])


# ---------------------------------------------------------------------------
# Nested-plans invariant + reached_target
# ---------------------------------------------------------------------------


class TestNestedness:
    def test_lower_k_clusters_contain_higher_k_clusters(
        self, tmp_path: Path
    ):
        # 6 features, with bridging pairs so the trajectory has a peak
        # then shrinks. Sorted by polygram_overlap (all confirmed
        # pairs share 0.8), the algorithm processes (0,1), (1,2),
        # (2,3), (3,4), (4,5) in min-fid tiebreak order, building one
        # giant chain that merges everything.
        report = build_report(
            n_features=6,
            confirmed=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        )
        report = _strip_to_confirmed(
            report,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
        )
        pr = compressor.plan_pareto([3, 2, 1])

        # Feature counts weakly decrease (larger K = at least as many
        # clusters as smaller K).
        counts = [o.plan.n_features_kept for o in pr.outcomes]
        assert counts == sorted(counts, reverse=True), counts

        # Nestedness: every cluster at higher K is a subset of some
        # cluster at lower K.
        for higher, lower in zip(pr.outcomes, pr.outcomes[1:]):
            higher_members = [
                frozenset(c.members) for c in higher.plan.clusters
            ]
            lower_members = [
                frozenset(c.members) for c in lower.plan.clusters
            ]
            for hm in higher_members:
                assert any(hm <= lm for lm in lower_members), (
                    f"K={higher.target_k} cluster {hm} is not a subset of "
                    f"any K={lower.target_k} cluster {lower_members}"
                )

    def test_reached_target_per_outcome(self, tmp_path: Path):
        # 3 disjoint pairs, no bridges. Trajectory peaks at
        # n_clusters == 3 (one compound cluster per pair) and never
        # exceeds that. K=10 > peak — reached_target is False because
        # the walk never had to merge down to K. K=2 < terminal cluster
        # count — also not reached.
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3), (4, 5)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
        )
        pr = compressor.plan_pareto([10, 2])
        assert pr.outcomes[0].target_k == 10
        assert pr.outcomes[0].reached_target is False
        assert pr.outcomes[0].plan.n_features_kept == 3
        assert pr.outcomes[1].target_k == 2
        assert pr.outcomes[1].reached_target is False
        assert pr.outcomes[1].plan.n_features_kept == 3

    def test_target_above_peak_returns_peak_state_plan(
        self, tmp_path: Path
    ):
        # 6 features. Confirmed pair list chosen so the (min, max)
        # tiebreak order processes pairs (0,1), (0,4), (2,3), (2,5),
        # (4,5) in sequence. n_clusters trajectory: 1, 1, 2, 2, 1.
        # The peak (2) is reached after (2,3) — at which point parent
        # describes two components {0,1,4} and {2,3} (feature 5 has
        # not appeared yet). The bridging pair (4,5) at the end then
        # collapses everything into a single component {0..5}.
        #
        # For K=10 (above the observed peak of 2), the old behaviour
        # silently returned the final all-merged state (1 cluster);
        # the fix returns the peak snapshot (2 clusters) and flags
        # reached_target=False so callers can distinguish this from a
        # true greedy stop.
        report = build_report(
            n_features=6,
            confirmed=[(0, 1), (0, 4), (2, 3), (2, 5), (4, 5)],
        )
        report = _strip_to_confirmed(
            report, [(0, 1), (0, 4), (2, 3), (2, 5), (4, 5)]
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
        )
        pr = compressor.plan_pareto([10])
        outcome = pr.outcomes[0]
        assert outcome.target_k == 10
        assert outcome.reached_target is False
        assert outcome.plan.n_features_kept == 2, (
            "expected peak-state snapshot (2 clusters); got "
            f"{outcome.plan.n_features_kept} (regressed to silent "
            "all-merged final state)"
        )
        cluster_members = sorted(
            tuple(sorted(c.members)) for c in outcome.plan.clusters
        )
        assert cluster_members == [(0, 1, 4), (2, 3)], (
            f"expected pre-bridge clusters; got {cluster_members}"
        )

    def test_single_target_above_peak_matches_pareto(
        self, tmp_path: Path
    ):
        # Same fixture as test_target_above_peak_returns_peak_state_plan.
        # Verifies the plan_pareto([K]) == plan_with_target(K) invariant
        # is preserved through the peak-fallback path: both must
        # return the peak-state snapshot, not the final-state plan.
        report = build_report(
            n_features=6,
            confirmed=[(0, 1), (0, 4), (2, 3), (2, 5), (4, 5)],
        )
        report = _strip_to_confirmed(
            report, [(0, 1), (0, 4), (2, 3), (2, 5), (4, 5)]
        )
        sae = _checkpoint(tmp_path, 6)
        compressor_pareto = Compressor(
            validation_report=report, sae_checkpoint=sae,
        )
        compressor_single = Compressor(
            validation_report=report,
            sae_checkpoint=sae,
            config=CompressionConfig(target_n_features_kept=10),
        )
        pr = compressor_pareto.plan_pareto([10])
        plan_single = compressor_single.plan_with_target()
        assert pr.outcomes[0].plan.clusters == plan_single.clusters


# ---------------------------------------------------------------------------
# Sort-once invariant
# ---------------------------------------------------------------------------


class TestSortOnce:
    def test_sort_pairs_by_score_invoked_exactly_once(
        self, tmp_path: Path
    ):
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (2, 3), (4, 5), (6, 7)],
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 8),
        )

        with patch.object(
            Compressor,
            "_sort_pairs_by_score",
            wraps=Compressor._sort_pairs_by_score,
        ) as spy:
            compressor.plan_pareto([4, 3, 2, 1])
        assert spy.call_count == 1, (
            f"_sort_pairs_by_score was called {spy.call_count} times; "
            "plan_pareto must share the sort across all K"
        )


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestParetoReportJSON:
    def test_round_trip(self, tmp_path: Path):
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3), (4, 5)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3), (4, 5)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
        )
        pr = compressor.plan_pareto([10, 5, 2])
        text = pr.to_json()
        pr2 = ParetoReport.from_json(text)
        assert pr == pr2

    def test_round_trip_via_file(self, tmp_path: Path):
        report = build_report(
            n_features=6, confirmed=[(0, 1), (2, 3)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 6),
        )
        pr = compressor.plan_pareto([4, 2])

        out = tmp_path / "pareto.json"
        pr.to_json(out)
        pr2 = ParetoReport.from_json(out)
        assert pr == pr2

    def test_from_json_missing_key_raises(self):
        with pytest.raises(ValueError, match=r"missing required key"):
            ParetoReport.from_json('{"schema_version": 1}')

    def test_from_json_rejects_non_string_non_path(self):
        with pytest.raises(TypeError):
            ParetoReport.from_json(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Score field plumbing
# ---------------------------------------------------------------------------


class TestScoreField:
    def test_score_field_recorded_in_report(self, tmp_path: Path):
        report = build_report(
            n_features=4, confirmed=[(0, 1), (2, 3)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
            config=CompressionConfig(score_field="jaccard"),
        )
        pr = compressor.plan_pareto([10])
        assert pr.score_field == "jaccard"

    def test_default_score_field_is_polygram_overlap(self, tmp_path: Path):
        report = build_report(
            n_features=4, confirmed=[(0, 1), (2, 3)]
        )
        report = _strip_to_confirmed(report, [(0, 1), (2, 3)])
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path, 4),
            # No config → default polygram_overlap
        )
        pr = compressor.plan_pareto([10])
        assert pr.score_field == "polygram_overlap"
