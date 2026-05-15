"""Tests for ``rep_selection="kl_attribution"`` — pure behavioural-ablation
rep_selection mode added in polygram 0.5.0 (``add-kl-attribution-rep-selection``).

Covers: highest-kl wins, n_fires tiebreak, feature_id tiebreak, mean
aggregation robustness, per-feature NaN fallback to geometric proxy,
all-NaN cluster raises ValueError, CompressionConfig surface,
end-to-end smoke.

Byte-identity for the default (`scale_aware`) and `n_fires` paths is
covered implicitly by the existing test suite continuing to pass —
``_score_kl_attribution`` is reachable only when
``rep_selection == "kl_attribution"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from safetensors.numpy import load_file

from polygram import CompressionConfig, Compressor
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _setup_sae(tmp_path: Path, *, n_features: int = 8, d_model: int = 8) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=d_model)
    return sae_path


def _plan_clusters(compressor: Compressor) -> dict[int, int]:
    """Return {cluster_id: representative_feature_id} for the plan."""
    plan = compressor.plan()
    return {c.cluster_id: c.representative for c in plan.clusters}


# ---------------------------------------------------------------------------
# Per-feature scoring
# ---------------------------------------------------------------------------


class TestScoringAndTiebreaks:
    def test_picks_highest_mean_kl_ablate(self, tmp_path: Path):
        """A 3-feature cluster with kl_ablate [0.1, 0.5, 0.2] picks feature 1."""
        sae_path = _setup_sae(tmp_path)
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (0, 2), (1, 2)],
            kl_ablate={0: 0.1, 1: 0.5, 2: 0.2},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        clusters = _plan_clusters(compressor)
        assert len(clusters) == 1
        assert clusters[0] == 1  # feature 1 has the highest mean kl_ablate

    def test_tiebreak_on_n_fires_higher_wins(self, tmp_path: Path):
        """Two features tied on kl_ablate; higher n_fires_total wins."""
        sae_path = _setup_sae(tmp_path)
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (0, 2), (1, 2)],
            kl_ablate={0: 0.5, 1: 0.5, 2: 0.1},
            n_fires={0: 50, 1: 100, 2: 10},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        clusters = _plan_clusters(compressor)
        assert clusters[0] == 1  # higher n_fires breaks the kl_ablate tie

    def test_tiebreak_on_feature_id_lower_wins(self, tmp_path: Path):
        """Two features tied on kl_ablate AND n_fires; lower feature_id wins."""
        sae_path = _setup_sae(tmp_path)
        report = build_report(
            n_features=8,
            confirmed=[(2, 3), (2, 4), (3, 4)],
            kl_ablate={2: 0.5, 3: 0.5, 4: 0.1},
            n_fires={2: 100, 3: 100, 4: 10},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        clusters = _plan_clusters(compressor)
        assert clusters[0] == 2  # lower fid wins the secondary tie

    def test_mean_aggregation_smooths_noise(self, tmp_path: Path):
        """A feature appearing in multiple pairs with slightly noisy kl_ablate
        values is scored as the mean of those values, not max or min.

        Construct: feature 1 has kl_ablate=0.31 (close to feature 2's 0.30
        single measurement). Since 1 appears in 3 pairs (with features 0, 2, 3)
        and the per-feature kl_ablate column carries the SAME value across all
        pair occurrences in the fixture, the mean is exactly that value. The
        test pins that no aggregation choice (sum vs mean) bias is introduced
        by the implementation when measurements are uniform.
        """
        sae_path = _setup_sae(tmp_path)
        # All pairs touch feature 1 → confirmed cluster {0, 1, 2, 3}.
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (1, 2), (1, 3), (0, 2), (0, 3), (2, 3)],
            kl_ablate={0: 0.10, 1: 0.31, 2: 0.30, 3: 0.05},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        clusters = _plan_clusters(compressor)
        # Feature 1 (mean=0.31) beats feature 2 (mean=0.30) — mean aggregation
        # would not flip this even though feature 1 appears in 3 pairs vs
        # feature 2's 3 pairs (sum-based aggregation would also pick 1, but
        # only because the fixture has equal pair counts; the value of the
        # test is the deterministic ordering on the mean).
        assert clusters[0] == 1


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_per_feature_nan_falls_back_to_geometric_proxy(self, tmp_path: Path):
        """A feature with NaN kl_ablate still competes — scored via the
        geometric proxy normalised to [0, 1] across the NaN-only subset.

        Fixture: 3-feature cluster {0, 1, 2}. Features 0 and 2 have finite
        kl_ablate; feature 1 is NaN. The cluster has only one NaN feature,
        so its [0, 1] subset normalisation maps it to score=0 (single
        element). Meanwhile features 0 and 2 normalise within their own
        subset — the highest finite-kl feature scores 1.0. So feature 1
        loses, and the highest-kl finite feature wins. This is the
        "subset-isolation" semantic: with only one NaN feature, it cannot
        out-compete the finite features' top.
        """
        sae_path = _setup_sae(tmp_path)
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (0, 2), (1, 2)],
            kl_ablate={0: 0.1, 2: 0.5},  # feature 1 missing → NaN
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        clusters = _plan_clusters(compressor)
        # Feature 2 has the highest finite kl_ablate; feature 1 is NaN and
        # gets normalised to 0 within its 1-element subset. Feature 2 wins.
        assert clusters[0] == 2

    def test_per_feature_nan_competes_when_multiple_nan(self, tmp_path: Path):
        """With multiple NaN features, the NaN subset normalises non-trivially
        and the geometric proxy can produce a winner.

        Fixture: 4-feature cluster. Features 0 and 1 are NaN; features 2 and 3
        are finite. Feature 1 has the higher n_fires (geometric proxy
        log_freq), and the W_dec norms are constructed (via the synth_sae
        fixture's seed) to be roughly similar, so log_freq dominates the
        proxy. Feature 1 wins its NaN-subset normalisation. Feature 3 has
        the highest finite kl_ablate. Both have score=1.0 after subset
        normalisation; tiebreak prefers higher n_fires (feature 1 has 1000
        vs feature 3 has 30).
        """
        sae_path = _setup_sae(tmp_path)
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            kl_ablate={2: 0.1, 3: 0.9},  # 0 and 1 NaN
            n_fires={0: 100, 1: 1000, 2: 20, 3: 30},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        clusters = _plan_clusters(compressor)
        # Feature 3 (finite, kl=0.9) and feature 1 (NaN, top of geo_proxy)
        # both score 1.0 after subset normalisation. Tiebreak on n_fires
        # picks feature 1 (1000 > 30). Confirms the fallback semantic is
        # not silently dominated by the finite path.
        assert clusters[0] == 1

    def test_all_nan_cluster_raises(self, tmp_path: Path):
        """When every cluster member has NaN kl_ablate (e.g. geometry-only
        confirmer), Compressor raises with an actionable message.
        """
        sae_path = _setup_sae(tmp_path)
        # No kl_ablate map → every feature gets NaN by default.
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (0, 2), (1, 2)],
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        with pytest.raises(ValueError) as exc:
            compressor.plan()
        msg = str(exc.value)
        assert "kl_attribution" in msg
        assert "behavioural confirmation" in msg
        assert "DecoderGeometryConfirmer" in msg
        assert "scale_aware" in msg
        assert "n_fires" in msg


# ---------------------------------------------------------------------------
# CompressionConfig surface
# ---------------------------------------------------------------------------


class TestCompressionConfigSurface:
    def test_accepts_kl_attribution(self):
        cfg = CompressionConfig(rep_selection="kl_attribution")
        assert cfg.rep_selection == "kl_attribution"

    def test_rejects_unknown_rep_selection(self):
        with pytest.raises(ValueError) as exc:
            CompressionConfig(rep_selection="bogus")
        msg = str(exc.value)
        # Error message names all three supported values.
        assert "n_fires" in msg
        assert "scale_aware" in msg
        assert "kl_attribution" in msg

    def test_default_unchanged(self):
        assert CompressionConfig().rep_selection == "scale_aware"

    def test_round_trip_via_dict(self):
        cfg = CompressionConfig(rep_selection="kl_attribution")
        rt = CompressionConfig.from_dict(cfg.to_dict())
        assert rt == cfg


# ---------------------------------------------------------------------------
# Compressor surface
# ---------------------------------------------------------------------------


class TestCompressorSurface:
    def test_compressor_accepts_kl_attribution_via_kwarg(self, tmp_path: Path):
        sae_path = _setup_sae(tmp_path)
        report = build_report(
            n_features=8,
            confirmed=[(0, 1)],
            kl_ablate={0: 0.1, 1: 0.5},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        )
        assert compressor.rep_selection == "kl_attribution"

    def test_compressor_accepts_kl_attribution_via_config(self, tmp_path: Path):
        sae_path = _setup_sae(tmp_path)
        report = build_report(
            n_features=8,
            confirmed=[(0, 1)],
            kl_ablate={0: 0.1, 1: 0.5},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            config=CompressionConfig(rep_selection="kl_attribution"),
        )
        assert compressor.rep_selection == "kl_attribution"

    def test_compressor_rejects_unknown_rep_selection(self, tmp_path: Path):
        sae_path = _setup_sae(tmp_path)
        report = build_report(n_features=8, confirmed=[(0, 1)])
        with pytest.raises(ValueError):
            Compressor(
                validation_report=report,
                sae_checkpoint=sae_path,
                rep_selection="bogus",
            )


# ---------------------------------------------------------------------------
# End-to-end smoke + plan_with_target / plan_pareto integration
# ---------------------------------------------------------------------------


class TestPlanIntegration:
    def test_plan_with_target_uses_kl_attribution(self, tmp_path: Path):
        """kl_attribution flows transparently through plan_with_target.

        Confirms the dispatch reaches `_score_kl_attribution` under the
        target-K planning path (not just default `plan()`). We assign
        every feature a known kl_ablate so the greedy reducer's eventual
        cluster structure is fully scored — regardless of which features
        the reducer ends up unioning at the chosen target K, each cluster's
        rep should be the in-cluster member with the highest mean kl_ablate.
        """
        sae_path = _setup_sae(tmp_path)
        # All 8 features get a known kl_ablate value, increasing with fid.
        # Whatever cluster shape plan_with_target produces, the highest-fid
        # member in each cluster has the highest kl_ablate and should win.
        kls = {fid: 0.1 + 0.1 * fid for fid in range(8)}
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (2, 3), (4, 5)],
            kl_ablate=kls,
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            config=CompressionConfig(
                rep_selection="kl_attribution",
                target_n_features_kept=3,
            ),
        )
        plan = compressor.plan_with_target()
        for cluster in plan.clusters:
            members = list(cluster.members)
            if len(members) <= 1:
                continue
            # Highest-fid member has the highest kl_ablate, so it should win.
            expected = max(members, key=lambda m: kls[m])
            assert cluster.representative == expected, (
                f"cluster {cluster.cluster_id} members={members} "
                f"expected_rep={expected} actual={cluster.representative}"
            )

    def test_end_to_end_apply_writes_correct_reps(self, tmp_path: Path):
        """Full plan + apply round-trip preserves the kl_attribution-chosen
        rep in the output checkpoint.
        """
        sae_path = _setup_sae(tmp_path)
        original = load_file(str(sae_path))
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (0, 2), (1, 2)],
            kl_ablate={0: 0.1, 1: 0.5, 2: 0.2},
        )
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="kl_attribution",
        ).run(out_path)
        new = load_file(str(out_path))

        # Feature 1 (kl=0.5) is the rep; features 0 and 2 (the non-reps)
        # should be zeroed in W_enc / W_dec / b_enc.
        import numpy as np

        for zeroed_fid in (0, 2):
            assert np.all(new["W_enc"][:, zeroed_fid] == 0)
            assert new["b_enc"][zeroed_fid] == 0
            assert np.all(new["W_dec"][zeroed_fid, :] == 0)
        # Feature 1's rows / cols should be retained from the original.
        assert np.array_equal(new["W_enc"][:, 1], original["W_enc"][:, 1])
