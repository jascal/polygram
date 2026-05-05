"""Unit tests for `Compressor.plan()` — the cheap, no-I/O stage."""

from __future__ import annotations

from pathlib import Path

import pytest

from polygram import Compressor
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _checkpoint(tmp_path: Path, n_features: int = 8) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=8)
    return sae_path


class TestPlanStructure:
    def test_three_clusters_two_singletons_and_a_clique(self, tmp_path: Path):
        """8 features; confirmed pairs `(0,1)`, `(2,3)`, and the
        4-clique on {4,5,6,7}. plan() must return 3 clusters in
        ascending min-fid order."""
        report = build_report(
            n_features=8,
            confirmed=[
                (0, 1),
                (2, 3),
                (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
            ],
        )
        compressor = Compressor(
            validation_report=report, sae_checkpoint=_checkpoint(tmp_path)
        )
        plan = compressor.plan()

        assert len(plan.clusters) == 3
        assert [c.cluster_id for c in plan.clusters] == [0, 1, 2]
        assert plan.clusters[0].members == (0, 1)
        assert plan.clusters[1].members == (2, 3)
        assert plan.clusters[2].members == (4, 5, 6, 7)

    def test_singletons_excluded(self, tmp_path: Path):
        """Features that never appear in a confirmed pair are not
        listed in the plan (nothing to compress)."""
        report = build_report(n_features=8, confirmed=[(0, 1)])
        compressor = Compressor(
            validation_report=report, sae_checkpoint=_checkpoint(tmp_path)
        )
        plan = compressor.plan()

        assert len(plan.clusters) == 1
        assert plan.clusters[0].members == (0, 1)

    def test_zero_clusters_when_no_confirmed_pairs(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[])
        compressor = Compressor(
            validation_report=report, sae_checkpoint=_checkpoint(tmp_path)
        )
        plan = compressor.plan()

        assert plan.clusters == ()
        assert plan.feature_ids == tuple(range(8))


class TestRepresentativeSelection:
    def test_default_picks_highest_summed_n_fires(self, tmp_path: Path):
        """Cluster {3,4,5} with hand-set firing counts; rep should be
        the one with the highest summed n_fires."""
        report = build_report(
            n_features=8,
            confirmed=[(3, 4), (3, 5), (4, 5)],
            n_fires={3: 100, 4: 50, 5: 1},
        )
        compressor = Compressor(
            validation_report=report, sae_checkpoint=_checkpoint(tmp_path)
        )
        plan = compressor.plan()

        cluster = plan.clusters[0]
        assert cluster.members == (3, 4, 5)
        assert cluster.representative == 3
        assert cluster.zeroed == (4, 5)

    def test_tiebreak_is_lowest_fid(self, tmp_path: Path):
        """Equal n_fires sums → lowest fid wins."""
        report = build_report(
            n_features=8,
            confirmed=[(2, 3)],
            n_fires={2: 50, 3: 50},
        )
        compressor = Compressor(
            validation_report=report, sae_checkpoint=_checkpoint(tmp_path)
        )
        plan = compressor.plan()
        assert plan.clusters[0].representative == 2

    def test_override_honored(self, tmp_path: Path):
        """User-supplied `representatives[cluster_id] = fid` overrides
        the firing-count rule."""
        report = build_report(
            n_features=8,
            confirmed=[(3, 4), (3, 5), (4, 5)],
            n_fires={3: 100, 4: 50, 5: 1},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path),
            representatives={0: 5},
        )
        plan = compressor.plan()
        assert plan.clusters[0].representative == 5
        assert sorted(plan.clusters[0].zeroed) == [3, 4]


class TestOverrideValidation:
    def test_override_with_unknown_cluster_id_raises(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1)])
        with pytest.raises(ValueError, match="cluster_id=99"):
            Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path),
                representatives={99: 0},
            )

    def test_override_with_fid_not_in_cluster_raises(self, tmp_path: Path):
        report = build_report(
            n_features=8, confirmed=[(0, 1), (2, 3)]
        )
        with pytest.raises(ValueError, match="not a member"):
            Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path),
                # cluster 0 is {0, 1}; fid 7 isn't in it
                representatives={0: 7},
            )


class TestDeterminism:
    def test_plan_is_idempotent(self, tmp_path: Path):
        """Two plan() calls return equal CompressionPlan objects."""
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (3, 4), (3, 5), (4, 5)],
        )
        compressor = Compressor(
            validation_report=report, sae_checkpoint=_checkpoint(tmp_path)
        )
        a = compressor.plan()
        b = compressor.plan()
        assert a == b
