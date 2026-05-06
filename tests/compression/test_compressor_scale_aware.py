"""`Compressor(rep_selection="scale_aware")` — unit tests."""

from __future__ import annotations

import warnings
from pathlib import Path

from polygram import Compressor
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _checkpoint(
    tmp_path: Path,
    n_features: int = 8,
    dec_norms: dict[int, float] | None = None,
) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(
        sae_path,
        n_features=n_features,
        d_model=8,
        dec_norms=dec_norms,
    )
    return sae_path


class TestScaleAwarePicksMedianNormCandidate:
    def test_norm_proximity_dominates_when_freq_and_ablation_equal(
        self, tmp_path: Path
    ):
        """Cluster {3,4,5}: equal n_fires and equal kl_ablate.
        Decoder norms 1.0 / 5.0 / 9.0 → median = 5.0 → fid 4 wins."""
        sae_path = _checkpoint(
            tmp_path,
            n_features=8,
            dec_norms={3: 1.0, 4: 5.0, 5: 9.0},
        )
        report = build_report(
            n_features=8,
            confirmed=[(3, 4), (3, 5), (4, 5)],
            n_fires={3: 50, 4: 50, 5: 50},
            kl_ablate={3: 1.0, 4: 1.0, 5: 1.0},
        )
        plan = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="scale_aware",
        ).plan()
        assert plan.clusters[0].representative == 4

    def test_high_ablation_overcomes_low_norm_proximity(
        self, tmp_path: Path
    ):
        """Cluster {3,4}: norms 5.0 / 1.0 → norm_proximity favors 3.
        But kl_ablate of 4 is 100x higher → ablation should pull
        the rep towards 4."""
        sae_path = _checkpoint(
            tmp_path, n_features=8, dec_norms={3: 5.0, 4: 1.0}
        )
        report = build_report(
            n_features=8,
            confirmed=[(3, 4)],
            n_fires={3: 10, 4: 10},
            kl_ablate={3: 0.01, 4: 1.0},
        )
        plan = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="scale_aware",
        ).plan()
        # With a 2-member cluster, median = mean of the two; both are
        # equidistant from the median so norm_proximity ties. Ablation
        # then picks 4.
        assert plan.clusters[0].representative == 4


class TestScaleAwareNanFallback:
    def test_nan_kl_ablate_emits_warning(self, tmp_path: Path):
        """All kl_ablate are NaN (geometry-only). scale_aware SHALL
        emit a UserWarning and fall through to n_fires-style scoring."""
        sae_path = _checkpoint(
            tmp_path, n_features=8, dec_norms={3: 5.0, 4: 5.0}
        )
        report = build_report(
            n_features=8,
            confirmed=[(3, 4)],
            n_fires={3: 100, 4: 50},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = Compressor(
                validation_report=report,
                sae_checkpoint=sae_path,
                rep_selection="scale_aware",
            ).plan()
        assert any(
            issubclass(w.category, UserWarning)
            and "kl_ablate" in str(w.message)
            for w in caught
        )
        # Equal norms → norm_proximity ties; n_fires breaks the tie:
        # 3 has 100 fires, 4 has 50 → rep should be 3.
        assert plan.clusters[0].representative == 3


class TestScaleAwareDoesNotAffectDefault:
    def test_n_fires_default_unchanged(self, tmp_path: Path):
        """Default rep_selection is `n_fires`; behaviour identical to
        pre-change code (no W_dec load, no warning, n_fires wins)."""
        sae_path = _checkpoint(
            tmp_path, n_features=8, dec_norms={3: 1.0, 4: 5.0, 5: 9.0}
        )
        report = build_report(
            n_features=8,
            confirmed=[(3, 4), (3, 5), (4, 5)],
            n_fires={3: 100, 4: 50, 5: 1},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = Compressor(
                validation_report=report,
                sae_checkpoint=sae_path,
            ).plan()
        # No scale_aware warnings.
        assert not any(
            "kl_ablate" in str(w.message) for w in caught
        )
        # n_fires wins → 3.
        assert plan.clusters[0].representative == 3


class TestScaleAwareReproducible:
    def test_two_plan_calls_return_equal_plans(self, tmp_path: Path):
        sae_path = _checkpoint(
            tmp_path, n_features=8, dec_norms={3: 1.0, 4: 5.0, 5: 9.0}
        )
        report = build_report(
            n_features=8,
            confirmed=[(3, 4), (3, 5), (4, 5)],
            n_fires={3: 50, 4: 50, 5: 50},
            kl_ablate={3: 0.5, 4: 0.5, 5: 0.5},
        )
        compressor = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="scale_aware",
        )
        a = compressor.plan()
        b = compressor.plan()
        assert a == b


class TestScaleAwareLowestFidTiebreak:
    def test_equal_scores_pick_lowest_fid(self, tmp_path: Path):
        """All candidates score identically → lowest fid wins."""
        sae_path = _checkpoint(
            tmp_path, n_features=8, dec_norms={2: 5.0, 3: 5.0, 4: 5.0}
        )
        report = build_report(
            n_features=8,
            confirmed=[(2, 3), (2, 4), (3, 4)],
            n_fires={2: 10, 3: 10, 4: 10},
            kl_ablate={2: 1.0, 3: 1.0, 4: 1.0},
        )
        plan = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            rep_selection="scale_aware",
        ).plan()
        assert plan.clusters[0].representative == 2
