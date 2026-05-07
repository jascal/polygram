"""`EpochCompressor.__post_init__` rejection paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from polygram import EpochCompressor
from tests._synth_sae import synth_sae


def _checkpoint(tmp_path: Path) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=16, d_model=8)
    return sae_path


class TestPostInit:
    def test_missing_checkpoint_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not found"):
            EpochCompressor(
                sae_checkpoint=tmp_path / "missing.safetensors",
                prompts=["x"], layer=10,
            )

    def test_empty_prompts_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="non-empty"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=[], layer=10,
            )

    def test_unsupported_strategy_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="zero"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=["x"], layer=10, strategy="merge",
            )

    def test_coverage_target_out_of_range(self, tmp_path: Path):
        with pytest.raises(ValueError, match="coverage_target"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=["x"], layer=10, coverage_target=1.5,
            )

    def test_cosine_threshold_out_of_range(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cosine_threshold"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=["x"], layer=10, cosine_threshold=2.0,
            )

    def test_n_visits_below_one(self, tmp_path: Path):
        with pytest.raises(ValueError, match="n_visits_per_feature"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=["x"], layer=10, n_visits_per_feature=0,
            )

    def test_max_iterations_below_one(self, tmp_path: Path):
        with pytest.raises(ValueError, match="max_iterations"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=["x"], layer=10, max_iterations=0,
            )

    def test_quality_delta_multiplier_non_positive(self, tmp_path: Path):
        with pytest.raises(ValueError, match="quality_delta_multiplier"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=["x"], layer=10, quality_delta_multiplier=0,
            )

    def test_layer_zero_without_override_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="layer 0"):
            EpochCompressor(
                sae_checkpoint=_checkpoint(tmp_path),
                prompts=["x"], layer=0,
            )

    def test_layer_zero_with_override_succeeds(self, tmp_path: Path):
        EpochCompressor(
            sae_checkpoint=_checkpoint(tmp_path),
            prompts=["x"], layer=0, allow_layer_zero=True,
        )


# ---------------------------------------------------------------------------
# Tasks §4.3 / §5.2 / §5.3 — EpochCompressor accepts EpochCompressionConfig
# with override precedence; named presets `.fast()` / `.thorough()` bundle
# tuning defaults so callers don't repeat the same kwargs.
# ---------------------------------------------------------------------------


class TestEpochCompressionConfigPassthrough:
    def test_no_config_uses_iterative_preset_defaults(self, tmp_path: Path):
        ec = EpochCompressor(
            sae_checkpoint=_checkpoint(tmp_path), prompts=["x"], layer=10,
        )
        # New defaults after polygram-tuning-config (was 0.95 / 3 / 5).
        assert ec.coverage_target == 0.5
        assert ec.n_visits_per_feature == 1
        assert ec.max_iterations == 1
        # cosine_threshold default unchanged.
        assert ec.cosine_threshold == 0.30

    def test_config_supplies_unset_fields(self, tmp_path: Path):
        from polygram import EpochCompressionConfig

        cfg = EpochCompressionConfig(coverage_target=0.9, max_iterations=3)
        ec = EpochCompressor(
            sae_checkpoint=_checkpoint(tmp_path), prompts=["x"], layer=10,
            config=cfg,
        )
        assert ec.coverage_target == 0.9
        assert ec.max_iterations == 3
        # other fields take config defaults
        assert ec.n_visits_per_feature == 1

    def test_per_field_kwarg_overrides_config(self, tmp_path: Path):
        from polygram import EpochCompressionConfig

        cfg = EpochCompressionConfig(coverage_target=0.9)
        ec = EpochCompressor(
            sae_checkpoint=_checkpoint(tmp_path), prompts=["x"], layer=10,
            config=cfg, coverage_target=0.7,
        )
        # kwarg wins.
        assert ec.coverage_target == 0.7

    def test_validation_config_supplies_validator_knobs(self, tmp_path: Path):
        from polygram import EpochCompressionConfig, ValidationConfig

        cfg = EpochCompressionConfig(
            validation=ValidationConfig(
                polygram_overlap_threshold=0.85,
                jaccard_threshold=0.5,
                min_both_fire=8,
            )
        )
        ec = EpochCompressor(
            sae_checkpoint=_checkpoint(tmp_path), prompts=["x"], layer=10,
            config=cfg,
        )
        assert ec.polygram_overlap_threshold == 0.85
        assert ec.jaccard_threshold == 0.5
        assert ec.min_both_fire == 8


class TestEpochCompressorPresets:
    def test_fast_matches_default_construction(self, tmp_path: Path):
        # Spec scenario: a = .fast(), b = default; tuning fields equal.
        ckpt = _checkpoint(tmp_path)
        a = EpochCompressor.fast(sae_checkpoint=ckpt, prompts=["x"], layer=10)
        b = EpochCompressor(sae_checkpoint=ckpt, prompts=["x"], layer=10)
        assert a.coverage_target == b.coverage_target
        assert a.n_visits_per_feature == b.n_visits_per_feature
        assert a.max_iterations == b.max_iterations

    def test_thorough_restores_legacy_defaults(self, tmp_path: Path):
        ec = EpochCompressor.thorough(
            sae_checkpoint=_checkpoint(tmp_path), prompts=["x"], layer=10,
        )
        assert ec.coverage_target == 0.95
        assert ec.n_visits_per_feature == 3
        assert ec.max_iterations == 5

    def test_fast_accepts_overrides(self, tmp_path: Path):
        ec = EpochCompressor.fast(
            sae_checkpoint=_checkpoint(tmp_path), prompts=["x"], layer=10,
            coverage_target=0.6,
        )
        # override wins...
        assert ec.coverage_target == 0.6
        # ...others stay at fast() values
        assert ec.n_visits_per_feature == 1
        assert ec.max_iterations == 1
