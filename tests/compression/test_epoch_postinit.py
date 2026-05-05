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
