"""`Regrower.__post_init__` rejection paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polygram import Regrower
from tests._synth_sae import synth_sae


def _checkpoint(tmp_path: Path, n_features: int = 16) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=8)
    return sae_path


class TestPostInit:
    def test_missing_checkpoint_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not found"):
            Regrower(
                sae_checkpoint=tmp_path / "missing.safetensors",
                strategy="residual_kmeans",
                zeroed={0},
                cached_residuals=np.zeros((10, 8), dtype=np.float32),
            )

    def test_unsupported_strategy_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unsupported strategy"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="merge",
                zeroed={0},
                cached_residuals=np.zeros((10, 8), dtype=np.float32),
            )

    def test_negative_seed_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="seed"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans",
                zeroed={0}, seed=-1,
                cached_residuals=np.zeros((10, 8), dtype=np.float32),
            )

    def test_n_init_below_one_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="n_init"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans",
                zeroed={0}, n_init=0,
                cached_residuals=np.zeros((10, 8), dtype=np.float32),
            )

    def test_both_residual_sources_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="exactly one"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans", zeroed={0},
                prompts=["x"], cached_residuals=np.zeros((10, 8), np.float32),
            )

    def test_neither_residual_source_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="exactly one"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans", zeroed={0},
            )

    def test_empty_prompts_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="non-empty"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans", zeroed={0},
                prompts=[],
            )

    def test_zeroed_fid_out_of_range_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="out of range"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path, n_features=16),
                strategy="residual_kmeans",
                zeroed={20},  # 20 >= 16
                cached_residuals=np.zeros((10, 8), np.float32),
            )

    def test_negative_zeroed_fid_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="non-negative"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans", zeroed={-1},
                cached_residuals=np.zeros((10, 8), np.float32),
            )

    def test_cached_residuals_wrong_ndim_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="2D"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans", zeroed={0},
                cached_residuals=np.zeros((10,), np.float32),
            )

    def test_cached_residuals_wrong_dtype_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="dtype"):
            Regrower(
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="residual_kmeans", zeroed={0},
                cached_residuals=np.zeros((10, 8), np.int32),
            )
