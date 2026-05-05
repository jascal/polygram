"""End-to-end tests for `Regrower.apply()` / `Regrower.run()`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from polygram import Regrower
from tests._synth_sae import synth_sae


def _setup(tmp_path: Path):
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=16, d_model=8)
    state = load_file(str(sae_path))
    for fid in (2, 5, 9, 13):
        state["W_enc"][:, fid] = 0
        state["b_enc"][fid] = 0
        state["W_dec"][fid, :] = 0
    from safetensors.numpy import save_file
    save_file(state, str(sae_path))
    return sae_path


class TestSourceImmutability:
    def test_source_bytes_unchanged(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        before = sae_path.read_bytes()
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
        )
        r.run(tmp_path / "out.safetensors")
        assert sae_path.read_bytes() == before


class TestApplyRejections:
    def test_output_equals_source_raises(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32)
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2}, cached_residuals=residuals,
        )
        with pytest.raises(ValueError, match="must differ"):
            r.apply(output_checkpoint=sae_path)

    def test_apply_without_output_path_raises(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32)
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2}, cached_residuals=residuals,
        )
        with pytest.raises(ValueError, match="output_checkpoint is required"):
            r.apply()


class TestProvenance:
    def test_sha256_fields_populated(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32)
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        rep = result.report
        assert len(rep.source_checkpoint_sha256) == 64
        assert len(rep.output_checkpoint_sha256) == 64
        assert rep.source_checkpoint_sha256 != rep.output_checkpoint_sha256
        assert rep.strategy_params["seed"] == 0
        assert rep.strategy_params["n_init"] == 4

    def test_dictionary_rebuilt_on_zeroed_subset(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        # Dictionary rebuilt on the first ≤8 zeroed fids.
        assert len(result.dictionary.features) == 4
