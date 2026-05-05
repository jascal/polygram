"""Determinism: identical inputs produce byte-identical outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


class TestDeterminism:
    def test_two_runs_produce_byte_identical_tensors(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((200, 8)).astype(np.float32)

        r1 = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
            seed=0, n_init=4,
        )
        r2 = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
            seed=0, n_init=4,
        )
        out1 = r1.run(tmp_path / "out1.safetensors")
        out2 = r2.run(tmp_path / "out2.safetensors")

        s1 = load_file(str(out1.output_checkpoint))
        s2 = load_file(str(out2.output_checkpoint))
        for k in s1:
            assert np.array_equal(s1[k], s2[k]), (
                f"tensor {k} differs across deterministic runs"
            )
        assert (
            out1.report.output_checkpoint_sha256
            == out2.report.output_checkpoint_sha256
        )

    def test_different_seeds_produce_different_outputs(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32)

        r1 = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals, seed=0,
        )
        r2 = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals, seed=42,
        )
        out1 = r1.run(tmp_path / "out1.safetensors")
        out2 = r2.run(tmp_path / "out2.safetensors")

        # At least one populated slot's decoder direction should differ
        s1 = load_file(str(out1.output_checkpoint))
        s2 = load_file(str(out2.output_checkpoint))
        any_diff = any(
            not np.array_equal(s1["W_dec"][fid, :], s2["W_dec"][fid, :])
            for fid in (2, 5, 9, 13)
        )
        assert any_diff, "different seeds should produce different directions"
