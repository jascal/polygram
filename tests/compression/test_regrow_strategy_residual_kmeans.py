"""`residual_kmeans` strategy unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from polygram import Regrower
from tests._synth_sae import synth_sae


def _setup(tmp_path: Path, *, n_features: int = 16, d_model: int = 8):
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=d_model)
    # Pre-zero a deterministic subset
    state = load_file(str(sae_path))
    for fid in (2, 5, 9, 13):
        state["W_enc"][:, fid] = 0
        state["b_enc"][fid] = 0
        state["W_dec"][fid, :] = 0
    from safetensors.numpy import save_file

    save_file(state, str(sae_path))
    return sae_path


class TestZeroedPopulation:
    def test_unit_norm_decoder_rows(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        rng = np.random.default_rng(7)
        residuals = rng.standard_normal((200, 8)).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals, seed=0,
        )
        result = r.run(tmp_path / "out.safetensors")
        new = load_file(str(result.output_checkpoint))
        for fid in (2, 5, 9, 13):
            norm = float(np.linalg.norm(new["W_dec"][fid, :]))
            assert 0.999 <= norm <= 1.001, f"fid {fid} decoder_norm = {norm}"

    def test_encoder_equals_decoder_transpose(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        new = load_file(str(result.output_checkpoint))
        for fid in (2, 5, 9, 13):
            assert np.array_equal(new["W_enc"][:, fid], new["W_dec"][fid, :])

    def test_encoder_bias_zero(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        new = load_file(str(result.output_checkpoint))
        for fid in (2, 5, 9, 13):
            assert new["b_enc"][fid] == 0.0

    def test_b_dec_untouched(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        original = load_file(str(sae_path))
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        new = load_file(str(result.output_checkpoint))
        assert np.array_equal(new["b_dec"], original["b_dec"])

    def test_non_zeroed_features_byte_equal(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        original = load_file(str(sae_path))
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2, 5, 9, 13}, cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        new = load_file(str(result.output_checkpoint))
        for fid in (0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15):
            assert np.array_equal(new["W_enc"][:, fid], original["W_enc"][:, fid])
            assert np.array_equal(new["W_dec"][fid, :], original["W_dec"][fid, :])
            assert new["b_enc"][fid] == original["b_enc"][fid]


class TestFailureModes:
    def test_flat_residual_stream_raises_runtime_error(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        flat = np.zeros((200, 8), dtype=np.float32)
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={2}, cached_residuals=flat,
        )
        with pytest.raises(RuntimeError, match="no signal"):
            r.plan()

    def test_n_tokens_below_K_raises_value_error(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        # 4 tokens < 6 zeroed slots
        residuals = np.random.default_rng(0).standard_normal(
            (4, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed={0, 1, 2, 3, 4, 5}, cached_residuals=residuals,
        )
        with pytest.raises(ValueError, match=r"n_residual_tokens=4"):
            r.plan()


class TestEmptyZeroed:
    def test_no_op_produces_tensor_identical_output(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        original = load_file(str(sae_path))
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32)
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans",
            zeroed=set(), cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        new = load_file(str(result.output_checkpoint))
        for k in original:
            assert np.array_equal(new[k], original[k])
        assert result.report.n_slots_repopulated == 0
        assert result.report.n_slots_left_zero == 0
        assert result.report.plan.slots == ()


class TestReservedStrategies:
    @pytest.mark.parametrize(
        "strategy", ["high_decoder_norm_random", "orthogonal_noise_scaled"]
    )
    def test_reserved_strategies_raise_not_implemented(
        self, tmp_path: Path, strategy: str
    ):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32) * 2.0
        r = Regrower(
            sae_checkpoint=sae_path, strategy=strategy,
            zeroed={2}, cached_residuals=residuals,
        )
        with pytest.raises(NotImplementedError, match=strategy):
            r.run(tmp_path / "out.safetensors")
