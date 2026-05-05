"""End-to-end iteration loop convergence tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from polygram import EpochCompressor


def _build_redundant_sae(tmp_path: Path, *, n_features: int = 32,
                         d_model: int = 768) -> Path:
    """Synthesize an SAE with one engineered redundancy cluster
    (features 4–7 share a decoder direction). d_model=768 to match
    GPT-2 small."""
    sae_path = tmp_path / "sae.safetensors"
    rng = np.random.default_rng(0)
    base = rng.standard_normal(d_model).astype(np.float32)
    state = {
        "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32) * 0.1,
        "b_enc": rng.standard_normal(n_features).astype(np.float32) * 0.1,
        "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32) * 0.1,
        "b_dec": np.zeros(d_model, dtype=np.float32),
    }
    for fid in (4, 5, 6, 7):
        state["W_dec"][fid, :] = base + rng.standard_normal(d_model).astype(np.float32) * 0.02
        state["W_enc"][:, fid] = state["W_dec"][fid, :]
        state["b_enc"][fid] = 0.5
    save_file(state, str(sae_path))
    return sae_path


@pytest.fixture(scope="module")
def torch_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("[behavioural] extra not installed")
    return True


class TestEpochLoop:
    def test_run_succeeds_and_compresses_redundancies(
        self, tmp_path: Path, torch_available,
    ):
        sae_path = _build_redundant_sae(tmp_path)
        epoch = EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=["hello world this is a test prompt",
                     "a second prompt for variation here"],
            layer=10, device="cpu",
            n_panels_max=4, max_iterations=2,
            min_firing_rate=0.0,
            coverage_target=1.0,
            cosine_threshold=0.50,
        )
        out_path = tmp_path / "epoch.safetensors"
        result = epoch.run(out_path)
        assert result.report.n_features_zeroed_total >= 1
        # Source bytes unchanged
        assert sae_path.is_file()

    def test_max_iterations_terminates_loop(
        self, tmp_path: Path, torch_available,
    ):
        sae_path = _build_redundant_sae(tmp_path)
        epoch = EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=["hello world this is a test prompt",
                     "another prompt with different words to vary the residuals"],
            layer=10, device="cpu",
            n_panels_max=4, max_iterations=1,
            min_firing_rate=0.0,
            coverage_target=1.0,
            cosine_threshold=0.50,
        )
        result = epoch.run(tmp_path / "epoch.safetensors")
        # max_iterations=1 → either max_iterations or no_more_priority_candidates
        assert result.report.convergence_reason in (
            "max_iterations",
            "no_more_priority_candidates",
            "stable_clusters",
        )
        assert len(result.report.iterations) <= 1

    def test_source_immutable_after_run(
        self, tmp_path: Path, torch_available,
    ):
        sae_path = _build_redundant_sae(tmp_path)
        before = sae_path.read_bytes()
        epoch = EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=["hello world test prompt"],
            layer=10, device="cpu",
            n_panels_max=2, max_iterations=1,
            min_firing_rate=0.0,
        )
        epoch.run(tmp_path / "epoch.safetensors")
        assert sae_path.read_bytes() == before

    def test_output_equals_source_raises(
        self, tmp_path: Path, torch_available,
    ):
        sae_path = _build_redundant_sae(tmp_path)
        epoch = EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=["hello"], layer=10, device="cpu",
        )
        with pytest.raises(ValueError, match="must differ"):
            epoch.run(sae_path)


class TestNoOpPath:
    def test_no_redundancies_returns_no_priority_candidates(
        self, tmp_path: Path, torch_available,
    ):
        # SAE with all-different decoder rows; no clique → no
        # confirmed pairs at any threshold. Use very strict gates
        # so the validator finds nothing.
        sae_path = tmp_path / "sae.safetensors"
        rng = np.random.default_rng(42)
        d_model = 768
        n_features = 16
        state = {
            "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32) * 0.1,
            "b_enc": rng.standard_normal(n_features).astype(np.float32) * 0.1,
            "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32) * 0.1,
            "b_dec": np.zeros(d_model, dtype=np.float32),
        }
        save_file(state, str(sae_path))
        epoch = EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=["hello"], layer=10, device="cpu",
            n_panels_max=2, max_iterations=2,
            min_firing_rate=0.0,
            coverage_target=1.0,
            cosine_threshold=0.95,                   # very strict
            polygram_overlap_threshold=0.99,         # very strict gate
        )
        result = epoch.run(tmp_path / "epoch.safetensors")
        assert result.report.n_features_zeroed_total == 0
