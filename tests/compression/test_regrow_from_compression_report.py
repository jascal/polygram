"""Chained constructor: `Regrower.from_compression_report`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polygram import (
    ClusterPlan,
    CompressionPlan,
    CompressionReport,
    Regrower,
)
from tests._synth_sae import synth_sae


def _setup(tmp_path: Path):
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=16, d_model=8)
    from safetensors.numpy import load_file, save_file
    state = load_file(str(sae_path))
    for fid in (2, 9):
        state["W_enc"][:, fid] = 0
        state["b_enc"][fid] = 0
        state["W_dec"][fid, :] = 0
    save_file(state, str(sae_path))
    return sae_path


def _hand_built_compression_report(sae_path: Path) -> CompressionReport:
    return CompressionReport(
        schema_version=1,
        source_checkpoint="/somewhere/source.safetensors",
        source_checkpoint_sha256="a" * 64,
        output_checkpoint=str(sae_path),
        output_checkpoint_sha256="b" * 64,
        validation_report_dictionary_name="UpstreamDict",
        validation_report_schema_version=1,
        strategy="zero",
        plan=CompressionPlan(
            clusters=(
                ClusterPlan(cluster_id=0, members=(2, 5),
                            representative=5, zeroed=(2,)),
                ClusterPlan(cluster_id=1, members=(9, 13),
                            representative=13, zeroed=(9,)),
            ),
            feature_ids=tuple(range(16)),
        ),
        n_features_zeroed=2, n_features_kept=2, n_clusters=2,
    )


class TestChainedConstructor:
    def test_zeroed_extracted_from_clusters(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path)
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans", model_name="gpt2", layer=10,
            cached_residuals=residuals,
        )
        assert sorted(r.zeroed) == [2, 9]

    def test_provenance_populated(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path)
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans", model_name="gpt2", layer=10,
            cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        prov = result.report.provenance
        assert prov["compression_report_source_sha256"] == "a" * 64
        assert prov["compression_report_output_sha256"] == "b" * 64
        assert prov["compression_report_dictionary_name"] == "UpstreamDict"

    def test_direct_constructor_leaves_provenance_empty(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        r = Regrower(
            sae_checkpoint=sae_path, strategy="residual_kmeans", model_name="gpt2", layer=10,
            zeroed={2, 9}, cached_residuals=residuals,
        )
        result = r.run(tmp_path / "out.safetensors")
        assert result.report.provenance == {}


# ---------------------------------------------------------------------------
# Tasks §8 — Regrower.from_compression_report requires explicit model_name
# and layer (no GPT-2 fallback) and accepts a RegrowConfig with the
# documented kwarg > config > defaults precedence.
# ---------------------------------------------------------------------------


class TestFromCompressionReportRequiredKwargs:
    def test_missing_model_name_raises(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path)
        with pytest.raises(TypeError, match="model_name"):
            Regrower.from_compression_report(
                report, sae_checkpoint=sae_path,
                strategy="residual_kmeans",
                cached_residuals=np.zeros((10, 8), dtype=np.float32),
                layer=10,  # model_name missing
            )

    def test_missing_layer_raises(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path)
        with pytest.raises(TypeError, match="layer"):
            Regrower.from_compression_report(
                report, sae_checkpoint=sae_path,
                strategy="residual_kmeans",
                cached_residuals=np.zeros((10, 8), dtype=np.float32),
                model_name="gpt2",  # layer missing
            )

    def test_missing_both_raises_with_both_in_message(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path)
        with pytest.raises(TypeError) as excinfo:
            Regrower.from_compression_report(
                report, sae_checkpoint=sae_path,
                strategy="residual_kmeans",
                cached_residuals=np.zeros((10, 8), dtype=np.float32),
            )
        assert "model_name" in str(excinfo.value)
        assert "layer" in str(excinfo.value)


class TestFromCompressionReportConfigPassthrough:
    def test_config_supplies_model_name_and_layer(self, tmp_path: Path):
        from polygram import RegrowConfig

        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path)
        cfg = RegrowConfig(model_name="pythia-160m", layer=4, seed=42)
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans",
            cached_residuals=residuals,
            config=cfg,
        )
        assert r.model_name == "pythia-160m"
        assert r.layer == 4
        assert r.seed == 42

    def test_per_field_kwarg_overrides_config(self, tmp_path: Path):
        from polygram import RegrowConfig

        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path)
        cfg = RegrowConfig(model_name="pythia-160m", layer=4)
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans",
            cached_residuals=residuals,
            config=cfg, layer=10,
        )
        assert r.model_name == "pythia-160m"  # from config
        assert r.layer == 10  # kwarg wins
