"""`polygram regrow` subcommand: argument parsing + end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from polygram.cli import main
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

    residuals_path = tmp_path / "residuals.npy"
    rng = np.random.default_rng(0)
    np.save(residuals_path, rng.standard_normal((100, 8)).astype(np.float32))
    return sae_path, residuals_path


class TestEndToEnd:
    def test_zeroed_list_plus_cached_residuals(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        out_ckpt = tmp_path / "regrown.safetensors"
        out_report = tmp_path / "regrow.json"

        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_ckpt),
            "--output", str(out_report),
            "--strategy", "residual_kmeans",
            "--zeroed-list", "2,5,9,13",
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 0
        assert out_ckpt.is_file()
        assert out_report.is_file()
        payload = json.loads(out_report.read_text())
        assert payload["strategy"] == "residual_kmeans"
        assert payload["n_slots_repopulated"] >= 1
        assert payload["provenance"] == {}

    def test_compression_report_chains_provenance(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)

        # Hand-build a CompressionReport JSON
        from polygram import (
            ClusterPlan, CompressionPlan, CompressionReport,
        )
        cr = CompressionReport(
            schema_version=1,
            source_checkpoint="/somewhere/source.safetensors",
            source_checkpoint_sha256="a" * 64,
            output_checkpoint=str(sae_path),
            output_checkpoint_sha256="b" * 64,
            validation_report_dictionary_name="ScaleupBlocks10",
            validation_report_schema_version=1,
            strategy="zero",
            plan=CompressionPlan(
                clusters=(
                    ClusterPlan(0, (2, 5), 5, (2,)),
                    ClusterPlan(1, (9, 13), 13, (9,)),
                ),
                feature_ids=tuple(range(16)),
            ),
            n_features_zeroed=2, n_features_kept=2, n_clusters=2,
        )
        cr_path = tmp_path / "compression.json"
        cr.to_json(cr_path)

        out_ckpt = tmp_path / "regrown.safetensors"
        out_report = tmp_path / "regrow.json"
        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_ckpt),
            "--output", str(out_report),
            "--strategy", "residual_kmeans",
            "--compression-report", str(cr_path),
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 0
        payload = json.loads(out_report.read_text())
        assert payload["provenance"]["compression_report_dictionary_name"] == "ScaleupBlocks10"
        assert payload["provenance"]["compression_report_source_sha256"] == "a" * 64

    def test_zero_pattern_in_rewritten_checkpoint(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        out_ckpt = tmp_path / "regrown.safetensors"
        out_report = tmp_path / "regrow.json"

        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_ckpt),
            "--output", str(out_report),
            "--strategy", "residual_kmeans",
            "--zeroed-list", "2,5,9,13",
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 0

        new = load_file(str(out_ckpt))
        for fid in (2, 5, 9, 13):
            # Either populated (norm ≈ 1) or left zero (cluster_size=0)
            norm = float(np.linalg.norm(new["W_dec"][fid, :]))
            assert 0.999 <= norm <= 1.001 or norm == 0.0


class TestRejectionPaths:
    def test_missing_sae_checkpoint(self, tmp_path: Path):
        residuals_path = tmp_path / "residuals.npy"
        np.save(residuals_path, np.zeros((10, 8), np.float32))
        rc = main([
            "regrow",
            "--sae-checkpoint", str(tmp_path / "missing.safetensors"),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
            "--strategy", "residual_kmeans",
            "--zeroed-list", "0",
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 2

    def test_output_equals_source(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(sae_path),
            "--output", str(tmp_path / "report.json"),
            "--strategy", "residual_kmeans",
            "--zeroed-list", "2",
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 2

    def test_both_zeroed_sources_rejected(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        # Synthesize a fake compression-report path; we shouldn't get
        # past the mutual-exclusion check.
        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
            "--strategy", "residual_kmeans",
            "--zeroed-list", "2",
            "--compression-report", str(tmp_path / "any.json"),
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 2

    def test_neither_zeroed_source_rejected(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
            "--strategy", "residual_kmeans",
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 2

    def test_both_residual_sources_rejected(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        prompts_path = tmp_path / "prompts.txt"
        prompts_path.write_text("hello\n")
        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
            "--strategy", "residual_kmeans",
            "--zeroed-list", "2",
            "--cached-residuals", str(residuals_path),
            "--prompts", str(prompts_path),
        ])
        assert rc == 2

    def test_unknown_strategy_argparse_rejection(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main([
                "regrow",
                "--sae-checkpoint", str(sae_path),
                "--output-checkpoint", str(tmp_path / "out.safetensors"),
                "--output", str(tmp_path / "report.json"),
                "--strategy", "merge",
                "--zeroed-list", "2",
                "--cached-residuals", str(residuals_path),
            ])
        assert exc.value.code == 2

    def test_malformed_zeroed_list(self, tmp_path: Path):
        sae_path, residuals_path = _setup(tmp_path)
        rc = main([
            "regrow",
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
            "--strategy", "residual_kmeans",
            "--zeroed-list", "1,foo,3",
            "--cached-residuals", str(residuals_path),
        ])
        assert rc == 2
