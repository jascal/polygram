"""`polygram compress-epoch` subcommand: argument parsing + skip paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from polygram.cli import main
from tests._synth_sae import synth_sae


def _setup(tmp_path: Path):
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=16, d_model=8)
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello world\n")
    return sae_path, prompts_path


class TestRejectionPaths:
    def test_missing_sae_checkpoint(self, tmp_path: Path):
        _, prompts_path = _setup(tmp_path)
        rc = main([
            "compress-epoch",
            "--sae-checkpoint", str(tmp_path / "missing.safetensors"),
            "--prompts", str(prompts_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
        ])
        assert rc == 2

    def test_missing_prompts(self, tmp_path: Path):
        sae_path, _ = _setup(tmp_path)
        rc = main([
            "compress-epoch",
            "--sae-checkpoint", str(sae_path),
            "--prompts", str(tmp_path / "missing.txt"),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
        ])
        assert rc == 2

    def test_output_equals_source(self, tmp_path: Path):
        sae_path, prompts_path = _setup(tmp_path)
        rc = main([
            "compress-epoch",
            "--sae-checkpoint", str(sae_path),
            "--prompts", str(prompts_path),
            "--output-checkpoint", str(sae_path),
            "--output", str(tmp_path / "report.json"),
        ])
        assert rc == 2

    def test_unknown_strategy_argparse_rejection(self, tmp_path: Path):
        sae_path, prompts_path = _setup(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main([
                "compress-epoch",
                "--sae-checkpoint", str(sae_path),
                "--prompts", str(prompts_path),
                "--output-checkpoint", str(tmp_path / "out.safetensors"),
                "--output", str(tmp_path / "report.json"),
                "--strategy", "merge",
            ])
        assert exc.value.code == 2

    def test_out_of_range_coverage_target(self, tmp_path: Path):
        sae_path, prompts_path = _setup(tmp_path)
        rc = main([
            "compress-epoch",
            "--sae-checkpoint", str(sae_path),
            "--prompts", str(prompts_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
            "--coverage-target", "1.5",
        ])
        assert rc == 2

    def test_layer_zero_without_override(self, tmp_path: Path):
        sae_path, prompts_path = _setup(tmp_path)
        rc = main([
            "compress-epoch",
            "--sae-checkpoint", str(sae_path),
            "--prompts", str(prompts_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "report.json"),
            "--layer", "0",
        ])
        assert rc == 2
