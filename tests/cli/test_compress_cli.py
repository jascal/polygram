"""`polygram compress` subcommand: argument parsing + end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.numpy import load_file
import numpy as np

from polygram.cli import main
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _setup(tmp_path: Path):
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=8, d_model=8)
    report = build_report(
        n_features=8,
        confirmed=[(0, 1), (3, 4), (3, 5), (4, 5)],
    )
    vreport_path = tmp_path / "validation.json"
    report.to_json(vreport_path)
    return sae_path, vreport_path


class TestEndToEnd:
    def test_happy_path_writes_checkpoint_and_report(self, tmp_path: Path):
        sae_path, vreport_path = _setup(tmp_path)
        out_ckpt = tmp_path / "out.safetensors"
        out_report = tmp_path / "compression.json"

        rc = main([
            "compress",
            "--validation-report", str(vreport_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_ckpt),
            "--strategy", "zero",
            "--output", str(out_report),
        ])
        assert rc == 0
        assert out_ckpt.is_file()
        assert out_report.is_file()
        payload = json.loads(out_report.read_text())
        assert payload["strategy"] == "zero"
        assert payload["n_clusters"] == 2

    def test_representatives_override_applied(self, tmp_path: Path):
        sae_path, vreport_path = _setup(tmp_path)
        out_ckpt = tmp_path / "out.safetensors"
        out_report = tmp_path / "compression.json"

        rc = main([
            "compress",
            "--validation-report", str(vreport_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_ckpt),
            "--strategy", "zero",
            "--output", str(out_report),
            "--representatives", "0=0,1=3",
        ])
        assert rc == 0
        payload = json.loads(out_report.read_text())
        # Cluster 0 reps fid 0; cluster 1 reps fid 3.
        assert payload["clusters"][0]["representative"] == 0
        assert payload["clusters"][1]["representative"] == 3

    def test_zero_pattern_in_rewritten_checkpoint(self, tmp_path: Path):
        sae_path, vreport_path = _setup(tmp_path)
        out_ckpt = tmp_path / "out.safetensors"
        out_report = tmp_path / "compression.json"

        rc = main([
            "compress",
            "--validation-report", str(vreport_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_ckpt),
            "--strategy", "zero",
            "--output", str(out_report),
        ])
        assert rc == 0

        new = load_file(str(out_ckpt))
        payload = json.loads(out_report.read_text())
        for cluster in payload["clusters"]:
            for fid in cluster["zeroed"]:
                assert np.all(new["W_enc"][:, fid] == 0)
                assert new["b_enc"][fid] == 0
                assert np.all(new["W_dec"][fid, :] == 0)


class TestRejectionPaths:
    def test_missing_validation_report(self, tmp_path: Path):
        sae_path, _ = _setup(tmp_path)
        rc = main([
            "compress",
            "--validation-report", str(tmp_path / "nope.json"),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--strategy", "zero",
            "--output", str(tmp_path / "report.json"),
        ])
        assert rc == 2

    def test_missing_sae_checkpoint(self, tmp_path: Path):
        _, vreport_path = _setup(tmp_path)
        rc = main([
            "compress",
            "--validation-report", str(vreport_path),
            "--sae-checkpoint", str(tmp_path / "nope.safetensors"),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--strategy", "zero",
            "--output", str(tmp_path / "report.json"),
        ])
        assert rc == 2

    def test_output_equals_source(self, tmp_path: Path):
        sae_path, vreport_path = _setup(tmp_path)
        rc = main([
            "compress",
            "--validation-report", str(vreport_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(sae_path),
            "--strategy", "zero",
            "--output", str(tmp_path / "report.json"),
        ])
        assert rc == 2

    def test_unknown_strategy_argparse_rejection(self, tmp_path: Path):
        sae_path, vreport_path = _setup(tmp_path)
        # argparse choices=("zero",) → exits 2 directly.
        with pytest.raises(SystemExit) as exc:
            main([
                "compress",
                "--validation-report", str(vreport_path),
                "--sae-checkpoint", str(sae_path),
                "--output-checkpoint", str(tmp_path / "out.safetensors"),
                "--strategy", "merge",
                "--output", str(tmp_path / "report.json"),
            ])
        assert exc.value.code == 2

    def test_malformed_representatives_rejected(self, tmp_path: Path):
        sae_path, vreport_path = _setup(tmp_path)
        rc = main([
            "compress",
            "--validation-report", str(vreport_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--strategy", "zero",
            "--output", str(tmp_path / "report.json"),
            "--representatives", "foo=bar",
        ])
        assert rc == 2

    def test_representatives_unknown_cluster_rejected(self, tmp_path: Path):
        sae_path, vreport_path = _setup(tmp_path)
        rc = main([
            "compress",
            "--validation-report", str(vreport_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--strategy", "zero",
            "--output", str(tmp_path / "report.json"),
            "--representatives", "99=0",
        ])
        assert rc == 2
