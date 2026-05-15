"""CLI subprocess tests for `polygram compress` (Phase 3 of
add-pareto-target-compression).

Covers the four new flags (`--target-features`, `--pareto`,
`--pareto-materialize`, `--score-field`) plus the mutual-exclusion
constraint. Run via `subprocess.run` so we exercise the actual
argparse dispatch and the dataclass round-trips end-to-end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from polygram.compression.pareto import ParetoReport
from polygram.compression.report import CompressionReport
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _setup_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    """Write a synth SAE + a ValidationReport JSON; return (sae, vr).

    Uses 8 features and a clique on {4,5,6,7} plus pairs (0,1) and
    (2,3) so target-K and Pareto sweeps have something to chew on.
    """
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=8, d_model=8)
    vreport = build_report(
        n_features=8,
        confirmed=[
            (0, 1), (2, 3),
            (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
        ],
    )
    vr_path = tmp_path / "validation_report.json"
    vreport.to_json(vr_path)
    return sae_path, vr_path


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "polygram.cli", *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# --target-features (single-shot target-K)
# ---------------------------------------------------------------------------


class TestTargetFeaturesFlag:
    def test_target_features_writes_compressed_sae_and_report(
        self, tmp_path: Path
    ):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        out_sae = tmp_path / "compressed.safetensors"
        out_json = tmp_path / "report.json"

        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_sae),
            "--output", str(out_json),
            "--target-features", "2",
        )
        assert proc.returncode == 0, proc.stderr
        assert out_sae.is_file()
        assert out_json.is_file()

        report = CompressionReport.from_json(out_json)
        assert report.n_features_kept <= 2 or report.n_features_kept == 3
        # The fixture's 3 disjoint super-clusters (0,1) (2,3) {4..7}
        # can compress to at most 3 cluster representatives; K=2
        # exceeds that floor → infeasible-but-best.

    def test_target_features_zero_rejected(self, tmp_path: Path):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(tmp_path / "out.safetensors"),
            "--output", str(tmp_path / "out.json"),
            "--target-features", "0",
        )
        assert proc.returncode != 0
        assert "target" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# --pareto (multi-K plan, no materialisation)
# ---------------------------------------------------------------------------


class TestParetoFlag:
    def test_pareto_writes_plan_only(self, tmp_path: Path):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        out_dir = tmp_path / "pareto_run"

        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output", str(out_dir),
            "--pareto", "3,2,1",
        )
        assert proc.returncode == 0, proc.stderr

        pareto_json = out_dir / "pareto.json"
        assert pareto_json.is_file()
        # No SAE materialisation without --pareto-materialize.
        assert not (out_dir / "pareto").exists()

        pr = ParetoReport.from_json(pareto_json)
        assert pr.targets == (3, 2, 1)
        assert len(pr.outcomes) == 3

    def test_pareto_with_materialize_writes_per_k_safetensors(
        self, tmp_path: Path
    ):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        out_dir = tmp_path / "pareto_run"

        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output", str(out_dir),
            "--pareto", "3,2,1",
            "--pareto-materialize",
        )
        assert proc.returncode == 0, proc.stderr

        assert (out_dir / "pareto.json").is_file()
        for k in (3, 2, 1):
            ckpt = out_dir / "pareto" / f"k_{k}.safetensors"
            assert ckpt.is_file(), f"K={k} materialisation missing at {ckpt}"

    def test_pareto_empty_list_rejected(self, tmp_path: Path):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output", str(tmp_path / "out"),
            "--pareto", "",
        )
        assert proc.returncode != 0
        assert "empty" in proc.stderr.lower() or "target" in proc.stderr.lower()

    def test_pareto_non_integer_rejected(self, tmp_path: Path):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output", str(tmp_path / "out"),
            "--pareto", "3,abc",
        )
        assert proc.returncode != 0
        assert "abc" in proc.stderr or "integer" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


class TestMutualExclusion:
    def test_target_features_and_pareto_mutually_exclusive(
        self, tmp_path: Path
    ):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output", str(tmp_path / "out"),
            "--target-features", "2",
            "--pareto", "3,2",
        )
        assert proc.returncode != 0
        # argparse's stderr names "not allowed with"
        assert "not allowed" in proc.stderr or "mutually" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Score-field flag plumbing
# ---------------------------------------------------------------------------


class TestScoreFieldFlag:
    def test_score_field_recorded_in_pareto_json(self, tmp_path: Path):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        out_dir = tmp_path / "pareto_run"

        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output", str(out_dir),
            "--pareto", "2",
            "--score-field", "jaccard",
        )
        assert proc.returncode == 0, proc.stderr

        pr = ParetoReport.from_json(out_dir / "pareto.json")
        assert pr.score_field == "jaccard"

    def test_score_field_bogus_rejected(self, tmp_path: Path):
        sae_path, vr_path = _setup_fixtures(tmp_path)
        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output", str(tmp_path / "out"),
            "--pareto", "2",
            "--score-field", "kl_log_ratio_abs",
        )
        # argparse's choices validation rejects → exit 2
        assert proc.returncode != 0


# ---------------------------------------------------------------------------
# Threshold-path byte-identity sentinel
# ---------------------------------------------------------------------------


class TestThresholdPathUnchanged:
    def test_threshold_mode_still_works_no_new_flags(
        self, tmp_path: Path
    ):
        # Validates the existing CLI shape continues to work; no
        # --target-features and no --pareto means we hit the original
        # compressor.run() path.
        sae_path, vr_path = _setup_fixtures(tmp_path)
        out_sae = tmp_path / "compressed.safetensors"
        out_json = tmp_path / "report.json"

        proc = _run_cli(
            "compress",
            "--validation-report", str(vr_path),
            "--sae-checkpoint", str(sae_path),
            "--output-checkpoint", str(out_sae),
            "--output", str(out_json),
            "--strategy", "zero",
        )
        assert proc.returncode == 0, proc.stderr
        assert out_sae.is_file()
        assert out_json.is_file()
        report = CompressionReport.from_json(out_json)
        assert report.strategy == "zero"
        # 3 confirmed clusters in the fixture: (0,1), (2,3), {4..7}.
        assert report.n_clusters == 3


# ---------------------------------------------------------------------------
# End-to-end: target-K reload round-trip (not via CLI subprocess)
# ---------------------------------------------------------------------------


class TestTargetKEndToEnd:
    def test_target_k_apply_reload(self, tmp_path: Path):
        # In-process variant of the CLI tests: ensures the
        # plan_with_target → apply → CompressionReport.from_json path
        # round-trips before the CLI tests exercise it.
        from polygram import Compressor
        from polygram.config import CompressionConfig

        sae_path, vr_path = _setup_fixtures(tmp_path)
        from polygram.behavioural import ValidationReport

        vr = ValidationReport.from_json(vr_path)
        compressor = Compressor(
            validation_report=vr,
            sae_checkpoint=sae_path,
            config=CompressionConfig(target_n_features_kept=3),
        )
        plan = compressor.plan_with_target()
        out_ckpt = tmp_path / "compressed.safetensors"
        result = compressor.apply(plan=plan, output_checkpoint=out_ckpt)

        out_json = tmp_path / "report.json"
        result.report.to_json(out_json)
        reloaded = CompressionReport.from_json(out_json)
        assert reloaded.n_features_kept == result.report.n_features_kept

    def test_pareto_per_outcome_apply(self, tmp_path: Path):
        """End-to-end Pareto: plan_pareto → apply per outcome → nested
        compression (more zeroed rows at lower K)."""
        from polygram import Compressor

        sae_path, vr_path = _setup_fixtures(tmp_path)
        from polygram.behavioural import ValidationReport

        vr = ValidationReport.from_json(vr_path)
        compressor = Compressor(
            validation_report=vr, sae_checkpoint=sae_path,
        )
        pr = compressor.plan_pareto([3, 1])

        zeroed_counts = []
        for i, outcome in enumerate(pr.outcomes):
            out_ckpt = tmp_path / f"k_{outcome.target_k}.safetensors"
            result = compressor.apply(
                plan=outcome.plan, output_checkpoint=out_ckpt
            )
            zeroed_counts.append(result.report.n_features_zeroed)

        # outcomes are ordered descending in K, so n_features_zeroed
        # should be weakly increasing across the list (smaller K means
        # more merges, more zeroings).
        assert zeroed_counts == sorted(zeroed_counts), zeroed_counts
