"""`Compressor.__post_init__` rejection paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from polygram import Compressor
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _checkpoint(tmp_path: Path) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path)
    return sae_path


class TestPostinit:
    def test_missing_checkpoint_raises(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1)])
        with pytest.raises(ValueError, match="not found"):
            Compressor(
                validation_report=report,
                sae_checkpoint=tmp_path / "does_not_exist.safetensors",
            )

    def test_unsupported_strategy_raises(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1)])
        with pytest.raises(ValueError, match="unsupported strategy"):
            Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path),
                strategy="merge",
            )

    def test_override_unknown_cluster_raises(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1)])
        with pytest.raises(ValueError, match="cluster_id"):
            Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path),
                representatives={5: 0},
            )

    def test_override_fid_not_in_cluster_raises(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1), (2, 3)])
        with pytest.raises(ValueError, match="not a member"):
            Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path),
                representatives={0: 7},
            )
