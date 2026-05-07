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
                strategy="not_a_strategy",
            )

    def test_unsupported_rep_selection_raises(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1)])
        with pytest.raises(ValueError, match="unsupported rep_selection"):
            Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path),
                rep_selection="bogus",
            )

    def test_unsupported_merge_mode_raises(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1)])
        with pytest.raises(ValueError, match="unsupported merge_mode"):
            Compressor(
                validation_report=report,
                sae_checkpoint=_checkpoint(tmp_path),
                merge_mode="kl_weighted",
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


# ---------------------------------------------------------------------------
# Tasks §4.3 — Compressor accepts CompressionConfig with override precedence.
# Note: Compressor's legacy defaults (strategy="zero", rep_selection="n_fires")
# differ from CompressionConfig's defaults, so a no-config call preserves the
# historical defaults; supplying a config switches in the config's values.
# ---------------------------------------------------------------------------


class TestCompressionConfigPassthrough:
    def test_no_config_preserves_legacy_defaults(self, tmp_path: Path):
        report = build_report(n_features=8, confirmed=[(0, 1)])
        c = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path),
        )
        assert c.strategy == "zero"
        assert c.rep_selection == "n_fires"
        assert c.merge_mode == "freq_weighted"

    def test_config_overrides_legacy_defaults(self, tmp_path: Path):
        from polygram import CompressionConfig

        report = build_report(n_features=8, confirmed=[(0, 1)])
        cfg = CompressionConfig(strategy="merge", rep_selection="scale_aware")
        c = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path),
            config=cfg,
        )
        assert c.strategy == "merge"
        assert c.rep_selection == "scale_aware"
        assert c.merge_mode == "freq_weighted"  # CompressionConfig default

    def test_per_field_kwarg_overrides_config(self, tmp_path: Path):
        from polygram import CompressionConfig

        report = build_report(n_features=8, confirmed=[(0, 1)])
        cfg = CompressionConfig(strategy="merge", rep_selection="scale_aware")
        c = Compressor(
            validation_report=report,
            sae_checkpoint=_checkpoint(tmp_path),
            config=cfg,
            strategy="zero",
        )
        # kwarg wins for strategy; config still supplies rep_selection.
        assert c.strategy == "zero"
        assert c.rep_selection == "scale_aware"
