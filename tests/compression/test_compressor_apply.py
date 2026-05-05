"""End-to-end tests for `Compressor.apply()` / `Compressor.run()`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from polygram import Compressor
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _setup(tmp_path: Path, *, n_features: int = 8, d_model: int = 8):
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=d_model)
    return sae_path


class TestZeroPattern:
    def test_zeroed_rows_columns_and_biases(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        original = load_file(str(sae_path))
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (3, 4), (3, 5), (4, 5)],
            n_fires={1: 100, 5: 200},
        )
        out_path = tmp_path / "out.safetensors"
        result = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        new = load_file(str(out_path))

        # Cluster 0 = {0, 1}, rep = 1 (n_fires 100), zeroed = {0}.
        assert np.all(new["W_enc"][:, 0] == 0)
        assert new["b_enc"][0] == 0
        assert np.all(new["W_dec"][0, :] == 0)
        assert np.array_equal(new["W_enc"][:, 1], original["W_enc"][:, 1])
        assert np.array_equal(new["W_dec"][1, :], original["W_dec"][1, :])

        # Cluster 1 = {3, 4, 5}, rep = 5 (n_fires 200), zeroed = {3, 4}.
        for fid in (3, 4):
            assert np.all(new["W_enc"][:, fid] == 0)
            assert new["b_enc"][fid] == 0
            assert np.all(new["W_dec"][fid, :] == 0)
        assert np.array_equal(new["W_enc"][:, 5], original["W_enc"][:, 5])

        # Three features were zeroed in total.
        assert result.report.n_features_zeroed == 3
        assert result.report.n_features_kept == 2
        assert result.report.n_clusters == 2

    def test_singleton_features_untouched(self, tmp_path: Path):
        """Features outside every confirmed cluster pass through
        unchanged."""
        sae_path = _setup(tmp_path)
        original = load_file(str(sae_path))
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        new = load_file(str(out_path))
        for fid in (2, 3, 4, 5, 6, 7):
            assert np.array_equal(
                new["W_enc"][:, fid], original["W_enc"][:, fid]
            )
            assert np.array_equal(
                new["W_dec"][fid, :], original["W_dec"][fid, :]
            )

    def test_b_dec_untouched(self, tmp_path: Path):
        """`b_dec` is global, never feature-specific."""
        sae_path = _setup(tmp_path)
        original = load_file(str(sae_path))
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        new = load_file(str(out_path))
        assert np.array_equal(new["b_dec"], original["b_dec"])


class TestSourceImmutability:
    def test_source_bytes_unchanged_after_run(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        before = sae_path.read_bytes()
        report = build_report(
            n_features=8,
            confirmed=[(0, 1), (3, 4), (3, 5), (4, 5)],
        )
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        after = sae_path.read_bytes()
        assert before == after


class TestProvenanceFields:
    def test_sha256_fields_populated(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        result = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        rep = result.report
        assert len(rep.source_checkpoint_sha256) == 64
        assert len(rep.output_checkpoint_sha256) == 64
        assert rep.source_checkpoint_sha256 != rep.output_checkpoint_sha256
        assert rep.validation_report_dictionary_name == "FixtureDict"
        assert rep.validation_report_schema_version == 1
        assert rep.strategy == "zero"

    def test_rebuilt_dictionary_has_expected_features(
        self, tmp_path: Path
    ):
        sae_path = _setup(tmp_path)
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        result = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        # 8 features round-trip through from_sae_lens.
        assert len(result.dictionary.features) == 8
        assert result.dictionary.name == "FixtureDict"


class TestApplyRejections:
    def test_output_equals_source_raises(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = build_report(n_features=8, confirmed=[(0, 1)])
        compressor = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        )
        with pytest.raises(ValueError, match="must differ"):
            compressor.apply(output_checkpoint=sae_path)

    def test_apply_without_output_path_raises(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = build_report(n_features=8, confirmed=[(0, 1)])
        compressor = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        )
        with pytest.raises(ValueError, match="output_checkpoint is required"):
            compressor.apply()
