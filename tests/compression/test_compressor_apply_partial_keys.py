"""`Compressor.apply()` accepts SAE checkpoints with partial key sets.

The strategy-dependent required-key set (W_dec for shipping strategies)
is strict-loaded; the optional-key set (W_enc / b_enc / b_dec) is
probed. The output safetensors mirrors the input's key set: a
W_dec-only input produces a W_dec-only output, with no placeholder
synthesis.

See `openspec/changes/compressor-partial-key-sae/specs/compressor-apply/spec.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from polygram import Compressor
from tests.compression._fixtures import build_report


def _write_partial_sae(
    path: Path,
    *,
    keys: set[str],
    n_features: int = 8,
    d_model: int = 8,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Write a synth SAE containing exactly ``keys``. Returns the
    dict written, so tests can compare against it."""
    rng = np.random.default_rng(seed)
    full = {
        "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32),
        "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32),
        "b_enc": np.zeros((n_features,), dtype=np.float32),
        "b_dec": np.zeros((d_model,), dtype=np.float32),
    }
    payload = {k: full[k] for k in keys}
    save_file(payload, str(path))
    return payload


class TestWDecOnlyInput:
    """Decoder-only inputs (sae-forge synth-basis case) are accepted."""

    def test_zero_strategy_compresses(self, tmp_path: Path):
        sae_path = tmp_path / "wdec_only.safetensors"
        original = _write_partial_sae(sae_path, keys={"W_dec"})
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        result = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            strategy="zero",
        ).run(out_path)
        assert result.output_checkpoint == out_path
        # Cluster {0, 1}, rep picked by n_fires_total (defaults make
        # higher fid win); the non-rep row is zeroed.
        loaded = load_file(str(out_path))
        assert set(loaded) == {"W_dec"}
        zeroed_rows = [
            i for i in range(8)
            if np.all(loaded["W_dec"][i] == 0)
            and not np.all(original["W_dec"][i] == 0)
        ]
        assert len(zeroed_rows) == 1, (
            f"expected exactly one row zeroed; got {zeroed_rows}"
        )

    def test_merge_strategy_compresses(self, tmp_path: Path):
        sae_path = tmp_path / "wdec_only_m.safetensors"
        _write_partial_sae(sae_path, keys={"W_dec"})
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            strategy="merge",
        ).run(out_path)
        loaded = load_file(str(out_path))
        assert set(loaded) == {"W_dec"}

    def test_output_mirrors_input_key_set(self, tmp_path: Path):
        """Mirror-the-input: W_dec in → W_dec out. No synthesised keys."""
        sae_path = tmp_path / "wdec_only_mirror.safetensors"
        _write_partial_sae(sae_path, keys={"W_dec"})
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        loaded = load_file(str(out_path))
        assert set(loaded) == {"W_dec"}
        for key in ("W_enc", "b_enc", "b_dec"):
            assert key not in loaded, (
                f"output should not synthesise {key!r}"
            )

    def test_rep_row_unchanged_other_zeroed(self, tmp_path: Path):
        """Functional correctness on the zero strategy's W_dec rows."""
        sae_path = tmp_path / "wdec_rep.safetensors"
        original = _write_partial_sae(sae_path, keys={"W_dec"})
        report = build_report(
            n_features=8,
            confirmed=[(0, 1)],
            n_fires={1: 999},  # rep = 1
        )
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        loaded = load_file(str(out_path))
        # Non-rep row zeroed:
        assert np.all(loaded["W_dec"][0] == 0)
        # Rep row preserved:
        assert np.array_equal(loaded["W_dec"][1], original["W_dec"][1])
        # Singletons untouched:
        for fid in range(2, 8):
            assert np.array_equal(
                loaded["W_dec"][fid], original["W_dec"][fid]
            )


class TestPartialInputs:
    """Inputs with some-but-not-all optional keys still compress."""

    def test_wdec_plus_wenc_no_biases(self, tmp_path: Path):
        sae_path = tmp_path / "wdec_wenc.safetensors"
        original = _write_partial_sae(
            sae_path, keys={"W_dec", "W_enc"}
        )
        report = build_report(
            n_features=8,
            confirmed=[(0, 1)],
            n_fires={1: 999},
        )
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        loaded = load_file(str(out_path))
        assert set(loaded) == {"W_dec", "W_enc"}
        # W_dec row 0 zeroed; W_enc col 0 zeroed; rep cols/rows intact.
        assert np.all(loaded["W_dec"][0] == 0)
        assert np.all(loaded["W_enc"][:, 0] == 0)
        assert np.array_equal(
            loaded["W_enc"][:, 1], original["W_enc"][:, 1]
        )


class TestFullSaeByteEquivalence:
    """Full-SAE inputs continue to work — partial-key support is
    purely additive."""

    def test_full_input_full_output(self, tmp_path: Path):
        sae_path = tmp_path / "full.safetensors"
        _write_partial_sae(
            sae_path, keys={"W_dec", "W_enc", "b_enc", "b_dec"}
        )
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(out_path)
        loaded = load_file(str(out_path))
        # All four keys round-trip.
        assert set(loaded) == {"W_dec", "W_enc", "b_enc", "b_dec"}


class TestMissingRequiredKey:
    """Missing W_dec — the only required key — still raises clearly."""

    def test_only_encoder_inputs_raise(self, tmp_path: Path):
        """No W_dec → `_load_sae_checkpoint` raises with the focused
        error (the optional-key probe does not protect required keys)."""
        sae_path = tmp_path / "wenc_only.safetensors"
        _write_partial_sae(sae_path, keys={"W_enc"})
        report = build_report(n_features=8, confirmed=[(0, 1)])
        out_path = tmp_path / "out.safetensors"
        with pytest.raises(ValueError, match="W_dec"):
            Compressor(
                validation_report=report, sae_checkpoint=sae_path
            ).run(out_path)
