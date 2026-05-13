"""Smoke tests for `epoch-compressor-configurable-encoding`.

Verifies that `EpochCompressor(encoding=Rung3())` actually engages
the new path: panels of >8 features (proving the neighbour cap
scales), panels capped at 16 features (proving the cap caps), and
that the run completes successfully on a synthetic Rung3-sized
fixture.

The default encoding=None / encoding=MPSRung1() byte-identity is
locked separately in `test_epoch_clustered_consume.py`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from polygram import EpochCompressor

from tests.compression._rung3_fixture import (
    CANONICAL_PROMPTS,
    EPOCH_KWARGS,
    build_rung3_synth_sae,
    make_rung3_synth_prepass_patch,
)


def _run_epoch(tmp_path: Path, encoding):
    """Run the configured `EpochCompressor` on the Rung3 fixture.
    Returns the captured per-iteration `ClusteredDictionary` views
    plus the `EpochResult`."""
    sae_path = build_rung3_synth_sae(tmp_path / "sae.safetensors")
    epoch = EpochCompressor(
        sae_checkpoint=sae_path,
        prompts=CANONICAL_PROMPTS,
        encoding=encoding,
        **EPOCH_KWARGS,
    )
    captured: list = []
    from polygram.compression import epoch as _epoch_mod

    original_validate = _epoch_mod.EpochCompressor._validate_panels

    def _capturing_validate(self, *, clustered, **kwargs):
        captured.append(clustered)
        return original_validate(self, clustered=clustered, **kwargs)

    with patch(
        "polygram.compression.epoch._compute_firing_rates_and_residuals",
        new=make_rung3_synth_prepass_patch(),
    ), patch.object(
        _epoch_mod.EpochCompressor,
        "_validate_panels",
        _capturing_validate,
    ):
        out_path = tmp_path / "epoch_out.safetensors"
        result = epoch.run(out_path)
    return captured, result


def test_rung3_run_produces_panels_larger_than_eight(tmp_path):
    """The Rung3 fixture has two engineered redundancy clusters of
    10 features. With `encoding=Rung3()` (`max_features=16`), at
    least one returned panel must have more than 8 features —
    proving the scaled neighbour cap (`max_panel_size - 1 = 15`)
    actually engages."""
    from polygram.encoding import Rung3

    captured, _ = _run_epoch(tmp_path, Rung3())

    assert captured, "no _validate_panels invocations captured"
    largest = max(
        len(block.features)
        for cd in captured
        for block in cd.blocks
    )
    assert largest > 8, (
        f"Rung3 fixture's largest panel only had {largest} features "
        f"— the new neighbour cap didn't engage. Expected >8."
    )


def test_rung3_run_respects_max_features_cap(tmp_path):
    """Every block in every per-iteration ClusteredDictionary has
    at most `Rung3.max_features` features. Pins the cap-caps
    invariant."""
    from polygram.encoding import Rung3

    captured, _ = _run_epoch(tmp_path, Rung3())

    assert captured
    cap = Rung3.max_features
    for cd in captured:
        for block in cd.blocks:
            assert len(block.features) <= cap, (
                f"block size {len(block.features)} exceeds "
                f"Rung3.max_features={cap}"
            )


def test_rung3_run_zeros_features(tmp_path):
    """Sanity check: the Rung3 fixture is engineered with enough
    redundancy that compression should actually zero at least one
    feature. If this trips, the synthetic SAE has drifted to the
    point where compression no longer finds any confirmed pairs —
    revisit the fixture."""
    from polygram.encoding import Rung3

    _, result = _run_epoch(tmp_path, Rung3())
    assert result.report.n_features_zeroed_total > 0, (
        "Rung3 run zeroed 0 features — fixture redundancy may be "
        "insufficient for confirmation under the configured "
        "validator thresholds."
    )


def test_rung3_run_carries_encoding_through_blocks(tmp_path):
    """Every block's Dictionary reports `encoding=Rung3()`. Pins
    that `EpochCompressor.run` actually threads `self.encoding`
    through `ClusteredDictionary.from_compression_panels`."""
    from polygram.encoding import Rung3

    captured, _ = _run_epoch(tmp_path, Rung3())

    assert captured
    for cd in captured:
        for block in cd.blocks:
            assert isinstance(block.encoding, Rung3), (
                f"block.encoding is "
                f"{type(block.encoding).__name__}, expected Rung3"
            )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_encoding_without_max_features_rejected(tmp_path):
    """Passing an object that doesn't expose `max_features` as an
    integer is rejected at construction time."""
    sae_path = build_rung3_synth_sae(tmp_path / "sae.safetensors")

    class _BadEncoding:
        pass

    with pytest.raises(ValueError, match="max_features"):
        EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=CANONICAL_PROMPTS,
            encoding=_BadEncoding(),
            **EPOCH_KWARGS,
        )


def test_encoding_with_degenerate_max_features_rejected(tmp_path):
    """`max_features < 2` is rejected (panels of size 1 are
    degenerate; the cosine-pair coverage logic assumes ≥2-feature
    panels)."""
    sae_path = build_rung3_synth_sae(tmp_path / "sae.safetensors")

    class _OneFeatureEncoding:
        max_features = 1

    with pytest.raises(ValueError, match="max_features must be >= 2"):
        EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=CANONICAL_PROMPTS,
            encoding=_OneFeatureEncoding(),
            **EPOCH_KWARGS,
        )
