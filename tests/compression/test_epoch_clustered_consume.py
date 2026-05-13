"""Differential regression for `compression-consumes-clustered-dictionary`.

`EpochCompressor.run` now internally builds a `ClusteredDictionary`
from `_select_panels`'s output and threads it through `_validate_panels`
and `_synthesize_validation_report` (which consume `ClusteredDictionary`
instead of `list[Panel]` after the refactor).

The contract: post-refactor `EpochCompressor.run` produces a
byte-identical `EpochResult` on the deterministic fixture compared
to the pre-refactor implementation, modulo a small set of explicitly
non-deterministic fields (tempfile paths, wall_seconds). The frozen
reference at `tests/compression/data/epoch_result_reference.json`
was captured on the pre-refactor code path.

If this test fails, the refactor has drifted from byte-identity —
investigate the diff, not the test.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from polygram import EpochCompressor

from tests.compression._clustered_fixture import (
    CANONICAL_PROMPTS,
    EPOCH_KWARGS,
    build_synth_sae,
    make_synth_prepass_patch,
)


REFERENCE_PATH = (
    Path(__file__).parent / "data" / "epoch_result_reference.json"
)


# Fields whose values vary across runs even when compression logic is
# identical. Excluded from the byte-equality check. See
# `tests/compression/data/README.md` for the full rationale.
NON_DETERMINISTIC_TOP_LEVEL = {
    "output_checkpoint",
    "source_checkpoint",
    "wall_seconds",
}
NON_DETERMINISTIC_PER_ITERATION = {
    "validation_report_paths",
}


def _strip_non_deterministic(payload: dict) -> dict:
    """Return a copy of `payload` with non-deterministic fields removed."""
    clean = {
        k: v
        for k, v in payload.items()
        if k not in NON_DETERMINISTIC_TOP_LEVEL
    }
    if "iterations" in clean:
        clean["iterations"] = [
            {k: v for k, v in it.items() if k not in NON_DETERMINISTIC_PER_ITERATION}
            for it in clean["iterations"]
        ]
    return clean


def _run_refactored_epoch(tmp_path: Path) -> dict:
    """Run the post-refactor pipeline on the deterministic fixture
    and return the resulting `EpochReport` as a JSON-parsed dict."""
    sae_path = build_synth_sae(tmp_path / "sae.safetensors")
    epoch = EpochCompressor(
        sae_checkpoint=sae_path,
        prompts=CANONICAL_PROMPTS,
        **EPOCH_KWARGS,
    )
    with patch(
        "polygram.compression.epoch._compute_firing_rates_and_residuals",
        new=make_synth_prepass_patch(),
    ):
        out_path = tmp_path / "epoch_out.safetensors"
        result = epoch.run(out_path)
    return json.loads(result.report.to_json())


# ---------------------------------------------------------------------------
# Load-bearing differential regression
# ---------------------------------------------------------------------------


def test_byte_identical_epoch_result_against_frozen_reference(tmp_path):
    """The load-bearing regression test for
    `compression-consumes-clustered-dictionary`. Re-runs the
    post-refactor pipeline on the deterministic fixture and asserts
    deterministic-field-equal vs the frozen pre-refactor reference."""
    reference = json.loads(REFERENCE_PATH.read_text())
    actual = _run_refactored_epoch(tmp_path)

    ref_clean = _strip_non_deterministic(reference)
    act_clean = _strip_non_deterministic(actual)

    if ref_clean != act_clean:
        # Diff-friendly assertion to make CI failure logs actionable.
        ref_pretty = json.dumps(ref_clean, indent=2, sort_keys=True)
        act_pretty = json.dumps(act_clean, indent=2, sort_keys=True)
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                ref_pretty.split("\n"),
                act_pretty.split("\n"),
                "reference",
                "actual",
                n=2,
            )
        )
        raise AssertionError(
            f"EpochResult drift detected — refactor is no longer "
            f"byte-identical to the frozen reference.\n\n{diff}"
        )


# ---------------------------------------------------------------------------
# Structural invariants on the per-iteration ClusteredDictionary
# ---------------------------------------------------------------------------


def test_per_iteration_clustered_dictionary_shape(tmp_path):
    """While we're running the refactored pipeline, instrument
    `_validate_panels` to capture the `ClusteredDictionary` argument
    and assert its structural invariants:

    - Every block has ≤ `MPSRung1.max_features` features.
    - `clustered.n_blocks == len(panels)` (the load-bearing ordering
      invariant from `from_compression_panels`).
    - Cross-block adjacency uses the canonical `bi < bj` ordering.
    """
    captured: list = []
    from polygram.compression import epoch as _epoch_mod

    original_validate = _epoch_mod.EpochCompressor._validate_panels

    def _capturing_validate(self, *, clustered, **kwargs):
        captured.append(clustered)
        return original_validate(self, clustered=clustered, **kwargs)

    with patch.object(
        _epoch_mod.EpochCompressor,
        "_validate_panels",
        _capturing_validate,
    ):
        _ = _run_refactored_epoch(tmp_path)

    assert captured, "no _validate_panels invocations captured"
    from polygram.encoding import MPSRung1

    cap = MPSRung1.max_features
    for cd in captured:
        for block in cd.blocks:
            assert len(block.features) <= cap, (
                f"block size {len(block.features)} exceeds cap {cap}"
            )
        for bi, _, bj, _ in cd.cross_block_pairs:
            assert bi < bj, (
                f"non-canonical cross-block ordering: bi={bi} bj={bj}"
            )


def test_validate_panels_signature_takes_clustered_kwarg():
    """`_validate_panels` now takes `clustered: ClusteredDictionary`
    as a keyword argument (renamed from `panels: list[Panel]`).
    This test pins the API change."""
    import inspect

    from polygram.compression.epoch import EpochCompressor

    sig = inspect.signature(EpochCompressor._validate_panels)
    assert "clustered" in sig.parameters
    assert "panels" not in sig.parameters


def test_synthesize_validation_report_signature():
    """`_synthesize_validation_report` now takes
    `(clustered, block_reports, sae_checkpoint)` instead of
    `(panels, per_panel_reports, sae_checkpoint)`."""
    import inspect

    from polygram.compression.epoch import _synthesize_validation_report

    sig = inspect.signature(_synthesize_validation_report)
    params = list(sig.parameters)
    assert params[0] == "clustered"
    assert params[1] == "block_reports"
    assert params[2] == "sae_checkpoint"
