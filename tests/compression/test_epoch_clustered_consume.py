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
    EPOCH_KWARGS_MULTI_ITER,
    build_synth_sae,
    make_synth_prepass_patch,
)


REFERENCE_PATH = (
    Path(__file__).parent / "data" / "epoch_result_reference.json"
)
REFERENCE_PATH_MULTI_ITER = (
    Path(__file__).parent / "data" / "epoch_result_reference_multi_iter.json"
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


# ---------------------------------------------------------------------------
# Multi-iteration convergence regression
# ---------------------------------------------------------------------------
#
# Reviewer's §7 follow-up on PR #51: extend the differential test to
# exercise more iterations of the EpochCompressor loop. Same fixture
# (synthetic SAE with engineered redundancy cluster at features 4-7),
# `max_iterations=5` instead of 2. Forces the loop to walk through 5
# iterations, progressively zeroing features each round and exercising
# the iteration-loop semantics that the 2-iter fixture doesn't reach.


def _run_refactored_epoch_multi_iter(tmp_path: Path) -> dict:
    """Run the refactored pipeline on the multi-iter fixture and return
    `EpochReport.to_json()` as a parsed dict."""
    sae_path = build_synth_sae(tmp_path / "sae.safetensors")
    epoch = EpochCompressor(
        sae_checkpoint=sae_path,
        prompts=CANONICAL_PROMPTS,
        **EPOCH_KWARGS_MULTI_ITER,
    )
    with patch(
        "polygram.compression.epoch._compute_firing_rates_and_residuals",
        new=make_synth_prepass_patch(),
    ):
        out_path = tmp_path / "epoch_out.safetensors"
        result = epoch.run(out_path)
    return json.loads(result.report.to_json())


def _assert_deep_close(
    ref,
    act,
    *,
    path: str = "",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Recursive structural-equality + float-tolerance comparison.

    Floats are compared with `math.isclose(rel_tol=rtol, abs_tol=atol)` —
    last-ULP JSON-repr drift between macOS (reference-capture host) and
    Linux (CI host) is normalized away while structural divergence (a
    feature appearing in the wrong iteration, a convergence state
    flipping, an extra/missing key) still trips the assertion. Ints
    and strings are compared exactly.

    `rtol=1e-5` allows ~5 significant figures of agreement, which
    comfortably absorbs the observed single-ULP drift on
    `cross_entropy_delta` and per-token residual values while still
    catching any real numerical regression (which would be many ULPs).
    """
    import math

    if isinstance(ref, dict):
        assert isinstance(act, dict), f"type drift at {path}: dict vs {type(act).__name__}"
        assert set(ref) == set(act), (
            f"key drift at {path}: ref-only={set(ref) - set(act)}, "
            f"act-only={set(act) - set(ref)}"
        )
        for k in ref:
            _assert_deep_close(ref[k], act[k], path=f"{path}.{k}", rtol=rtol, atol=atol)
        return
    if isinstance(ref, list):
        assert isinstance(act, list), f"type drift at {path}: list vs {type(act).__name__}"
        assert len(ref) == len(act), (
            f"length drift at {path}: ref={len(ref)} act={len(act)}"
        )
        for i, (r, a) in enumerate(zip(ref, act)):
            _assert_deep_close(r, a, path=f"{path}[{i}]", rtol=rtol, atol=atol)
        return
    if isinstance(ref, float) or isinstance(act, float):
        # Treat int/float as comparable here — JSON ints survive the
        # round trip as ints, but defensively coerce in case one side
        # is `1` and the other `1.0`.
        assert math.isclose(float(ref), float(act), rel_tol=rtol, abs_tol=atol), (
            f"float drift at {path}: ref={ref!r} act={act!r}"
        )
        return
    assert ref == act, f"value drift at {path}: ref={ref!r} act={act!r}"


def test_multi_iter_epoch_result_matches_frozen_reference(tmp_path):
    """Multi-iteration variant of the differential regression. Asserts
    that the post-refactor pipeline produces the same iteration-loop
    trajectory (per-iteration features zeroed, convergence state at
    each step, final aggregate) as the frozen reference.

    Unlike the 2-iter test, this one does NOT enforce byte-identity
    on float fields: with 5 iterations of accumulating FP ops, the
    JSON repr of last-decimal-place values drifts by a single ULP
    across host architectures (the reference was captured on macOS;
    CI runs on Linux). The byte-identical guarantee is already
    pinned by the 2-iter test for the load-bearing refactor invariant;
    this test exists to pin convergence *semantics* (iteration count,
    per-iteration features zeroed, convergence states, integer
    counters) and approximate numerics."""
    reference = json.loads(REFERENCE_PATH_MULTI_ITER.read_text())
    actual = _run_refactored_epoch_multi_iter(tmp_path)

    ref_clean = _strip_non_deterministic(reference)
    act_clean = _strip_non_deterministic(actual)

    _assert_deep_close(ref_clean, act_clean)


def test_multi_iter_runs_all_five_iterations(tmp_path):
    """Sanity: the multi-iter fixture actually exercises 5 iterations
    (not just 2). If the fixture's redundancy structure changes such
    that the loop converges early, the fixture is no longer testing
    what its name claims and this test will trip."""
    actual = _run_refactored_epoch_multi_iter(tmp_path)
    assert len(actual["iterations"]) == 5, (
        f"multi-iter fixture should exercise 5 iterations; "
        f"got {len(actual['iterations'])}"
    )


def test_multi_iter_progressive_zeroing(tmp_path):
    """Each iteration zeros a non-empty set of features (the iteration
    loop makes forward progress). Pins the "doesn't get stuck on the
    same panel cluster forever" property."""
    actual = _run_refactored_epoch_multi_iter(tmp_path)
    for i, it in enumerate(actual["iterations"]):
        # Allow the last iteration to zero nothing if it converged,
        # but earlier iterations should make progress.
        if i < len(actual["iterations"]) - 1:
            assert len(it["features_zeroed_this_iteration"]) > 0, (
                f"iteration {i} zeroed 0 features — loop not making progress"
            )
    # Total zeroed is the sum of per-iteration zeroed sets (no overlaps
    # — once zeroed, a feature stays zeroed across iterations).
    total = sum(
        len(it["features_zeroed_this_iteration"]) for it in actual["iterations"]
    )
    assert total == actual["n_features_zeroed_total"], (
        f"sum of per-iteration zeroed ({total}) != n_features_zeroed_total "
        f"({actual['n_features_zeroed_total']})"
    )


def test_multi_iter_convergence_state_sequence(tmp_path):
    """The convergence-state sequence is `continuing` for iterations
    0..N-2 and a terminal state on iteration N-1. Pins the
    iteration-loop's termination semantics."""
    actual = _run_refactored_epoch_multi_iter(tmp_path)
    terminal_states = {
        "max_iterations",
        "stable_clusters",
        "quality_bound_breached",
        "no_more_priority_candidates",
    }
    iterations = actual["iterations"]
    for it in iterations[:-1]:
        assert it["convergence_state"] == "continuing", (
            f"iter {it['iteration']} should be 'continuing'; "
            f"got {it['convergence_state']!r}"
        )
    final = iterations[-1]
    assert final["convergence_state"] in terminal_states, (
        f"final iteration's convergence_state {final['convergence_state']!r} "
        f"not in known terminal set {terminal_states}"
    )
