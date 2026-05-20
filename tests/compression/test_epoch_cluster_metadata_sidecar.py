"""`EpochCompressor.run()` writes a `_compression_report.json` sidecar
alongside the compressed safetensors, mirroring the convention used by
`Compressor.apply()`.

Closes the Wave B blocker surfaced by sae-forge PR #69's §8.4 smoke —
without the sidecar, `FeatureBasis.from_polygram_checkpoint` reads
`n_clusters=0` (defensive default) and the polygram-clusters label
source refuses the basis.

See `openspec/changes/emit-cluster-metadata-from-epoch-compressor/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from polygram.compression.epoch import EpochCompressor, _final_cluster_assignments
from polygram.compression.epoch_report import EpochReport, SCHEMA_VERSION
from tests.compression._clustered_fixture import (
    EPOCH_KWARGS,
    CANONICAL_PROMPTS,
    build_synth_sae,
    make_synth_prepass_patch,
)


def _run(tmp_path: Path, epoch_kwargs: dict) -> tuple[Path, EpochReport]:
    """Run the canonical synth epoch pipeline; return (out_path, report)."""
    sae_path = build_synth_sae(tmp_path / "sae.safetensors")
    out_path = tmp_path / "epoch_out.safetensors"
    epoch = EpochCompressor(
        sae_checkpoint=sae_path,
        prompts=CANONICAL_PROMPTS,
        **epoch_kwargs,
    )
    with patch(
        "polygram.compression.epoch._compute_firing_rates_and_residuals",
        new=make_synth_prepass_patch(),
    ):
        result = epoch.run(out_path)
    return out_path, result.report


def test_sidecar_written_alongside_safetensors(tmp_path):
    out_path, _ = _run(tmp_path, EPOCH_KWARGS)
    sidecar = out_path.with_name(
        out_path.stem + "_compression_report.json"
    )
    assert out_path.exists(), "safetensors not written"
    assert sidecar.exists(), f"sidecar not written: {sidecar}"


def test_sidecar_parses_as_v4_epoch_report(tmp_path):
    out_path, _ = _run(tmp_path, EPOCH_KWARGS)
    sidecar = out_path.with_name(
        out_path.stem + "_compression_report.json"
    )
    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == 4
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "n_clusters" in payload
    assert "cluster_assignments" in payload


def test_sidecar_cluster_assignments_length_matches_n_features_input(tmp_path):
    out_path, report = _run(tmp_path, EPOCH_KWARGS)
    assert report.cluster_assignments is not None
    assert len(report.cluster_assignments) == report.n_features_input


def test_sidecar_n_clusters_matches_distinct_non_negative_ids(tmp_path):
    _, report = _run(tmp_path, EPOCH_KWARGS)
    assert report.cluster_assignments is not None
    expected = len({cid for cid in report.cluster_assignments if cid >= 0})
    assert report.n_clusters == expected


def test_sidecar_minus_one_assignments_are_subset_of_zeroed(tmp_path):
    """The invariant is one-way: `cluster_assignments[i] == -1` IMPLIES
    feature i is in the zeroed set. The converse doesn't have to hold —
    a zeroed feature can also be a member of a cluster in the final
    fingerprint (the cluster it was zeroed as part of).
    """
    _, report = _run(tmp_path, EPOCH_KWARGS)
    assert report.cluster_assignments is not None
    minus_ones = sum(1 for cid in report.cluster_assignments if cid == -1)
    # Every -1 must correspond to a zeroed feature; otherwise the
    # materialiser is mis-categorising surviving singletons.
    assert minus_ones <= report.n_features_zeroed_total, (
        f"Saw {minus_ones} -1 entries but only {report.n_features_zeroed_total} "
        f"features were zeroed. -1 must be a subset of zeroed."
    )


def test_sidecar_atomic_write_no_orphans_on_failure(tmp_path, monkeypatch):
    """If the sidecar JSON write fails (simulated via monkeypatch on
    os.replace targeting the sidecar), the temp file SHALL be cleaned
    up — no orphan .tmp files left around."""
    import os as _os

    sae_path = build_synth_sae(tmp_path / "sae.safetensors")
    out_path = tmp_path / "epoch_out.safetensors"
    epoch = EpochCompressor(
        sae_checkpoint=sae_path,
        prompts=CANONICAL_PROMPTS,
        **EPOCH_KWARGS,
    )

    real_replace = _os.replace
    call_count = {"n": 0}

    def replace_fail_on_sidecar(src, dst):
        # Two replaces happen: safetensors first, sidecar second.
        # Fail the second.
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated sidecar-write failure")
        return real_replace(src, dst)

    monkeypatch.setattr(_os, "replace", replace_fail_on_sidecar)

    with patch(
        "polygram.compression.epoch._compute_firing_rates_and_residuals",
        new=make_synth_prepass_patch(),
    ):
        with pytest.raises(OSError, match="simulated sidecar-write failure"):
            epoch.run(out_path)

    # What matters: the SIDECAR-specific temp file is cleaned up
    # (orphans of the form `.<stem>_compression_report.<rand>.tmp`).
    # Other per-iteration intermediates may leak depending on where
    # the failure lands — that's a separate concern, not regressed by
    # this change. Narrow the assertion to the sidecar.
    sidecar_orphans = [
        p for p in tmp_path.glob(".*tmp")
        if "_compression_report" in p.name
    ]
    assert not sidecar_orphans, (
        f"orphan sidecar tmp file leaked on failure: {sidecar_orphans}"
    )


# ---------------------------------------------------------------------------
# _final_cluster_assignments unit tests (the materialiser helper)
# ---------------------------------------------------------------------------


class TestFinalClusterAssignments:
    def test_simple_multi_cluster_plus_singleton_plus_zeroed(self):
        """Two multi-feature clusters, one surviving singleton, two
        zeroed features."""
        fingerprints = [
            frozenset({frozenset({0, 1, 2}), frozenset({3, 4})})
        ]
        result = _final_cluster_assignments(
            fingerprints, n_features_input=8, zeroed_set={6, 7},
        )
        # Features 0,1,2 → cluster 0; 3,4 → cluster 1; 5 → cluster 2
        # (singleton); 6, 7 → -1.
        assert result == (0, 0, 0, 1, 1, 2, -1, -1)

    def test_empty_fingerprints_all_zeroed(self):
        """max_iterations=0 case with everything zeroed (defensive)."""
        result = _final_cluster_assignments(
            [], n_features_input=4, zeroed_set={0, 1, 2, 3},
        )
        assert result == (-1, -1, -1, -1)

    def test_empty_fingerprints_all_survived(self):
        result = _final_cluster_assignments(
            [], n_features_input=4, zeroed_set=set(),
        )
        # Every surviving feature → its own singleton cluster.
        assert result == (0, 1, 2, 3)

    def test_deterministic_ordering_by_min_member(self):
        """Cluster ids assigned in order of ascending min member."""
        fingerprints = [
            frozenset({frozenset({3, 4}), frozenset({0, 1})}),
        ]
        result = _final_cluster_assignments(
            fingerprints, n_features_input=5, zeroed_set=set(),
        )
        # {0,1} has min=0 → cid 0; {3,4} has min=3 → cid 1;
        # feature 2 surviving singleton → cid 2.
        assert result == (0, 0, 2, 1, 1)

    def test_singletons_assigned_after_multi_clusters(self):
        """Surviving singletons get ids strictly greater than any
        multi-feature cluster id."""
        fingerprints = [
            frozenset({frozenset({0, 1})}),
        ]
        result = _final_cluster_assignments(
            fingerprints, n_features_input=5, zeroed_set={4},
        )
        # {0,1} → cid 0; 2,3 → cid 1,2 (singletons); 4 → -1
        assert result == (0, 0, 1, 2, -1)
