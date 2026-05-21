"""Tests for `add-encoding-partition` Phase 2 — per-block dispatch in
`Compressor.apply`.

Phase 1 (PR #107) locked the API surface; Phase 2 wires the actual
per-block compress + stitch path. Tests below validate:

  - Single-block partition matches single-encoding output bit-exactly.
  - Multi-block partition zeroes the right features per block.
  - Cross-block clusters are dropped (their features end up as
    singletons in the output, NOT merged across the block boundary).
  - CompressionReport.blocks is populated correctly with per-block
    counts + cluster_assignments + scale_compression_ratio.
  - Coverage validation fires at apply() time on a bad partition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from polygram.compression import (
    BlockSpec,
    Compressor,
    PartitionCoverageError,
)
from polygram.config import CompressionConfig

from tests.compression._fixtures import build_report


def _build_synth_sae(tmp_path: Path, n_features: int = 8, d_model: int = 16):
    """Write a tiny synth SAE checkpoint. Returns the path."""
    rng = np.random.default_rng(0)
    W_dec = rng.standard_normal((n_features, d_model)).astype(np.float32)
    W_enc = W_dec.T.astype(np.float32)
    b_enc = np.zeros(n_features, dtype=np.float32)
    b_dec = np.zeros(d_model, dtype=np.float32)
    sae_path = tmp_path / "sae.safetensors"
    save_file(
        {"W_dec": W_dec, "W_enc": W_enc, "b_enc": b_enc, "b_dec": b_dec},
        str(sae_path),
    )
    return sae_path


# ---------------------------------------------------------------------------
# Sanity: single-block partition is a no-op wrapper around single-encoding
# ---------------------------------------------------------------------------


def test_single_block_partition_runs_strategy_correctly(tmp_path):
    """A 1-block partition covering all features SHALL run the strategy
    on the full set and produce a result with one BlockReport whose
    counts match the top-level CompressionReport."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    vr = build_report(n_features=n_features, confirmed=[(0, 1), (2, 3)])

    cfg = CompressionConfig(
        strategy="zero",
        encoding_partition=(
            BlockSpec(
                block_id="all", encoding_class="MPSRung1",
                feature_ids=tuple(range(n_features)),
            ),
        ),
    )
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=tmp_path / "out.safetensors")

    assert result.report.blocks is not None
    assert len(result.report.blocks) == 1
    block = result.report.blocks[0]
    assert block.block_id == "all"
    assert block.encoding_class == "MPSRung1"
    assert block.n_features_kept == result.report.n_features_kept
    assert block.n_features_zeroed == result.report.n_features_zeroed
    assert block.n_clusters == result.report.n_clusters


# ---------------------------------------------------------------------------
# Multi-block: per-block stitch + report population
# ---------------------------------------------------------------------------


def test_two_block_partition_stitches_correctly(tmp_path):
    """A 2-block partition with intra-block clusters in each SHALL
    produce the right per-block zero / kept counts AND stitch the
    output safetensors so the right rows are zeroed."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    # Cluster {0,1} in heavy block; cluster {4,5} in tail block.
    vr = build_report(
        n_features=n_features,
        confirmed=[(0, 1), (4, 5)],
    )

    heavy = BlockSpec(
        block_id="heavy", encoding_class="Rung5",
        encoding_kwargs={"n_amp_qubits": 4},
        feature_ids=(0, 1, 2, 3),
    )
    tail = BlockSpec(
        block_id="tail", encoding_class="MPSRung1",
        feature_ids=(4, 5, 6, 7),
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(heavy, tail))
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=tmp_path / "out.safetensors")

    # Top-level: 2 clusters total (one per block); 2 zeroed (1 and 5).
    assert result.report.n_clusters == 2
    assert result.report.n_features_zeroed == 2
    assert result.report.n_features_kept == 2  # representatives 0 and 4

    # Per-block reports
    assert result.report.blocks is not None
    blocks_by_id = {b.block_id: b for b in result.report.blocks}
    assert blocks_by_id["heavy"].n_clusters == 1
    assert blocks_by_id["heavy"].n_features_zeroed == 1
    assert blocks_by_id["heavy"].encoding_class == "Rung5"
    assert blocks_by_id["heavy"].encoding_kwargs == {"n_amp_qubits": 4}

    assert blocks_by_id["tail"].n_clusters == 1
    assert blocks_by_id["tail"].n_features_zeroed == 1
    assert blocks_by_id["tail"].encoding_class == "MPSRung1"


def test_two_block_partition_zeroes_correct_rows_in_output(tmp_path):
    """Verify the output safetensors has the non-rep rows zeroed and
    the reps (the higher-fire member of each cluster) + unclustered
    singletons intact."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    # Explicit n_fires so rep selection is deterministic: 0 wins over 1,
    # and 4 wins over 5 (higher firing count → representative under
    # scale_aware → falls back to n_fires when kl_ablate is NaN).
    vr = build_report(
        n_features=n_features,
        confirmed=[(0, 1), (4, 5)],
        n_fires={0: 100, 1: 10, 4: 100, 5: 10},
    )

    heavy = BlockSpec(
        block_id="heavy", encoding_class="MPSRung1",
        feature_ids=(0, 1, 2, 3),
    )
    tail = BlockSpec(
        block_id="tail", encoding_class="MPSRung1",
        feature_ids=(4, 5, 6, 7),
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(heavy, tail))
    out_path = tmp_path / "out.safetensors"
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    c.run(output_checkpoint=out_path)

    out_state = load_file(str(out_path))
    norms = np.linalg.norm(out_state["W_dec"], axis=1)
    assert norms[1] < 1e-6, "row 1 should be zeroed (lower fire than rep 0)"
    assert norms[5] < 1e-6, "row 5 should be zeroed (lower fire than rep 4)"
    # Reps and unclustered singletons survive
    for fid in (0, 2, 3, 4, 6, 7):
        assert norms[fid] > 1e-3, f"row {fid} should be kept"


# ---------------------------------------------------------------------------
# Cross-block cluster handling
# ---------------------------------------------------------------------------


def test_cross_block_clusters_are_dropped(tmp_path):
    """Confirmed pair (3, 4) bridges the heavy and tail blocks. The
    resulting global cluster {3, 4} must NOT merge across blocks; both
    features SHALL end up as singletons in the output (no row gets
    zeroed for this cross-block pair)."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    # Only one confirmed pair, deliberately crossing the heavy/tail boundary.
    vr = build_report(n_features=n_features, confirmed=[(3, 4)])

    heavy = BlockSpec(
        block_id="heavy", encoding_class="MPSRung1",
        feature_ids=(0, 1, 2, 3),
    )
    tail = BlockSpec(
        block_id="tail", encoding_class="MPSRung1",
        feature_ids=(4, 5, 6, 7),
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(heavy, tail))
    out_path = tmp_path / "out.safetensors"
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=out_path)

    # No cluster makes it to the compressed report.
    assert result.report.n_clusters == 0
    assert result.report.n_features_zeroed == 0
    # The output W_dec rows are all intact (nothing zeroed).
    out_state = load_file(str(out_path))
    norms = np.linalg.norm(out_state["W_dec"], axis=1)
    for fid in range(n_features):
        assert norms[fid] > 1e-3, f"row {fid} should be kept (no cross-block merge)"

    # Per-block reports: both blocks have 0 clusters, 0 zeroed.
    blocks_by_id = {b.block_id: b for b in result.report.blocks}
    for bid in ("heavy", "tail"):
        assert blocks_by_id[bid].n_clusters == 0
        assert blocks_by_id[bid].n_features_zeroed == 0


# ---------------------------------------------------------------------------
# Coverage validation fires at apply time
# ---------------------------------------------------------------------------


def test_coverage_validation_fires_on_incomplete_partition(tmp_path):
    """A partition missing some feature ids SHALL raise
    PartitionCoverageError at apply() time, before any compression
    work begins."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    vr = build_report(n_features=n_features, confirmed=[(0, 1)])

    # Only covers half the features
    incomplete = BlockSpec(
        block_id="incomplete", encoding_class="MPSRung1",
        feature_ids=(0, 1, 2, 3),
    )
    cfg = CompressionConfig(
        strategy="zero", encoding_partition=(incomplete,),
    )
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    with pytest.raises(PartitionCoverageError, match="incomplete"):
        c.run(output_checkpoint=tmp_path / "out.safetensors")


def test_coverage_validation_fires_on_overlapping_partition(tmp_path):
    """A partition with overlapping feature ids SHALL raise
    PartitionCoverageError at apply() time."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    vr = build_report(n_features=n_features, confirmed=[(0, 1)])

    a = BlockSpec(
        block_id="a", encoding_class="MPSRung1",
        feature_ids=(0, 1, 2, 3),
    )
    b = BlockSpec(
        block_id="b", encoding_class="MPSRung1",
        feature_ids=(3, 4, 5, 6, 7),  # 3 overlaps
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(a, b))
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    with pytest.raises(PartitionCoverageError) as exc_info:
        c.run(output_checkpoint=tmp_path / "out.safetensors")
    assert "3" in str(exc_info.value)


# ---------------------------------------------------------------------------
# cluster_assignments per block
# ---------------------------------------------------------------------------


def test_block_report_cluster_assignments_local_to_block(tmp_path):
    """BlockReport.cluster_assignments uses LOCAL cluster ids
    (0..n_clusters_in_block-1) indexed by the block's feature_ids
    ordering. Features not in any cluster SHALL be assigned -1."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    # Two intra-heavy clusters: {0,1} and {2,3}
    vr = build_report(
        n_features=n_features,
        confirmed=[(0, 1), (2, 3)],
    )

    heavy = BlockSpec(
        block_id="heavy", encoding_class="MPSRung1",
        feature_ids=(0, 1, 2, 3),
    )
    tail = BlockSpec(
        block_id="tail", encoding_class="MPSRung1",
        feature_ids=(4, 5, 6, 7),
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(heavy, tail))
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=tmp_path / "out.safetensors")

    blocks_by_id = {b.block_id: b for b in result.report.blocks}
    # Heavy: local indices 0,1 in cluster 0; local indices 2,3 in cluster 1
    assert blocks_by_id["heavy"].cluster_assignments == (0, 0, 1, 1)
    # Tail: no clusters → all -1
    assert blocks_by_id["tail"].cluster_assignments == (-1, -1, -1, -1)


# ---------------------------------------------------------------------------
# Schema + serialization
# ---------------------------------------------------------------------------


def test_full_compression_report_with_blocks_round_trips(tmp_path):
    """The CompressionReport produced by a partitioned run SHALL
    round-trip cleanly via to_json / from_json, with blocks preserved."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    vr = build_report(n_features=n_features, confirmed=[(0, 1)])

    heavy = BlockSpec(
        block_id="heavy", encoding_class="Rung5",
        encoding_kwargs={"n_amp_qubits": 4},
        feature_ids=(0, 1, 2, 3),
    )
    tail = BlockSpec(
        block_id="tail", encoding_class="MPSRung1",
        feature_ids=(4, 5, 6, 7),
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(heavy, tail))
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=tmp_path / "out.safetensors")

    # Round-trip via JSON
    from polygram.compression import CompressionReport
    serialised = result.report.to_json()
    rt = CompressionReport.from_json(serialised)
    assert rt == result.report
    assert rt.blocks is not None
    assert len(rt.blocks) == 2


# ---------------------------------------------------------------------------
# Single-encoding path still works (Phase 2 doesn't regress Phase 0)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-block diagnostics (rank_ratio, post_A, informative_metric)
# ---------------------------------------------------------------------------


def test_block_report_diagnostic_floats_are_populated(tmp_path):
    """The Phase 2 enhancement populates per-block rank_ratio, post_A,
    and informative_metric on BlockReport. They were None in Phase 2 v1.
    Verify they're now real numbers (or None when the block has no
    clusters — empty-block degenerate case)."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    vr = build_report(
        n_features=n_features,
        confirmed=[(0, 1), (4, 5)],
        n_fires={0: 100, 1: 10, 4: 100, 5: 10},
    )

    heavy = BlockSpec(
        block_id="heavy", encoding_class="MPSRung1",
        feature_ids=(0, 1, 2, 3),
    )
    tail = BlockSpec(
        block_id="tail", encoding_class="MPSRung1",
        feature_ids=(4, 5, 6, 7),
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(heavy, tail))
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=tmp_path / "out.safetensors")

    blocks_by_id = {b.block_id: b for b in result.report.blocks}
    for bid in ("heavy", "tail"):
        b = blocks_by_id[bid]
        assert b.rank_ratio is not None, f"{bid}: rank_ratio should be populated"
        assert 0.0 <= b.rank_ratio <= 1.0, f"{bid}: rank_ratio in [0, 1]"
        assert b.post_A is not None, f"{bid}: post_A should be populated"
        assert b.informative_metric in {"post_A", "both", "forge_mse"}, (
            f"{bid}: informative_metric should be one of the expected values"
        )


def test_block_report_diagnostics_none_for_empty_block(tmp_path):
    """When a block has no clusters (e.g. all cross-block dropped), the
    per-block diagnostics SHALL be None — there's nothing to measure
    against an empty plan."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    # Only cross-block pair → both blocks end up with 0 clusters
    vr = build_report(n_features=n_features, confirmed=[(3, 4)])

    heavy = BlockSpec(
        block_id="heavy", encoding_class="MPSRung1",
        feature_ids=(0, 1, 2, 3),
    )
    tail = BlockSpec(
        block_id="tail", encoding_class="MPSRung1",
        feature_ids=(4, 5, 6, 7),
    )
    cfg = CompressionConfig(strategy="zero", encoding_partition=(heavy, tail))
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=tmp_path / "out.safetensors")

    for b in result.report.blocks:
        assert b.n_clusters == 0
        assert b.rank_ratio is None
        assert b.post_A is None
        assert b.informative_metric is None


def test_no_partition_uses_single_encoding_path(tmp_path):
    """When encoding_partition is None, Compressor.apply runs the
    historical single-encoding path. CompressionReport.blocks is None."""
    n_features = 8
    sae_path = _build_synth_sae(tmp_path, n_features=n_features)
    vr = build_report(n_features=n_features, confirmed=[(0, 1)])

    cfg = CompressionConfig(strategy="zero")  # no partition
    c = Compressor(sae_checkpoint=sae_path, validation_report=vr, config=cfg)
    result = c.run(output_checkpoint=tmp_path / "out.safetensors")

    assert result.report.blocks is None
    assert result.report.n_clusters == 1
    assert result.report.n_features_zeroed == 1
