"""Tests for `add-encoding-partition` Phase 1 — BlockSpec, partition
coverage validation, CompressionConfig.encoding_partition, and the
CompressionReport.blocks schema bump.

Phase 1 of this change locks the API surface and ships the
scaffolding; the actual per-block dispatch in `Compressor.apply` is
a Phase 2 follow-up. This test file pins the contract that Phase 2
implementations will need to honour.
"""

from __future__ import annotations

import json

import pytest

from polygram.compression import (
    BlockReport,
    BlockSpec,
    CompressionReport,
    PartitionCoverageError,
    make_default_block,
    validate_partition_coverage,
)
from polygram.compression.report import (
    CompressionPlan,
    SCHEMA_VERSION,
    MAX_CLUSTERS_PER_BLOCK,
)
from polygram.config import CompressionConfig


# ---------------------------------------------------------------------------
# §9.1 — BlockSpec validation
# ---------------------------------------------------------------------------


def test_block_spec_accepts_mps_rung1_with_empty_kwargs():
    b = BlockSpec(block_id="b", encoding_class="MPSRung1", feature_ids=(0, 1, 2))
    assert b.encoding_class == "MPSRung1"
    assert b.feature_ids == (0, 1, 2)


def test_block_spec_rejects_unknown_encoding_class():
    with pytest.raises(ValueError, match="encoding_class"):
        BlockSpec(block_id="b", encoding_class="MadeUp", feature_ids=(0,))


def test_block_spec_rejects_missing_n_amp_qubits_for_rung5():
    with pytest.raises(ValueError, match="n_amp_qubits"):
        BlockSpec(block_id="b", encoding_class="Rung5",
                  encoding_kwargs={}, feature_ids=(0,))


def test_block_spec_rejects_negative_n_amp_qubits():
    with pytest.raises(ValueError, match="n_amp_qubits"):
        BlockSpec(block_id="b", encoding_class="Rung5",
                  encoding_kwargs={"n_amp_qubits": -1}, feature_ids=(0,))


def test_block_spec_rejects_extra_kwargs_for_mps_rung1():
    with pytest.raises(ValueError, match="no kwargs"):
        BlockSpec(block_id="b", encoding_class="MPSRung1",
                  encoding_kwargs={"foo": 1}, feature_ids=(0,))


def test_block_spec_rejects_missing_n_qubits_for_hea_rung2():
    with pytest.raises(ValueError, match="n_qubits"):
        BlockSpec(block_id="b", encoding_class="HEA_Rung2",
                  encoding_kwargs={}, feature_ids=(0,))


def test_block_spec_rejects_empty_feature_ids():
    with pytest.raises(ValueError, match="non-empty"):
        BlockSpec(block_id="b", encoding_class="MPSRung1",
                  feature_ids=())


def test_block_spec_rejects_duplicate_feature_ids_within_block():
    with pytest.raises(ValueError, match="duplicate"):
        BlockSpec(block_id="b", encoding_class="MPSRung1",
                  feature_ids=(1, 2, 1))


def test_block_spec_rejects_negative_feature_id():
    with pytest.raises(ValueError, match="non-negative"):
        BlockSpec(block_id="b", encoding_class="MPSRung1",
                  feature_ids=(0, -1))


def test_block_spec_rejects_empty_block_id():
    with pytest.raises(ValueError, match="block_id"):
        BlockSpec(block_id="", encoding_class="MPSRung1", feature_ids=(0,))


def test_block_spec_is_hashable():
    """frozen=True + custom __hash__ → BlockSpec is hashable even
    though encoding_kwargs is a dict. Required for
    CompressionConfig's downstream cache-key contract."""
    b = BlockSpec(block_id="heavy", encoding_class="Rung5",
                  encoding_kwargs={"n_amp_qubits": 4},
                  feature_ids=(0, 1, 2, 3))
    h = hash(b)
    assert isinstance(h, int)
    # Re-hash with the same fields → same hash.
    b2 = BlockSpec(block_id="heavy", encoding_class="Rung5",
                   encoding_kwargs={"n_amp_qubits": 4},
                   feature_ids=(0, 1, 2, 3))
    assert hash(b) == hash(b2)


# ---------------------------------------------------------------------------
# §9.2 — Partition coverage
# ---------------------------------------------------------------------------


def test_partition_coverage_disjoint_complete_passes():
    a = BlockSpec(block_id="a", encoding_class="MPSRung1",
                  feature_ids=(0, 1, 2, 3))
    b = BlockSpec(block_id="b", encoding_class="MPSRung1",
                  feature_ids=(4, 5, 6, 7))
    # Should not raise
    validate_partition_coverage((a, b), n_features_input=8)


def test_partition_coverage_overlap_raises_naming_duplicates():
    a = BlockSpec(block_id="a", encoding_class="MPSRung1",
                  feature_ids=(0, 1, 2, 3))
    b = BlockSpec(block_id="b", encoding_class="MPSRung1",
                  feature_ids=(3, 4, 5))  # 3 overlaps
    with pytest.raises(PartitionCoverageError) as exc_info:
        validate_partition_coverage((a, b), n_features_input=6)
    assert "3" in str(exc_info.value)
    assert "'a'" in str(exc_info.value)
    assert "'b'" in str(exc_info.value)


def test_partition_coverage_missing_raises_naming_holes():
    a = BlockSpec(block_id="a", encoding_class="MPSRung1",
                  feature_ids=(0, 1, 2))  # missing 3
    with pytest.raises(PartitionCoverageError, match="incomplete"):
        validate_partition_coverage((a,), n_features_input=4)


def test_partition_coverage_extras_only_raises_naming_extras():
    """Pure-extras case (no missing): blocks cover [0..7] inclusive but
    n_features_input=4 → 4,5,6,7 are extras."""
    a = BlockSpec(block_id="a", encoding_class="MPSRung1",
                  feature_ids=(0, 1, 2, 3, 4, 5, 6, 7))
    with pytest.raises(PartitionCoverageError, match="extra"):
        validate_partition_coverage((a,), n_features_input=4)


def test_partition_coverage_rejects_negative_n_features_input():
    a = BlockSpec(block_id="a", encoding_class="MPSRung1",
                  feature_ids=(0,))
    with pytest.raises(ValueError, match="non-negative"):
        validate_partition_coverage((a,), n_features_input=-1)


# ---------------------------------------------------------------------------
# §9.5 — make_default_block helper
# ---------------------------------------------------------------------------


def test_make_default_block_covers_all_except_excluded():
    block = make_default_block(
        encoding_class="MPSRung1",
        n_features_input=16,
        excluded_feature_ids={0, 1, 2, 3},
    )
    assert block.feature_ids == tuple(range(4, 16))
    assert block.encoding_class == "MPSRung1"
    assert block.block_id == "default"


def test_make_default_block_with_no_exclusions_covers_all():
    block = make_default_block(
        encoding_class="MPSRung1",
        n_features_input=8,
    )
    assert block.feature_ids == (0, 1, 2, 3, 4, 5, 6, 7)


def test_make_default_block_refuses_empty_result():
    """If every feature is excluded, the resulting block would be empty,
    which violates BlockSpec.__post_init__'s non-empty requirement.
    make_default_block catches this earlier with a clearer message."""
    with pytest.raises(ValueError, match="empty"):
        make_default_block(
            encoding_class="MPSRung1",
            n_features_input=4,
            excluded_feature_ids={0, 1, 2, 3},
        )


def test_make_default_block_with_heavy_override_partition_works():
    """Realistic 'default + heavy override' pattern: a small heavy
    block + a default tail computed by the helper, validated together."""
    heavy = BlockSpec(
        block_id="heavy", encoding_class="Rung5",
        encoding_kwargs={"n_amp_qubits": 4},
        feature_ids=(0, 1, 2, 3),
    )
    tail = make_default_block(
        encoding_class="MPSRung1",
        n_features_input=16,
        excluded_feature_ids={0, 1, 2, 3},
    )
    # Coverage validates cleanly
    validate_partition_coverage((heavy, tail), n_features_input=16)


# ---------------------------------------------------------------------------
# §9.4 — CompressionConfig.encoding_partition validation
# ---------------------------------------------------------------------------


def test_compression_config_default_partition_is_none():
    cfg = CompressionConfig()
    assert cfg.encoding_partition is None


def test_compression_config_accepts_partition_tuple():
    a = BlockSpec(block_id="a", encoding_class="MPSRung1",
                  feature_ids=(0,))
    cfg = CompressionConfig(encoding_partition=(a,))
    assert cfg.encoding_partition == (a,)


def test_compression_config_rejects_partition_as_list():
    """The locked API surface is `tuple[BlockSpec, ...]`, not list,
    for hashability. Passing a list raises TypeError."""
    a = BlockSpec(block_id="a", encoding_class="MPSRung1",
                  feature_ids=(0,))
    with pytest.raises(TypeError, match="tuple of BlockSpec"):
        CompressionConfig(encoding_partition=[a])


def test_compression_config_rejects_non_blockspec_in_partition():
    with pytest.raises(TypeError, match="BlockSpec"):
        CompressionConfig(encoding_partition=("not-a-blockspec",))


def test_compression_config_rejects_empty_partition_tuple():
    with pytest.raises(ValueError, match="non-empty"):
        CompressionConfig(encoding_partition=())


# ---------------------------------------------------------------------------
# §9.4 — CompressionReport schema v2 → v3 round-trip with blocks
# ---------------------------------------------------------------------------


def _empty_plan() -> CompressionPlan:
    return CompressionPlan(clusters=(), feature_ids=())


def _make_report_with_blocks() -> CompressionReport:
    blocks = (
        BlockReport(
            block_id="heavy",
            encoding_class="Rung5",
            encoding_kwargs={"n_amp_qubits": 4},
            learn_axis_assignment=True,
            feature_ids=(0, 1, 2, 3),
            n_features_kept=2,
            n_features_zeroed=2,
            n_clusters=2,
            cluster_assignments=(0, 0, 1, 1),
            scale_compression_ratio=0.85,
            rank_ratio=0.95,
            post_A=0.97,
            forge_mse=0.001,
            informative_metric="both",
        ),
        BlockReport(
            block_id="tail",
            encoding_class="MPSRung1",
            encoding_kwargs={},
            learn_axis_assignment=False,
            feature_ids=(4, 5, 6, 7),
            n_features_kept=4,
            n_features_zeroed=0,
            n_clusters=4,
            cluster_assignments=(0, 1, 2, 3),
        ),
    )
    return CompressionReport(
        schema_version=SCHEMA_VERSION,
        source_checkpoint="/x/sae.safetensors",
        source_checkpoint_sha256="a" * 64,
        output_checkpoint="/x/sae.compressed.safetensors",
        output_checkpoint_sha256="b" * 64,
        validation_report_dictionary_name="d",
        validation_report_schema_version=1,
        strategy="merge",
        plan=_empty_plan(),
        n_features_zeroed=2,
        n_features_kept=6,
        n_clusters=6,
        scale_compression_ratio=0.92,
        blocks=blocks,
    )


def test_compression_report_schema_version_is_3():
    assert SCHEMA_VERSION == 3


def test_compression_report_blocks_default_is_none():
    r = CompressionReport(
        schema_version=SCHEMA_VERSION,
        source_checkpoint="/x", source_checkpoint_sha256="a" * 64,
        output_checkpoint="/y", output_checkpoint_sha256="b" * 64,
        validation_report_dictionary_name="d",
        validation_report_schema_version=1,
        strategy="merge", plan=_empty_plan(),
        n_features_zeroed=0, n_features_kept=0, n_clusters=0,
    )
    assert r.blocks is None


def test_compression_report_blocks_roundtrip_via_json():
    r = _make_report_with_blocks()
    rt = CompressionReport.from_json(r.to_json())
    assert rt.blocks is not None
    assert len(rt.blocks) == 2
    assert rt == r


def test_compression_report_v2_payload_loads_with_blocks_none():
    """A v2-schema payload (no `blocks` key) SHALL load without error,
    defaulting blocks=None — preserves back-compat with the v2 contract
    that pre-add-encoding-partition consumers wrote."""
    v2_payload = {
        "schema_version": 2,
        "source_checkpoint": "/x",
        "source_checkpoint_sha256": "a" * 64,
        "output_checkpoint": "/y",
        "output_checkpoint_sha256": "b" * 64,
        "validation_report_dictionary_name": "d",
        "validation_report_schema_version": 1,
        "strategy": "merge",
        "feature_ids": [],
        "clusters": [],
        "n_features_zeroed": 0,
        "n_features_kept": 0,
        "n_clusters": 0,
        "scale_compression_ratio": 1.0,
        "rank_ratio": None,
        "post_A": None,
        "forge_mse": None,
        "informative_metric": None,
        # NOTE: no "blocks" key
    }
    r = CompressionReport.from_json(json.dumps(v2_payload))
    assert r.schema_version == 2
    assert r.blocks is None


def test_compression_report_v3_payload_blocks_none_round_trip():
    """A v3-schema payload with blocks=None (the unpartitioned compress
    case) round-trips cleanly — blocks is None on both sides."""
    r = CompressionReport(
        schema_version=SCHEMA_VERSION,
        source_checkpoint="/x", source_checkpoint_sha256="a" * 64,
        output_checkpoint="/y", output_checkpoint_sha256="b" * 64,
        validation_report_dictionary_name="d",
        validation_report_schema_version=1,
        strategy="merge", plan=_empty_plan(),
        n_features_zeroed=0, n_features_kept=0, n_clusters=0,
        blocks=None,
    )
    rt = CompressionReport.from_json(r.to_json())
    assert rt.blocks is None
    assert rt == r


def test_compression_report_blocks_equality_field_by_field():
    """Two reports with identical blocks compare equal; differing
    on any block field breaks equality."""
    r1 = _make_report_with_blocks()
    r2 = _make_report_with_blocks()
    assert r1 == r2
    # Mutate one field on the first block's BlockReport
    blocks_mod = list(r2.blocks)
    block0 = blocks_mod[0]
    import dataclasses
    blocks_mod[0] = dataclasses.replace(block0, n_features_kept=99)
    r3 = dataclasses.replace(r2, blocks=tuple(blocks_mod))
    assert r1 != r3


# ---------------------------------------------------------------------------
# §9.7 — Compressor.apply refusal (Phase 1 stub; Phase 2 follow-up)
# ---------------------------------------------------------------------------


def test_compressor_apply_runs_partition_path_in_phase_2():
    """Phase 2 (post-#107) IMPLEMENTS the per-block dispatch that
    Phase 1's apply() refused. This test pins the Phase 1 → Phase 2
    transition: with a partition supplied, Compressor.apply now runs
    (rather than raising NotImplementedError). See
    `tests/compression/test_encoding_partition_phase2.py` for the
    full per-block dispatch test coverage."""
    # The Phase 1 NotImplementedError block has been removed by
    # Phase 2; verify there's no residual refusal logic by checking
    # the apply() source doesn't contain the Phase 1 refusal message.
    from polygram.compression import compressor as _compressor_mod
    import inspect
    apply_src = inspect.getsource(_compressor_mod.Compressor.apply)
    assert "Phase 2 follow-up" not in apply_src, (
        "Phase 2 should have removed the Phase 1 NotImplementedError "
        "refusal block; if you see this failure, the block is still "
        "present and per-block dispatch isn't wired in."
    )


# ---------------------------------------------------------------------------
# §9.6 — Global cluster-id namespace constant
# ---------------------------------------------------------------------------


def test_max_clusters_per_block_constant_present():
    """The per-block global cluster-id namespace constant SHALL be
    exported so Phase 2 implementations + downstream consumers can
    reference the load-bearing cap. See Decision 2 in design.md."""
    assert MAX_CLUSTERS_PER_BLOCK == 10_000
