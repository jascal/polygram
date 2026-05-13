"""SAE import — fixture loading, selection, clustering, fidelity report."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from polygram import (
    SAEFeatureRecord,
    SelectionReport,
    from_sae_lens,
    load_toy_sae,
)

FIXTURE = Path(__file__).parent / "fixtures" / "toy_sae.json"


def test_fixture_loads_clean():
    records = load_toy_sae(FIXTURE)
    assert len(records) == 16
    for rec in records.values():
        assert isinstance(rec, SAEFeatureRecord)
        assert rec.projection.shape == (8,)
        assert rec.label and "/" in rec.label
        assert rec.activation_mean is not None
        assert rec.activation_std is not None


def test_record_rejects_2d_projection():
    with pytest.raises(ValueError, match="must be 1D"):
        SAEFeatureRecord(
            feature_id=0, name="x", projection=np.zeros((2, 2))
        )


def test_record_rejects_nan_projection():
    with pytest.raises(ValueError, match="non-finite"):
        SAEFeatureRecord(
            feature_id=0, name="x",
            projection=np.array([1.0, np.nan, 0.0]),
        )


def test_select_too_many_features_rejected():
    records = load_toy_sae(FIXTURE)
    with pytest.raises(ValueError, match="caps a Dictionary at 8"):
        from_sae_lens(records, list(range(9)))


def test_select_empty_rejected():
    records = load_toy_sae(FIXTURE)
    with pytest.raises(ValueError, match="empty"):
        from_sae_lens(records, [])


def test_select_unknown_id_rejected():
    records = load_toy_sae(FIXTURE)
    with pytest.raises(ValueError, match="not in records"):
        from_sae_lens(records, [0, 999])


def test_explicit_cluster_assignments_honored():
    records = load_toy_sae(FIXTURE)
    d, report = from_sae_lens(
        records, [0, 1, 4, 5],
        cluster_assignments={0: "A", 1: "A", 4: "B", 5: "B"},
        name="Custom",
    )
    assert report.cluster_method == "user"
    assert d.hierarchy == {
        "A": ["dog_poodle", "dog_beagle"],
        "B": ["hawk_red", "hawk_cooper"],
    }
    assert d.name == "Custom"


def test_user_assignments_must_cover_selection():
    records = load_toy_sae(FIXTURE)
    with pytest.raises(ValueError, match="missing entry"):
        from_sae_lens(
            records, [0, 1, 4, 5],
            cluster_assignments={0: "A", 1: "A"},
        )


def test_from_labels_path():
    records = load_toy_sae(FIXTURE)
    d, report = from_sae_lens(records, [0, 1, 4, 5])
    assert report.cluster_method == "from_labels"
    assert set(d.hierarchy.keys()) == {"mammals", "birds"}
    assert d.hierarchy["mammals"] == ["dog_poodle", "dog_beagle"]
    assert d.hierarchy["birds"] == ["hawk_red", "hawk_cooper"]


def test_kmeans_default_separates_toy_fixture():
    records = load_toy_sae(FIXTURE)
    no_labels = {k: replace(v, label=None) for k, v in records.items()}
    d, report = from_sae_lens(no_labels, [0, 1, 4, 5], n_clusters=2)
    assert report.cluster_method == "kmeans"
    assert len(d.hierarchy) == 2
    by_cluster = {tuple(sorted(v)): k for k, v in d.hierarchy.items()}
    assert ("dog_beagle", "dog_poodle") in by_cluster
    assert ("hawk_cooper", "hawk_red") in by_cluster
    assert report.beta_variance_explained > 0.9


def test_beta_variance_in_unit_interval():
    records = load_toy_sae(FIXTURE)
    _, report = from_sae_lens(records, [0, 1, 4, 5])
    assert 0.0 <= report.beta_variance_explained <= 1.0


def test_identical_projections_var_explained_is_one():
    records = {
        i: SAEFeatureRecord(
            feature_id=i, name=f"f{i}",
            projection=np.array([1.0, 0.0, 0.0]),
            label=f"{'A' if i < 2 else 'B'}/x",
        )
        for i in range(4)
    }
    _, report = from_sae_lens(records, [0, 1, 2, 3])
    assert abs(report.beta_variance_explained - 1.0) < 1e-9


def test_warning_on_overspecified_n_clusters():
    records = load_toy_sae(FIXTURE)
    no_labels = {k: replace(v, label=None) for k, v in records.items()}
    _, report = from_sae_lens(no_labels, [0, 1], n_clusters=4)
    assert any("clamping" in w for w in report.warnings)


def test_returned_dictionary_is_valid_and_grams():
    records = load_toy_sae(FIXTURE)
    d, _ = from_sae_lens(records, [0, 1, 4, 5])
    g = d.gram()
    assert g.shape == (4, 4)
    np.testing.assert_allclose(np.abs(np.diag(g)), 1.0, atol=1e-9)


def test_report_records_input_count():
    records = load_toy_sae(FIXTURE)
    _, report = from_sae_lens(records, [0, 1, 4, 5])
    assert report.n_input_features == 16
    assert report.n_selected == 4
    assert isinstance(report, SelectionReport)


def test_assign_gamma_writes_nonzero_gammas():
    records = load_toy_sae(FIXTURE)
    d, report = from_sae_lens(records, [0, 1, 4, 5], assign_gamma=True)
    assert report.gamma_method == "projection_pca"
    gammas = [f.gamma for f in d.features]
    assert any(abs(g) > 1e-9 for g in gammas)
    assert all(-0.25 - 1e-12 <= g <= 0.25 + 1e-12 for g in gammas)


def test_default_assign_gamma_now_true():
    # ``assign_gamma`` default flipped from False → True per the
    # polygram-tuning-config change. Matches the README guidance that
    # γ=0 is "almost always wrong" on real SAEs.
    records = load_toy_sae(FIXTURE)
    d, report = from_sae_lens(records, [0, 1, 4, 5])
    assert report.gamma_method == "projection_pca"
    assert any(abs(f.gamma) > 1e-9 for f in d.features)


def test_explicit_assign_gamma_false_keeps_legacy_behaviour():
    # Pinning the pre-change behaviour: callers who genuinely want γ=0
    # pass ``assign_gamma=False`` explicitly.
    records = load_toy_sae(FIXTURE)
    d, report = from_sae_lens(records, [0, 1, 4, 5], assign_gamma=False)
    assert report.gamma_method == "zero"
    assert all(f.gamma == 0.0 for f in d.features)


def test_reconstruction_error_per_feature():
    records = load_toy_sae(FIXTURE)
    _, report = from_sae_lens(records, [0, 1, 4, 5])
    assert set(report.reconstruction_error.keys()) == {
        "dog_poodle", "dog_beagle", "hawk_red", "hawk_cooper"
    }
    for v in report.reconstruction_error.values():
        assert v >= 0.0
        assert np.isfinite(v)


def test_reconstruction_error_zero_for_identical_projections():
    records = {
        i: SAEFeatureRecord(
            feature_id=i, name=f"f{i}",
            projection=np.array([1.0, 0.0, 0.0]),
            label=f"{'A' if i < 2 else 'B'}/x",
        )
        for i in range(4)
    }
    _, report = from_sae_lens(records, [0, 1, 2, 3])
    for v in report.reconstruction_error.values():
        assert v < 1e-12


def test_tier_preservation_in_unit_interval_or_none():
    records = load_toy_sae(FIXTURE)
    _, report = from_sae_lens(records, [0, 1, 4, 5])
    assert report.tier_preservation is not None
    assert -1.0 - 1e-12 <= report.tier_preservation <= 1.0 + 1e-12 or np.isnan(
        report.tier_preservation
    )


def test_tier_preservation_none_for_singleton():
    records = load_toy_sae(FIXTURE)
    _, report = from_sae_lens(records, [0])
    assert report.tier_preservation is None


# ---------------------------------------------------------------------------
# Tasks §7 — from_sae_lens accepts SAEImportConfig with override
# precedence (per-field kwarg > config > dataclass-default).
# ---------------------------------------------------------------------------


class TestSAEImportConfigPassthrough:
    def test_config_supplies_assign_gamma_false(self):
        from polygram import SAEImportConfig

        records = load_toy_sae(FIXTURE)
        cfg = SAEImportConfig(assign_gamma=False)
        d, report = from_sae_lens(records, [0, 1, 4, 5], config=cfg)
        assert report.gamma_method == "zero"
        assert all(f.gamma == 0.0 for f in d.features)

    def test_per_field_kwarg_overrides_config(self):
        from polygram import SAEImportConfig

        records = load_toy_sae(FIXTURE)
        # Config says False, but the explicit kwarg wins.
        cfg = SAEImportConfig(assign_gamma=False)
        d, report = from_sae_lens(
            records, [0, 1, 4, 5], config=cfg, assign_gamma=True
        )
        assert report.gamma_method == "projection_pca"
        assert any(abs(f.gamma) > 1e-9 for f in d.features)

    def test_config_supplies_gamma_range(self):
        from polygram import SAEImportConfig

        records = load_toy_sae(FIXTURE)
        cfg = SAEImportConfig(assign_gamma=True, gamma_range=(-0.1, 0.1))
        d, _ = from_sae_lens(records, [0, 1, 4, 5], config=cfg)
        for f in d.features:
            assert -0.1 - 1e-12 <= f.gamma <= 0.1 + 1e-12


# ---------------------------------------------------------------------------
# §6 — Clustered loader path (clustered-dictionary-analysis)
# ---------------------------------------------------------------------------


class TestFromSaeLensClustered:
    def test_default_clustered_false_returns_dictionary(self):
        # Backwards-compatible: omitting `clustered` returns a
        # `Dictionary`, not a `ClusteredDictionary`.
        from polygram.dictionary import Dictionary

        records = load_toy_sae(FIXTURE)
        result, report = from_sae_lens(records, [0, 1, 4, 5])
        assert isinstance(result, Dictionary)
        assert report.n_blocks is None
        assert report.mean_block_size is None
        assert report.n_cross_block_edges is None

    def test_clustered_true_returns_clustered_dictionary(self):
        from polygram.clustered_dictionary import ClusteredDictionary

        records = load_toy_sae(FIXTURE)
        # 16 features (more than the legacy 8-cap) — clustered path
        # accepts.
        result, report = from_sae_lens(
            records, list(range(16)), clustered=True
        )
        assert isinstance(result, ClusteredDictionary)
        assert report.n_blocks is not None
        assert report.mean_block_size is not None
        assert report.n_cross_block_edges is not None
        assert report.n_blocks == result.n_blocks
        assert report.mean_block_size == result.mean_block_size
        assert report.n_cross_block_edges == result.n_cross_block_edges

    def test_clustered_true_skips_8_cap(self):
        # Without clustered=True, 16 features raises. With it, no raise.
        records = load_toy_sae(FIXTURE)
        with pytest.raises(ValueError, match="caps a Dictionary"):
            from_sae_lens(records, list(range(16)))
        # Same call with clustered=True succeeds.
        _result, _report = from_sae_lens(
            records, list(range(16)), clustered=True
        )

    def test_clustered_block_size_respects_encoding_cap(self):
        # Default block_formation.block_size_max=None → uses encoding's
        # max_features (legacy 8). 16 features → at least 2 blocks.
        records = load_toy_sae(FIXTURE)
        result, _report = from_sae_lens(
            records, list(range(16)), clustered=True
        )
        for block in result.blocks:
            assert len(block.features) <= 8

    def test_clustered_with_custom_block_formation(self):
        from polygram.clustered_dictionary import BlockFormation

        records = load_toy_sae(FIXTURE)
        bf = BlockFormation(
            strategy="cosine", cosine_threshold=0.5, block_size_max=4
        )
        result, _report = from_sae_lens(
            records, list(range(16)), clustered=True, block_formation=bf
        )
        for block in result.blocks:
            assert len(block.features) <= 4

    def test_clustered_error_message_points_to_clustered(self):
        records = load_toy_sae(FIXTURE)
        with pytest.raises(ValueError, match="clustered=True"):
            from_sae_lens(records, list(range(16)))


# ---------------------------------------------------------------------------
# Per-encoding-feature-cap (loader side)
# ---------------------------------------------------------------------------


class TestPerEncodingFeatureCapLoader:
    def test_mpsrung1_eight_features_still_loads(self):
        # The pre-change happy path: 8 features against MPSRung1 (the
        # default encoding). Should be byte-identical to before.
        records = load_toy_sae(FIXTURE)
        d, report = from_sae_lens(records, [0, 1, 2, 3, 4, 5, 6, 7])
        assert len(d.features) == 8
        assert report.n_selected == 8

    def test_rung3_twelve_features_now_loads(self):
        # 12 features against Rung3 was previously rejected by the
        # uniform 8-cap. Now accepted (Rung3.max_features == 16).
        from polygram.encoding import Rung3

        records = load_toy_sae(FIXTURE)
        # 12 features at indices 0..11 (the toy fixture has 16).
        d, _report = from_sae_lens(
            records, list(range(12)), encoding=Rung3()
        )
        assert len(d.features) == 12

    def test_rung3_seventeen_features_raises_with_encoding_name(self):
        # 17 against Rung3 (cap 16). The toy fixture only has 16
        # features so we need to fake a record set. Just exercise the
        # cap check directly via from_sae_lens.
        from polygram.encoding import Rung3

        records = load_toy_sae(FIXTURE)
        # The fixture only has 16 features; we'd need 17 to trip the
        # cap. The cap check fires BEFORE record-id validation, so we
        # can pass 17 ids (the 16 real ones plus a duplicate) — the
        # ValueError surfaces from the cap path, not from id lookup.
        ids = list(range(16)) + [0]  # 17 entries, last is a dup
        # The duplicate would normally cause other validation issues,
        # so we test the cap raises FIRST.
        with pytest.raises(ValueError) as exc_info:
            from_sae_lens(records, ids, encoding=Rung3())
        msg = str(exc_info.value)
        # Error names the encoding and its cap.
        assert "Rung3" in msg
        assert "16" in msg

    def test_mpsrung1_nine_features_raises_with_encoding_name(self):
        records = load_toy_sae(FIXTURE)
        with pytest.raises(ValueError) as exc_info:
            from_sae_lens(records, list(range(9)))
        msg = str(exc_info.value)
        assert "MPSRung1" in msg
        assert "8" in msg

    def test_error_message_suggests_clustered_path(self):
        records = load_toy_sae(FIXTURE)
        with pytest.raises(ValueError) as exc_info:
            from_sae_lens(records, list(range(9)))
        assert "clustered=True" in str(exc_info.value)
