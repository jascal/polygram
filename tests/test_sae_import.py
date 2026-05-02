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
