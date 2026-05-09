"""Byte-equality regression test for the `clustered` profile.

The `tests/fixtures/golden_clustered.json` snapshot locks in the v0.1.0
defaults that the `clustered` profile reproduces. **Regenerate the
fixture only on intentional default changes** to the `clustered`
profile (renamed kwargs, retuned β-spread, etc.) — never as a side
effect of refactoring. If a future change needs to ship multiple
golden sets (e.g. a `clustered_v2` profile with different defaults),
version the filename (`golden_clustered_v2.json`) rather than
overwriting this one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polygram import from_sae_lens, load_toy_sae
from polygram.sae_import import SAEFeatureRecord

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_clustered.json"
TOY_SAE_PATH = Path(__file__).parent / "fixtures" / "toy_sae.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _strip_labels(records: dict[int, SAEFeatureRecord]) -> dict[int, SAEFeatureRecord]:
    return {
        fid: SAEFeatureRecord(
            feature_id=r.feature_id,
            name=r.name,
            projection=r.projection,
            label=None,
            activation_mean=r.activation_mean,
            activation_std=r.activation_std,
        )
        for fid, r in records.items()
    }


def _assert_features_equal(features, golden_features) -> None:
    assert len(features) == len(golden_features)
    for f, gf in zip(features, golden_features):
        assert f.name == gf["name"]
        assert f.cluster == gf["cluster"]
        np.testing.assert_allclose(f.beta, gf["beta"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(f.gamma, gf["gamma"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(f.alpha, gf["alpha"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(f.phi, gf["phi"], rtol=0, atol=1e-12)


def _assert_report_equal(report, g) -> None:
    assert dict(report.cluster_assignments) == dict(g["cluster_assignments"])
    assert report.cluster_method == g["cluster_method"]
    np.testing.assert_allclose(
        report.beta_variance_explained, g["beta_variance_explained"], atol=1e-12
    )
    if g["tier_preservation"] is None:
        assert report.tier_preservation is None
    else:
        np.testing.assert_allclose(
            report.tier_preservation, g["tier_preservation"], atol=1e-12
        )
    assert report.gamma_method == g["gamma_method"]
    assert report.profile == g["profile"]
    if g["geometric_fidelity"] is None:
        assert report.geometric_fidelity is None
    else:
        np.testing.assert_allclose(
            report.geometric_fidelity, g["geometric_fidelity"], atol=1e-12
        )
    for fname, expected in g["reconstruction_error"].items():
        np.testing.assert_allclose(
            report.reconstruction_error[fname], expected, atol=1e-12
        )


def test_clustered_from_labels_byte_equal(golden):
    records = load_toy_sae(str(TOY_SAE_PATH))
    d, rep = from_sae_lens(records, [0, 1, 4, 5], profile="clustered")
    _assert_features_equal(d.features, golden["from_labels"]["features"])
    _assert_report_equal(rep, golden["from_labels"])


def test_clustered_kmeans_byte_equal(golden):
    records = _strip_labels(load_toy_sae(str(TOY_SAE_PATH)))
    d, rep = from_sae_lens(records, [0, 1, 4, 5], profile="clustered")
    _assert_features_equal(d.features, golden["kmeans"]["features"])
    _assert_report_equal(rep, golden["kmeans"])


def test_omitting_profile_matches_clustered():
    """`from_sae_lens(records, ids)` with no `profile=` kwarg must be
    bit-equal to passing `profile="clustered"`."""
    records = load_toy_sae(str(TOY_SAE_PATH))
    d_default, rep_default = from_sae_lens(records, [0, 1, 4, 5])
    d_explicit, rep_explicit = from_sae_lens(
        records, [0, 1, 4, 5], profile="clustered"
    )
    assert rep_default.profile == "clustered"
    assert rep_default.profile == rep_explicit.profile
    assert rep_default.tier_preservation == rep_explicit.tier_preservation
    assert rep_default.geometric_fidelity == rep_explicit.geometric_fidelity
    assert rep_default.beta_variance_explained == rep_explicit.beta_variance_explained
    for f1, f2 in zip(d_default.features, d_explicit.features):
        assert f1.beta == f2.beta
        assert f1.gamma == f2.gamma
        assert f1.cluster == f2.cluster
