"""Tests for `from_sae_lens` profile dispatch and resolution order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polygram import (
    SAEImportConfig,
    from_sae_lens,
    get_profile,
    load_toy_sae,
)
from polygram.sae_import import SAEFeatureRecord

TOY_SAE_PATH = Path(__file__).parent / "fixtures" / "toy_sae.json"


def _strip_labels(records):
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


def test_omitting_profile_resolves_to_clustered():
    records = load_toy_sae(str(TOY_SAE_PATH))
    _, rep = from_sae_lens(records, [0, 1, 4, 5])
    assert rep.profile == "clustered"


def test_explicit_profile_string_resolves_via_registry():
    records = load_toy_sae(str(TOY_SAE_PATH))
    _, rep_str = from_sae_lens(records, [0, 1, 4, 5], profile="uniform-sphere")
    _, rep_obj = from_sae_lens(
        records, [0, 1, 4, 5], profile=get_profile("uniform-sphere")
    )
    assert rep_str.profile == "uniform-sphere"
    assert rep_obj.profile == "uniform-sphere"
    assert rep_str.geometric_fidelity == rep_obj.geometric_fidelity


def test_unknown_profile_string_raises_keyerror():
    records = load_toy_sae(str(TOY_SAE_PATH))
    with pytest.raises(KeyError, match="no profile named"):
        from_sae_lens(records, [0, 1, 4, 5], profile="not-a-profile")


def test_invalid_profile_type_raises_typeerror():
    records = load_toy_sae(str(TOY_SAE_PATH))
    with pytest.raises(TypeError, match="profile must be"):
        from_sae_lens(records, [0, 1, 4, 5], profile=42)


def test_per_field_n_clusters_overrides_profile_default():
    """profile="uniform-sphere" defaults to k=16, but per-field n_clusters=4
    must override (and warn since 4 > selected=4 is at the clamp boundary)."""
    records = _strip_labels(load_toy_sae(str(TOY_SAE_PATH)))
    _, rep = from_sae_lens(
        records, [0, 1, 4, 5], profile="uniform-sphere", n_clusters=4
    )
    # 4 features, n_clusters=4 → each its own cluster (after clamping).
    assert len({c for c in rep.cluster_assignments.values()}) <= 4


def test_config_profile_field_resolves():
    records = load_toy_sae(str(TOY_SAE_PATH))
    cfg = SAEImportConfig(profile="uniform-sphere")
    _, rep = from_sae_lens(records, [0, 1, 4, 5], config=cfg)
    assert rep.profile == "uniform-sphere"


def test_kwarg_profile_overrides_config_profile():
    records = load_toy_sae(str(TOY_SAE_PATH))
    cfg = SAEImportConfig(profile="uniform-sphere")
    _, rep = from_sae_lens(
        records, [0, 1, 4, 5], config=cfg, profile="clustered"
    )
    assert rep.profile == "clustered"


def test_cluster_assignments_bypasses_strategy_but_fidelity_still_computed():
    """cluster_assignments runs upstream of strategy dispatch — but the
    profile's geometric_fidelity SHALL still be computed."""
    records = load_toy_sae(str(TOY_SAE_PATH))
    _, rep = from_sae_lens(
        records,
        [0, 1, 4, 5],
        profile="uniform-sphere",
        cluster_assignments={0: "A", 1: "A", 4: "B", 5: "B"},
    )
    assert rep.profile == "uniform-sphere"
    assert rep.cluster_method == "user"
    # tier_preservation is None for uniform-sphere; geometric_fidelity is set.
    assert rep.tier_preservation is None
    assert rep.geometric_fidelity is not None
    assert 0.0 <= rep.geometric_fidelity <= 1.0


def test_clustered_profile_populates_both_tier_pres_and_geometric_fidelity():
    records = load_toy_sae(str(TOY_SAE_PATH))
    _, rep = from_sae_lens(records, [0, 1, 4, 5], profile="clustered")
    # For the clustered profile, tier_preservation IS the geometric_fidelity.
    assert rep.tier_preservation == rep.geometric_fidelity


def test_uniform_sphere_clears_tier_preservation():
    records = load_toy_sae(str(TOY_SAE_PATH))
    _, rep = from_sae_lens(records, [0, 1, 4, 5], profile="uniform-sphere")
    assert rep.profile == "uniform-sphere"
    assert rep.tier_preservation is None
    assert rep.geometric_fidelity is not None
