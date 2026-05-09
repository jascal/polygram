"""Tests for `polygram.geometry` registry + protocols."""

from __future__ import annotations

import pytest

from polygram import (
    GeometricProfile,
    available_profiles,
    clustered,
    get_profile,
    register_profile,
    uniform_sphere,
)
from polygram.geometry.protocols import (
    GeometricFidelity,
    KnobAssignment,
    KnobAssignmentResult,
)


def test_builtins_registered_at_import():
    profiles = available_profiles()
    assert "clustered" in profiles
    assert "uniform-sphere" in profiles


def test_get_profile_returns_clustered_with_v0_1_0_defaults():
    p = get_profile("clustered")
    assert p.name == "clustered"
    assert p.default_n_clusters == 2
    assert p.default_gamma_range == (-0.25, 0.25)


def test_get_profile_returns_uniform_sphere_with_audio_defaults():
    p = get_profile("uniform-sphere")
    assert p.name == "uniform-sphere"
    assert p.default_n_clusters == 16
    assert p.default_gamma_range == (-0.25, 0.25)


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_profile(clustered())


def test_get_profile_missing_lists_available_names():
    with pytest.raises(KeyError) as exc_info:
        get_profile("nonexistent-profile")
    msg = str(exc_info.value)
    assert "nonexistent-profile" in msg
    assert "clustered" in msg
    assert "uniform-sphere" in msg


def test_geometric_profile_is_frozen_and_hashable():
    p = clustered()
    assert {p, clustered()} == {p}  # equal-by-content collapses
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        p.name = "renamed"


def test_geometric_profile_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        GeometricProfile(
            name="",
            knob_assignment=clustered().knob_assignment,
            geometric_fidelity=clustered().geometric_fidelity,
            default_n_clusters=2,
            default_gamma_range=(-0.25, 0.25),
        )


def test_strategy_objects_satisfy_protocols():
    c = clustered()
    u = uniform_sphere()
    assert isinstance(c.knob_assignment, KnobAssignment)
    assert isinstance(c.geometric_fidelity, GeometricFidelity)
    assert isinstance(u.knob_assignment, KnobAssignment)
    assert isinstance(u.geometric_fidelity, GeometricFidelity)


def test_third_party_profile_can_be_registered():
    """Demonstrates the sae-forge-style consumer registration path."""

    class _MinimalKnob:
        def assign(self, projections, feature_names, *,
                   n_clusters, gamma_range, assign_gamma, seed):
            n = len(feature_names)
            return KnobAssignmentResult(
                cluster_per_feature=["cluster_0"] * n,
                betas=[0.0] * n,
                gammas=[0.0] * n,
                cluster_method="dummy",
                beta_variance_explained=0.0,
            )

    class _MinimalFidelity:
        def compute(self, projections, dictionary):
            return 0.5

    custom = GeometricProfile(
        name="test-custom",
        knob_assignment=_MinimalKnob(),
        geometric_fidelity=_MinimalFidelity(),
        default_n_clusters=4,
        default_gamma_range=(-0.1, 0.1),
    )
    register_profile(custom)
    try:
        assert "test-custom" in available_profiles()
        assert get_profile("test-custom") is custom
    finally:
        # Manually clean up — the registry doesn't expose unregister_profile.
        from polygram.geometry.registry import _REGISTRY
        _REGISTRY.pop("test-custom", None)
