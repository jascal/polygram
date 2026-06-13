"""Tests for the `readout-aligned` geometric profile (add-readout-aligned-geometry-profile).

Builds the dictionary geometry in the model's readout subspace (top-`r` SVD of `gain ⊙ U`) instead of on raw
decoder vectors. Synthetic fixtures only (no host download): a random unembed `U` whose readout subspace
re-orders the decoder directions, so the readout-aligned clustering provably differs from the raw one.
"""

from __future__ import annotations

import numpy as np
import pytest

from polygram import from_sae_lens
from polygram.geometry import available_profiles, readout_variance_captured
from polygram.geometry.readout_aligned import ReadoutAlignedKnobAssignment, readout_subspace
from polygram.sae_import import SAEFeatureRecord


def _records(seed: int = 0, n: int = 24, d_model: int = 32):
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((n, d_model))
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    recs = {
        i: SAEFeatureRecord(feature_id=i, name=f"feat_{i}", projection=proj[i].astype(float))
        for i in range(n)
    }
    # Select a small subset so from_sae_lens returns a flat Dictionary (with `.features`),
    # not an auto-promoted ClusteredDictionary (mirrors test_uniform_sphere_profile).
    return recs, list(range(8)), proj


def _unembed(seed: int = 1, vocab: int = 200, d_model: int = 32):
    return np.random.default_rng(seed).standard_normal((vocab, d_model))


def test_readout_aligned_is_registered():
    assert "readout-aligned" in available_profiles()


def test_from_sae_lens_requires_u_matrix():
    records, ids, _ = _records()
    with pytest.raises(ValueError, match="u_matrix is required"):
        from_sae_lens(records, ids, profile="readout-aligned")


def test_from_sae_lens_rejects_non_2d_u_matrix():
    records, ids, _ = _records()
    with pytest.raises(ValueError, match="2-D"):
        from_sae_lens(records, ids, profile="readout-aligned", u_matrix=np.ones(32))


def test_readout_aligned_builds_and_reports_profile():
    records, ids, _ = _records()
    U = _unembed(d_model=32)
    d, rep = from_sae_lens(records, ids, profile="readout-aligned", u_matrix=U)
    assert rep.profile == "readout-aligned"
    assert len(d.features) == len(ids)
    # fidelity (if computed) is a valid correlation or NaN, never out of range
    assert rep.geometric_fidelity is None or np.isnan(rep.geometric_fidelity) \
        or -1.0 <= rep.geometric_fidelity <= 1.0


def test_readout_aligned_differs_from_clustered():
    """The whole point: building geometry in the readout subspace changes the knobs vs the raw basis."""
    records, ids, _ = _records()
    U = _unembed(d_model=32)
    d_ro, _ = from_sae_lens(records, ids, profile="readout-aligned", u_matrix=U, readout_rank=4)
    d_cl, _ = from_sae_lens(records, ids, profile="clustered")
    betas_ro = [f.beta for f in d_ro.features]
    betas_cl = [f.beta for f in d_cl.features]
    assert betas_ro != betas_cl


def test_default_path_needs_no_u_matrix_and_stays_clustered():
    records, ids, _ = _records()
    _, rep = from_sae_lens(records, ids)
    assert rep.profile == "clustered"


def test_readout_subspace_edge_cases():
    U = _unembed(vocab=50, d_model=16)
    # rank capped at min(rank, d_model, vocab)
    assert readout_subspace(U, None, 16, 999).shape == (16, 16)
    # gain scalar and vector both accepted, shape (r, d_model)
    assert readout_subspace(U, 2.0, 16, 4).shape == (4, 16)
    assert readout_subspace(U, np.ones(16), 16, 4).shape == (4, 16)
    # wrong d_model raises
    with pytest.raises(ValueError, match="vocab, d_model"):
        readout_subspace(U, None, 8, 4)


def test_variance_captured_in_unit_range_and_monotone_in_rank():
    _, _, proj = _records(d_model=32)
    U = _unembed(d_model=32)
    vc_small = readout_variance_captured(proj, U, rank=2)
    vc_big = readout_variance_captured(proj, U, rank=16)
    assert 0.0 <= vc_small <= vc_big <= 1.0  # more rank captures at least as much


def test_assign_is_deterministic():
    _, _, proj = _records(d_model=32)
    U = _unembed(d_model=32)
    ka = ReadoutAlignedKnobAssignment(u_matrix=U, readout_rank=4)
    kw = dict(n_clusters=2, gamma_range=(-0.25, 0.25), assign_gamma=True, seed=0)
    a = ka.assign(proj, [f"f{i}" for i in range(len(proj))], **kw)
    b = ka.assign(proj, [f"f{i}" for i in range(len(proj))], **kw)
    assert a.cluster_per_feature == b.cluster_per_feature
    assert a.betas == b.betas and a.gammas == b.gammas


def test_assign_without_u_matrix_raises():
    _, _, proj = _records(d_model=32)
    with pytest.raises(ValueError, match="u_matrix is unset"):
        ReadoutAlignedKnobAssignment().assign(
            proj, [f"f{i}" for i in range(len(proj))],
            n_clusters=2, gamma_range=(-0.25, 0.25), assign_gamma=False, seed=0,
        )
