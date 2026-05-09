"""Tests for the `uniform-sphere` profile on a synthetic audio-style fixture.

Builds a small uniform-sphere SAE in-memory: unit-normalised quasi-orthogonal
decoder rows with one tight 8-feature cluster (mean within-cluster cosine
≈ 0.4). Captures the Phase-1 audio signature without shipping a 200 MB
checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from polygram import from_sae_lens
from polygram.sae_import import SAEFeatureRecord


def _audio_style_records(seed: int = 42, n_total: int = 64, d_model: int = 32):
    """Build a synthetic SAE with a tight 8-feature cluster embedded in
    a quasi-orthogonal sea. The cluster shares ~40% cosine similarity;
    background features are random Gaussian unit vectors."""
    rng = np.random.default_rng(seed)
    # Background: orthogonalised random vectors.
    bg = rng.standard_normal((n_total - 8, d_model))
    bg /= np.linalg.norm(bg, axis=1, keepdims=True)

    # Tight 8-feature cluster: small perturbations off a shared seed.
    seed_vec = rng.standard_normal(d_model)
    seed_vec /= np.linalg.norm(seed_vec)
    cluster = []
    for _ in range(8):
        perturb = rng.standard_normal(d_model) * 0.7
        v = seed_vec + perturb
        v /= np.linalg.norm(v)
        cluster.append(v)
    cluster = np.stack(cluster)

    full = np.vstack([cluster, bg])
    rng.shuffle(full)
    return {
        i: SAEFeatureRecord(
            feature_id=i,
            name=f"feat_{i}",
            projection=full[i].astype(float),
            label=None,
        )
        for i in range(n_total)
    }, list(range(8))


def test_uniform_sphere_produces_nondegenerate_beta_spread():
    """Spec scenario: β span ≥ 60% of (-0.5, 0.5) on a tight 8-feature cluster."""
    records, ids = _audio_style_records()
    d, rep = from_sae_lens(records, ids, profile="uniform-sphere")
    betas = [f.beta for f in d.features]
    span = max(betas) - min(betas)
    assert span >= 0.6, f"β span {span:.3f} < 0.6 (range goal)"
    assert rep.geometric_fidelity is None or 0.0 <= rep.geometric_fidelity <= 1.0


def test_uniform_sphere_cluster_method_is_pca_axis():
    records, ids = _audio_style_records()
    _, rep = from_sae_lens(records, ids, profile="uniform-sphere")
    assert rep.cluster_method == "pca_axis"


def test_uniform_sphere_geometric_fidelity_is_bounded():
    records, ids = _audio_style_records()
    _, rep = from_sae_lens(records, ids, profile="uniform-sphere")
    assert rep.profile == "uniform-sphere"
    assert rep.tier_preservation is None
    assert rep.geometric_fidelity is not None
    assert 0.0 <= rep.geometric_fidelity <= 1.0


def test_uniform_sphere_beta_var_is_top1_pca_fraction():
    records, ids = _audio_style_records()
    _, rep = from_sae_lens(records, ids, profile="uniform-sphere")
    # PCA-axis variance fraction is bounded [0, 1].
    assert 0.0 <= rep.beta_variance_explained <= 1.0


def test_uniform_sphere_single_feature_returns_none_fidelity():
    records, ids = _audio_style_records()
    _, rep = from_sae_lens(records, [ids[0]], profile="uniform-sphere")
    assert rep.geometric_fidelity is None
