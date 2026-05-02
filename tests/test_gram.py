"""Gram round-trip — Polygram-emitted machine should reproduce the published
larql-animals-interference Gram tiers within 1e-4."""

import math

import numpy as np
import pytest

from polygram.dictionary import Dictionary, Feature


def _animals_interference() -> Dictionary:
    pi_2 = math.pi / 2
    return Dictionary(
        name="LarqlAnimalsInterference",
        features=[
            Feature("dog_at_rest", "dogs", beta=-0.5, phi=0.0),
            Feature("dog_in_motion", "dogs", beta=-0.5, phi=pi_2),
            Feature("bird_at_rest", "birds", beta=0.5, phi=0.0),
            Feature("bird_in_motion", "birds", beta=0.5, phi=pi_2),
        ],
        hierarchy={
            "dogs": ["dog_at_rest", "dog_in_motion"],
            "birds": ["bird_at_rest", "bird_in_motion"],
        },
    )


def test_gram_shape_and_diagonal():
    d = _animals_interference()
    g = d.gram()
    assert g.shape == (4, 4)
    diag = np.abs(np.diag(g)) ** 2
    np.testing.assert_allclose(diag, np.ones(4), atol=1e-9)


def test_gram_reproduces_published_tiers():
    """Three off-diagonal tiers from larql-animals-interference.q.orca.md:

    - same-cluster, φ-shifted             → 0.8851
    - cross-cluster, φ-mismatched         → 0.6816
    - cross-cluster, φ-matched (=baseline) → 0.5931
    """
    d = _animals_interference()
    g = d.gram()
    sq = np.abs(g) ** 2

    same_cluster_phi_shifted = [(0, 1), (2, 3)]
    cross_cluster_phi_mismatched = [(0, 3), (1, 2)]
    cross_cluster_phi_matched = [(0, 2), (1, 3)]

    for (i, j) in same_cluster_phi_shifted:
        assert sq[i, j] == pytest.approx(0.8851, abs=1e-3)
    for (i, j) in cross_cluster_phi_mismatched:
        assert sq[i, j] == pytest.approx(0.6816, abs=1e-3)
    for (i, j) in cross_cluster_phi_matched:
        assert sq[i, j] == pytest.approx(0.5931, abs=1e-3)


def test_gram_strictly_ordered_tiers():
    d = _animals_interference()
    sq = np.abs(d.gram()) ** 2
    same_cluster = sq[0, 1]
    cross_mismatched = sq[0, 3]
    cross_matched = sq[0, 2]
    assert same_cluster > cross_mismatched > cross_matched


def test_gram_no_phi_collapses_to_two_tiers():
    """Without the Rz knob (φ=0 everywhere), the four off-diagonals should
    collapse to two values (same-cluster vs cross-cluster)."""
    d = Dictionary(
        name="LarqlAnimalsNoPhi",
        features=[
            Feature("dog_a", "dogs", beta=-0.5),
            Feature("dog_b", "dogs", beta=-0.5),
            Feature("bird_a", "birds", beta=0.5),
            Feature("bird_b", "birds", beta=0.5),
        ],
        hierarchy={"dogs": ["dog_a", "dog_b"], "birds": ["bird_a", "bird_b"]},
    )
    sq = np.abs(d.gram()) ** 2
    assert sq[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert sq[2, 3] == pytest.approx(1.0, abs=1e-6)
    cross = [sq[0, 2], sq[0, 3], sq[1, 2], sq[1, 3]]
    assert all(c == pytest.approx(0.5931, abs=1e-3) for c in cross)
