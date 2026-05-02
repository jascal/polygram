"""Local statevector simulator — sanity checks against q-orca's analytic
Gram (which builds states via its own private code path)."""

import math

import numpy as np

from polygram._state import build_statevector, schmidt_rank
from polygram.dictionary import Feature


def test_zero_angles_gives_ground_state():
    f = Feature("f", "c", beta=0.0)
    state = build_statevector(f)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_state_normalized():
    f = Feature("f", "c", beta=0.5, alpha=0.3, gamma=-0.2, phi=1.0)
    state = build_statevector(f)
    assert np.abs(np.vdot(state, state)) == np.float64(1) or np.isclose(
        np.abs(np.vdot(state, state)), 1.0, atol=1e-12
    )


def test_overlap_matches_published_phi_baseline():
    """φ-matched cross-cluster overlap = cos⁴(0.5) ≈ 0.5931 (q1 and q2
    angles each differ by 1.0)."""
    a = Feature("a", "dogs", beta=-0.5)
    b = Feature("b", "birds", beta=0.5)
    s_a = build_statevector(a)
    s_b = build_statevector(b)
    sq = np.abs(np.vdot(s_a, s_b)) ** 2
    expected = np.cos(0.5) ** 4
    assert np.isclose(sq, expected, atol=1e-6)


def test_schmidt_rank_product_state_is_one():
    f = Feature("f", "c", beta=0.0, alpha=0.0, gamma=0.5)
    state = build_statevector(f)
    assert schmidt_rank(state, cut=1) == 1


def test_schmidt_rank_entangled_state_is_two():
    f = Feature("f", "c", beta=0.5, alpha=math.pi / 3, gamma=0.0, phi=0.0)
    state = build_statevector(f)
    assert schmidt_rank(state, cut=1) == 2
