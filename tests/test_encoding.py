import pytest

from polygram.encoding import HEA_Rung2, MPSRung1


def test_default_is_bond_2_with_phase_knobs():
    e = MPSRung1()
    assert e.bond_dim == 2
    assert e.phase_knobs is True


def test_bond_dim_other_than_2_rejected():
    with pytest.raises(ValueError, match="bond_dim must be 2"):
        MPSRung1(bond_dim=3)


def test_phase_knobs_can_be_disabled():
    e = MPSRung1(phase_knobs=False)
    assert e.phase_knobs is False


class TestHEARung2:
    def test_defaults_match_q_orca_lang_spike(self):
        e = HEA_Rung2(depth=3)
        assert e.depth == 3
        assert e.entangler == "ring"
        assert e.rotations == ("Ry", "Rz")
        assert e.tier_separation_bound == 0.025
        assert e.n_qubits == 3

    def test_theta_shape(self):
        e = HEA_Rung2(depth=3, rotations=("Ry", "Rz"))
        assert e.theta_shape == (2, 3, 3)
        e2 = HEA_Rung2(depth=2, rotations=("Rx", "Ry", "Rz"), n_qubits=4)
        assert e2.theta_shape == (3, 2, 4)

    def test_invalid_depth_rejected(self):
        with pytest.raises(ValueError, match="depth >= 1"):
            HEA_Rung2(depth=0)
        with pytest.raises(ValueError, match="depth >= 1"):
            HEA_Rung2(depth=-2)

    def test_unknown_entangler_rejected(self):
        with pytest.raises(ValueError, match="entangler"):
            HEA_Rung2(depth=2, entangler="all-to-all")

    def test_unknown_rotation_rejected(self):
        with pytest.raises(ValueError, match="'Rq'"):
            HEA_Rung2(depth=2, rotations=("Ry", "Rq"))

    def test_empty_rotations_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            HEA_Rung2(depth=2, rotations=())

    def test_tier_bound_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            HEA_Rung2(depth=2, tier_separation_bound=1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            HEA_Rung2(depth=2, tier_separation_bound=-0.1)

    def test_tier_bound_none_permitted(self):
        e = HEA_Rung2(depth=2, tier_separation_bound=None)
        assert e.tier_separation_bound is None

    def test_frozen_equality(self):
        a = HEA_Rung2(depth=2)
        b = HEA_Rung2(depth=2)
        c = HEA_Rung2(depth=3)
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# Per-encoding-feature-cap
# ---------------------------------------------------------------------------


class TestPerEncodingFeatureCap:
    """Each encoding declares `max_features` matching its reachable
    Hilbert-space dimension. See `docs/research/rung3-rank-bound.md`
    for the empirical basis."""

    def test_mpsrung1_cap_is_eight(self):
        from polygram.encoding import MPSRung1

        assert MPSRung1.max_features == 8
        assert MPSRung1().max_features == 8

    def test_rung3_cap_is_sixteen(self):
        from polygram.encoding import Rung3

        assert Rung3.max_features == 16
        assert Rung3().max_features == 16

    def test_hea_cap_scales_with_n_qubits(self):
        from polygram.encoding import HEA_Rung2

        assert HEA_Rung2(depth=1, n_qubits=3).max_features == 8
        assert HEA_Rung2(depth=1, n_qubits=4).max_features == 16
        assert HEA_Rung2(depth=1, n_qubits=5).max_features == 32
        assert HEA_Rung2(depth=2, n_qubits=10).max_features == 1024

    def test_back_compat_constant_matches_mpsrung1(self):
        from polygram.encoding import MPSRung1
        from polygram.sae_import import MAX_FEATURES_PER_DICTIONARY

        assert MAX_FEATURES_PER_DICTIONARY == MPSRung1.max_features
        assert MAX_FEATURES_PER_DICTIONARY == 8


# ---------------------------------------------------------------------------
# Rung4 encoding (add-rung4-encoding-mvp)
# ---------------------------------------------------------------------------


class TestRung4Encoding:
    def test_max_features_is_32(self):
        from polygram.encoding import Rung4

        assert Rung4.max_features == 32
        assert Rung4().max_features == 32

    def test_bond_dim_default_is_2(self):
        from polygram.encoding import Rung4

        assert Rung4().bond_dim == 2

    def test_bond_dim_other_than_2_raises(self):
        import pytest as _pytest

        from polygram.encoding import Rung4

        with _pytest.raises(ValueError, match="bond_dim must be 2"):
            Rung4(bond_dim=4)

    def test_rung4_amp_overlap_default_is_one(self):
        from polygram.encoding import rung4_amp_overlap

        z = rung4_amp_overlap(0, 0, 0, 0, 0, 0, 0, 0)
        assert abs(z - 1.0) < 1e-12

    def test_rung4_amp_overlap_factors_through_single_qubit_overlaps(self):
        from polygram.encoding import (
            _single_qubit_overlap,
            rung4_amp_overlap,
        )

        # Pick arbitrary non-default knobs.
        t_a3, p_a3 = 0.3, 0.4
        t_a4, p_a4 = 0.5, 0.6
        t_b3, p_b3 = 0.7, 0.8
        t_b4, p_b4 = 0.9, 1.0
        expected = (
            _single_qubit_overlap(t_a3, p_a3, t_b3, p_b3)
            * _single_qubit_overlap(t_a4, p_a4, t_b4, p_b4)
        )
        actual = rung4_amp_overlap(
            t_a3, p_a3, t_a4, p_a4, t_b3, p_b3, t_b4, p_b4
        )
        assert abs(actual - expected) < 1e-12

    def test_rung4_amp_overlap_squared_matches_complex_abs2(self):
        import math

        from polygram.encoding import (
            rung4_amp_overlap,
            rung4_amp_overlap_squared,
        )

        args = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        sq = rung4_amp_overlap_squared(*args)
        z = rung4_amp_overlap(*args)
        assert math.isclose(sq, abs(z) ** 2, abs_tol=1e-12)

    def test_rung4state_default_amp_overlap_is_one(self):
        from polygram.encoding import Rung4State

        s = Rung4State(alpha=0.1, beta=0.2, gamma=0.3, phi=0.4)
        assert s.amp_overlap_squared(s) == 1.0

    def test_rung4state_amp_overlap_equals_complex_product(self):
        import math

        from polygram.encoding import Rung4State, rung4_amp_overlap

        a = Rung4State(0.1, 0.2, 0.3, 0.4, theta_amp=0.5, psi_aux=0.6,
                       theta_amp_b=0.7, psi_amp_b=0.8)
        b = Rung4State(0.1, 0.2, 0.3, 0.4, theta_amp=0.2, psi_aux=0.1,
                       theta_amp_b=0.9, psi_amp_b=1.1)
        expected = (
            abs(rung4_amp_overlap(
                a.theta_amp, a.psi_aux, a.theta_amp_b, a.psi_amp_b,
                b.theta_amp, b.psi_aux, b.theta_amp_b, b.psi_amp_b,
            )) ** 2
        )
        assert math.isclose(a.amp_overlap_squared(b), expected, abs_tol=1e-12)

    def test_rung4state_from_mps_knobs_defaults(self):
        from polygram.encoding import Rung4State

        s = Rung4State.from_mps_knobs(0.1, 0.2, 0.3, 0.4)
        assert s.theta_amp == 0.0
        assert s.psi_aux == 0.0
        assert s.theta_amp_b == 0.0
        assert s.psi_amp_b == 0.0


class TestRung3AmpOverlapBackcompat:
    """Pin that the existing `rung3_amp_overlap` math is unchanged by
    the refactor that extracted `_single_qubit_overlap`. Same inputs
    must produce the same outputs as before the extraction."""

    def test_rung3_amp_overlap_matches_explicit_formula(self):
        import math

        from polygram.encoding import rung3_amp_overlap

        # Hand-computed against the docstring formula.
        ta, pa, tb, pb = 0.5, 0.3, 0.7, 0.2
        ca, sa = math.cos(ta), math.sin(ta)
        cb, sb = math.cos(tb), math.sin(tb)
        delta = pb - pa
        expected = complex(
            ca * cb + sa * sb * math.cos(delta),
            sa * sb * math.sin(delta),
        )
        actual = rung3_amp_overlap(ta, pa, tb, pb)
        assert abs(actual - expected) < 1e-12

    def test_rung3_amp_overlap_default_is_one(self):
        import math

        from polygram.encoding import rung3_amp_overlap

        # Rung3 default knobs (π/4, 0) give amp factor = 1.
        z = rung3_amp_overlap(math.pi / 4, 0.0, math.pi / 4, 0.0)
        assert abs(z - 1.0) < 1e-12


class TestRung5Encoding:
    def test_constructible_with_explicit_k(self):
        from polygram.encoding import Rung5

        r = Rung5(n_amp_qubits=3)
        assert r.bond_dim == 2
        assert r.n_amp_qubits == 3
        assert r.max_features == 64

    def test_max_features_scales_with_k(self):
        from polygram.encoding import Rung5

        expected = {1: 16, 2: 32, 3: 64, 4: 128, 5: 256, 16: 524288}
        for k, mf in expected.items():
            assert Rung5(n_amp_qubits=k).max_features == mf

    def test_bond_dim_not_2_rejected(self):
        from polygram.encoding import Rung5

        with pytest.raises(ValueError, match="bond_dim must be 2"):
            Rung5(bond_dim=3, n_amp_qubits=2)

    def test_n_amp_qubits_zero_rejected_with_mpsrung1_hint(self):
        from polygram.encoding import Rung5

        with pytest.raises(ValueError, match="MPSRung1 directly"):
            Rung5(n_amp_qubits=0)

    def test_n_amp_qubits_negative_rejected(self):
        from polygram.encoding import Rung5

        with pytest.raises(ValueError, match="n_amp_qubits must be >= 1"):
            Rung5(n_amp_qubits=-1)

    def test_n_amp_qubits_above_cap_rejected(self):
        from polygram.encoding import RUNG5_MAX_N_AMP_QUBITS, Rung5

        with pytest.raises(ValueError, match="n_amp_qubits must be <="):
            Rung5(n_amp_qubits=RUNG5_MAX_N_AMP_QUBITS + 1)

    def test_n_amp_qubits_at_cap_accepted(self):
        from polygram.encoding import RUNG5_MAX_N_AMP_QUBITS, Rung5

        r = Rung5(n_amp_qubits=RUNG5_MAX_N_AMP_QUBITS)
        assert r.max_features == 8 * 2 ** RUNG5_MAX_N_AMP_QUBITS

    def test_rung5_amp_overlap_kfold_product(self):
        from polygram.encoding import (
            _single_qubit_overlap,
            rung5_amp_overlap,
        )

        amp_a = ((0.3, 0.1), (0.5, 0.2), (0.7, 0.4))
        amp_b = ((0.4, 0.0), (0.6, 0.5), (0.8, 0.3))
        expected = complex(1.0, 0.0)
        for (ta, pa), (tb, pb) in zip(amp_a, amp_b):
            expected *= _single_qubit_overlap(ta, pa, tb, pb)
        assert abs(rung5_amp_overlap(amp_a, amp_b) - expected) < 1e-12

    def test_rung5_amp_overlap_default_is_one(self):
        from polygram.encoding import rung5_amp_overlap

        for k in (1, 2, 3, 5):
            amp = ((0.0, 0.0),) * k
            assert abs(rung5_amp_overlap(amp, amp) - 1.0) < 1e-12

    def test_rung5_amp_overlap_length_mismatch_rejected(self):
        from polygram.encoding import rung5_amp_overlap

        with pytest.raises(ValueError, match="same length"):
            rung5_amp_overlap(((0.0, 0.0),), ((0.0, 0.0), (0.0, 0.0)))

    def test_rung5_amp_overlap_squared_equals_abs_squared(self):
        import math

        from polygram.encoding import rung5_amp_overlap, rung5_amp_overlap_squared

        amp_a = ((0.3, 0.1), (0.5, 0.2))
        amp_b = ((0.4, 0.0), (0.6, 0.5))
        z = rung5_amp_overlap(amp_a, amp_b)
        assert math.isclose(
            rung5_amp_overlap_squared(amp_a, amp_b),
            abs(z) ** 2,
            abs_tol=1e-12,
        )

    def test_rung5state_amp_overlap_squared_matches_helper(self):
        import math

        from polygram.encoding import (
            Rung5State,
            rung5_amp_overlap_squared,
        )

        a = Rung5State(0.1, 0.2, 0.3, 0.4, amp_knobs=((0.3, 0.1), (0.5, 0.2)))
        b = Rung5State(0.1, 0.2, 0.3, 0.4, amp_knobs=((0.4, 0.0), (0.6, 0.5)))
        assert math.isclose(
            a.amp_overlap_squared(b),
            rung5_amp_overlap_squared(a.amp_knobs, b.amp_knobs),
            abs_tol=1e-12,
        )

    def test_rung5state_from_mps_knobs_defaults_to_empty_tuple(self):
        from polygram.encoding import Rung5State

        s = Rung5State.from_mps_knobs(0.1, 0.2, 0.3, 0.4)
        assert s.amp_knobs == ()

    def test_rung5state_from_mps_knobs_with_amp_knobs(self):
        from polygram.encoding import Rung5State

        amp = ((0.3, 0.1), (0.5, 0.2))
        s = Rung5State.from_mps_knobs(0.1, 0.2, 0.3, 0.4, amp_knobs=amp)
        assert s.amp_knobs == amp
