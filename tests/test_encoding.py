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
