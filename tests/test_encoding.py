import pytest

from polygram.encoding import MPSRung1


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
