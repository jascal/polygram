"""Encoding markers — config tags that downstream emitters dispatch on."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MPSRung1:
    """Rung-1 MPS encoding (bond dimension 2) with optional `Rz` phase knobs.

    Maps to q-orca's cross-coupled CNOT-staircase preparation:

        Ry(qs[0], α); CNOT(qs[0], qs[1]);
        Ry(qs[1], α + β); Rz(qs[1], φ);
        CNOT(qs[1], qs[2]); Ry(qs[2], β + γ)

    on a 3-qubit register. v0 supports `bond_dim=2` only — the
    safe-`Rz` matcher in q-orca >= 0.7.1 is fixed at χ=2.
    """

    bond_dim: int = 2
    phase_knobs: bool = True

    def __post_init__(self) -> None:
        if self.bond_dim != 2:
            raise ValueError(
                f"MPSRung1.bond_dim must be 2 (rung-1 / χ=2 only in v0); "
                f"got {self.bond_dim}"
            )
