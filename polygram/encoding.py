"""Encoding markers — config tags that downstream emitters dispatch on."""

from dataclasses import dataclass


_HEA_VALID_ROTATIONS = ("Rx", "Ry", "Rz")
_HEA_VALID_ENTANGLERS = ("ring", "chain")


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


@dataclass(frozen=True)
class HEA_Rung2:
    """Rung-2 hardware-efficient ansatz encoding.

    Mirrors the q-orca-lang ``examples/larql-hea-minimal.q.orca.md`` shape:
    a stack of ``depth`` rotation+entangler layers on ``n_qubits`` qubits,
    with the rotation set drawn from ``rotations`` per layer and the
    entangler chosen per topology.

    A non-``None`` ``tier_separation_bound`` causes the emitter to declare
    a ``concept_gram_tier_separation >= bound`` invariant in the produced
    ``.q.orca.md``; passing ``None`` suppresses invariant emission. The
    default ``0.025`` matches the q-orca-lang spike's ``HEA_TIER_TOLERANCE``.
    """

    depth: int
    entangler: str = "ring"
    rotations: tuple[str, ...] = ("Ry", "Rz")
    tier_separation_bound: float | None = 0.025
    n_qubits: int = 3

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(
                f"HEA_Rung2.depth must satisfy depth >= 1; got {self.depth}"
            )
        if self.entangler not in _HEA_VALID_ENTANGLERS:
            raise ValueError(
                f"HEA_Rung2.entangler must be one of "
                f"{_HEA_VALID_ENTANGLERS!r}; got {self.entangler!r}"
            )
        if not self.rotations:
            raise ValueError("HEA_Rung2.rotations must be non-empty")
        for rot in self.rotations:
            if rot not in _HEA_VALID_ROTATIONS:
                raise ValueError(
                    f"HEA_Rung2.rotations entries must be in "
                    f"{_HEA_VALID_ROTATIONS!r}; got {rot!r}"
                )
        if self.tier_separation_bound is not None:
            if not (0.0 <= self.tier_separation_bound <= 1.0):
                raise ValueError(
                    f"HEA_Rung2.tier_separation_bound must lie in [0, 1] "
                    f"or be None; got {self.tier_separation_bound}"
                )
        if self.n_qubits < 1:
            raise ValueError(
                f"HEA_Rung2.n_qubits must be >= 1; got {self.n_qubits}"
            )

    @property
    def theta_shape(self) -> tuple[int, int, int]:
        """Expected shape of a per-feature θ tensor: ``(|rotations|, depth, n_qubits)``."""
        return (len(self.rotations), self.depth, self.n_qubits)
