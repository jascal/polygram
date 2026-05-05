"""Encoding markers — config tags that downstream emitters dispatch on."""

import math
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


# Default values for the rung-3 amplitude branch — chosen so the branch
# reduces to identity overlap (factor = 1) when *both* paired features hold
# the defaults, leaving Rung3 gram values equal to the MPSRung1-equivalent
# gram on (α, β, γ, φ).
RUNG3_DEFAULT_THETA_AMP = math.pi / 4
RUNG3_DEFAULT_PSI_AUX = 0.0


@dataclass(frozen=True)
class Rung3:
    """Rung-3 encoding: MPSRung1 on qubits 0–2 plus an amplitude branch on
    qubits 3–4.

    The amplitude branch state per feature is

        |amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩

    on qubits 3–4, parameterized by per-feature ``theta_amp`` and
    ``psi_aux`` knobs (carried on the ``Feature`` rows, not on this tag).
    With ``θ = π/4`` and ``ψ = 0`` for every feature the amplitude
    overlap factor is identically 1, so the rung-3 dictionary's gram
    matches the MPSRung1-equivalent gram on the same (α, β, γ, φ).

    The rung-3 viability spike (`docs/research/rung3-viability-spike.md`,
    landed via `add-rung3-encoding-mvp` / PR #29) tests whether joint
    (φ, θ_amp, ψ_aux) optimization breaks below the
    ``MPSRung1.structural_floor`` of ``M − |V|`` on real GPT-2-small SAE
    pairs.
    """

    bond_dim: int = 2

    def __post_init__(self) -> None:
        if self.bond_dim != 2:
            raise ValueError(
                f"Rung3.bond_dim must be 2 (rung-1 / χ=2 only in v0); "
                f"got {self.bond_dim}"
            )


def rung3_amp_overlap_squared(
    theta_a: float, psi_a: float, theta_b: float, psi_b: float
) -> float:
    """Analytic ``|⟨amp_a|amp_b⟩|²`` for the rung-3 amplitude branch.

    With ``|amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩`` the squared
    overlap is

        cos²(θ_a) cos²(θ_b)
        + sin²(θ_a) sin²(θ_b)
        + 2 cos(θ_a) cos(θ_b) sin(θ_a) sin(θ_b) cos(ψ_b − ψ_a).

    At default knobs (θ = π/4 for both, ψ = 0 for both) the value is
    exactly 1 — so the rung-3 gram reduces to the MPSRung1-equivalent
    gram per ``Rung3``'s docstring.
    """
    ca, sa = math.cos(theta_a), math.sin(theta_a)
    cb, sb = math.cos(theta_b), math.sin(theta_b)
    return float(
        ca * ca * cb * cb
        + sa * sa * sb * sb
        + 2.0 * ca * cb * sa * sb * math.cos(psi_b - psi_a)
    )


def rung3_amp_overlap(
    theta_a: float, psi_a: float, theta_b: float, psi_b: float
) -> complex:
    """Analytic complex ``⟨amp_a|amp_b⟩`` for the rung-3 amp branch.

    ``|amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩`` gives

        ⟨amp_a|amp_b⟩ = cos(θ_a) cos(θ_b)
                        + e^(i(ψ_b − ψ_a)) sin(θ_a) sin(θ_b).

    The complex form is what ``Dictionary.gram()`` multiplies into
    the MPS overlap path so that ``np.abs(gram)**2`` factorizes
    correctly into ``|⟨mps|mps⟩|² · |⟨amp|amp⟩|²``.
    """
    ca, sa = math.cos(theta_a), math.sin(theta_a)
    cb, sb = math.cos(theta_b), math.sin(theta_b)
    delta = psi_b - psi_a
    return complex(
        ca * cb + sa * sb * math.cos(delta),
        sa * sb * math.sin(delta),
    )


@dataclass(frozen=True)
class Rung3State:
    """Per-feature rung-3 state — analytic ``compute_concept_gram(other)``.

    Parallel to the per-feature shape the spec describes. Carries the
    ``MPSRung1``-on-qubits-0–2 knobs (α, β, γ, φ) plus the amplitude-
    branch knobs (θ_amp, ψ_aux). Used by ``Dictionary.gram()``'s
    rung-3 dispatch and by the cancellation primitive's joint
    optimizer; not part of ``Dictionary``'s persisted shape.
    """

    alpha: float
    beta: float
    gamma: float
    phi: float
    theta_amp: float = RUNG3_DEFAULT_THETA_AMP
    psi_aux: float = RUNG3_DEFAULT_PSI_AUX

    def amp_overlap_squared(self, other: "Rung3State") -> float:
        """Analytic ``|⟨amp_self|amp_other⟩|²`` for the amplitude branch."""
        return rung3_amp_overlap_squared(
            self.theta_amp, self.psi_aux, other.theta_amp, other.psi_aux
        )

    @classmethod
    def from_mps_knobs(
        cls,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float,
        theta_amp: float = RUNG3_DEFAULT_THETA_AMP,
        psi_aux: float = RUNG3_DEFAULT_PSI_AUX,
    ) -> "Rung3State":
        """Construct a rung-3 state from MPSRung1-equivalent knobs plus
        the amplitude branch defaults."""
        return cls(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=phi,
            theta_amp=theta_amp,
            psi_aux=psi_aux,
        )
