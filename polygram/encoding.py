"""Encoding markers — config tags that downstream emitters dispatch on.

Each encoding declares its `max_features` cap — the maximum number of
linearly-independent features the encoding's state space can hold.
Loaders and validators query this attribute rather than a hardcoded
constant. See `docs/research/rung3-rank-bound.md` for the empirical
basis of the per-encoding values.
"""

import math
from dataclasses import dataclass
from typing import ClassVar


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

    The per-encoding feature cap is `dim(C^8) = 8` — every additional
    feature past 8 is forced to live in a linear combination of the
    existing 8 (rank-deficient Gram).
    """

    bond_dim: int = 2
    phase_knobs: bool = True

    max_features: ClassVar[int] = 8

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

    @property
    def max_features(self) -> int:
        """Per-encoding feature cap: ``2 ** n_qubits`` (full Hilbert dim
        of the qubit register). Scales with the existing ``n_qubits``
        knob, no per-encoding constant needed."""
        return 2 ** self.n_qubits


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

    The per-encoding feature cap is **16**, not 32. The amp branch's
    parameterization ``|amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩``
    is restricted to the 2-dim subspace ``span{|00⟩, |11⟩}`` of the
    2-qubit Hilbert space; ``|01⟩`` and ``|10⟩`` are structurally
    unreachable. Total: ``C^8 ⊗ C^2 = C^16``. See
    ``docs/research/rung3-rank-bound.md`` for the empirical confirmation
    (sharp algebraic limit at N=16 across two seeds).
    """

    bond_dim: int = 2

    max_features: ClassVar[int] = 16

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


def _single_qubit_overlap(
    theta_a: float, psi_a: float, theta_b: float, psi_b: float
) -> complex:
    """Analytic complex overlap of two states parameterised as
    ``cos(θ)|0⟩ + e^(iψ) sin(θ)|1⟩`` (a Schmidt-style single-qubit
    family — note the *full* angle θ, not the Bloch half-angle θ/2).

        ⟨u_a | u_b⟩ = cos(θ_a) cos(θ_b)
                      + e^(i(ψ_b − ψ_a)) sin(θ_a) sin(θ_b).

    Shared building block for Rung3's amp-branch overlap (which uses
    this exact formula on the |00⟩/|11⟩ subspace, with θ as the
    Schmidt angle) and Rung4's product amp branch (which uses two
    independent invocations, one per qubit factor).
    """
    ca, sa = math.cos(theta_a), math.sin(theta_a)
    cb, sb = math.cos(theta_b), math.sin(theta_b)
    delta = psi_b - psi_a
    return complex(
        ca * cb + sa * sb * math.cos(delta),
        sa * sb * math.sin(delta),
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

    Numerically identical to the inlined pre-Rung4 implementation;
    the body now delegates to ``_single_qubit_overlap`` which Rung4
    also consumes (its product amp branch is two independent
    single-qubit overlaps).
    """
    return _single_qubit_overlap(theta_a, psi_a, theta_b, psi_b)


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


# Default values for the Rung4 product amplitude branch — both
# single-qubit amps at |0⟩ make each overlap factor = 1, so the
# Rung4 gram with all features at defaults equals the MPSRung1-
# equivalent gram on the same (α, β, γ, φ).
RUNG4_DEFAULT_THETA_AMP = 0.0
RUNG4_DEFAULT_PSI_AUX = 0.0
RUNG4_DEFAULT_THETA_AMP_B = 0.0
RUNG4_DEFAULT_PSI_AMP_B = 0.0


@dataclass(frozen=True)
class Rung4:
    """Rung-4 encoding: MPSRung1 on qubits 0–2 plus a **product**
    amplitude branch on qubits 3 and 4 (two independent single-qubit
    amps; no entanglement between q3 and q4, unlike Rung3's Bell-
    pattern amp).

    Each feature's amp state factorises as

        |amp(θ_a, ψ_a, θ_b, ψ_b)⟩
            = (cos(θ_a)|0⟩ + e^(iψ_a) sin(θ_a)|1⟩)_{q3}
              ⊗ (cos(θ_b)|0⟩ + e^(iψ_b) sin(θ_b)|1⟩)_{q4}

    where each single-qubit factor's parameter family linearly spans
    its C². The product spans the full ``C^2 ⊗ C^2 = C^4`` amp
    subspace (vs Rung3's restricted 2-dim ``span{|00⟩, |11⟩}``), so
    the Rung4 per-feature Hilbert dim is ``8 · 4 = 32`` — twice
    Rung3's, four times MPSRung1's.

    **Default knobs** are ``theta_amp = theta_amp_b = 0`` and
    ``psi_aux = psi_amp_b = 0`` for every feature; each single-qubit
    amp reduces to ``|0⟩`` and the amp overlap factor equals 1, so a
    default-knob Rung4 dictionary's gram matches the MPSRung1-
    equivalent gram on the same (α, β, γ, φ). This mirrors Rung3's
    "default reduces to MPS" property at a different fixed point.

    The amp parameterization uses *full* angles (not the Bloch
    half-angle convention) to match Rung3's existing conventions —
    both encodings consume the shared ``_single_qubit_overlap``
    helper.

    See ``docs/research/rung3-rank-bound.md`` for the dimensional
    analysis that motivates this design.
    """

    bond_dim: int = 2

    max_features: ClassVar[int] = 32

    def __post_init__(self) -> None:
        if self.bond_dim != 2:
            raise ValueError(
                f"Rung4.bond_dim must be 2 (rung-1 / χ=2 only in v0); "
                f"got {self.bond_dim}"
            )


def rung4_amp_overlap(
    theta_a3: float,
    psi_a3: float,
    theta_a4: float,
    psi_a4: float,
    theta_b3: float,
    psi_b3: float,
    theta_b4: float,
    psi_b4: float,
) -> complex:
    """Analytic complex ``⟨amp_a|amp_b⟩`` for the Rung4 product amp.

    Product of two independent single-qubit overlaps — one for the
    q3 amp factor, one for q4:

        ⟨amp_a|amp_b⟩ = ⟨u_a | u_b⟩_{q3} · ⟨v_a | v_b⟩_{q4}

    where each factor is a ``_single_qubit_overlap`` evaluation.
    """
    return _single_qubit_overlap(
        theta_a3, psi_a3, theta_b3, psi_b3
    ) * _single_qubit_overlap(
        theta_a4, psi_a4, theta_b4, psi_b4
    )


def rung4_amp_overlap_squared(
    theta_a3: float,
    psi_a3: float,
    theta_a4: float,
    psi_a4: float,
    theta_b3: float,
    psi_b3: float,
    theta_b4: float,
    psi_b4: float,
) -> float:
    """``|⟨amp_a|amp_b⟩|²`` for the Rung4 product amp.

    Equivalent to ``abs(rung4_amp_overlap(...)) ** 2`` and also to
    the product of the two single-qubit *squared* overlaps. Either
    form is mathematically identical; we compute via the complex
    product for floating-point stability with `abs(complex)`.
    """
    z = rung4_amp_overlap(
        theta_a3, psi_a3, theta_a4, psi_a4,
        theta_b3, psi_b3, theta_b4, psi_b4,
    )
    return float(abs(z) ** 2)


@dataclass(frozen=True)
class Rung4State:
    """Per-feature Rung4 state — parallel to ``Rung3State``.

    Carries the MPSRung1-on-qubits-0–2 knobs (α, β, γ, φ) plus the
    Rung4 product amp's four per-feature knobs:

    - ``theta_amp``, ``psi_aux`` — q3 single-qubit amp (reuses the
      Rung3-shipped ``Feature.theta_amp`` / ``Feature.psi_aux``
      fields).
    - ``theta_amp_b``, ``psi_amp_b`` — q4 single-qubit amp (Rung4's
      additions to ``Feature``).

    Not part of ``Dictionary``'s persisted shape; constructed on
    demand by ``Dictionary.gram()``'s Rung4 dispatch and by the
    cancellation primitive's joint optimiser.
    """

    alpha: float
    beta: float
    gamma: float
    phi: float
    theta_amp: float = RUNG4_DEFAULT_THETA_AMP
    psi_aux: float = RUNG4_DEFAULT_PSI_AUX
    theta_amp_b: float = RUNG4_DEFAULT_THETA_AMP_B
    psi_amp_b: float = RUNG4_DEFAULT_PSI_AMP_B

    def amp_overlap_squared(self, other: "Rung4State") -> float:
        """Analytic ``|⟨amp_self|amp_other⟩|²`` for the Rung4 amp branch."""
        return rung4_amp_overlap_squared(
            self.theta_amp,
            self.psi_aux,
            self.theta_amp_b,
            self.psi_amp_b,
            other.theta_amp,
            other.psi_aux,
            other.theta_amp_b,
            other.psi_amp_b,
        )

    @classmethod
    def from_mps_knobs(
        cls,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float,
        *,
        theta_amp: float = RUNG4_DEFAULT_THETA_AMP,
        psi_aux: float = RUNG4_DEFAULT_PSI_AUX,
        theta_amp_b: float = RUNG4_DEFAULT_THETA_AMP_B,
        psi_amp_b: float = RUNG4_DEFAULT_PSI_AMP_B,
    ) -> "Rung4State":
        """Construct a Rung4 state from MPSRung1-equivalent knobs plus
        the product amp's four default knobs."""
        return cls(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=phi,
            theta_amp=theta_amp,
            psi_aux=psi_aux,
            theta_amp_b=theta_amp_b,
            psi_amp_b=psi_amp_b,
        )


# Per-feature Hilbert dim for Rung5 is 8 * 2**k where k = n_amp_qubits.
# Cap k at 16 → max_features = 524288. The cap is exposed as a module
# constant so sae-forge (and other callers) can validate sweep ranges
# without hard-coding the value.
RUNG5_MAX_N_AMP_QUBITS: int = 16


@dataclass(frozen=True)
class Rung5:
    """Rung-5 encoding: MPSRung1 on qubits 0–2 plus a product amplitude
    branch of ``n_amp_qubits`` independent single-qubit amps on qubits
    q3..q3+n_amp_qubits-1.

    Generalises ``Rung4`` (the fixed ``n_amp_qubits=2`` case) to an
    arbitrary amp-register width fixed at construction time. Per-feature
    Hilbert dim is ``8 · 2^n_amp_qubits`` — ``max_features`` scales
    accordingly.

    Each feature's amp state factorises as

        |amp(θ_0, ψ_0, …, θ_{k−1}, ψ_{k−1})⟩
            = ⊗_{i=0}^{k−1} (cos(θ_i)|0⟩ + e^(iψ_i) sin(θ_i)|1⟩)_{q(3+i)}

    where k = ``n_amp_qubits``. No entangling gates are applied between
    amp qubits.

    Default knobs (every ``(θ_i, ψ_i) == (0, 0)``) reduce each
    single-qubit overlap factor to 1, so a default-knob Rung5
    dictionary's gram equals the MPSRung1-equivalent gram on the same
    (α, β, γ, φ) — the same fixed-point property Rung3/Rung4 ship.

    ``n_amp_qubits=0`` is rejected: that case is numerically identical
    to ``MPSRung1`` and should be spelled that way directly. The
    discriminator on Rung5 is the *presence* of an amp branch.
    """

    bond_dim: int = 2
    n_amp_qubits: int = 0

    def __post_init__(self) -> None:
        if self.bond_dim != 2:
            raise ValueError(
                f"Rung5.bond_dim must be 2 (rung-1 / χ=2 only in v0); "
                f"got {self.bond_dim}"
            )
        if self.n_amp_qubits < 1:
            raise ValueError(
                f"Rung5.n_amp_qubits must be >= 1 "
                f"(use MPSRung1 directly for the 3-qubit MPS-only case); "
                f"got {self.n_amp_qubits}"
            )
        if self.n_amp_qubits > RUNG5_MAX_N_AMP_QUBITS:
            raise ValueError(
                f"Rung5.n_amp_qubits must be <= "
                f"{RUNG5_MAX_N_AMP_QUBITS} "
                f"(max_features = 8 * 2**{RUNG5_MAX_N_AMP_QUBITS} "
                f"= {8 * 2 ** RUNG5_MAX_N_AMP_QUBITS}); "
                f"got {self.n_amp_qubits}"
            )

    @property
    def max_features(self) -> int:
        """Per-encoding feature cap: ``8 · 2^n_amp_qubits`` (MPSRung1
        core's 8-dim Hilbert space times each amp qubit's 2-dim
        Hilbert space). Scales with the ``n_amp_qubits`` knob."""
        return 8 * 2 ** self.n_amp_qubits


def rung5_amp_overlap(
    amp_a: tuple[tuple[float, float], ...],
    amp_b: tuple[tuple[float, float], ...],
) -> complex:
    """Analytic complex ``⟨amp_a|amp_b⟩`` for the Rung5 product amp.

    Product of ``len(amp_a)`` independent single-qubit overlaps:

        ⟨amp_a|amp_b⟩ = ∏_i ⟨u_a_i | u_b_i⟩

    where each factor is a ``_single_qubit_overlap`` evaluation on the
    i-th amp-qubit's (θ, ψ) pair.

    Raises ``ValueError`` if the two tuples have different lengths.
    """
    if len(amp_a) != len(amp_b):
        raise ValueError(
            f"rung5_amp_overlap: amp_a and amp_b must have the same "
            f"length; got {len(amp_a)} and {len(amp_b)}"
        )
    result: complex = complex(1.0, 0.0)
    for (theta_a, psi_a), (theta_b, psi_b) in zip(amp_a, amp_b):
        result *= _single_qubit_overlap(theta_a, psi_a, theta_b, psi_b)
    return result


def rung5_amp_overlap_squared(
    amp_a: tuple[tuple[float, float], ...],
    amp_b: tuple[tuple[float, float], ...],
) -> float:
    """``|⟨amp_a|amp_b⟩|²`` for the Rung5 product amp.

    Equivalent to ``abs(rung5_amp_overlap(...)) ** 2`` and also to the
    product of the per-qubit *squared* overlaps.
    """
    z = rung5_amp_overlap(amp_a, amp_b)
    return float(abs(z) ** 2)


@dataclass(frozen=True)
class Rung5State:
    """Per-feature Rung5 state — parallel to ``Rung4State``.

    Carries the MPSRung1-on-qubits-0–2 knobs (α, β, γ, φ) plus the
    Rung5 product amp's ``k`` per-qubit (θ, ψ) pairs as a tuple. Not
    part of ``Dictionary``'s persisted shape; constructed on demand by
    ``Dictionary.gram()``'s Rung5 dispatch and the cancellation
    primitive's joint optimiser.
    """

    alpha: float
    beta: float
    gamma: float
    phi: float
    amp_knobs: tuple[tuple[float, float], ...] = ()

    def amp_overlap_squared(self, other: "Rung5State") -> float:
        """Analytic ``|⟨amp_self|amp_other⟩|²`` for the Rung5 amp branch."""
        return rung5_amp_overlap_squared(self.amp_knobs, other.amp_knobs)

    @classmethod
    def from_mps_knobs(
        cls,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float,
        *,
        amp_knobs: tuple[tuple[float, float], ...] = (),
    ) -> "Rung5State":
        """Construct a Rung5 state from MPSRung1-equivalent knobs plus
        the product amp's per-qubit (θ, ψ) pairs."""
        return cls(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=phi,
            amp_knobs=amp_knobs,
        )
