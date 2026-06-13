"""`readout-aligned` profile — build the dictionary geometry on the basis the
model *reads through*, not on raw decoder vectors.

Before the existing `clustered` k-means + β/γ assignment, project the decoder
vectors onto the **readout subspace**: the top-`r` right singular directions of
`gain ⊙ U` (the host unembed, optionally weighted by the final-norm gain). The
powered cross-model R2 result (fieldrun `tau_star_powered.py`) shows this
subspace — not the energy-weighted decoder spectrum — is what governs the
model's argmax (GPT-2 +52pp / Pythia-70m +31pp / Pythia-160m +40pp open-class
R@32 vs the frozen SVD lens). So fitting the geometry here, rather than on raw
decoder norm, tests whether Polygram's geometry-tracks-behaviour Spearman was
partly measuring the (wrong) basis.

Scope: the projection is a pre-processing of the per-feature projections; the
clustering, β spread, per-cluster-PCA γ, and the fidelity's cosine-overlap target
all run in the projected space, so the resulting knobs (and therefore the Gram /
Q-Orca emission, which consume the knobs) reflect readout geometry. Nothing about
the encoding structure or the quantum machine changes. `u_matrix`/`gain` are
supplied by the caller (Polygram stays numpy-light — no host load in the core).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from polygram.geometry.clustered import ClusteredKnobAssignment, TierPreservationFidelity
from polygram.geometry.profile import GeometricProfile
from polygram.geometry.protocols import KnobAssignmentResult

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary

_DEFAULT_READOUT_RANK = 64


def _resolve_gain(gain: "np.ndarray | float | None", d_model: int) -> np.ndarray:
    """Final-norm gain → a `(d_model,)` vector. None → ones; scalar → broadcast."""
    if gain is None:
        return np.ones(d_model, dtype=float)
    g = np.asarray(gain, dtype=float)
    if g.ndim == 0:
        return np.full(d_model, float(g))
    if g.shape != (d_model,):
        raise ValueError(
            f"readout-aligned gain must be a scalar or a ({d_model},) vector "
            f"matching d_model; got shape {g.shape}"
        )
    return g


def readout_subspace(
    u_matrix: np.ndarray, gain: "np.ndarray | float | None", d_model: int, rank: int
) -> np.ndarray:
    """Return `Vt` `(r, d_model)` — the top-`r` right singular directions of
    `gain ⊙ U`. `r = min(rank, d_model, vocab)` (so `rank > rank(gain⊙U)` is safe).
    `R = Vt.T`; projecting a row vector is `v @ Vt.T`."""
    U = np.asarray(u_matrix, dtype=float)
    if U.ndim != 2 or U.shape[1] != d_model:
        raise ValueError(
            f"readout-aligned u_matrix must be (vocab, d_model) with "
            f"d_model={d_model}; got {U.shape}"
        )
    weighted = U * _resolve_gain(gain, d_model)[None, :]  # (vocab, d_model)
    r = int(min(max(1, rank), d_model, U.shape[0]))
    vt = np.linalg.svd(weighted, full_matrices=False)[2]  # (min(vocab,d_model), d_model)
    return vt[:r]


def _project(projections: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """`(n_features, d_model) @ (r, d_model)ᵀ -> (n_features, r)`."""
    return np.asarray(projections, dtype=float) @ vt.T


def readout_variance_captured(
    projections: np.ndarray,
    u_matrix: np.ndarray,
    gain: "np.ndarray | float | None" = None,
    rank: int = _DEFAULT_READOUT_RANK,
) -> float:
    """Diagnostic: fraction of decoder-vector Frobenius energy captured by the
    top-`r` readout subspace = ‖P·proj‖²_F / ‖proj‖²_F ∈ [0, 1]. Near 1 ⇒ the SAE
    features already live in the readout subspace (little to gain); low ⇒ raw and
    readout geometries diverge sharply."""
    proj = np.asarray(projections, dtype=float)
    total = float(np.sum(proj ** 2))
    if total < 1e-12:
        return 1.0
    vt = readout_subspace(u_matrix, gain, proj.shape[1], rank)
    captured = float(np.sum(_project(proj, vt) ** 2))
    return float(np.clip(captured / total, 0.0, 1.0))


@dataclass(frozen=True)
class ReadoutAlignedKnobAssignment:
    """k-means + β/γ assignment (the `clustered` logic) run in the readout
    subspace. `u_matrix`/`gain` are injected by `from_sae_lens` at import time;
    a `None` `u_matrix` at assign time is a usage error."""

    beta_range: tuple[float, float] = (-0.5, 0.5)
    u_matrix: np.ndarray | None = None
    gain: "np.ndarray | float | None" = None
    readout_rank: int = _DEFAULT_READOUT_RANK

    def assign(
        self,
        projections: np.ndarray,
        feature_names: list[str],
        *,
        n_clusters: int | None,
        gamma_range: tuple[float, float],
        assign_gamma: bool,
        seed: int,
        assign_amp_knobs: bool = False,
        assign_phase_knobs: bool = False,
        encoding: object = None,
    ) -> KnobAssignmentResult:
        if self.u_matrix is None:
            raise ValueError(
                "readout-aligned profile: u_matrix is unset. Pass "
                "from_sae_lens(..., profile='readout-aligned', u_matrix=U) so "
                "the readout subspace can be built."
            )
        vt = readout_subspace(
            self.u_matrix, self.gain, np.asarray(projections).shape[1], self.readout_rank
        )
        proj_readout = _project(projections, vt)
        # Delegate to the calibrated clustered logic — in the projected space.
        return ClusteredKnobAssignment(self.beta_range).assign(
            proj_readout,
            feature_names,
            n_clusters=n_clusters,
            gamma_range=gamma_range,
            assign_gamma=assign_gamma,
            seed=seed,
            assign_amp_knobs=assign_amp_knobs,
            assign_phase_knobs=assign_phase_knobs,
            encoding=encoding,
        )


@dataclass(frozen=True)
class ReadoutAlignedFidelity:
    """`tier_preservation` Pearson fidelity computed on the readout-projected
    cosine-overlap target (consistent with the readout-space knobs)."""

    u_matrix: np.ndarray | None = None
    gain: "np.ndarray | float | None" = None
    readout_rank: int = _DEFAULT_READOUT_RANK

    def compute(
        self, projections: np.ndarray, dictionary: "Dictionary"
    ) -> float | None:
        if self.u_matrix is None:
            return None
        vt = readout_subspace(
            self.u_matrix, self.gain, np.asarray(projections).shape[1], self.readout_rank
        )
        proj_readout = _project(projections, vt)
        return TierPreservationFidelity().compute(proj_readout, dictionary)


def readout_aligned() -> GeometricProfile:
    """Built-in profile: cluster + assign knobs in the model's readout subspace.

    Registered with placeholder (`u_matrix=None`) strategies; `from_sae_lens`
    injects the caller-supplied `u_matrix`/`gain`/`readout_rank` via
    `dataclasses.replace` and raises if `u_matrix` is missing.
    """
    return GeometricProfile(
        name="readout-aligned",
        knob_assignment=ReadoutAlignedKnobAssignment(),
        geometric_fidelity=ReadoutAlignedFidelity(),
        default_n_clusters=2,
        default_gamma_range=(-0.25, 0.25),
    )
