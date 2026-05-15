"""Strategy protocols + result type for geometric-regime profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary


@dataclass(frozen=True)
class KnobAssignmentResult:
    cluster_per_feature: list[str]
    betas: list[float]
    gammas: list[float]
    cluster_method: str
    beta_variance_explained: float
    # encoding-aware-knob-assignment. None (default) signals "use
    # encoding defaults"; a populated list overrides the encoding's
    # amp-branch knob value per-feature. Length matches
    # cluster_per_feature when populated. theta_amp_bs / psi_amp_bs
    # remain None for encodings without a branch-B amp (Rung3).
    theta_amps: list[float] | None = None
    psi_auxes: list[float] | None = None
    theta_amp_bs: list[float] | None = None
    psi_amp_bs: list[float] | None = None
    # add-phase-knob-assignment. None (default) signals "use encoding
    # defaults" (typically α=0, φ=0); a populated list overrides the
    # encoding's MPS-substrate phase knob per-feature. Length matches
    # cluster_per_feature when populated. HEA_Rung2 leaves these as
    # None.
    alphas: list[float] | None = None
    phis: list[float] | None = None


@runtime_checkable
class KnobAssignment(Protocol):
    """Maps selected projection vectors to per-feature `(β, γ, cluster)`,
    optionally extending to amp-branch knobs for higher-rung encodings.

    Implementations are invoked by the k-means dispatch path of
    `from_sae_lens` only — `cluster_assignments` and `from_labels`
    paths run upstream and bypass the strategy entirely.

    Strategies that support `assign_amp_knobs=True` populate the
    result's `theta_amps`, `psi_auxes`, `theta_amp_bs`, `psi_amp_bs`
    fields. Strategies that don't leave them as `None`; the loader
    falls back to the encoding's default knob values.
    """

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
        ...


@runtime_checkable
class GeometricFidelity(Protocol):
    """Profile-specific scalar fidelity metric. Returning `None`
    signals "not defined for this geometry / sample size"."""

    def compute(
        self, projections: np.ndarray, dictionary: "Dictionary"
    ) -> float | None:
        ...
