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
    # Rung5 product-amp knobs as one length-k tuple-of-pairs per
    # feature. `None` (default) signals "use encoding defaults"
    # (which for Rung5 fails Dictionary validation unless populated).
    # Non-Rung5 encodings leave this as None.
    amp_knobs_list: list[tuple[tuple[float, float], ...]] | None = None
    # add-learned-axis-assignment. `LearnedKnobAssignment` populates
    # these to surface what was learned and how well it scored;
    # `ClusteredKnobAssignment` / `UniformSphereKnobAssignment` leave
    # them all `None`. `axis_assignment` carries either a
    # knob → PCA-axis-index map (greedy solver) or a
    # knob → axis-coefficient-vector map (scipy solver).
    # `objective_value` is the validation-set objective when
    # `validation_fraction > 0`, otherwise equal to
    # `training_objective_value`.
    axis_assignment: dict[str, int | list[float]] | None = None
    objective_value: float | None = None
    objective_baseline: float | None = None
    training_objective_value: float | None = None
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
class LearnedAxisObjective(Protocol):
    """Scalar fidelity score for a candidate analytic gram against a
    reference geometry. ``LearnedKnobAssignment`` maximises this
    objective during its axis-to-knob search.

    ``feature_names`` is keyword-only with a ``None`` default so
    simple objectives (Spearman, Pearson, behavioural) can ignore it.
    Custom objectives that need cluster context still receive it
    when the strategy invokes them.
    """

    def __call__(
        self,
        analytic_gram: np.ndarray,
        decoder_geom: np.ndarray,
        *,
        feature_names: list[str] | None = None,
    ) -> float:
        ...


@runtime_checkable
class GeometricFidelity(Protocol):
    """Profile-specific scalar fidelity metric. Returning `None`
    signals "not defined for this geometry / sample size"."""

    def compute(
        self, projections: np.ndarray, dictionary: "Dictionary"
    ) -> float | None:
        ...
