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


@runtime_checkable
class KnobAssignment(Protocol):
    """Maps selected projection vectors to per-feature `(β, γ, cluster)`.

    Implementations are invoked by the k-means dispatch path of
    `from_sae_lens` only — `cluster_assignments` and `from_labels`
    paths run upstream and bypass the strategy entirely.
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
