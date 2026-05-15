"""`clustered` profile — calibrated for small dense LM SAEs at
GPT-2-small scale (d_model ≤ 768, ≤24K features). The v0.1.0 default.

K-means on raw projections + antipodal β spread over `(-0.5, 0.5)` +
optional per-cluster PCA γ + Pearson `tier_preservation` fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from polygram.geometry.profile import GeometricProfile
from polygram.geometry.protocols import KnobAssignmentResult

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary


# --- Helpers (lifted verbatim from polygram/sae_import.py for v0.1.0
# byte-equivalence; signatures and float behaviour preserved) -----------------


def _kmeans(
    points: np.ndarray, k: int, seed: int = 0, max_iter: int = 100
) -> tuple[np.ndarray, list[int]]:
    """Tiny Lloyd's-algorithm k-means in pure numpy with k-means++ init.

    Returns `(assignments, empty_cluster_indices)`. Deterministic
    given the seed.
    """
    n = len(points)
    if k <= 1:
        return np.zeros(n, dtype=int), []
    rng = np.random.default_rng(seed)

    centroids = np.empty((k, points.shape[1]), dtype=points.dtype)
    centroids[0] = points[rng.integers(0, n)]
    for ci in range(1, k):
        d2 = np.min(
            np.sum((points[:, None, :] - centroids[None, :ci, :]) ** 2, axis=2),
            axis=1,
        )
        total = d2.sum()
        if total <= 0:
            centroids[ci] = points[rng.integers(0, n)]
            continue
        probs = d2 / total
        idx = int(rng.choice(n, p=probs))
        centroids[ci] = points[idx]

    assignments = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_assignments = np.argmin(dists, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for ci in range(k):
            mask = assignments == ci
            if mask.any():
                centroids[ci] = points[mask].mean(axis=0)

    empties = [int(ci) for ci in range(k) if not (assignments == ci).any()]
    return assignments, empties


def _spread_betas(
    cluster_order: list[str], beta_range: tuple[float, float]
) -> dict[str, float]:
    n = len(cluster_order)
    lo, hi = beta_range
    if n == 0:
        return {}
    if n == 1:
        return {cluster_order[0]: 0.5 * (lo + hi)}
    return {c: lo + (hi - lo) * i / (n - 1) for i, c in enumerate(cluster_order)}


def _centroids(
    projs: np.ndarray, cluster_per_feature: list[str]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cluster in set(cluster_per_feature):
        mask = np.array([c == cluster for c in cluster_per_feature])
        out[cluster] = projs[mask].mean(axis=0)
    return out


def _variance_explained(
    projs: np.ndarray,
    centroids_by_cluster: dict[str, np.ndarray],
    cluster_per_feature: list[str],
) -> float:
    overall_centroid = projs.mean(axis=0)
    ss_total = float(np.sum((projs - overall_centroid) ** 2))
    if ss_total < 1e-12:
        return 1.0
    ss_residual = 0.0
    for i, c in enumerate(cluster_per_feature):
        diff = projs[i] - centroids_by_cluster[c]
        ss_residual += float(np.sum(diff ** 2))
    return float(np.clip(1.0 - ss_residual / ss_total, 0.0, 1.0))


def _gamma_via_cluster_pca(
    projs: np.ndarray,
    cluster_per_feature: list[str],
    gamma_range: tuple[float, float],
) -> list[float]:
    """Per-cluster PCA on centered projections; γ for each feature is
    its coefficient on the cluster's first PC, rescaled into
    `gamma_range`. Singletons get γ = 0."""
    lo, hi = gamma_range
    n = len(cluster_per_feature)
    raw = np.zeros(n, dtype=float)
    for cluster in set(cluster_per_feature):
        idx = [i for i, c in enumerate(cluster_per_feature) if c == cluster]
        if len(idx) < 2:
            continue
        sub = projs[idx]
        centered = sub - sub.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0]
        coeffs = centered @ pc1
        for k, val in zip(idx, coeffs):
            raw[k] = float(val)
    if not np.any(raw):
        return raw.tolist()
    abs_max = float(np.max(np.abs(raw)))
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    scaled = raw / abs_max * half + mid
    return scaled.tolist()


# --- Strategy + profile ------------------------------------------------------


@dataclass(frozen=True)
class ClusteredKnobAssignment:
    """k-means + antipodal β spread + per-cluster PCA γ. v0.1.0 default."""

    beta_range: tuple[float, float] = (-0.5, 0.5)

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
        n = len(feature_names)
        k = n_clusters if n_clusters is not None else 2
        if k > n:
            k = n
        labels, _ = _kmeans(projections, k, seed=seed)
        cluster_per_feature = [f"cluster_{int(label)}" for label in labels]

        cluster_order: list[str] = []
        seen: set[str] = set()
        for c in cluster_per_feature:
            if c not in seen:
                cluster_order.append(c)
                seen.add(c)

        betas_by_cluster = _spread_betas(cluster_order, self.beta_range)
        centroids_by_cluster = _centroids(projections, cluster_per_feature)
        var_explained = _variance_explained(
            projections, centroids_by_cluster, cluster_per_feature
        )

        if assign_gamma:
            gammas = _gamma_via_cluster_pca(
                projections, cluster_per_feature, gamma_range
            )
        else:
            gammas = [0.0] * n

        betas = [betas_by_cluster[c] for c in cluster_per_feature]

        amp_assignments: dict[str, list[float] | None] = {
            "theta_amps": None,
            "psi_auxes": None,
            "theta_amp_bs": None,
            "psi_amp_bs": None,
        }
        if assign_amp_knobs and encoding is not None:
            from polygram.geometry.amp_assignment import assign_amp_knobs_pca

            amp_assignments = assign_amp_knobs_pca(projections, encoding)

        phase_assignments: dict[str, list[float] | None] = {
            "alphas": None,
            "phis": None,
        }
        if assign_phase_knobs and encoding is not None:
            from polygram.geometry.phase_assignment import (
                assign_phase_knobs_pca,
            )

            phase_assignments = assign_phase_knobs_pca(projections, encoding)

        return KnobAssignmentResult(
            cluster_per_feature=cluster_per_feature,
            betas=betas,
            gammas=gammas,
            cluster_method="kmeans",
            beta_variance_explained=var_explained,
            theta_amps=amp_assignments["theta_amps"],
            psi_auxes=amp_assignments["psi_auxes"],
            theta_amp_bs=amp_assignments["theta_amp_bs"],
            psi_amp_bs=amp_assignments["psi_amp_bs"],
            alphas=phase_assignments["alphas"],
            phis=phase_assignments["phis"],
        )


@dataclass(frozen=True)
class TierPreservationFidelity:
    """v0.1.0 Pearson correlation between off-diagonal `|G|²` of the
    projection-space cosine-overlap matrix and the analytic Polygram
    Gram of the built `Dictionary` at φ=0."""

    def compute(
        self, projections: np.ndarray, dictionary: "Dictionary"
    ) -> float | None:
        n = projections.shape[0]
        if n <= 1:
            return None
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        proj_unit = projections / norms
        cos_overlap = np.abs(proj_unit @ proj_unit.T) ** 2

        gram = np.abs(dictionary.gram()) ** 2

        iu = np.triu_indices(n, k=1)
        a = cos_overlap[iu]
        b = gram[iu]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])


def clustered() -> GeometricProfile:
    """Built-in profile: small dense LM SAEs at GPT-2-small scale.

    The v0.1.0 default — k=2 k-means, β = ±0.5 antipodal spread,
    Pearson `tier_preservation` fidelity, γ via per-cluster PCA.
    """
    return GeometricProfile(
        name="clustered",
        knob_assignment=ClusteredKnobAssignment(),
        geometric_fidelity=TierPreservationFidelity(),
        default_n_clusters=2,
        default_gamma_range=(-0.25, 0.25),
    )
