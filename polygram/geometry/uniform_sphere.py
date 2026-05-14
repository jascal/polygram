"""`uniform-sphere` profile — calibrated for SAEs whose decoder rows
sit on a near-uniform sphere (cosine std ≤ ~0.06). Empirical scope:
any SAE with d_model ≥ ~1K and n_features ≥ ~16K, regardless of
modality (audio + text), training recipe (TopK + JumpReLU), decoder
normalization, or layer position. See
`docs/research/sae-geometry-regimes.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from polygram.geometry.clustered import _gamma_via_cluster_pca, _kmeans
from polygram.geometry.profile import GeometricProfile
from polygram.geometry.protocols import KnobAssignmentResult

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary


@dataclass(frozen=True)
class UniformSphereKnobAssignment:
    """k-means on unit-normalised projections + β via top-1 PCA
    coordinate of the centered selected subset, rescaled into
    `(-0.5, 0.5)`. β carries continuous geometric position; clusters
    carry tier identity but not β ordinal.

    `beta_variance_explained` is the fraction of selected-subset
    variance captured by the top-1 PCA component (not the cluster
    centroids — k-means residual is meaningless on uniform-sphere
    data).
    """

    beta_range: tuple[float, float] = (-0.5, 0.5)
    # n_init=4 is a cost/quality compromise: sklearn defaults to 10, but
    # our pure-numpy k-means is ~5x slower per run, and on the uniform-
    # sphere geometries this profile targets, runs converge to similar
    # inertia within 3-4 seeds (cluster identity is itself ambiguous on
    # near-orthogonal inputs). Bump if a downstream calibration shows
    # seed-sensitivity in the resulting fidelity.
    n_init: int = 4

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
        encoding: object = None,
    ) -> KnobAssignmentResult:
        n = len(feature_names)
        k = n_clusters if n_clusters is not None else 16
        if k > n:
            k = n

        # Cluster on unit vectors so cluster identity tracks angular
        # geometry (cosine ≈ Euclidean for unit norms).
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit = projections / norms

        # n_init>=4 — pick the lowest-inertia run from independent seeds.
        best_labels = None
        best_inertia = np.inf
        for s in range(self.n_init):
            labels, _ = _kmeans(unit, k, seed=seed + s)
            # Inertia: sum of within-cluster squared distances.
            inertia = 0.0
            for ci in range(int(labels.max()) + 1):
                mask = labels == ci
                if mask.any():
                    centroid = unit[mask].mean(axis=0)
                    inertia += float(np.sum((unit[mask] - centroid) ** 2))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
        cluster_per_feature = [f"cluster_{int(label)}" for label in best_labels]

        # β via top-1 PCA component of the centered selected-subset
        # projections (using raw projections, not unit, so the PCA picks
        # up magnitude variation when present). Rescale into beta_range.
        centered = projections - projections.mean(axis=0)
        if n >= 2:
            _, sv, vt = np.linalg.svd(centered, full_matrices=False)
            pc1 = vt[0]
            coords = centered @ pc1
            total_var = float(np.sum(sv ** 2))
            top1_var = float(sv[0] ** 2)
            beta_var_explained = (
                top1_var / total_var if total_var > 1e-12 else 1.0
            )
            abs_max = float(np.max(np.abs(coords)))
            if abs_max < 1e-12:
                betas = [0.5 * (self.beta_range[0] + self.beta_range[1])] * n
            else:
                lo, hi = self.beta_range
                half = 0.5 * (hi - lo)
                mid = 0.5 * (hi + lo)
                betas = (coords / abs_max * half + mid).tolist()
        else:
            betas = [0.5 * (self.beta_range[0] + self.beta_range[1])] * n
            beta_var_explained = 1.0

        if assign_gamma:
            gammas = _gamma_via_cluster_pca(
                projections, cluster_per_feature, gamma_range
            )
        else:
            gammas = [0.0] * n

        amp_assignments: dict[str, list[float] | None] = {
            "theta_amps": None,
            "psi_auxes": None,
            "theta_amp_bs": None,
            "psi_amp_bs": None,
        }
        if assign_amp_knobs and encoding is not None:
            from polygram.geometry.amp_assignment import assign_amp_knobs_pca

            amp_assignments = assign_amp_knobs_pca(projections, encoding)

        return KnobAssignmentResult(
            cluster_per_feature=cluster_per_feature,
            betas=list(betas),
            gammas=list(gammas),
            cluster_method="pca_axis",
            beta_variance_explained=float(np.clip(beta_var_explained, 0.0, 1.0)),
            theta_amps=amp_assignments["theta_amps"],
            psi_auxes=amp_assignments["psi_auxes"],
            theta_amp_bs=amp_assignments["theta_amp_bs"],
            psi_amp_bs=amp_assignments["psi_amp_bs"],
        )


@dataclass(frozen=True)
class RankRecallAtKFidelity:
    """Top-k off-diagonal pairs by Polygram Gram `|G|²` ∩ top-k by
    projection-space cosine, divided by k. Bounded `[0, 1]`, higher
    is better. `k = max(3, n_pairs // 2)`. Returns `None` when fewer
    than `k+1` off-diagonal pairs exist.
    """

    def compute(
        self, projections: np.ndarray, dictionary: "Dictionary"
    ) -> float | None:
        n = projections.shape[0]
        if n <= 1:
            return None
        n_pairs = n * (n - 1) // 2
        k = max(3, n_pairs // 2)
        if n_pairs < k + 1:
            return None

        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        proj_unit = projections / norms
        cos_overlap = np.abs(proj_unit @ proj_unit.T)
        gram_sq = np.abs(dictionary.gram()) ** 2

        iu = np.triu_indices(n, k=1)
        cos_pairs = cos_overlap[iu]
        gram_pairs = gram_sq[iu]

        top_k_cos = set(np.argsort(-cos_pairs)[:k].tolist())
        top_k_gram = set(np.argsort(-gram_pairs)[:k].tolist())
        return float(len(top_k_cos & top_k_gram) / k)


def uniform_sphere() -> GeometricProfile:
    """Built-in profile: SAEs with `d_model ≥ ~1K`, `n_features ≥ ~16K`.

    k≥16 k-means on unit-normalised projections; β via top-1 PCA-axis
    coordinate; γ via per-cluster PCA when `assign_gamma=True`;
    `rank_recall_at_k` fidelity replaces the Pearson tier_preservation
    that collapses on this regime.

    Empirical scope: audio TopK SAEs, Qwen-Scope, Llama-Scope (TopK +
    JumpReLU). See `docs/research/sae-geometry-regimes.md`.
    """
    return GeometricProfile(
        name="uniform-sphere",
        knob_assignment=UniformSphereKnobAssignment(),
        geometric_fidelity=RankRecallAtKFidelity(),
        default_n_clusters=16,
        default_gamma_range=(-0.25, 0.25),
    )
