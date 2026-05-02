"""SAE import utilities — bridge SAE-Lens / Anthropic-style sparse
autoencoder dictionaries into Polygram `Dictionary` objects.

The bridge is *selection-first*: real SAEs ship 16k–1M features but
Polygram's rung-1 MPS encoding holds at most 8 features per
dictionary. The user names a small subset by feature id; this module
clusters their projection vectors to assign β, surfaces fidelity stats
in a `SelectionReport`, and refuses oversized subsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polygram.dictionary import Dictionary, Feature
from polygram.encoding import MPSRung1

MAX_FEATURES_PER_DICTIONARY = 8


@dataclass(frozen=True)
class SAEFeatureRecord:
    """One feature pulled from an SAE.

    `projection` is the decoder column (or other unit-direction
    vector) for the feature in residual-stream space — what Polygram
    actually consumes.
    """

    feature_id: int
    name: str
    projection: np.ndarray
    label: str | None = None
    activation_mean: float | None = None
    activation_std: float | None = None

    def __post_init__(self) -> None:
        proj = np.asarray(self.projection, dtype=float)
        if proj.ndim != 1:
            raise ValueError(
                f"SAEFeatureRecord {self.name!r}: projection must be 1D, "
                f"got shape {proj.shape}"
            )
        if not np.all(np.isfinite(proj)):
            raise ValueError(
                f"SAEFeatureRecord {self.name!r}: projection contains "
                f"non-finite values"
            )
        # frozen dataclass — bypass to coerce dtype
        object.__setattr__(self, "projection", proj)


@dataclass(frozen=True)
class SelectionReport:
    """Fidelity stats for a `from_sae_lens(...)` call.

    `beta_variance_explained` is `1 - SS_residual / SS_total`, where
    `SS_total` is the sum of squared distances of selected projection
    vectors from their collective centroid and `SS_residual` is the
    sum of squared distances from each vector to *its assigned
    cluster's centroid*. Higher = the cluster partition captures more
    of the projection-space variance the user-selected subset carries.
    1.0 means clusters are noise-free (e.g., identical projections per
    cluster). 0.0 means the partition explains nothing.

    `reconstruction_error` is per-feature Euclidean distance from each
    projection vector to its assigned cluster centroid. `tier_preservation`
    is the Pearson correlation between off-diagonal `|G|²` entries of
    the projection-space cosine-overlap matrix and the analytic
    Polygram Gram of the built `Dictionary` at φ=0; `None` when there
    is only one selected feature so no off-diagonals exist.
    `gamma_method` records `"zero"` (default) or `"projection_pca"`.
    """

    n_input_features: int
    n_selected: int
    cluster_assignments: dict[str, str]
    cluster_method: str
    beta_variance_explained: float
    reconstruction_error: dict[str, float] = field(default_factory=dict)
    tier_preservation: float | None = None
    gamma_method: str = "zero"
    warnings: list[str] = field(default_factory=list)


def load_toy_sae(path: str | Path) -> dict[int, SAEFeatureRecord]:
    """Load a JSON file in the bundled toy-SAE schema.

    Schema: `{"features": [{"feature_id", "name", "projection", ...},
    ...]}`. Returns a dict keyed by `feature_id` for O(1) lookup.
    """
    p = Path(path)
    raw: dict[str, Any] = json.loads(p.read_text())
    if "features" not in raw:
        raise ValueError(f"{path}: missing top-level 'features' list")
    out: dict[int, SAEFeatureRecord] = {}
    for entry in raw["features"]:
        rec = SAEFeatureRecord(
            feature_id=int(entry["feature_id"]),
            name=str(entry["name"]),
            projection=np.asarray(entry["projection"], dtype=float),
            label=entry.get("label"),
            activation_mean=entry.get("activation_mean"),
            activation_std=entry.get("activation_std"),
        )
        if rec.feature_id in out:
            raise ValueError(
                f"{path}: duplicate feature_id {rec.feature_id}"
            )
        out[rec.feature_id] = rec
    return out


def from_sae_lens(
    records: dict[int, SAEFeatureRecord],
    feature_ids: list[int],
    *,
    name: str = "ImportedSAE",
    cluster_assignments: dict[int, str] | None = None,
    n_clusters: int | None = None,
    encoding: MPSRung1 | None = None,
    beta_range: tuple[float, float] = (-0.5, 0.5),
    assign_gamma: bool = False,
    gamma_range: tuple[float, float] = (-0.25, 0.25),
) -> tuple[Dictionary, SelectionReport]:
    """Build a `Dictionary` from an explicit subset of SAE features.

    Cluster assignment precedence:

    1. `cluster_assignments` (user) — `dict[feature_id, cluster_name]`
    2. Labels of the form `"<cluster>/<name>"` — parse the prefix
    3. K-means with `n_clusters` (default 2) on projection vectors

    β values are spread evenly across cluster means within `beta_range`.
    α, φ default to 0. γ defaults to 0 unless `assign_gamma=True`, in
    which case each feature's γ is its projection vector's coefficient
    on the first principal component of its assigned cluster's
    centered projection vectors, rescaled into `gamma_range`. Refuses
    subsets larger than 8 features.
    """
    if len(feature_ids) > MAX_FEATURES_PER_DICTIONARY:
        raise ValueError(
            f"selected {len(feature_ids)} features, but Polygram's "
            f"rung-1 MPS encoding caps a Dictionary at "
            f"{MAX_FEATURES_PER_DICTIONARY} features. Pick a smaller "
            f"subset."
        )
    if len(feature_ids) == 0:
        raise ValueError("feature_ids is empty; nothing to import")

    missing = [fid for fid in feature_ids if fid not in records]
    if missing:
        raise ValueError(f"feature_id(s) not in records: {missing}")

    selected = [records[fid] for fid in feature_ids]
    projs = np.stack([r.projection for r in selected])
    n_features_input = len(records)

    warnings: list[str] = []

    if cluster_assignments is not None:
        method = "user"
        for fid in feature_ids:
            if fid not in cluster_assignments:
                raise ValueError(
                    f"cluster_assignments missing entry for feature_id {fid}"
                )
        cluster_per_feature = [cluster_assignments[fid] for fid in feature_ids]
    elif all(_label_has_cluster_prefix(r.label) for r in selected):
        method = "from_labels"
        cluster_per_feature = [r.label.split("/", 1)[0] for r in selected]
    else:
        method = "kmeans"
        k = n_clusters if n_clusters is not None else 2
        if k > len(selected):
            warnings.append(
                f"n_clusters={k} > selected={len(selected)}; "
                f"clamping to {len(selected)}"
            )
            k = len(selected)
        labels, empties = _kmeans(projs, k, seed=0)
        if empties:
            warnings.append(
                f"k-means produced {len(empties)} empty cluster(s) "
                f"(k={k}, n={len(selected)})"
            )
        cluster_per_feature = [f"cluster_{int(label)}" for label in labels]

    cluster_order: list[str] = []
    seen: set[str] = set()
    for c in cluster_per_feature:
        if c not in seen:
            cluster_order.append(c)
            seen.add(c)

    betas_by_cluster = _spread_betas(cluster_order, beta_range)
    centroids_by_cluster = _centroids(projs, cluster_per_feature)
    var_explained = _variance_explained(projs, centroids_by_cluster, cluster_per_feature)

    if assign_gamma:
        gammas = _gamma_via_cluster_pca(
            projs, cluster_per_feature, gamma_range
        )
        gamma_method = "projection_pca"
    else:
        gammas = [0.0] * len(selected)
        gamma_method = "zero"

    features = [
        Feature(name=r.name, cluster=c, beta=betas_by_cluster[c], gamma=g)
        for r, c, g in zip(selected, cluster_per_feature, gammas)
    ]
    hierarchy: dict[str, list[str]] = {c: [] for c in cluster_order}
    for f in features:
        hierarchy[f.cluster].append(f.name)

    dictionary = Dictionary(
        name=name,
        features=features,
        hierarchy=hierarchy,
        encoding=encoding or MPSRung1(),
    )

    reconstruction_error = {
        r.name: float(np.linalg.norm(r.projection - centroids_by_cluster[c]))
        for r, c in zip(selected, cluster_per_feature)
    }
    tier_preservation = _tier_preservation(projs, dictionary)

    report = SelectionReport(
        n_input_features=n_features_input,
        n_selected=len(selected),
        cluster_assignments={r.name: c for r, c in zip(selected, cluster_per_feature)},
        cluster_method=method,
        beta_variance_explained=var_explained,
        reconstruction_error=reconstruction_error,
        tier_preservation=tier_preservation,
        gamma_method=gamma_method,
        warnings=warnings,
    )
    return dictionary, report


def _label_has_cluster_prefix(label: str | None) -> bool:
    return isinstance(label, str) and "/" in label and label.split("/", 1)[0]


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
        # Top right-singular vector = first principal component.
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


def _tier_preservation(
    projs: np.ndarray, dictionary: Dictionary
) -> float | None:
    """Pearson correlation between off-diagonal `|G|²` entries of the
    projection-space cosine-overlap matrix and the analytic Polygram
    Gram of the built `Dictionary` at φ=0. None when there are no
    off-diagonals (N ≤ 1)."""
    n = projs.shape[0]
    if n <= 1:
        return None
    norms = np.linalg.norm(projs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    proj_unit = projs / norms
    cos_overlap = np.abs(proj_unit @ proj_unit.T) ** 2

    gram = np.abs(dictionary.gram()) ** 2

    iu = np.triu_indices(n, k=1)
    a = cos_overlap[iu]
    b = gram[iu]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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

    # k-means++ init: first centroid uniform random; subsequent
    # centroids weighted by D² to nearest existing centroid.
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
