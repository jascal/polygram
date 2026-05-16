"""`LearnedKnobAssignment` — production strategy that calibrates the
PCA-axis-to-polygram-knob projection from data instead of using the
hardcoded permutation in ``assign_amp_knobs_pca`` and
``assign_phase_knobs_pca``.

See ``openspec/changes/add-learned-axis-assignment/`` for the full
proposal and design. Empirical motivation lives in
``docs/research/rung5-pareto-scans.md`` scan 4.

Two solvers ship:

- ``solver="greedy"`` (default) — deterministic per-knob permutation
  search; no extra dependencies. Reproduces the prototype's published
  Spearman lift to within float-noise.
- ``solver="scipy"`` — continuous optimisation on a small linear map
  ``W ∈ R^{n_knobs × n_axes}`` initialised from the greedy result.
  Requires ``polygram[opt]``.

The strategy implements the ``KnobAssignment`` protocol from
``polygram.geometry.protocols``, so it slots into ``from_sae_lens``
behind the ``learn_axis_assignment`` kwarg without disturbing the
existing strategy classes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from polygram.geometry.clustered import (
    _centroids,
    _gamma_via_cluster_pca,
    _kmeans,
    _spread_betas,
    _variance_explained,
)
from polygram.geometry.objectives import spearman_objective
from polygram.geometry.protocols import (
    KnobAssignmentResult,
    LearnedAxisObjective,
)

logger = logging.getLogger(__name__)

# INFO-once flag so HEA users see a single explanation, not one per
# import. Mirrors the existing assign_*_pca pattern.
_INFO_LOGGED_HEA: set[str] = set()


_SCIPY_INSTALL_HINT = (
    "scipy is required for LearnedKnobAssignment(solver='scipy'); "
    "install with `pip install polygram[opt]`."
)


# Threshold below which a PCA axis is treated as degenerate (all
# coords ≈ 0). Configurable module-level constant so future callers
# with extreme-precision projection matrices can override without
# patching `_project_to_knob` directly. Matches the FP-noise floor
# used by `assign_amp_knobs_pca`'s `abs_max < 1e-12` check; raise it
# if your projection matrix sits closer to that floor for legitimate
# data reasons.
_DEGENERATE_AXIS_ABS_MAX_EPS: float = 1e-12


# Knob → linear-rescale range (matches the hardcoded helpers'
# convention so callers see consistent magnitudes regardless of which
# strategy populated the knobs).
_KNOB_RANGE: dict[str, tuple[float, float]] = {
    "alpha": (0.0, 2 * math.pi),
    "phi": (0.0, 2 * math.pi),
    "amp_theta": (0.0, math.pi / 2),
    "amp_psi": (0.0, 2 * math.pi),
}


def _decoder_cosine_gram_squared(projs: np.ndarray) -> np.ndarray:
    """``|cos(v_i, v_j)|²`` for unit-normalised projection vectors.

    The reference geometry the default Spearman / Pearson objectives
    correlate against. Matches the helper in
    `docs/research/decoder-gram-validity.md`.
    """
    norms = np.linalg.norm(projs, axis=1, keepdims=True) + 1e-12
    u = projs / norms
    cos = u @ u.T
    return cos ** 2


def _knob_order_for_encoding(encoding: object) -> list[str]:
    """Canonical knob slot order for the learned search.

    α and φ always come first (MPS-substrate). Amp knobs follow
    only for encodings that have an amp branch — Rung3, Rung4,
    Rung5(k). MPSRung1 contributes only α and φ.
    """
    knobs = ["alpha", "phi"]
    from polygram.encoding import Rung3, Rung4, Rung5

    if isinstance(encoding, Rung5):
        for i in range(encoding.n_amp_qubits):
            knobs.append(f"amp_{i}_theta")
            knobs.append(f"amp_{i}_psi")
    elif isinstance(encoding, Rung4):
        # Rung4 has two amp qubits with named knobs theta_amp /
        # psi_aux (q3) and theta_amp_b / psi_amp_b (q4). The learned
        # strategy expresses both via the indexed amp_<i>_{theta,psi}
        # schema; the downstream apply step routes each to the right
        # Feature field.
        for i in (0, 1):
            knobs.append(f"amp_{i}_theta")
            knobs.append(f"amp_{i}_psi")
    elif isinstance(encoding, Rung3):
        # Rung3 has one (θ_amp, ψ_aux) pair on q3.
        knobs.append("amp_0_theta")
        knobs.append("amp_0_psi")
    return knobs


def _baseline_axis_for_knob(knob: str) -> int:
    """Hardcoded baseline axis index for `knob`, matching
    ``assign_phase_knobs_pca`` + ``assign_amp_knobs_pca``.

    α ← PC2 (axis 1), φ ← PC3 (axis 2), amp_0_θ ← PC4 (axis 3), etc.
    Used both for the baseline-score comparison and as the fallback
    when greedy early-stop terminates before assigning every knob.
    """
    if knob == "alpha":
        return 1
    if knob == "phi":
        return 2
    # amp_<i>_{theta,psi} — first PC4+2i, then PC5+2i, ...
    parts = knob.split("_")
    i = int(parts[1])
    return 3 + 2 * i + (1 if parts[2] == "psi" else 0)


def _knob_range(knob: str) -> tuple[float, float]:
    if knob in ("alpha", "phi"):
        return _KNOB_RANGE[knob]
    if knob.endswith("_theta"):
        return _KNOB_RANGE["amp_theta"]
    return _KNOB_RANGE["amp_psi"]


def _project_to_knob(
    centered: np.ndarray, vt: np.ndarray, axis_idx: int | None, knob: str
) -> np.ndarray:
    """Project mean-centered projections along PCA axis `axis_idx`
    and linearly rescale into `knob`'s natural range. Returns an
    array of length n_features.

    `None` axis or an axis past `vt`'s row count → returns the range
    midpoint per feature (matches the hardcoded helpers' fallback
    when an axis is unavailable).
    """
    lo, hi = _knob_range(knob)
    n = centered.shape[0]
    if axis_idx is None or axis_idx >= vt.shape[0]:
        return np.full(n, 0.5 * (lo + hi))
    pc = vt[axis_idx]
    coords = centered @ pc
    abs_max = float(np.max(np.abs(coords)))
    if abs_max < _DEGENERATE_AXIS_ABS_MAX_EPS:
        return np.full(n, 0.5 * (lo + hi))
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    return (coords / abs_max) * half + mid


def _apply_axis_map(
    centered: np.ndarray,
    vt: np.ndarray,
    knob_order: list[str],
    axis_map: dict[str, int],
    encoding: object,
) -> dict:
    """Apply a knob → axis map and produce per-feature knob arrays in
    the same shape the hardcoded helpers return.

    Returns a dict with keys ``alphas``, ``phis``, ``theta_amps``,
    ``psi_auxes``, ``theta_amp_bs``, ``psi_amp_bs``,
    ``amp_knobs_list`` — same shape as
    ``assign_amp_knobs_pca`` + ``assign_phase_knobs_pca`` combined.
    Slots not present in ``knob_order`` (e.g. amp knobs on MPSRung1)
    are left ``None``.
    """
    from polygram.encoding import Rung3, Rung4, Rung5

    out: dict[str, list[float] | list[tuple[tuple[float, float], ...]] | None] = {
        "alphas": None,
        "phis": None,
        "theta_amps": None,
        "psi_auxes": None,
        "theta_amp_bs": None,
        "psi_amp_bs": None,
        "amp_knobs_list": None,
    }
    if "alpha" in knob_order:
        out["alphas"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("alpha"), "alpha")
        ]
    if "phi" in knob_order:
        out["phis"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("phi"), "phi")
        ]

    if isinstance(encoding, Rung5):
        k = encoding.n_amp_qubits
        per_feature: list[list[tuple[float, float]]] = [
            [(0.0, 0.0)] * k for _ in range(centered.shape[0])
        ]
        for i in range(k):
            thetas = _project_to_knob(
                centered, vt, axis_map.get(f"amp_{i}_theta"), f"amp_{i}_theta"
            )
            psis = _project_to_knob(
                centered, vt, axis_map.get(f"amp_{i}_psi"), f"amp_{i}_psi"
            )
            for f_idx in range(centered.shape[0]):
                per_feature[f_idx][i] = (float(thetas[f_idx]), float(psis[f_idx]))
        out["amp_knobs_list"] = [tuple(p) for p in per_feature]
        return out

    if isinstance(encoding, Rung4):
        # Map amp_0_* → theta_amp / psi_aux (q3), amp_1_* → theta_amp_b / psi_amp_b (q4).
        out["theta_amps"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("amp_0_theta"), "amp_0_theta")
        ]
        out["psi_auxes"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("amp_0_psi"), "amp_0_psi")
        ]
        out["theta_amp_bs"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("amp_1_theta"), "amp_1_theta")
        ]
        out["psi_amp_bs"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("amp_1_psi"), "amp_1_psi")
        ]
        return out

    if isinstance(encoding, Rung3):
        out["theta_amps"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("amp_0_theta"), "amp_0_theta")
        ]
        out["psi_auxes"] = [
            float(v) for v in _project_to_knob(centered, vt, axis_map.get("amp_0_psi"), "amp_0_psi")
        ]
        return out

    # MPSRung1 / unknown encoding: only α and φ are populated.
    return out


def _build_analytic_gram(
    *,
    encoding: object,
    cluster_per_feature: list[str],
    betas_by_cluster: dict[str, float],
    gammas: list[float],
    feature_names: list[str],
    knob_arrays: dict,
) -> np.ndarray:
    """Construct a `Dictionary` from the given per-feature knob arrays
    and return its analytic gram. Used inside the objective loop.
    """
    from polygram.dictionary import Dictionary, Feature
    from polygram.encoding import MPSRung1

    n = len(feature_names)
    alphas = knob_arrays["alphas"] if knob_arrays["alphas"] is not None else [0.0] * n
    phis = knob_arrays["phis"] if knob_arrays["phis"] is not None else [0.0] * n
    theta_amps = knob_arrays["theta_amps"]
    psi_auxes = knob_arrays["psi_auxes"]
    theta_amp_bs = knob_arrays["theta_amp_bs"]
    psi_amp_bs = knob_arrays["psi_amp_bs"]
    amp_knobs_list = knob_arrays["amp_knobs_list"]

    features = []
    for i in range(n):
        kw: dict[str, object] = {
            "name": feature_names[i],
            "cluster": cluster_per_feature[i],
            "beta": betas_by_cluster[cluster_per_feature[i]],
            "alpha": alphas[i],
            "gamma": gammas[i],
            "phi": phis[i],
        }
        if theta_amps is not None:
            kw["theta_amp"] = theta_amps[i]
        if psi_auxes is not None:
            kw["psi_aux"] = psi_auxes[i]
        if theta_amp_bs is not None:
            kw["theta_amp_b"] = theta_amp_bs[i]
        if psi_amp_bs is not None:
            kw["psi_amp_b"] = psi_amp_bs[i]
        if amp_knobs_list is not None:
            kw["amp_knobs"] = amp_knobs_list[i]
        features.append(Feature(**kw))

    cluster_order: list[str] = []
    seen: set[str] = set()
    for c in cluster_per_feature:
        if c not in seen:
            cluster_order.append(c)
            seen.add(c)
    hierarchy = {c: [] for c in cluster_order}
    for f in features:
        hierarchy[f.cluster].append(f.name)
    d = Dictionary(
        name="learned_assignment_candidate",
        features=features,
        hierarchy=hierarchy,
        encoding=encoding or MPSRung1(),
    )
    return d.gram()


def _validation_mask(
    n_features: int, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_mask, val_mask)`` boolean arrays over the
    upper-triangle off-diagonal pairs of an n×n matrix. Both masks
    have shape ``(C(n, 2),)``. When ``fraction == 0`` returns a
    full-True train mask and a full-False val mask.

    ``seed`` is the same value ``assign()`` receives from its caller
    (typically ``from_sae_lens``'s seed=0 default). Reusing the same
    seed each call makes the validation split deterministic and
    reproducible. Callers who want stochastic per-call splits should
    pass a varying ``seed`` to ``from_sae_lens`` (or construct
    multiple ``LearnedKnobAssignment`` instances and average).
    """
    n_pairs = n_features * (n_features - 1) // 2
    train = np.ones(n_pairs, dtype=bool)
    val = np.zeros(n_pairs, dtype=bool)
    if fraction <= 0.0 or n_pairs < 2:
        return train, val
    rng = np.random.default_rng(seed)
    n_val = int(round(fraction * n_pairs))
    val_idx = rng.choice(n_pairs, size=n_val, replace=False)
    val[val_idx] = True
    train = ~val
    return train, val


def _objective_on_mask(
    objective: LearnedAxisObjective,
    analytic_gram: np.ndarray,
    decoder_geom: np.ndarray,
    mask: np.ndarray,
    feature_names: list[str],
) -> float:
    """Evaluate `objective` only on the pairs where `mask` is True.

    The objective's protocol consumes the full off-diagonal pairs.
    To restrict to a subset we zero out the masked-out positions in
    both matrices — Spearman/Pearson on equal-ranked zeros adds no
    information.
    """
    if mask.all():
        return objective(
            analytic_gram, decoder_geom, feature_names=feature_names
        )
    n = analytic_gram.shape[0]
    iu = np.triu_indices(n, k=1)
    g_masked = np.zeros_like(analytic_gram)
    d_masked = np.zeros_like(decoder_geom)
    # Symmetrise the mask back onto the (n, n) matrix.
    g_flat = analytic_gram[iu]
    d_flat = decoder_geom[iu]
    g_keep = np.where(mask, g_flat, 0.0)
    d_keep = np.where(mask, d_flat, 0.0)
    g_masked[iu] = g_keep
    d_masked[iu] = d_keep
    g_masked = g_masked + g_masked.T
    d_masked = d_masked + d_masked.T
    return objective(
        g_masked, d_masked, feature_names=feature_names
    )


@dataclass(frozen=True)
class LearnedKnobAssignment:
    """`KnobAssignment` strategy that calibrates the
    PCA-axis-to-polygram-knob projection from data.

    Two solvers:

    - ``solver="greedy"`` — deterministic permutation search; for
      each knob slot in canonical order, tries every still-unused
      PCA axis and locks in the axis whose addition maximises the
      objective.
    - ``solver="scipy"`` — continuous optimisation on a linear map
      ``W ∈ R^{n_knobs × n_axes}`` initialised from the greedy
      result. Requires ``polygram[opt]``.

    Defaults reproduce the published prototype numbers: greedy
    solver, Spearman-against-decoder-cosine² objective,
    ``max_axes=32``, no validation split, ``early_stop_eps=1e-4``.

    The strategy honours the ``KnobAssignment`` protocol; pass it
    through ``from_sae_lens(learn_axis_assignment=...)`` (or
    instantiate inline for direct use). HEA_Rung2 encodings fall back
    to the hardcoded helpers with an INFO-once log.

    See ``docs/research/rung5-pareto-scans.md`` scan 4 for the
    empirical motivation.
    """

    solver: str = "greedy"
    objective: LearnedAxisObjective = field(default=spearman_objective)
    max_axes: int = 32
    validation_fraction: float = 0.0
    scipy_restarts: int = 1
    early_stop_eps: float = 1e-4
    beta_range: tuple[float, float] = (-0.5, 0.5)

    def __post_init__(self) -> None:
        if self.solver not in ("greedy", "scipy"):
            raise ValueError(
                f"LearnedKnobAssignment.solver must be one of "
                f"{{'greedy', 'scipy'}}; got {self.solver!r}"
            )
        if not (0.0 <= self.validation_fraction <= 0.5):
            raise ValueError(
                f"LearnedKnobAssignment.validation_fraction must be "
                f"in [0.0, 0.5]; got {self.validation_fraction}"
            )
        if self.max_axes < 1:
            raise ValueError(
                f"LearnedKnobAssignment.max_axes must be >= 1; "
                f"got {self.max_axes}"
            )
        if self.scipy_restarts < 1:
            raise ValueError(
                f"LearnedKnobAssignment.scipy_restarts must be >= 1; "
                f"got {self.scipy_restarts}"
            )
        if self.early_stop_eps < 0.0:
            raise ValueError(
                f"LearnedKnobAssignment.early_stop_eps must be >= 0; "
                f"got {self.early_stop_eps}"
            )

    def assign(
        self,
        projections: np.ndarray,
        feature_names: list[str],
        *,
        n_clusters: int | None,
        gamma_range: tuple[float, float],
        assign_gamma: bool,
        seed: int,
        assign_amp_knobs: bool = False,  # noqa: ARG002 — consumed via encoding
        assign_phase_knobs: bool = False,  # noqa: ARG002
        encoding: object = None,
    ) -> KnobAssignmentResult:
        """Implement the `KnobAssignment` protocol via learned
        axis-to-knob calibration."""
        from polygram.encoding import HEA_Rung2

        # HEA fallback — different per-feature θ tensor shape; out of
        # scope for v1 of the learned strategy. Use the hardcoded
        # helpers + clustered strategy's β/γ pipeline so the caller
        # gets a usable result rather than an error.
        if isinstance(encoding, HEA_Rung2):
            enc_name = type(encoding).__name__
            if enc_name not in _INFO_LOGGED_HEA:
                logger.info(
                    "LearnedKnobAssignment: HEA_Rung2's per-feature θ "
                    "tensor has a different shape than the MPS-substrate "
                    "knobs the learned strategy targets; falling back to "
                    "the hardcoded helpers. This is a known v1 limitation."
                )
                _INFO_LOGGED_HEA.add(enc_name)
            from polygram.geometry.clustered import ClusteredKnobAssignment

            return ClusteredKnobAssignment(
                beta_range=self.beta_range
            ).assign(
                projections,
                feature_names,
                n_clusters=n_clusters,
                gamma_range=gamma_range,
                assign_gamma=assign_gamma,
                seed=seed,
                assign_amp_knobs=False,
                assign_phase_knobs=False,
                encoding=encoding,
            )

        # Cluster step (k-means, antipodal β spread) — same shape as
        # ClusteredKnobAssignment. The learned strategy's contribution
        # is purely on the α/φ/amp-knob side.
        n_features = len(feature_names)
        n_clusters_eff = n_clusters if n_clusters is not None else 2
        if n_clusters_eff > n_features:
            n_clusters_eff = n_features
        labels, _ = _kmeans(projections, n_clusters_eff, seed=seed)
        cluster_per_feature = [f"cluster_{int(label)}" for label in labels]

        cluster_order: list[str] = []
        seen: set[str] = set()
        for c in cluster_per_feature:
            if c not in seen:
                cluster_order.append(c)
                seen.add(c)
        betas_by_cluster = _spread_betas(cluster_order, self.beta_range)
        betas = [betas_by_cluster[c] for c in cluster_per_feature]
        centroids_by_cluster = _centroids(projections, cluster_per_feature)
        var_explained = _variance_explained(
            projections, centroids_by_cluster, cluster_per_feature
        )
        if assign_gamma:
            gammas = _gamma_via_cluster_pca(
                projections, cluster_per_feature, gamma_range
            )
        else:
            gammas = [0.0] * n_features

        # PCA once for the learned-axis search.
        centered = projections - projections.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        n_candidate_axes = min(self.max_axes, vt.shape[0])

        knob_order = _knob_order_for_encoding(encoding)
        decoder_geom = _decoder_cosine_gram_squared(projections)

        train_mask, val_mask = _validation_mask(
            n_features, self.validation_fraction, seed
        )

        # Baseline evaluation — the hardcoded permutation we're
        # measuring against.
        baseline_map = {
            knob: _baseline_axis_for_knob(knob) for knob in knob_order
        }
        baseline_knobs = _apply_axis_map(
            centered, vt, knob_order, baseline_map, encoding
        )
        baseline_gram = _build_analytic_gram(
            encoding=encoding,
            cluster_per_feature=cluster_per_feature,
            betas_by_cluster=betas_by_cluster,
            gammas=gammas,
            feature_names=feature_names,
            knob_arrays=baseline_knobs,
        )
        baseline_train = _objective_on_mask(
            self.objective, baseline_gram, decoder_geom,
            train_mask, feature_names,
        )

        # Solver dispatch.
        if self.solver == "greedy":
            learned_map, _trajectory = self._greedy(
                centered=centered,
                vt=vt,
                knob_order=knob_order,
                encoding=encoding,
                cluster_per_feature=cluster_per_feature,
                betas_by_cluster=betas_by_cluster,
                gammas=gammas,
                feature_names=feature_names,
                decoder_geom=decoder_geom,
                train_mask=train_mask,
                n_candidate_axes=n_candidate_axes,
            )
            axis_assignment_out: dict[str, int | list[float]] = {
                k: int(v) for k, v in learned_map.items()
            }
        else:
            learned_map, weighted_map = self._scipy(
                centered=centered,
                vt=vt,
                knob_order=knob_order,
                encoding=encoding,
                cluster_per_feature=cluster_per_feature,
                betas_by_cluster=betas_by_cluster,
                gammas=gammas,
                feature_names=feature_names,
                decoder_geom=decoder_geom,
                train_mask=train_mask,
                n_candidate_axes=n_candidate_axes,
            )
            axis_assignment_out = {
                k: [float(x) for x in v] for k, v in weighted_map.items()
            }

        learned_knobs = _apply_axis_map(
            centered, vt, knob_order, learned_map, encoding
        )
        learned_gram = _build_analytic_gram(
            encoding=encoding,
            cluster_per_feature=cluster_per_feature,
            betas_by_cluster=betas_by_cluster,
            gammas=gammas,
            feature_names=feature_names,
            knob_arrays=learned_knobs,
        )
        training_score = _objective_on_mask(
            self.objective, learned_gram, decoder_geom,
            train_mask, feature_names,
        )
        if val_mask.any():
            validation_score = _objective_on_mask(
                self.objective, learned_gram, decoder_geom,
                val_mask, feature_names,
            )
        else:
            validation_score = training_score

        return KnobAssignmentResult(
            cluster_per_feature=cluster_per_feature,
            betas=betas,
            gammas=gammas,
            cluster_method="kmeans",
            beta_variance_explained=var_explained,
            theta_amps=learned_knobs["theta_amps"],  # type: ignore[arg-type]
            psi_auxes=learned_knobs["psi_auxes"],  # type: ignore[arg-type]
            theta_amp_bs=learned_knobs["theta_amp_bs"],  # type: ignore[arg-type]
            psi_amp_bs=learned_knobs["psi_amp_bs"],  # type: ignore[arg-type]
            amp_knobs_list=learned_knobs["amp_knobs_list"],  # type: ignore[arg-type]
            alphas=learned_knobs["alphas"],
            phis=learned_knobs["phis"],
            axis_assignment=axis_assignment_out,
            objective_value=float(validation_score),
            objective_baseline=float(baseline_train),
            training_objective_value=float(training_score),
        )

    def _greedy(
        self,
        *,
        centered: np.ndarray,
        vt: np.ndarray,
        knob_order: list[str],
        encoding: object,
        cluster_per_feature: list[str],
        betas_by_cluster: dict[str, float],
        gammas: list[float],
        feature_names: list[str],
        decoder_geom: np.ndarray,
        train_mask: np.ndarray,
        n_candidate_axes: int,
    ) -> tuple[dict[str, int], list[dict]]:
        """Greedy per-knob permutation search with early-stop on flat
        marginal gains.
        """
        assigned: dict[str, int] = {}
        used: set[int] = set()
        trajectory: list[dict] = []
        flat_streak = 0
        prev_score: float | None = None

        for knob in knob_order:
            best_axis: int | None = None
            best_score = -float("inf")
            for axis in range(n_candidate_axes):
                if axis in used:
                    continue
                trial = dict(assigned)
                trial[knob] = axis
                # Fill remaining knobs with baseline so the gram is
                # complete and the objective is well-defined.
                for k in knob_order:
                    if k not in trial:
                        trial[k] = _baseline_axis_for_knob(k)
                knobs = _apply_axis_map(centered, vt, knob_order, trial, encoding)
                gram = _build_analytic_gram(
                    encoding=encoding,
                    cluster_per_feature=cluster_per_feature,
                    betas_by_cluster=betas_by_cluster,
                    gammas=gammas,
                    feature_names=feature_names,
                    knob_arrays=knobs,
                )
                score = _objective_on_mask(
                    self.objective, gram, decoder_geom,
                    train_mask, feature_names,
                )
                if score > best_score:
                    best_score = score
                    best_axis = axis
            if best_axis is None:
                break
            assigned[knob] = best_axis
            used.add(best_axis)
            trajectory.append({
                "knob": knob, "axis": best_axis, "score": float(best_score),
            })
            if prev_score is not None and self.early_stop_eps > 0.0:
                gain = best_score - prev_score
                if gain < self.early_stop_eps:
                    flat_streak += 1
                else:
                    flat_streak = 0
                if flat_streak >= 2:
                    break
            prev_score = best_score

        # Fill any remaining knobs with the baseline axis (used by
        # early-stop and by knob orders that exceed available axes).
        for knob in knob_order:
            if knob not in assigned:
                assigned[knob] = _baseline_axis_for_knob(knob)
        return assigned, trajectory

    def _scipy(
        self,
        *,
        centered: np.ndarray,
        vt: np.ndarray,
        knob_order: list[str],
        encoding: object,
        cluster_per_feature: list[str],
        betas_by_cluster: dict[str, float],
        gammas: list[float],
        feature_names: list[str],
        decoder_geom: np.ndarray,
        train_mask: np.ndarray,
        n_candidate_axes: int,
    ) -> tuple[dict[str, int], dict[str, list[float]]]:
        """Scipy continuous optimisation on a linear axis-coefficient
        map. Initialised from the greedy result; returns both the
        nearest-permutation integer map (for the in-tree applier) and
        the full weight matrix (surfaced in ``axis_assignment``)."""
        try:
            from scipy.optimize import differential_evolution, minimize
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_SCIPY_INSTALL_HINT) from exc

        # Initialise from greedy.
        greedy_map, _ = self._greedy(
            centered=centered, vt=vt, knob_order=knob_order, encoding=encoding,
            cluster_per_feature=cluster_per_feature,
            betas_by_cluster=betas_by_cluster, gammas=gammas,
            feature_names=feature_names, decoder_geom=decoder_geom,
            train_mask=train_mask, n_candidate_axes=n_candidate_axes,
        )

        n_knobs = len(knob_order)
        n_axes = n_candidate_axes
        # x0: a flat vector of length n_knobs * n_axes; reshape on
        # use. Start as a one-hot encoding of greedy_map.
        x0 = np.zeros(n_knobs * n_axes)
        for i, knob in enumerate(knob_order):
            x0[i * n_axes + greedy_map[knob]] = 1.0

        def _eval_W(flat: np.ndarray) -> float:
            W = flat.reshape(n_knobs, n_axes)
            # For each knob, pick a single representative axis as
            # argmax(W[knob, :]) to feed `_apply_axis_map`. This keeps
            # the parameter→state map differentiable enough for
            # scipy's gradient-free solvers while preserving the
            # downstream rescaling semantics.
            axis_map: dict[str, int] = {
                knob: int(np.argmax(W[i])) for i, knob in enumerate(knob_order)
            }
            knobs = _apply_axis_map(
                centered, vt, knob_order, axis_map, encoding
            )
            gram = _build_analytic_gram(
                encoding=encoding, cluster_per_feature=cluster_per_feature,
                betas_by_cluster=betas_by_cluster, gammas=gammas,
                feature_names=feature_names, knob_arrays=knobs,
            )
            score = _objective_on_mask(
                self.objective, gram, decoder_geom,
                train_mask, feature_names,
            )
            return -float(score)  # scipy minimises

        if n_knobs >= 8:
            bounds = [(0.0, 1.0)] * (n_knobs * n_axes)
            res = differential_evolution(
                _eval_W, bounds=bounds, x0=x0, seed=0,
                maxiter=10, polish=False, tol=1e-3,
                init="sobol",
            )
            x_best = res.x
        else:
            res = minimize(
                _eval_W, x0=x0, method="Nelder-Mead",
                options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200},
            )
            x_best = res.x

        W_best = x_best.reshape(n_knobs, n_axes)
        weighted_map: dict[str, list[float]] = {
            knob: list(W_best[i]) for i, knob in enumerate(knob_order)
        }
        # Integer map for the applier.
        int_map: dict[str, int] = {
            knob: int(np.argmax(W_best[i])) for i, knob in enumerate(knob_order)
        }
        return int_map, weighted_map
