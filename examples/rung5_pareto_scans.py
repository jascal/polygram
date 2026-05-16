"""Pareto scans probing the polygram 0.7.0 Rung5 encoding.

Three subcommands, all CPU-only (no torch / no behavioural validator) so
they run anywhere polygram itself runs:

- ``capacity-quality`` — at each feature count N, build a dictionary
  under every encoding with ``cap >= N`` and measure decoder-Gram
  fidelity (Spearman against the projection-space cosine matrix), tier
  separation, and gram condition number. Answers the strategic
  question: as you walk up the k-axis, does quality keep up with
  capacity?
- ``saturation-density`` — at each ``k``, scan ``N`` finely around
  ``cap = 8·2^k`` and report rank, smallest non-zero singular value,
  and condition number. Fills in the saturation transition that
  ``rung5_rank_verification.py`` only samples at sentinel points.
- ``pca-amp-ablation`` — for each k, import the same synthetic SAE
  twice (``assign_amp_knobs=False`` vs ``True``) and compare
  decoder-Gram Spearman. Measures whether the new
  ``_assign_amp_knobs_pca_rung5`` branch is load-bearing or inert
  plumbing.
- ``learned-assignment`` — small prototype of "learned PCA-axis
  assignment" (cf. ``rung5-pareto-scans.md`` discussion). For each
  k, greedy-searches over axis-to-knob permutations to maximise
  decoder-Gram Spearman, and compares against the hardcoded
  baseline (PC2→α, PC3→φ, PC4..→amp_knobs).

Usage::

    python examples/rung5_pareto_scans.py capacity-quality \\
        --json-out docs/research/data/rung5_pareto/capacity_quality.json
    python examples/rung5_pareto_scans.py saturation-density \\
        --json-out docs/research/data/rung5_pareto/saturation_density.json
    python examples/rung5_pareto_scans.py pca-amp-ablation \\
        --json-out docs/research/data/rung5_pareto/pca_amp_ablation.json

Each subcommand prints a summary table to stdout in addition to writing
the JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polygram import (
    Dictionary,
    Feature,
    HEA_Rung2,
    MPSRung1,
    Rung3,
    Rung4,
    Rung5,
)


# ---------------------------------------------------------------------------
# Synthetic clustered-SAE fixture (no torch, no I/O)
# ---------------------------------------------------------------------------


def synth_clustered_sae(
    *,
    n_clusters: int,
    cluster_size: int,
    d_model: int,
    seed: int = 0,
    noise_sigma: float = 0.08,
) -> tuple[np.ndarray, list[str]]:
    """Return ``(projections, cluster_labels)`` for ``n_clusters *
    cluster_size`` features.

    Each cluster lives in a disjoint 2-D subspace of ``R^{d_model}``;
    within-cluster siblings are perturbed by Gaussian noise. The shape
    mirrors ``tests/fixtures/toy_sae.json`` but is sized for arbitrary
    encoding caps. Centroids are unit-norm.
    """
    rng = np.random.default_rng(seed)
    n = n_clusters * cluster_size
    projs = np.zeros((n, d_model), dtype=np.float64)
    labels: list[str] = []
    # Pick one random unit-norm centroid per cluster; siblings live as
    # small Gaussian perturbations around it. This is the "natural"
    # SAE-feature regime polygram targets — high intra-cluster cosine,
    # low inter-cluster cosine. (The fixture file at
    # tests/fixtures/toy_sae.json uses the same construction.)
    for c in range(n_clusters):
        centroid = rng.standard_normal(d_model)
        centroid /= np.linalg.norm(centroid) + 1e-12
        for s in range(cluster_size):
            noise = rng.standard_normal(d_model) * noise_sigma
            v = centroid + noise
            v /= np.linalg.norm(v) + 1e-12
            projs[c * cluster_size + s] = v
            labels.append(f"c{c:02d}/f{c:02d}_{s:02d}")
    return projs, labels


def _records_from_projections(
    projs: np.ndarray, labels: list[str]
) -> dict:
    """Wrap a ``(n, d_model)`` projection matrix into the SAEFeatureRecord
    dict shape that ``from_sae_lens`` consumes."""
    from polygram.sae_import import SAEFeatureRecord

    return {
        i: SAEFeatureRecord(
            feature_id=i,
            name=labels[i].replace("/", "_"),
            label=labels[i],
            projection=projs[i].astype(np.float32),
            activation_mean=0.0,
            activation_std=1.0,
        )
        for i in range(len(labels))
    }


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------


def _decoder_cosine_gram(projs: np.ndarray) -> np.ndarray:
    """Cosine similarity squared (|<v_i, v_j>|^2) for unit projections.

    Matches the metric polygram's decoder-Gram-validity research note
    correlates against the analytic Polygram gram.
    """
    norms = np.linalg.norm(projs, axis=1, keepdims=True) + 1e-12
    u = projs / norms
    cos = u @ u.T
    return cos ** 2


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between off-diagonal entries of two
    square matrices. Returns 0 for degenerate inputs."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    x = a[iu].astype(float)
    y = b[iu].astype(float)
    if x.size < 2 or x.std() < 1e-15 or y.std() < 1e-15:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum()))
    if denom < 1e-15:
        return 0.0
    return float((rx_c * ry_c).sum() / denom)


def _rank_at_tol(s: np.ndarray, tol: float) -> int:
    if s.size == 0:
        return 0
    return int(np.sum(s > tol * float(np.max(s))))


def _cond_number(gram: np.ndarray) -> float:
    """Numeric condition number of a (possibly complex) gram matrix."""
    s = np.linalg.svd(gram, compute_uv=False)
    s_real = np.real(s)
    s_real = s_real[s_real > 0]
    if s_real.size == 0:
        return float("inf")
    return float(s_real.max() / s_real.min())


# ---------------------------------------------------------------------------
# Scan 1: capacity-quality Pareto
# ---------------------------------------------------------------------------


@dataclass
class EncodingSpec:
    label: str
    factory: callable  # noqa: E501 — () -> encoding instance
    max_features: int


def _encoding_grid() -> list[EncodingSpec]:
    return [
        EncodingSpec("MPSRung1", lambda: MPSRung1(), 8),
        EncodingSpec("Rung3", lambda: Rung3(), 16),
        EncodingSpec("Rung4", lambda: Rung4(), 32),
        EncodingSpec("Rung5(k=3)", lambda: Rung5(n_amp_qubits=3), 64),
        EncodingSpec("Rung5(k=4)", lambda: Rung5(n_amp_qubits=4), 128),
        EncodingSpec("Rung5(k=5)", lambda: Rung5(n_amp_qubits=5), 256),
    ]


def _build_dictionary(
    records: dict, ids: list[int], encoding
) -> Dictionary:
    """Wrapper around from_sae_lens with both knob-assignment flags on
    so amp_knobs / alphas / phis come from decoder PCA."""
    from polygram import from_sae_lens

    d, _ = from_sae_lens(
        records,
        ids,
        encoding=encoding,
        assign_amp_knobs=True,
        assign_phase_knobs=True,
    )
    return d


def _measure_quality(
    d: Dictionary, projs: np.ndarray, ids: list[int]
) -> dict:
    """Decoder-Gram Spearman + tier_separation + gram condition number."""
    sel_projs = projs[ids]
    cos_sq = _decoder_cosine_gram(sel_projs)
    g = d.gram()
    g_sq = np.abs(g) ** 2
    spearman = _spearman(g_sq, cos_sq)
    try:
        tier_sep = d.tier_separation()
        tier_sep = float(tier_sep) if tier_sep is not None else None
    except Exception:  # noqa: BLE001 — defensive; q-orca path may fail
        tier_sep = None
    cond = _cond_number(g)
    return {
        "spearman_gram_vs_cosine": spearman,
        "tier_separation": tier_sep,
        "condition_number": cond,
    }


def run_capacity_quality(args: argparse.Namespace) -> dict:
    """Sweep encodings × feature-count targets."""
    n_clusters = 64
    cluster_size = 4  # 256 features total, 4 siblings per cluster
    d_model = 64  # 64 PCA axes — enough for Rung5 k up to 30 (axes 4..63)
    projs, labels = synth_clustered_sae(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        d_model=d_model,
        seed=args.seed,
    )
    records = _records_from_projections(projs, labels)

    encodings = _encoding_grid()
    # N values to probe — geometric spread that hits each encoding's cap.
    targets = [8, 16, 32, 64, 128, 256]

    runs: list[dict] = []
    for n in targets:
        # Select N features stride-spread across clusters so each
        # target hits multiple clusters even when N < n_clusters.
        # Picks the *first* sibling of N evenly-spaced clusters when
        # N <= n_clusters, otherwise wraps. This keeps the inter-cluster
        # signal alive at small N (without the stride, ids=[0..7] would
        # land all in c00..c01 and the off-diagonal Spearman would be
        # dominated by tied intra-cluster pairs).
        if n <= n_clusters:
            stride = n_clusters // n
            ids = [c * cluster_size for c in range(0, n_clusters, stride)][:n]
        else:
            ids = list(range(n))
        for enc_spec in encodings:
            if enc_spec.max_features < n:
                continue  # encoding can't hold this many features
            encoding = enc_spec.factory()
            t0 = time.perf_counter()
            d = _build_dictionary(records, ids, encoding)
            build_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            quality = _measure_quality(d, projs, ids)
            measure_s = time.perf_counter() - t0
            row = {
                "n_features": n,
                "encoding": enc_spec.label,
                "encoding_cap": enc_spec.max_features,
                "build_seconds": build_s,
                "measure_seconds": measure_s,
                **quality,
            }
            runs.append(row)
            print(
                f"N={n:>3} | {enc_spec.label:<14} "
                f"spearman={quality['spearman_gram_vs_cosine']:+.4f}  "
                f"tier_sep={_fmt(quality['tier_separation'])}  "
                f"cond={quality['condition_number']:.2e}  "
                f"build={build_s * 1000:.0f}ms  "
                f"measure={measure_s * 1000:.0f}ms"
            )
        print()
    return {
        "schema": "polygram.rung5_pareto.capacity_quality.v1",
        "config": {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "d_model": d_model,
            "seed": args.seed,
            "targets": targets,
        },
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Scan 2: saturation-density
# ---------------------------------------------------------------------------


def _build_random_rung5_dict(
    n_features: int, k: int, seed: int
) -> Dictionary:
    rng = np.random.default_rng(seed)
    feats = []
    for i in range(n_features):
        amp = tuple(
            (
                float(rng.uniform(0.0, np.pi / 2)),
                float(rng.uniform(0.0, 2 * np.pi)),
            )
            for _ in range(k)
        )
        feats.append(
            Feature(
                name=f"f{i}",
                cluster="all",
                alpha=float(rng.uniform(0.0, 2 * np.pi)),
                beta=float(rng.uniform(0.0, 2 * np.pi)),
                gamma=float(rng.uniform(0.0, 2 * np.pi)),
                phi=float(rng.uniform(0.0, 2 * np.pi)),
                amp_knobs=amp,
            )
        )
    return Dictionary(
        name=f"sat_k{k}_n{n_features}",
        features=feats,
        hierarchy={"all": [f.name for f in feats]},
        encoding=Rung5(n_amp_qubits=k),
    )


def run_saturation_density(args: argparse.Namespace) -> dict:
    """Scan N around cap = 8·2^k for each k."""
    runs: list[dict] = []
    for k in args.k:
        cap = 8 * 2 ** k
        # 7-point density around the cap: 0.25x, 0.5x, 0.9x, 1.0x, 1.1x, 1.5x, 2x
        fracs = [0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0]
        n_values = sorted(set(max(2, int(round(f * cap))) for f in fracs))
        for n in n_values:
            for seed in args.seeds:
                t0 = time.perf_counter()
                d = _build_random_rung5_dict(n, k, seed)
                g = d.gram()
                build_s = time.perf_counter() - t0
                s = np.real(np.linalg.svd(g, compute_uv=False))
                nz = s[s > 1e-15 * float(np.max(s) if s.size else 1.0)]
                row = {
                    "k": k,
                    "cap": cap,
                    "n_features": n,
                    "frac_of_cap": n / cap,
                    "seed": seed,
                    "rank_1e_12": _rank_at_tol(s, 1e-12),
                    "rank_1e_9": _rank_at_tol(s, 1e-9),
                    "rank_1e_6": _rank_at_tol(s, 1e-6),
                    "sigma_max": float(s.max()) if s.size else 0.0,
                    "sigma_min_nonzero": float(nz.min()) if nz.size else 0.0,
                    "condition_number": _cond_number(g),
                    "build_seconds": build_s,
                }
                runs.append(row)
                print(
                    f"k={k} cap={cap:>4} | N={n:>4} "
                    f"({n / cap:.2f}× cap) seed={seed}  "
                    f"rank_1e-12={row['rank_1e_12']:>4}  "
                    f"σ_min>0={row['sigma_min_nonzero']:.2e}  "
                    f"cond={row['condition_number']:.2e}  "
                    f"build={build_s * 1000:.0f}ms"
                )
        print()
    return {
        "schema": "polygram.rung5_pareto.saturation_density.v1",
        "config": {"k": list(args.k), "seeds": list(args.seeds)},
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Scan 3: PCA amp-knob ablation
# ---------------------------------------------------------------------------


def run_pca_amp_ablation(args: argparse.Namespace) -> dict:
    """For each k, import the same SAE with assign_amp_knobs ∈ {False,
    True} and compare decoder-Gram Spearman. Measures whether the new
    Rung5 PCA branch is load-bearing."""
    n_clusters = 16
    cluster_size = 4
    d_model = 32
    projs, labels = synth_clustered_sae(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        d_model=d_model,
        seed=args.seed,
    )
    records = _records_from_projections(projs, labels)
    n_features = n_clusters * cluster_size  # 64

    runs: list[dict] = []
    for k in args.k:
        cap = 8 * 2 ** k
        if n_features > cap:
            continue
        ids = list(range(n_features))
        for assign in (False, True):
            from polygram import from_sae_lens

            encoding = Rung5(n_amp_qubits=k)
            d, _ = from_sae_lens(
                records,
                ids,
                encoding=encoding,
                assign_amp_knobs=assign,
                assign_phase_knobs=True,
            )
            quality = _measure_quality(d, projs, ids)
            row = {
                "k": k,
                "n_features": n_features,
                "assign_amp_knobs": assign,
                **quality,
            }
            runs.append(row)
            print(
                f"k={k} N={n_features} | assign_amp_knobs={assign!s:<5}  "
                f"spearman={quality['spearman_gram_vs_cosine']:+.4f}  "
                f"tier_sep={_fmt(quality['tier_separation'])}  "
                f"cond={quality['condition_number']:.2e}"
            )

    # Per-k delta summary.
    deltas: list[dict] = []
    for k in args.k:
        rows = [r for r in runs if r["k"] == k]
        if len(rows) != 2:
            continue
        off = next(r for r in rows if not r["assign_amp_knobs"])
        on = next(r for r in rows if r["assign_amp_knobs"])
        delta = on["spearman_gram_vs_cosine"] - off["spearman_gram_vs_cosine"]
        deltas.append({
            "k": k,
            "spearman_off": off["spearman_gram_vs_cosine"],
            "spearman_on": on["spearman_gram_vs_cosine"],
            "delta_spearman": delta,
        })
        print(
            f"k={k}: Spearman {off['spearman_gram_vs_cosine']:+.4f} → "
            f"{on['spearman_gram_vs_cosine']:+.4f}  "
            f"(Δ={delta:+.4f})"
        )
    return {
        "schema": "polygram.rung5_pareto.pca_amp_ablation.v1",
        "config": {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "d_model": d_model,
            "seed": args.seed,
            "k": list(args.k),
        },
        "runs": runs,
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# Scan 4: learned PCA-axis assignment (prototype)
# ---------------------------------------------------------------------------


# Knob ranges, matching the hardcoded `assign_*_pca` helpers.
_KNOB_RANGE = {
    "alpha": (0.0, 2 * np.pi),
    "phi": (0.0, 2 * np.pi),
    "amp_theta": (0.0, np.pi / 2),
    "amp_psi": (0.0, 2 * np.pi),
}


def _pca(projs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(centered, sv, vt)`` from SVD of mean-centered projs."""
    centered = projs - projs.mean(axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    return centered, sv, vt


def _axis_to_knob(
    centered: np.ndarray, vt: np.ndarray, axis_idx: int, lo: float, hi: float
) -> np.ndarray:
    """Linearly rescale the per-feature coord along PCA axis ``axis_idx``
    into ``[lo, hi]``. Returns zeros (= rescale midpoint at default
    knob) when the axis is unavailable or degenerate — matching the
    ``assign_*_pca`` helpers' fallback."""
    if axis_idx is None or axis_idx >= vt.shape[0]:
        return np.full(centered.shape[0], 0.5 * (lo + hi))
    pc = vt[axis_idx]
    coords = centered @ pc
    abs_max = float(np.max(np.abs(coords)))
    if abs_max < 1e-12:
        return np.full(centered.shape[0], 0.5 * (lo + hi))
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    return (coords / abs_max) * half + mid


def _build_dict_from_assignment(
    records: dict,
    ids: list[int],
    projs: np.ndarray,
    encoding,
    axis_for_knob: dict[str, int | None],
) -> Dictionary:
    """Build a Dictionary using an explicit knob → PCA-axis assignment.

    β is derived from cluster labels (same as the labels-bypass path
    in ``from_sae_lens``); α, φ, and per-amp-qubit (θ_i, ψ_i) are
    derived from ``axis_for_knob[knob_name]`` if present, otherwise
    default to the range midpoint.
    """
    sel = projs[ids]
    centered, _, vt = _pca(sel)

    # β: cluster-ordinal spread from labels (mirrors `_spread_betas`).
    cluster_per_feature = [records[i].label.split("/", 1)[0] for i in ids]
    cluster_order: list[str] = []
    seen: set[str] = set()
    for c in cluster_per_feature:
        if c not in seen:
            cluster_order.append(c)
            seen.add(c)
    n_clusters = len(cluster_order)
    if n_clusters >= 2:
        betas_by_cluster = {
            c: -0.5 + 1.0 * i / (n_clusters - 1)
            for i, c in enumerate(cluster_order)
        }
    else:
        betas_by_cluster = {cluster_order[0]: 0.0}

    alphas = _axis_to_knob(
        centered, vt, axis_for_knob.get("alpha"),
        *_KNOB_RANGE["alpha"],
    )
    phis = _axis_to_knob(
        centered, vt, axis_for_knob.get("phi"),
        *_KNOB_RANGE["phi"],
    )

    k = encoding.n_amp_qubits if isinstance(encoding, Rung5) else 0
    amp_thetas = [
        _axis_to_knob(
            centered, vt, axis_for_knob.get(f"amp_{i}_theta"),
            *_KNOB_RANGE["amp_theta"],
        )
        for i in range(k)
    ]
    amp_psis = [
        _axis_to_knob(
            centered, vt, axis_for_knob.get(f"amp_{i}_psi"),
            *_KNOB_RANGE["amp_psi"],
        )
        for i in range(k)
    ]

    features = []
    for f_idx, feat_id in enumerate(ids):
        amp_knobs = tuple(
            (float(amp_thetas[i][f_idx]), float(amp_psis[i][f_idx]))
            for i in range(k)
        )
        features.append(
            Feature(
                name=records[feat_id].name,
                cluster=cluster_per_feature[f_idx],
                beta=betas_by_cluster[cluster_per_feature[f_idx]],
                alpha=float(alphas[f_idx]),
                phi=float(phis[f_idx]),
                amp_knobs=amp_knobs,
            )
        )
    hierarchy: dict[str, list[str]] = {c: [] for c in cluster_order}
    for f in features:
        hierarchy[f.cluster].append(f.name)
    return Dictionary(
        name="learned_assignment",
        features=features,
        hierarchy=hierarchy,
        encoding=encoding,
    )


def _greedy_axis_assignment(
    records: dict,
    ids: list[int],
    projs: np.ndarray,
    encoding,
    knob_order: list[str],
    candidate_axes: list[int],
) -> tuple[dict[str, int], float, list[dict]]:
    """Greedy permutation search.

    For each knob in ``knob_order``, tries each unused axis from
    ``candidate_axes`` and locks in the axis whose addition gives the
    best decoder-Gram Spearman. Returns ``(assignment, best_spearman,
    trajectory)`` where trajectory records the best score at every
    step.
    """
    cos_sq = _decoder_cosine_gram(projs[ids])
    assigned: dict[str, int] = {}
    used: set[int] = set()
    trajectory: list[dict] = []

    for knob in knob_order:
        best_axis = None
        best_score = -2.0
        for axis in candidate_axes:
            if axis in used:
                continue
            trial = dict(assigned)
            trial[knob] = axis
            d = _build_dict_from_assignment(
                records, ids, projs, encoding, trial
            )
            g = d.gram()
            score = _spearman(np.abs(g) ** 2, cos_sq)
            if score > best_score:
                best_score = score
                best_axis = axis
        if best_axis is None:
            break
        assigned[knob] = best_axis
        used.add(best_axis)
        trajectory.append(
            {"knob": knob, "axis": best_axis, "spearman": best_score}
        )
    return assigned, best_score, trajectory


def run_learned_assignment(args: argparse.Namespace) -> dict:
    """Prototype: greedy axis-to-knob permutation vs hardcoded baseline."""
    n_clusters = 16
    cluster_size = 4
    d_model = 32
    projs, labels = synth_clustered_sae(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        d_model=d_model,
        seed=args.seed,
    )
    records = _records_from_projections(projs, labels)
    n_features = n_clusters * cluster_size  # 64

    runs: list[dict] = []
    for k in args.k:
        cap = 8 * 2 ** k
        if n_features > cap:
            continue
        ids = list(range(n_features))
        encoding = Rung5(n_amp_qubits=k)

        # Hardcoded baseline: α←PC2 (axis 1), φ←PC3 (axis 2),
        # amp_i_{theta,psi} ← PC{4+2i}, PC{5+2i} (axes 3+2i, 4+2i).
        baseline_assignment: dict[str, int] = {"alpha": 1, "phi": 2}
        for i in range(k):
            baseline_assignment[f"amp_{i}_theta"] = 3 + 2 * i
            baseline_assignment[f"amp_{i}_psi"] = 4 + 2 * i

        knob_order = ["alpha", "phi"]
        for i in range(k):
            knob_order.append(f"amp_{i}_theta")
            knob_order.append(f"amp_{i}_psi")
        max_axis = max(2 * (2 + k) + 4, d_model)
        candidate_axes = list(range(min(max_axis, d_model)))

        # Baseline metrics.
        d_base = _build_dict_from_assignment(
            records, ids, projs, encoding, baseline_assignment
        )
        cos_sq = _decoder_cosine_gram(projs[ids])
        g_base = d_base.gram()
        spearman_base = _spearman(np.abs(g_base) ** 2, cos_sq)
        cond_base = _cond_number(g_base)

        # Greedy learned assignment.
        t0 = time.perf_counter()
        learned_assignment, spearman_learned, trajectory = (
            _greedy_axis_assignment(
                records, ids, projs, encoding, knob_order, candidate_axes
            )
        )
        search_s = time.perf_counter() - t0

        d_learned = _build_dict_from_assignment(
            records, ids, projs, encoding, learned_assignment
        )
        g_learned = d_learned.gram()
        cond_learned = _cond_number(g_learned)

        runs.append({
            "k": k,
            "n_features": n_features,
            "baseline_assignment": baseline_assignment,
            "learned_assignment": learned_assignment,
            "spearman_baseline": spearman_base,
            "spearman_learned": spearman_learned,
            "delta_spearman": spearman_learned - spearman_base,
            "cond_baseline": cond_base,
            "cond_learned": cond_learned,
            "search_seconds": search_s,
            "trajectory": trajectory,
            "n_candidate_axes": len(candidate_axes),
        })
        print(
            f"k={k} N={n_features} cap={cap}:\n"
            f"  baseline   Spearman={spearman_base:+.4f}  cond={cond_base:.2e}  "
            f"assignment={baseline_assignment}\n"
            f"  learned    Spearman={spearman_learned:+.4f}  cond={cond_learned:.2e}  "
            f"assignment={learned_assignment}\n"
            f"  Δ Spearman={spearman_learned - spearman_base:+.4f}  "
            f"search={search_s:.2f}s ({len(candidate_axes)} candidate axes)"
        )
    return {
        "schema": "polygram.rung5_pareto.learned_assignment.v1",
        "config": {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "d_model": d_model,
            "seed": args.seed,
            "k": list(args.k),
        },
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# CLI scaffolding
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "  n/a "


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    subparsers = parser.add_subparsers(dest="scan", required=True)

    def _add_json_out(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--json-out",
            type=Path,
            default=None,
            help="Optional JSON output path.",
        )

    cq = subparsers.add_parser(
        "capacity-quality",
        help="Encoding-ladder Pareto: quality at fixed N for each encoding.",
    )
    cq.add_argument("--seed", type=int, default=0)
    _add_json_out(cq)
    cq.set_defaults(func=run_capacity_quality)

    sd = subparsers.add_parser(
        "saturation-density",
        help="Rank vs N around cap=8*2^k for each k.",
    )
    sd.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    sd.add_argument("--seeds", type=int, nargs="+", default=[0, 42])
    _add_json_out(sd)
    sd.set_defaults(func=run_saturation_density)

    pa = subparsers.add_parser(
        "pca-amp-ablation",
        help="assign_amp_knobs ∈ {False, True} delta at each k.",
    )
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--k", type=int, nargs="+", default=[3, 4])
    _add_json_out(pa)
    pa.set_defaults(func=run_pca_amp_ablation)

    la = subparsers.add_parser(
        "learned-assignment",
        help="Greedy axis-to-knob permutation vs hardcoded baseline.",
    )
    la.add_argument("--seed", type=int, default=0)
    la.add_argument("--k", type=int, nargs="+", default=[3, 4])
    _add_json_out(la)
    la.set_defaults(func=run_learned_assignment)

    args = parser.parse_args(argv)
    artifact = args.func(args)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(artifact, indent=2))
        print(f"\nJSON artifact written → {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
