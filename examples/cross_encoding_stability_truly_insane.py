"""Overnight cross-encoding stability spike — three-stage hardware push.

Targets a ~5h budget on 16 logical cores. Companion to
``cross_encoding_stability.py`` (small-fixture spike) and
``cross_encoding_stability_large.py`` (50K-subset variance test).

The 50K-subset run with the simpler script confirmed three things:

- HEA gram is depth-invariant under ``_default_hea_theta`` (extra
  layers are entangler-only with identity rotations; cancel by global
  unitary invariance).
- Cross-cluster Δ(current) MPS-vs-HEA(d=2) sits in [+0.04, +0.31]
  with a tight median around +0.18 — the small-spike Finding 2 holds
  at scale.
- Edge-set agreement at default thresholds is essentially 100%.

This script presses three new questions across three stages:

**Stage A — Random-selection variance baseline** (~3h, 500K subsets)
Pure variance push. Random 8-feature subsets from the SAE; MPS rung-1
vs HEA(d=2) under default theta. Looking for: rare edge-set
disagreements (any nonzero rate sets a real-world expectation),
tighter percentiles on the cross-cluster Δ distribution.

**Stage B — Projection-similarity-anchored selection** (~30 min,
100K subsets) Each subset is built by picking a random *anchor*
feature and adding its 7 nearest neighbours by projection cosine.
This is the geometry of the original `feat_7836` cluster that
surfaced the encoding-stability question — the small spike sampled
this regime once. Stage B asks: does the variance behavior under
this regime differ from random?

**Stage C — Real-depth sweep with non-default theta** (~1.5h,
50K subsets × HEA depths {2, 4, 8}) Steps outside the default-theta
heuristic. Each feature's θ tensor is filled with deterministic
*pseudo-random* angles (uniform [-π/8, π/8] seeded by
``(feature_id, layer)``) so the depth axis actually exercises
HEA expressivity. Caveat in the research note: pseudo-random theta
is noise rather than signal — this stage does not validate HEA's
depth utility for SAE compression. It only answers "does the gram
shape under HEA depend on depth when layers > 0 are not zero".

Output layout under ``./scratch/cross-encoding-truly-insane/``:

- ``stage_a/records.jsonl`` (one record per line, streamed)
- ``stage_a/summary.json`` (running aggregate, written periodically)
- ``stage_b/records.jsonl``, ``stage_b/summary.json``
- ``stage_c/records.jsonl``, ``stage_c/summary.json``
- ``run_log.json`` — top-level start/end timestamps + stage status

Each stage is checkpoint-resumable: re-running picks up where the
JSONL ended.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Per-subset comparison workers
# ---------------------------------------------------------------------------


def _trim_record(
    feature_ids: list[int],
    triage: dict[str, Any],
    edge_sets: dict[str, dict[str, set[tuple[str, str]]]],
) -> dict[str, Any]:
    """Compress a per-subset record to the minimum needed for
    aggregation: per-pair Δ values relative to MPS, plus boolean
    edge-set agreement flags. ~500 bytes JSON per record."""
    mps_pairs = triage["mps"]
    out: dict[str, Any] = {"feature_ids": feature_ids, "encodings": {}}
    for label, pairs in triage.items():
        if label == "mps":
            continue
        delta_within: list[float] = []
        delta_cross: list[float] = []
        floor_within: list[float] = []
        floor_cross: list[float] = []
        for p_mps, p_other in zip(mps_pairs, pairs):
            d_curr = p_other["current"] - p_mps["current"]
            d_floor = p_other["floor"] - p_mps["floor"]
            if p_mps["is_cross_cluster"]:
                delta_cross.append(d_curr)
                floor_cross.append(d_floor)
            else:
                delta_within.append(d_curr)
                floor_within.append(d_floor)
        out["encodings"][label] = {
            "dc_within": delta_within,
            "dc_cross": delta_cross,
            "df_within": floor_within,
            "df_cross": floor_cross,
            "sharing_agree": (
                edge_sets["mps"]["sharing"] == edge_sets[label]["sharing"]
            ),
            "separation_agree": (
                edge_sets["mps"]["separation"] == edge_sets[label]["separation"]
            ),
        }
    return out


def _build_dictionaries(
    sae_path: str,
    feature_ids: list[int],
    encoding_specs: list[dict[str, Any]],
):
    """Build a list of (label, Dictionary) pairs per encoding spec.

    encoding_specs items have keys:
      - "label": str
      - "kind": "mps" | "hea_default" | "hea_random"
      - "depth": int (HEA only)
      - "theta_seed": int (only for hea_random)
    """
    from polygram import Dictionary, HEA_Rung2, load_sae_safetensors
    from polygram.sae_import import from_sae_lens

    records = load_sae_safetensors(sae_path, feature_ids=feature_ids)
    base_d, _ = from_sae_lens(records, feature_ids, assign_gamma=True)
    out: dict[str, Any] = {}
    for spec in encoding_specs:
        if spec["kind"] == "mps":
            out[spec["label"]] = base_d
        elif spec["kind"] == "hea_default":
            out[spec["label"]] = Dictionary(
                name=base_d.name,
                features=base_d.features,
                hierarchy=base_d.hierarchy,
                encoding=HEA_Rung2(depth=spec["depth"]),
            )
        elif spec["kind"] == "hea_random":
            from dataclasses import replace as dataclass_replace

            from polygram.encoding import HEA_Rung2 as HEA

            encoding = HEA(depth=spec["depth"])
            shape = encoding.theta_shape
            features_with_theta = []
            for i, f in enumerate(base_d.features):
                rng = np.random.default_rng(int(spec["theta_seed"]) + i * 1000003)
                theta = rng.uniform(-np.pi / 8, np.pi / 8, size=shape)
                features_with_theta.append(dataclass_replace(f, theta=theta))
            out[spec["label"]] = Dictionary(
                name=base_d.name,
                features=features_with_theta,
                hierarchy=base_d.hierarchy,
                encoding=encoding,
            )
        else:
            raise ValueError(f"unknown encoding kind: {spec['kind']!r}")
    return out


def _compare_subset_worker(
    args: tuple[str, list[int], list[dict[str, Any]]],
) -> dict[str, Any]:
    sae_path, feature_ids, encoding_specs = args

    from polygram.analysis import (
        build_separation_graph,
        build_sharing_graph,
        triage_dictionary,
    )

    dictionaries = _build_dictionaries(sae_path, feature_ids, encoding_specs)

    triage = {}
    edge_sets: dict[str, dict[str, Any]] = {}
    for label, d in dictionaries.items():
        pred = triage_dictionary(d)
        rows = []
        for p in pred.pairs:
            rows.append(
                {
                    "is_cross_cluster": p.is_cross_cluster,
                    "current": float(p.current_overlap),
                    "floor": float(p.structural_floor),
                }
            )
        triage[label] = rows
        sharing = build_sharing_graph(pred, threshold=0.5)
        separation = build_separation_graph(pred, threshold=0.2)
        edge_sets[label] = {
            "sharing": frozenset(
                tuple(sorted([e.source, e.target])) for e in sharing.edges
            ),
            "separation": frozenset(
                tuple(sorted([e.source, e.target])) for e in separation.edges
            ),
        }
    return _trim_record(feature_ids, triage, edge_sets)


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def _random_selector(
    n_features: int, n_subsets: int, seed: int,
) -> Iterator[list[int]]:
    rng = np.random.default_rng(seed)
    for _ in range(n_subsets):
        yield sorted(int(x) for x in rng.choice(n_features, size=8, replace=False))


def _similarity_anchored_selector(
    sae_path: str, n_subsets: int, seed: int,
) -> Iterator[list[int]]:
    """Sample N subsets where each subset is an anchor feature plus its
    7 nearest neighbours by projection cosine (over a 1024-feature
    random sample, for compute tractability — exact NN over 24K
    features per subset would dominate runtime)."""
    from polygram import load_sae_safetensors

    rng = np.random.default_rng(seed)
    # Pre-cache projection vectors for the candidates we'll sample
    # against. Loading all 24K is fine — we eat the cost once.
    records = load_sae_safetensors(sae_path)
    n = len(records)
    proj = np.stack([records[i].projection for i in range(n)])
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    proj_unit = proj / norms

    for _ in range(n_subsets):
        anchor = int(rng.integers(0, n))
        candidates = rng.choice(n, size=1024, replace=False)
        if anchor not in candidates:
            candidates[0] = anchor
        sims = proj_unit[candidates] @ proj_unit[anchor]
        # Top 8 (anchor will be one of them, similarity=1)
        top = candidates[np.argsort(sims)[::-1][:8]]
        yield sorted(int(x) for x in top)


# ---------------------------------------------------------------------------
# Streaming aggregation
# ---------------------------------------------------------------------------


@dataclass
class RunningAgg:
    n_subsets: int = 0
    delta_current_within: list[float] = field(default_factory=list)
    delta_current_cross: list[float] = field(default_factory=list)
    delta_floor_within: list[float] = field(default_factory=list)
    delta_floor_cross: list[float] = field(default_factory=list)
    sharing_agree: int = 0
    separation_agree: int = 0
    sharing_disagree_examples: list[list[int]] = field(default_factory=list)
    separation_disagree_examples: list[list[int]] = field(default_factory=list)


def _summarize(arr: list[float]) -> dict[str, float]:
    if not arr:
        return {"n": 0}
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def _format_progress_metrics(agg: dict[str, RunningAgg]) -> list[str]:
    """One short line per encoding label: edge-set agreement and the
    cross-cluster Δ p05/median/p95. Designed to be printed under the
    checkpoint progress line at every log interval — enough signal for
    a human watcher to spot drift without opening summary.json."""
    lines: list[str] = []
    for label, ra in agg.items():
        n = max(ra.n_subsets, 1)
        share_pct = 100.0 * ra.sharing_agree / n
        sep_pct = 100.0 * ra.separation_agree / n
        dc = ra.delta_current_cross
        if dc:
            a = np.asarray(dc, dtype=float)
            p05 = float(np.percentile(a, 5))
            med = float(np.median(a))
            p95 = float(np.percentile(a, 95))
            cross_str = (
                f"Δ_cross[p05/med/p95]={p05:+.4f}/{med:+.4f}/{p95:+.4f} "
                f"(n={a.size})"
            )
        else:
            cross_str = "Δ_cross=<no cross-cluster pairs yet>"
        within = ra.delta_current_within
        within_med = (
            f"{float(np.median(np.asarray(within, dtype=float))):+.4f}"
            if within
            else "n/a"
        )
        lines.append(
            f"    {label:>14}: shareAgr={share_pct:6.2f}% "
            f"sepAgr={sep_pct:6.2f}% {cross_str} Δ_within_med={within_med}"
        )
    return lines


def _update_agg(agg: dict[str, RunningAgg], record: dict[str, Any]) -> None:
    for label, enc in record["encodings"].items():
        ra = agg.setdefault(label, RunningAgg())
        ra.n_subsets += 1
        ra.delta_current_within.extend(enc["dc_within"])
        ra.delta_current_cross.extend(enc["dc_cross"])
        ra.delta_floor_within.extend(enc["df_within"])
        ra.delta_floor_cross.extend(enc["df_cross"])
        if enc["sharing_agree"]:
            ra.sharing_agree += 1
        elif len(ra.sharing_disagree_examples) < 50:
            ra.sharing_disagree_examples.append(record["feature_ids"])
        if enc["separation_agree"]:
            ra.separation_agree += 1
        elif len(ra.separation_disagree_examples) < 50:
            ra.separation_disagree_examples.append(record["feature_ids"])


def _agg_to_summary(agg: dict[str, RunningAgg]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, ra in agg.items():
        out[label] = {
            "n_subsets": ra.n_subsets,
            "sharing_agree_pct": (
                100.0 * ra.sharing_agree / max(ra.n_subsets, 1)
            ),
            "separation_agree_pct": (
                100.0 * ra.separation_agree / max(ra.n_subsets, 1)
            ),
            "sharing_disagree_count": ra.n_subsets - ra.sharing_agree,
            "separation_disagree_count": ra.n_subsets - ra.separation_agree,
            "sharing_disagree_examples": ra.sharing_disagree_examples[:50],
            "separation_disagree_examples": ra.separation_disagree_examples[:50],
            "delta_current_within": _summarize(ra.delta_current_within),
            "delta_current_cross": _summarize(ra.delta_current_cross),
            "delta_floor_within": _summarize(ra.delta_floor_within),
            "delta_floor_cross": _summarize(ra.delta_floor_cross),
        }
    return out


# ---------------------------------------------------------------------------
# Stage runner — one Pool, JSONL streaming, periodic summary checkpoints
# ---------------------------------------------------------------------------


def run_stage(
    *,
    stage_name: str,
    out_dir: Path,
    sae_path: str,
    selector: Iterator[list[int]],
    n_subsets: int,
    encoding_specs: list[dict[str, Any]],
    workers: int,
    log_every: int = 5000,
) -> dict[str, Any]:
    stage_dir = out_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    records_path = stage_dir / "records.jsonl"
    summary_path = stage_dir / "summary.json"

    # Resume support: if records.jsonl exists, count its lines and
    # skip selector ahead by that many. Don't re-aggregate the
    # existing records — we trust the on-disk summary if it's there.
    existing = 0
    if records_path.exists():
        existing = sum(1 for _ in records_path.open())
        # Advance the selector past records we already have.
        for _ in range(existing):
            next(selector, None)

    print()
    print(f"{'='*78}")
    print(f"STAGE {stage_name.upper()}: {n_subsets} subsets, {len(encoding_specs)} encodings")
    if existing:
        print(f"  resuming from existing {existing} records on disk")
    print(f"{'='*78}", flush=True)

    agg: dict[str, RunningAgg] = {}
    if existing and summary_path.exists():
        # Re-read existing records to re-populate the aggregate.
        # Cost: O(existing) but disk-bound; once per stage.
        with records_path.open() as f:
            for line in f:
                _update_agg(agg, json.loads(line))

    args_iter = (
        (sae_path, ids, encoding_specs)
        for ids in selector
    )

    t0 = time.monotonic()
    with mp.Pool(processes=workers) as pool, records_path.open("a") as records_f:
        target = n_subsets
        completed = existing
        for rec in pool.imap_unordered(_compare_subset_worker, args_iter, chunksize=4):
            records_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            _update_agg(agg, rec)
            completed += 1
            if completed % log_every == 0 or completed == target:
                elapsed = time.monotonic() - t0
                rate = (completed - existing) / max(elapsed, 1e-9)
                eta = (target - completed) / max(rate, 1e-9)
                summary_path.write_text(json.dumps(_agg_to_summary(agg), indent=2))
                print(
                    f"  [{completed:7d}/{target}] "
                    f"{elapsed:.0f}s elapsed, {rate:.1f} subsets/s, "
                    f"ETA {eta/60:.1f} min",
                    flush=True,
                )
                for line in _format_progress_metrics(agg):
                    print(line, flush=True)
                # Force a JSONL flush so a crash leaves at most one
                # incomplete checkpoint behind, not the whole stage.
                records_f.flush()
            if completed >= target:
                break

    summary = _agg_to_summary(agg)
    summary_path.write_text(json.dumps(summary, indent=2))
    elapsed = time.monotonic() - t0
    print(f"stage {stage_name} done in {elapsed/60:.1f} min")
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-path",
        default="./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument(
        "--output-dir", default="./scratch/cross-encoding-truly-insane",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Per-stage knobs (overridable; defaults sized for ~5h on 16 cores).
    parser.add_argument("--n-stage-a", type=int, default=500_000)
    parser.add_argument("--n-stage-b", type=int, default=100_000)
    parser.add_argument("--n-stage-c", type=int, default=50_000)
    parser.add_argument(
        "--stage-c-depths", default="2,4,8",
        help="HEA depths for Stage C (random theta tensors)",
    )

    parser.add_argument(
        "--skip", default="",
        help="comma-separated stage names to skip (e.g. --skip a)",
    )

    args = parser.parse_args(argv)
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    sae_path = Path(args.sae_path).resolve()
    if not sae_path.exists():
        print(f"error: SAE not found at {sae_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover n_features lazily.
    from safetensors import safe_open

    with safe_open(str(sae_path), framework="numpy") as f:
        keys = list(f.keys())
        for cand in ("W_dec", "decoder.weight", "dec"):
            if cand in keys:
                shape = tuple(f.get_slice(cand).get_shape())
                break
        else:
            print("error: no decoder key", file=sys.stderr)
            return 2
    n_features = shape[0]
    print(f"SAE: {sae_path} — n_features={n_features} d_model={shape[1]}")

    run_log_path = out_dir / "run_log.json"
    log: dict[str, Any] = {
        "started_at": time.time(),
        "config": {
            "sae_path": str(sae_path),
            "workers": args.workers,
            "n_stage_a": args.n_stage_a,
            "n_stage_b": args.n_stage_b,
            "n_stage_c": args.n_stage_c,
            "stage_c_depths": args.stage_c_depths,
            "seed": args.seed,
        },
        "stages": {},
    }
    run_log_path.write_text(json.dumps(log, indent=2))

    # ---- Stage A: random selection, MPS + HEA(d=2) default theta ----
    if "a" not in skip:
        encoding_specs_a = [
            {"label": "mps", "kind": "mps"},
            {"label": "hea_d2", "kind": "hea_default", "depth": 2},
        ]
        selector_a = _random_selector(n_features, args.n_stage_a, args.seed)
        log["stages"]["a"] = run_stage(
            stage_name="stage_a",
            out_dir=out_dir,
            sae_path=str(sae_path),
            selector=selector_a,
            n_subsets=args.n_stage_a,
            encoding_specs=encoding_specs_a,
            workers=args.workers,
        )
        run_log_path.write_text(json.dumps(log, indent=2))

    # ---- Stage B: similarity-anchored, MPS + HEA(d=2) default theta ----
    if "b" not in skip:
        encoding_specs_b = [
            {"label": "mps", "kind": "mps"},
            {"label": "hea_d2", "kind": "hea_default", "depth": 2},
        ]
        selector_b = _similarity_anchored_selector(
            str(sae_path), args.n_stage_b, args.seed + 1
        )
        log["stages"]["b"] = run_stage(
            stage_name="stage_b",
            out_dir=out_dir,
            sae_path=str(sae_path),
            selector=selector_b,
            n_subsets=args.n_stage_b,
            encoding_specs=encoding_specs_b,
            workers=args.workers,
        )
        run_log_path.write_text(json.dumps(log, indent=2))

    # ---- Stage C: random selection, MPS + HEA at depths d ∈ {2,4,8} with
    # NON-DEFAULT random theta tensors ----
    if "c" not in skip:
        depths = [int(x) for x in args.stage_c_depths.split(",") if x]
        encoding_specs_c: list[dict[str, Any]] = [
            {"label": "mps", "kind": "mps"},
        ]
        for d in depths:
            encoding_specs_c.append({
                "label": f"hea_rand_d{d}",
                "kind": "hea_random",
                "depth": d,
                "theta_seed": args.seed + 2,
            })
        selector_c = _random_selector(n_features, args.n_stage_c, args.seed + 3)
        log["stages"]["c"] = run_stage(
            stage_name="stage_c",
            out_dir=out_dir,
            sae_path=str(sae_path),
            selector=selector_c,
            n_subsets=args.n_stage_c,
            encoding_specs=encoding_specs_c,
            workers=args.workers,
        )
        run_log_path.write_text(json.dumps(log, indent=2))

    log["completed_at"] = time.time()
    log["wall_seconds"] = log["completed_at"] - log["started_at"]
    run_log_path.write_text(json.dumps(log, indent=2))

    print()
    print(f"{'='*78}")
    print(f"ALL STAGES COMPLETE in {log['wall_seconds']/3600:.2f} h")
    print(f"{'='*78}")
    print(f"output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
