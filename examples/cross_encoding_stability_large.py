"""Large-scale cross-encoding stability spike — push hardware.

Companion to ``examples/cross_encoding_stability.py``. The small-fixture
spike found that MPSRung1 and HEA_Rung2(depth=2) agreed on kept-edge
classifications across every fixture, but cross-cluster *magnitudes*
differed by up to +0.30. The research note flagged two open questions:

1. **Variance.** The small spike tested 3 fixtures of 4 features
   each — does the kept-edge agreement hold over hundreds or
   thousands of randomly-sampled feature subsets, or do edge-case
   subsets show classification drift?
2. **Depth.** Does HEA `depth=4`, `8`, `16` recover MPS-like
   cross-cluster magnitude variation that depth=2 collapses?

This script answered (2) immediately during smoke-testing: under the
default ``_default_hea_theta`` heuristic, HEA gram is **provably
depth-invariant**. ``_default_hea_theta`` only populates layer 0
(``theta[*, 0, *]``); higher-depth slices are zero, so layers 1..N-1
are entangler-only with identity rotations. By the global-unitary
invariance of the gram (``|<U·a|U·b>|² = |<a|b>|²``), those extra
layers cancel exactly. Empirically: depth ∈ {2, 4, 8, 16} produce
gram entries identical to 6+ decimal places on every fixture tested.

That leaves question (1) — the variance question — as the actual
hardware-pushing axis. This script samples N feature subsets of
size 8 from a real SAE checkpoint, runs MPS rung-1 vs HEA(depth=2)
under default theta, and aggregates the per-subset comparison.
``--depths`` is exposed as a flag (so anyone re-running can confirm
the depth-invariance finding) but the default is ``[2]`` only.

Multiprocessing across CPU cores keeps wall time manageable. On
16-logical-core hardware, 1000 subsets × 5 encodings completes in
~90 seconds; 5000 subsets × 5 encodings in ~7 minutes; 10000
subsets in ~14 minutes.

Usage
-----

    python examples/cross_encoding_stability_large.py
    python examples/cross_encoding_stability_large.py --push
    python examples/cross_encoding_stability_large.py --insane
    python examples/cross_encoding_stability_large.py \\
        --n-subsets 200 --depths 2,4,8 --workers 4

The script writes ``scratch/cross-encoding-large/results.json``
(every per-subset record), ``...summary.json`` (the aggregated
distributions), and prints a console summary.
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
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Per-subset comparison (worker function — must be top-level for pickling)
# ---------------------------------------------------------------------------


def _compare_subset(args: tuple[str, list[int], list[int]]) -> dict[str, Any]:
    """Worker entry point. Loads the lazy-slice subset, builds N
    Dictionaries (one per encoding), runs triage_dictionary on each,
    and returns a per-pair record."""
    sae_path, feature_ids, hea_depths = args

    # Imports inside the worker so the parent stays cheap to fork.
    from polygram import Dictionary, HEA_Rung2, load_sae_safetensors
    from polygram.analysis import (
        build_separation_graph,
        build_sharing_graph,
        triage_dictionary,
    )
    from polygram.sae_import import from_sae_lens

    records = load_sae_safetensors(sae_path, feature_ids=feature_ids)
    d_mps, _ = from_sae_lens(records, feature_ids, assign_gamma=True)
    encodings: dict[str, Dictionary] = {"mps": d_mps}
    for depth in hea_depths:
        encodings[f"hea_d{depth}"] = Dictionary(
            name=d_mps.name,
            features=d_mps.features,
            hierarchy=d_mps.hierarchy,
            encoding=HEA_Rung2(depth=depth),
        )

    triage = {label: triage_dictionary(d) for label, d in encodings.items()}

    # Per-pair magnitudes per encoding.
    pairs_per_enc: dict[str, list[dict[str, float | bool]]] = {}
    for label, pred in triage.items():
        rows = []
        for p in pred.pairs:
            rows.append(
                {
                    "a": p.feature_a,
                    "b": p.feature_b,
                    "is_cross_cluster": p.is_cross_cluster,
                    "current": float(p.current_overlap),
                    "floor": float(p.structural_floor),
                    "gap": float(p.cancellation_gap),
                    "V": float(p.V),
                }
            )
        pairs_per_enc[label] = rows

    # Edge-set agreement at default thresholds.
    edge_sets: dict[str, dict[str, list[list[str]]]] = {}
    for label, pred in triage.items():
        sharing = build_sharing_graph(pred, threshold=0.5)
        separation = build_separation_graph(pred, threshold=0.2)
        edge_sets[label] = {
            "sharing": [[e.source, e.target] for e in sharing.edges],
            "separation": [[e.source, e.target] for e in separation.edges],
        }

    return {
        "feature_ids": feature_ids,
        "pairs_per_enc": pairs_per_enc,
        "edge_sets": edge_sets,
        "suitability": {
            label: float(p.encoding_suitability_score)
            for label, p in triage.items()
        },
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class Aggregate:
    n_subsets: int = 0
    pair_count: int = 0
    cross_count: int = 0
    within_count: int = 0
    delta_current_within: list[float] = field(default_factory=list)
    delta_current_cross: list[float] = field(default_factory=list)
    delta_floor_within: list[float] = field(default_factory=list)
    delta_floor_cross: list[float] = field(default_factory=list)
    sharing_agree: int = 0
    separation_agree: int = 0
    sharing_disagree_subsets: list[list[int]] = field(default_factory=list)
    separation_disagree_subsets: list[list[int]] = field(default_factory=list)


def _aggregate(records: list[dict[str, Any]], hea_label: str) -> Aggregate:
    """Aggregate per-subset records, comparing MPS vs one HEA depth."""
    agg = Aggregate()
    for rec in records:
        agg.n_subsets += 1
        mps_pairs = rec["pairs_per_enc"]["mps"]
        hea_pairs = rec["pairs_per_enc"].get(hea_label)
        if hea_pairs is None:
            continue
        for p_mps, p_hea in zip(mps_pairs, hea_pairs):
            assert p_mps["a"] == p_hea["a"] and p_mps["b"] == p_hea["b"]
            d_curr = p_hea["current"] - p_mps["current"]
            d_floor = p_hea["floor"] - p_mps["floor"]
            agg.pair_count += 1
            if p_mps["is_cross_cluster"]:
                agg.cross_count += 1
                agg.delta_current_cross.append(d_curr)
                agg.delta_floor_cross.append(d_floor)
            else:
                agg.within_count += 1
                agg.delta_current_within.append(d_curr)
                agg.delta_floor_within.append(d_floor)

        edge_mps = rec["edge_sets"]["mps"]
        edge_hea = rec["edge_sets"].get(hea_label, {})
        sharing_mps = {tuple(e) for e in edge_mps["sharing"]}
        sharing_hea = {tuple(e) for e in edge_hea.get("sharing", [])}
        separation_mps = {tuple(e) for e in edge_mps["separation"]}
        separation_hea = {tuple(e) for e in edge_hea.get("separation", [])}
        if sharing_mps == sharing_hea:
            agg.sharing_agree += 1
        else:
            agg.sharing_disagree_subsets.append(rec["feature_ids"])
        if separation_mps == separation_hea:
            agg.separation_agree += 1
        else:
            agg.separation_disagree_subsets.append(rec["feature_ids"])
    return agg


def _summarize(arr: list[float]) -> dict[str, float]:
    if not arr:
        return {"n": 0}
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-path",
        default="./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors",
        help="path to the safetensors file",
    )
    parser.add_argument(
        "--n-subsets", type=int, default=1000,
        help="number of 8-feature subsets to sample (default 1000)",
    )
    parser.add_argument(
        "--depths", default="2",
        help="comma-separated HEA depths to test (default: 2). Higher "
             "depths produce identical gram values under default theta — "
             "see module docstring. Set e.g. --depths 2,4,8 to confirm.",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 4,
        help="multiprocessing workers (default: os.cpu_count())",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="numpy seed for subset sampling",
    )
    parser.add_argument(
        "--output-dir", default="./scratch/cross-encoding-large",
        help="where to write per-subset records and the summary JSON",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="alias for --n-subsets 10000 (~3 min on 8 workers)",
    )
    parser.add_argument(
        "--insane", action="store_true",
        help="alias for --n-subsets 50000 (~15 min on 8 workers; "
             "really push hardware)",
    )
    args = parser.parse_args(argv)

    if args.insane:
        args.n_subsets = 50000
    elif args.push:
        args.n_subsets = 10000

    sae_path = Path(args.sae_path).resolve()
    if not sae_path.exists():
        print(f"error: SAE file not found at {sae_path}", file=sys.stderr)
        print(
            "hint: hf download jbloom/GPT2-Small-SAEs-Reformatted "
            "--include='blocks.0.hook_resid_pre/sae_weights.safetensors' "
            "--local-dir ./scratch/real-sae",
            file=sys.stderr,
        )
        return 2

    hea_depths = [int(x) for x in args.depths.split(",") if x]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover total feature count via lazy load metadata ---
    from safetensors import safe_open  # only used here

    with safe_open(str(sae_path), framework="numpy") as f:
        keys = list(f.keys())
        # Same precedence as Polygram's loader.
        for cand in ("W_dec", "decoder.weight", "dec"):
            if cand in keys:
                shape = tuple(f.get_slice(cand).get_shape())
                break
        else:
            print(f"error: no decoder key in {sae_path}", file=sys.stderr)
            return 2
    n_features = shape[0]
    print(
        f"SAE: {sae_path.name} — {n_features} features, "
        f"d_model={shape[1]}"
    )
    print(
        f"plan: {args.n_subsets} subsets × {1 + len(hea_depths)} encodings "
        f"(MPS + HEA depth {hea_depths}); {args.workers} workers"
    )

    # --- Sample subsets ---
    rng = np.random.default_rng(args.seed)
    subsets = []
    for _ in range(args.n_subsets):
        ids = sorted(int(x) for x in rng.choice(n_features, size=8, replace=False))
        subsets.append(ids)

    # --- Fan out to workers ---
    worker_args = [(str(sae_path), ids, hea_depths) for ids in subsets]
    t0 = time.monotonic()
    records: list[dict[str, Any]] = []
    print_every = max(1, args.n_subsets // 20)
    with mp.Pool(processes=args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_compare_subset, worker_args, chunksize=4)):
            records.append(rec)
            if (i + 1) % print_every == 0 or (i + 1) == args.n_subsets:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed
                eta = (args.n_subsets - (i + 1)) / max(rate, 1e-9)
                print(
                    f"  [{i + 1:5d}/{args.n_subsets}] "
                    f"{elapsed:.1f}s elapsed, {rate:.1f} subsets/s, "
                    f"ETA {eta:.0f}s",
                    flush=True,
                )
    elapsed = time.monotonic() - t0
    print(f"done in {elapsed:.1f}s ({len(records) / elapsed:.1f} subsets/s)")

    # --- Persist raw records (potentially large for --insane runs) ---
    raw_path = out_dir / "results.json"
    raw_path.write_text(json.dumps({
        "config": {
            "sae_path": str(sae_path),
            "n_subsets": args.n_subsets,
            "depths": hea_depths,
            "workers": args.workers,
            "seed": args.seed,
            "wall_seconds": elapsed,
        },
        "records": records,
    }))
    print(f"wrote raw records: {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")

    # --- Aggregate per HEA depth ---
    summary: dict[str, Any] = {
        "config": {
            "sae_path": str(sae_path),
            "n_subsets": args.n_subsets,
            "depths": hea_depths,
        },
        "per_depth": {},
    }
    print()
    print(f"{'='*78}")
    print(f"AGGREGATE: MPS vs HEA(depth=d), {args.n_subsets} subsets")
    print(f"{'='*78}")
    for depth in hea_depths:
        label = f"hea_d{depth}"
        agg = _aggregate(records, label)
        within_curr = _summarize(agg.delta_current_within)
        cross_curr = _summarize(agg.delta_current_cross)
        sharing_pct = 100.0 * agg.sharing_agree / max(agg.n_subsets, 1)
        separation_pct = 100.0 * agg.separation_agree / max(agg.n_subsets, 1)

        summary["per_depth"][label] = {
            "delta_current_within": within_curr,
            "delta_current_cross": cross_curr,
            "delta_floor_within": _summarize(agg.delta_floor_within),
            "delta_floor_cross": _summarize(agg.delta_floor_cross),
            "sharing_agree_pct": sharing_pct,
            "separation_agree_pct": separation_pct,
            "sharing_disagree_subsets": agg.sharing_disagree_subsets[:50],
            "separation_disagree_subsets": agg.separation_disagree_subsets[:50],
            "n_pairs_within": agg.within_count,
            "n_pairs_cross": agg.cross_count,
        }

        print()
        print(f"--- HEA depth={depth} -----------------------------------")
        print(f"  Δ(current) within-cluster (n={within_curr['n']}): "
              f"median={within_curr.get('median', 0):+.4f}, "
              f"|max|={max(abs(within_curr.get('min', 0)), abs(within_curr.get('max', 0))):.4f}")
        print(f"  Δ(current) cross-cluster  (n={cross_curr['n']}): "
              f"median={cross_curr.get('median', 0):+.4f}, "
              f"p05={cross_curr.get('p05', 0):+.4f}, "
              f"p95={cross_curr.get('p95', 0):+.4f}, "
              f"max={cross_curr.get('max', 0):+.4f}")
        print(f"  edge-set agreement: "
              f"sharing={sharing_pct:.1f}%, "
              f"separation={separation_pct:.1f}%")
        if agg.sharing_disagree_subsets:
            print(f"  ⚠ {len(agg.sharing_disagree_subsets)} sharing-graph "
                  f"disagreements (first: {agg.sharing_disagree_subsets[0]})")
        if agg.separation_disagree_subsets:
            print(f"  ⚠ {len(agg.separation_disagree_subsets)} separation-graph "
                  f"disagreements (first: {agg.separation_disagree_subsets[0]})")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
