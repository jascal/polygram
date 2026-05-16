"""Empirical rank verification for Rung5 dictionaries.

The `add-rung5-encoding` openspec change predicts Rung5 saturates
at **rank 8 · 2^k** for `Rung5(n_amp_qubits=k)`. Generalises the
Rung4 (k=2 → rank 32) result: the product amp branch on k qubits

    |amp(θ_0, ψ_0, ..., θ_{k-1}, ψ_{k-1})⟩
        = ⊗_{i=0}^{k-1} |u(θ_i, ψ_i)⟩_{q(3+i)}

spans the full ``C^{2^k}`` amp subspace, giving total dim
``C^8 ⊗ C^{2^k} = C^{8 · 2^k}``.

This probe verifies that empirically for a configurable ladder of
k values, computing Gram rank at `N ∈ {cap, 2·cap}` for each.

Usage:

    python examples/rung5_rank_verification.py
    python examples/rung5_rank_verification.py --k 2 3 4
    python examples/rung5_rank_verification.py \\
        --json-out docs/research/data/rung5_rank_verification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from polygram import Dictionary, Feature
from polygram.encoding import Rung5


def make_features(n: int, k: int, seed: int) -> list[Feature]:
    """N features with random params drawn from the full parameter
    range for all `4 + 2k` per-feature knobs."""
    rng = np.random.default_rng(seed)
    feats = []
    for i in range(n):
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
    return feats


def make_dict(n: int, k: int, seed: int) -> Dictionary:
    feats = make_features(n, k=k, seed=seed)
    return Dictionary(
        name=f"probe_k{k}_n{n}",
        features=feats,
        hierarchy={"all": [f.name for f in feats]},
        encoding=Rung5(n_amp_qubits=k),
    )


def rank_at_tol(s: np.ndarray, tol: float) -> int:
    """Count singular values above `tol * max(s)` (relative tol)."""
    if s.size == 0:
        return 0
    return int(np.sum(s > tol * float(np.max(s))))


def measure(k: int, sizes: list[int], seed: int) -> list[dict]:
    rows = []
    for n in sizes:
        d = make_dict(n, k=k, seed=seed)
        g = d.gram()
        s = np.real(np.linalg.svd(g, compute_uv=False))
        s_max = float(s.max()) if s.size else 0.0
        nz = s[s > 1e-15]
        s_min_nz = float(nz.min()) if nz.size else 0.0
        rows.append({
            "n": int(n),
            "rank_1e_12": rank_at_tol(s, 1e-12),
            "rank_1e_9": rank_at_tol(s, 1e-9),
            "rank_1e_6": rank_at_tol(s, 1e-6),
            "sigma_max": s_max,
            "sigma_min_nonzero": s_min_nz,
            "sigma_min": float(s.min()) if s.size else 0.0,
        })
    return rows


def print_table(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'N':>4}  {'rank@1e-12':>10}  {'rank@1e-9':>9}  "
          f"{'σ_max':>10}  {'σ_min>0':>11}  {'σ_min':>10}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['n']:>4}  {r['rank_1e_12']:>10}  "
            f"{r['rank_1e_9']:>9}  "
            f"{r['sigma_max']:>10.4e}  "
            f"{r['sigma_min_nonzero']:>11.4e}  "
            f"{r['sigma_min']:>10.4e}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the JSON data artifact.",
    )
    p.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Rung5 n_amp_qubits values to probe (default: 2, 3, 4).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 42],
        help="Seeds for the sanity-check sweep.",
    )
    return p.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args()

    runs: dict[str, dict] = {}
    for k in args.k:
        cap = 8 * 2 ** k
        sizes = [cap // 4, cap // 2, cap, 2 * cap]
        for seed in args.seeds:
            rows = measure(k, sizes, seed=seed)
            print_table(f"Rung5(k={k}) — seed={seed}", rows)
            runs[f"rung5_k{k}_seed{seed}"] = {
                "k": k,
                "cap": cap,
                "seed": seed,
                "rows": rows,
            }

        # Pinned verification: rank == cap at N = cap and at N = 2·cap.
        pinned = runs[f"rung5_k{k}_seed{args.seeds[0]}"]["rows"]
        by_n = {r["n"]: r for r in pinned}
        cap_rank = by_n[cap]["rank_1e_12"]
        sat_rank = by_n[2 * cap]["rank_1e_12"]
        print(
            f"\nk={k}: rank@N={cap} = {cap_rank} (expected: {cap}); "
            f"rank@N={2 * cap} = {sat_rank} (saturation check, "
            f"expected: {cap})"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema": "polygram.rung5_rank_verification.v1",
            "runs": runs,
        }
        args.json_out.write_text(json.dumps(artifact, indent=2))
        print(f"\nJSON artifact written → {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
