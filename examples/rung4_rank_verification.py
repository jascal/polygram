"""Empirical rank verification for Rung4 dictionaries.

The `add-rung4-encoding-mvp` openspec change predicts Rung4 saturates
at **rank 32**, twice Rung3's empirical 16-cap (per
`docs/research/rung3-rank-bound.md`). The product amp branch

    |amp(θ_3, ψ_3, θ_4, ψ_4)⟩ = |u(θ_3, ψ_3)⟩_{q3} ⊗ |v(θ_4, ψ_4)⟩_{q4}

spans the full ``C^2 ⊗ C^2 = C^4`` amp subspace (vs Rung3's
restricted 2-dim ``span{|00⟩, |11⟩}``), giving total dim
``C^8 ⊗ C^4 = C^32``.

This probe verifies that empirically by computing Gram rank as N
grows from 4 to 40 across two seeds.

Usage:

    python examples/rung4_rank_verification.py
    python examples/rung4_rank_verification.py --json-out docs/research/data/rung4_rank_verification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from polygram import Dictionary, Feature
from polygram.encoding import Rung4


def make_features(n: int, seed: int) -> list[Feature]:
    """N features with random params drawn from the full parameter range
    for all six per-feature knobs (α, β, γ, φ on the MPS side and the
    four amp knobs on q3 + q4)."""
    rng = np.random.default_rng(seed)
    feats = []
    for i in range(n):
        feats.append(
            Feature(
                name=f"f{i}",
                cluster="all",
                alpha=float(rng.uniform(0.0, 2 * np.pi)),
                beta=float(rng.uniform(0.0, 2 * np.pi)),
                gamma=float(rng.uniform(0.0, 2 * np.pi)),
                phi=float(rng.uniform(0.0, 2 * np.pi)),
                theta_amp=float(rng.uniform(0.0, np.pi / 2)),
                psi_aux=float(rng.uniform(0.0, 2 * np.pi)),
                theta_amp_b=float(rng.uniform(0.0, np.pi / 2)),
                psi_amp_b=float(rng.uniform(0.0, 2 * np.pi)),
            )
        )
    return feats


def make_dict(n: int, seed: int) -> Dictionary:
    feats = make_features(n, seed=seed)
    return Dictionary(
        name=f"probe_n{n}",
        features=feats,
        hierarchy={"all": [f.name for f in feats]},
        encoding=Rung4(),
    )


def rank_at_tol(s: np.ndarray, tol: float) -> int:
    """Count singular values above `tol * max(s)` (relative tol)."""
    if s.size == 0:
        return 0
    return int(np.sum(s > tol * float(np.max(s))))


def measure(sizes: list[int], seed: int) -> list[dict]:
    rows = []
    for n in sizes:
        d = make_dict(n, seed=seed)
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
            "singular_values": [float(v) for v in s],
        })
    return rows


def print_table(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'N':>3}  {'rank@1e-12':>10}  {'rank@1e-9':>9}  "
          f"{'σ_max':>10}  {'σ_min>0':>11}  {'σ_min':>10}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['n']:>3}  {r['rank_1e_12']:>10}  "
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
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 24, 32, 40],
        help="N values to probe (default: 4, 8, 16, 24, 32, 40).",
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
    args = parse_args() if argv is None else parse_args()  # ignore argv for CLI mode

    runs: dict[str, dict] = {}
    for seed in args.seeds:
        rows = measure(args.sizes, seed=seed)
        print_table(f"Rung4 — seed={seed}", rows)
        runs[f"rung4_seed{seed}"] = {"seed": seed, "rows": rows}

    # Verify the cap at N=32 and saturation at N=40.
    pinned = runs[f"rung4_seed{args.seeds[0]}"]["rows"]
    by_n = {r["n"]: r for r in pinned}
    if 32 in by_n:
        cap_rank = by_n[32]["rank_1e_12"]
        print(f"\nRank @ N=32 (seed {args.seeds[0]}): {cap_rank} "
              f"(expected: 32)")
    if 40 in by_n:
        sat_rank = by_n[40]["rank_1e_12"]
        print(f"Rank @ N=40 (saturation check): {sat_rank} "
              f"(expected: 32 — above-cap features land in linear span)")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema": "rung4_rank_verification.v1",
            "math_precheck": {
                "mps_subspace_dim": 8,
                "amp_subspace_dim": 4,  # full C^2 ⊗ C^2 = C^4
                "predicted_rung4_max_rank": 32,
            },
            "sizes": args.sizes,
            "runs": runs,
        }
        args.json_out.write_text(json.dumps(artifact, indent=2))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
