"""Empirical rank probe for Rung3 dictionaries.

Resolves the open question raised during the "lift the per-encoding
feature cap" design discussion: does Rung3 actually unlock 32
linearly-independent features (the naive ``2^5 = 32`` for "MPS-on-q0-2
+ amp-on-q3-4"), or is the amp-branch parameterization weaker than
that?

**Math precheck:** the amp branch carries states

    |amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩

which lives in the 2-dim subspace ``span{|00⟩, |11⟩}`` of the 2-qubit
Hilbert space C^4 — the parameterization can never reach ``|01⟩`` or
``|10⟩``. So the effective amp dimension is **2, not 4**. Tensor with
the MPSRung1 8-dim subspace gives ``C^8 ⊗ C^2 = C^16``, not C^32.

This probe verifies that empirically by computing Gram rank as N
grows from 4 to 32 with diverse, randomly-sampled parameters.

Companion writeup: ``docs/research/rung3-rank-bound.md``.

Usage:

    # Print the table to stdout:
    python examples/rung3_rank_probe.py

    # Also dump JSON for the research note's data artifact:
    python examples/rung3_rank_probe.py --json-out docs/research/data/rung3_rank_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from polygram import Dictionary, Feature
from polygram.encoding import MPSRung1, Rung3


def make_features(n: int, seed: int = 0, encoding_kind: str = "rung3") -> list[Feature]:
    """N features with random params drawn from the full parameter range.

    For Rung3, uniformly samples (α, β, γ, φ) over [0, 2π) and (θ_amp,
    ψ_aux) over [0, π/2] × [0, 2π). For MPSRung1, only the first four.
    """
    rng = np.random.default_rng(seed)
    feats = []
    for i in range(n):
        kwargs = dict(
            name=f"f{i}",
            cluster="all",
            alpha=float(rng.uniform(0.0, 2 * np.pi)),
            beta=float(rng.uniform(0.0, 2 * np.pi)),
            gamma=float(rng.uniform(0.0, 2 * np.pi)),
            phi=float(rng.uniform(0.0, 2 * np.pi)),
        )
        if encoding_kind == "rung3":
            kwargs["theta_amp"] = float(rng.uniform(0.0, np.pi / 2))
            kwargs["psi_aux"] = float(rng.uniform(0.0, 2 * np.pi))
        feats.append(Feature(**kwargs))
    return feats


def make_dict(n: int, encoding, seed: int = 0) -> Dictionary:
    kind = "rung3" if isinstance(encoding, Rung3) else "mps"
    feats = make_features(n, seed=seed, encoding_kind=kind)
    return Dictionary(
        name=f"probe_n{n}",
        features=feats,
        hierarchy={"all": [f.name for f in feats]},
        encoding=encoding,
    )


def rank_at_tol(s: np.ndarray, tol: float) -> int:
    """Number of singular values exceeding `tol * max(s)` (relative tol)."""
    if s.size == 0:
        return 0
    return int(np.sum(s > tol * float(np.max(s))))


def measure(encoding, sizes: list[int], seed: int = 0) -> list[dict]:
    rows = []
    for n in sizes:
        d = make_dict(n, encoding, seed=seed)
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


def spectrum_relative(rows: list[dict], n: int) -> list[float]:
    """Singular values at the requested N, normalized to σ_max."""
    for r in rows:
        if r["n"] == n:
            s = np.asarray(r["singular_values"], dtype=float)
            return (s / s.max()).tolist() if s.max() > 0 else s.tolist()
    return []


def print_table(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'N':>3}  {'rank@1e-12':>10}  {'rank@1e-9':>9}  {'rank@1e-6':>9}  "
          f"{'σ_max':>10}  {'σ_min>0':>11}  {'σ_min':>10}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['n']:>3}  {r['rank_1e_12']:>10}  "
            f"{r['rank_1e_9']:>9}  {r['rank_1e_6']:>9}  "
            f"{r['sigma_max']:>10.4e}  "
            f"{r['sigma_min_nonzero']:>11.4e}  "
            f"{r['sigma_min']:>10.4e}"
        )


def print_spectrum(label: str, rel: list[float]) -> None:
    print(f"\n=== {label} (singular values relative to σ_max) ===")
    for i, v in enumerate(rel):
        marker = ""
        if v < 1e-12:
            marker = "  ← below numerical floor"
        elif v < 1e-9:
            marker = "  ← below 1e-9"
        elif v < 1e-6:
            marker = "  ← below 1e-6"
        print(f"  σ[{i:>2}] = {v:.6e}{marker}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to dump measurement JSON (for the research-note data artifact).",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24, 28, 32],
        help="N values to probe.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 42],
        help="Seeds for the Rung3 sanity-check sweep (the first seed is also used for MPSRung1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    runs: dict[str, dict] = {}

    mps_rows = measure(MPSRung1(), args.sizes, seed=args.seeds[0])
    print_table(f"MPSRung1 (expected max rank 8) — seed={args.seeds[0]}", mps_rows)
    runs["mps_rung1"] = {"seed": args.seeds[0], "rows": mps_rows}

    for seed in args.seeds:
        rows = measure(Rung3(), args.sizes, seed=seed)
        print_table(f"Rung3 (claimed 32; suspect 16) — seed={seed}", rows)
        runs[f"rung3_seed{seed}"] = {"seed": seed, "rows": rows}

    rung3_rel_at_32 = spectrum_relative(runs[f"rung3_seed{args.seeds[0]}"]["rows"], n=32)
    mps_rel_at_16 = spectrum_relative(runs["mps_rung1"]["rows"], n=16)
    print_spectrum(f"Rung3 spectrum @ N=32 (seed {args.seeds[0]})", rung3_rel_at_32)
    print_spectrum(f"MPSRung1 spectrum @ N=16 (seed {args.seeds[0]})", mps_rel_at_16)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema": "rung3_rank_probe.v1",
            "math_precheck": {
                "mps_subspace_dim": 8,
                "amp_subspace_dim": 2,
                "predicted_rung3_max_rank": 16,
            },
            "sizes": args.sizes,
            "runs": runs,
            "diagnostic_spectra": {
                "rung3_n32_relative": rung3_rel_at_32,
                "mps_rung1_n16_relative": mps_rel_at_16,
            },
        }
        args.json_out.write_text(json.dumps(artifact, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
