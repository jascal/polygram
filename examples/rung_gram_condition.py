"""Rung-viability v2 — **Axis 2** (gram condition).

Per `docs/research/rung-viability-methodology.md` Axis 2: build a
`Dictionary` at exactly the encoding's `max_features` using a real-SAE
feature subset chosen for high mutual redundancy, then measure:

- λ_min(|gram|²) — smallest eigenvalue of the squared-modulus gram.
  Lower = more nearly-singular = features less distinguishable.
- Off-diagonal Frobenius mass / max_features — average off-diagonal
  modulus. Higher = features more "tangled" in the encoding's state
  space.

The hypothesis: higher rungs have larger state spaces (Rung4: 32-dim
per feature; Rung3: 16-dim; MPSRung1: 8-dim), so at K=max_features
the gram should be better conditioned (higher λ_min, lower
off-diagonal mass) compared to wedging the same K features into a
smaller-cap encoding via `clustered=True`.

The probe is torch-free — `Dictionary.gram()` is purely analytic.

Usage
-----

    python examples/rung_gram_condition.py \\
        --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \\
        --encoding rung4 \\
        --output docs/research/data/rung_gram_condition_rung4.json

When `--sae` is omitted, falls back to the bundled toy fixture.

JSON schema
-----------

```
{
  "encoding": "mps" | "rung3" | "rung4",
  "max_features": int,                # encoding's cap
  "k_selected": int,                  # = max_features (or smaller on toy)
  "feature_ids": [int, ...],
  "fixture": str,
  "selection_method": "top_cosine_cluster",
  "gram_squared_mod_max_off_diagonal": float,
  "gram_squared_mod_mean_off_diagonal": float,
  "gram_squared_mod_frobenius_off_diagonal": float,
  "gram_squared_mod_lambda_min": float,
  "gram_squared_mod_lambda_max": float,
  "gram_squared_mod_condition_number": float
}
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from polygram import (
    HEA_Rung2,
    MPSRung1,
    Rung3,
    Rung4,
    from_sae_lens,
    load_sae_safetensors,
)


FIXTURE_TOY = Path(__file__).parent.parent / "tests" / "fixtures" / "toy_sae.json"

ENCODING_REGISTRY: dict[str, tuple[callable, int]] = {
    "mps":   (lambda: MPSRung1(),                     MPSRung1.max_features),
    "rung3": (lambda: Rung3(),                        Rung3.max_features),
    "rung4": (lambda: Rung4(),                        Rung4.max_features),
    "hea":   (lambda: HEA_Rung2(depth=1, n_qubits=5), 32),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--sae",
        type=Path,
        default=None,
        help=(
            "Path to an SAE safetensors file with W_dec (n_features, "
            "d_model). When omitted, falls back to the bundled toy "
            "fixture (16 features × 8 d_model)."
        ),
    )
    p.add_argument(
        "--encoding",
        choices=sorted(ENCODING_REGISTRY.keys()),
        default="rung4",
        help="Encoding to probe (default: rung4).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Subset size. Defaults to the encoding's `max_features`. "
            "Override to probe sub-saturation conditioning."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for any tie-breaking; deterministic selection otherwise.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the result artifact.",
    )
    p.add_argument(
        "--assign-amp-knobs",
        action="store_true",
        help=(
            "Populate the encoding's amp-branch knobs from decoder "
            "geometry (PC4-PC7). When omitted, the loader uses encoding "
            "defaults — which for Rung3/Rung4 produce MPSRung1-"
            "equivalent gram, per the v2 results note."
        ),
    )
    p.add_argument(
        "--assign-phase-knobs",
        action="store_true",
        help=(
            "Populate the encoding's MPS-substrate α and φ from decoder "
            "geometry (PC2/PC3, per add-phase-knob-assignment). Without "
            "this, MPSRung1 collapses on activation-uncorrelated features "
            "even at K=8 (per the 2026-05-15 GPT-2 bug report root cause)."
        ),
    )
    return p.parse_args()


def load_decoder_vectors(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.sae is None:
        from polygram import load_toy_sae

        records = load_toy_sae(FIXTURE_TOY)
        ids = sorted(records.keys())
        vectors = np.stack([records[i].projection for i in ids], axis=0)
        return vectors.astype(np.float32), "toy_sae.json"
    try:
        from safetensors import safe_open
    except ImportError:
        raise SystemExit(
            "safetensors is required to load --sae checkpoints; "
            "install with `pip install polygram[sae]`"
        )
    with safe_open(args.sae, framework="numpy") as f:
        w_dec = f.get_tensor("W_dec")
    return w_dec.astype(np.float32), str(args.sae)


def select_high_redundancy_subset(
    vectors: np.ndarray, k: int, *, seed: int = 0
) -> list[int]:
    """Pick K feature IDs forming the highest-cosine subset of the SAE.

    Strategy: greedy density expansion. Start from the pair with the
    highest cosine, then iteratively add the feature whose mean cosine
    to the current subset is highest, until reaching size K.
    Deterministic given the same `vectors` and `k`. The `seed` argument
    is unused at present (no tie-breaking randomness) but reserved so
    the script's CLI surface matches its peers.
    """
    n = vectors.shape[0]
    if k > n:
        raise ValueError(
            f"k={k} exceeds the SAE's feature count ({n}). Use a smaller "
            f"--k or a larger SAE."
        )
    # Cosine matrix; small enough at K ≤ 32 to materialize on the
    # full row block.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = vectors / norms
    cosines = unit @ unit.T
    np.fill_diagonal(cosines, -np.inf)  # exclude self
    # Initial pair: argmax of the upper triangle.
    flat_idx = int(np.argmax(cosines))
    i, j = divmod(flat_idx, n)
    subset = [int(i), int(j)]
    while len(subset) < k:
        # Mean cosine of each candidate to the current subset.
        candidate_scores = cosines[:, subset].mean(axis=1)
        for idx in subset:
            candidate_scores[idx] = -np.inf  # exclude already-picked
        next_idx = int(np.argmax(candidate_scores))
        subset.append(next_idx)
    return sorted(subset)


def compute_gram_condition_metrics(
    sae_path: Path | None,
    feature_ids: list[int],
    encoding,
    *,
    name: str,
    assign_amp_knobs: bool = False,
    assign_phase_knobs: bool = False,
) -> dict:
    """Build a `Dictionary` on `feature_ids` with `encoding`, compute
    `np.abs(gram)**2`, and report the off-diagonal mass + spectral
    extrema (load-bearing metrics for Axis 2).

    `assign_amp_knobs` controls higher-rung amp-branch knob assignment
    (PC4-PC7 after add-phase-knob-assignment). `assign_phase_knobs`
    controls MPS-substrate α/φ assignment (PC2/PC3). Default False on
    both matches the v2 results note's pre-fix measurements.
    """
    if sae_path is not None:
        records = load_sae_safetensors(sae_path, feature_ids=feature_ids)
    else:
        from polygram import load_toy_sae

        all_records = load_toy_sae(FIXTURE_TOY)
        records = {fid: all_records[fid] for fid in feature_ids}
    dictionary, _ = from_sae_lens(
        records, feature_ids, assign_gamma=True, name=name,
        encoding=encoding,
        assign_amp_knobs=assign_amp_knobs,
        assign_phase_knobs=assign_phase_knobs,
    )
    gram = dictionary.gram()
    gram_sq = np.abs(gram) ** 2
    k = gram_sq.shape[0]
    off_diagonal_mask = ~np.eye(k, dtype=bool)
    off_diagonal_values = gram_sq[off_diagonal_mask]
    eigvals = np.linalg.eigvalsh(gram_sq)
    lambda_min = float(eigvals.min())
    lambda_max = float(eigvals.max())
    condition_number = (
        lambda_max / lambda_min if lambda_min > 0 else float("inf")
    )
    return {
        "gram_squared_mod_max_off_diagonal": float(off_diagonal_values.max()),
        "gram_squared_mod_mean_off_diagonal": float(off_diagonal_values.mean()),
        "gram_squared_mod_frobenius_off_diagonal": float(
            np.sqrt((off_diagonal_values ** 2).sum() / k)
        ),
        "gram_squared_mod_lambda_min": lambda_min,
        "gram_squared_mod_lambda_max": lambda_max,
        "gram_squared_mod_condition_number": condition_number,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else _parse(argv)
    encoding_factory, default_k = ENCODING_REGISTRY[args.encoding]
    encoding = encoding_factory()
    k = args.k if args.k is not None else default_k

    vectors, fixture_label = load_decoder_vectors(args)
    n_total = vectors.shape[0]
    if k > n_total:
        # Toy fallback can't reach Rung4's K=32; cap and proceed.
        print(
            f"rung_gram_condition: requested k={k} exceeds fixture's "
            f"{n_total} features; clamping to {n_total}",
            file=sys.stderr,
        )
        k = n_total

    feature_ids = select_high_redundancy_subset(vectors, k, seed=args.seed)

    print(f"fixture: {fixture_label}")
    print(f"  total features: {n_total}, d_model: {vectors.shape[1]}")
    print(f"  encoding={args.encoding} (max_features={default_k}), k={k}")
    print(f"  selected feature_ids: {feature_ids[:8]}{'...' if k > 8 else ''}")
    print()

    sae_path = args.sae if args.sae is not None else None
    metrics = compute_gram_condition_metrics(
        sae_path, feature_ids, encoding,
        name=f"RungGramCondition_{args.encoding}_k{k}",
        assign_amp_knobs=bool(args.assign_amp_knobs),
        assign_phase_knobs=bool(args.assign_phase_knobs),
    )

    print("== Gram condition metrics ==")
    print(
        f"  max off-diagonal |gram|²:   "
        f"{metrics['gram_squared_mod_max_off_diagonal']:.4f}"
    )
    print(
        f"  mean off-diagonal |gram|²:  "
        f"{metrics['gram_squared_mod_mean_off_diagonal']:.4f}"
    )
    print(
        f"  Frobenius off-diag / k:     "
        f"{metrics['gram_squared_mod_frobenius_off_diagonal']:.4f}"
    )
    print(
        f"  λ_min(|gram|²):             "
        f"{metrics['gram_squared_mod_lambda_min']:.4e}"
    )
    print(
        f"  λ_max(|gram|²):             "
        f"{metrics['gram_squared_mod_lambda_max']:.4e}"
    )
    print(
        f"  condition number:           "
        f"{metrics['gram_squared_mod_condition_number']:.4e}"
    )

    payload = {
        "encoding": args.encoding,
        "max_features": default_k,
        "k_selected": k,
        "feature_ids": feature_ids,
        "fixture": fixture_label,
        "selection_method": "top_cosine_cluster",
        "assign_amp_knobs": bool(args.assign_amp_knobs),
        "assign_phase_knobs": bool(args.assign_phase_knobs),
        **metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=_json_default))
        print(f"\nwrote {args.output}")
    return 0


def _parse(argv: list[str]) -> argparse.Namespace:
    # Variant of parse_args() that takes explicit argv (for tests).
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sae", type=Path, default=None)
    p.add_argument("--encoding", choices=sorted(ENCODING_REGISTRY.keys()), default="rung4")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--assign-amp-knobs", action="store_true")
    p.add_argument("--assign-phase-knobs", action="store_true")
    return p.parse_args(argv)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
