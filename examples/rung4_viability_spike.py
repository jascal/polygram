"""Worked example — `add-rung4-encoding-mvp` §7 viability spike.

Mirrors `examples/rung3_viability_spike.py` for the Rung4 encoding:
runs all 28 pairs of the §4.4 8-feature panel through both
`Cancellation(encoding="mps")` and `Cancellation(encoding="rung4")`,
materializes the optimized dictionaries to disk, runs
`BehaviouralValidator` on each, and prints the four-criterion
(A, B, C, D) decision banner per the proposal's calibrated rule.

The Rung4 amp branch is a **product** of two single-qubit amps:

    |amp(θ_amp, ψ_aux, θ_amp_b, ψ_amp_b)⟩
        = |sq(θ_amp, ψ_aux)⟩ ⊗ |sq(θ_amp_b, ψ_amp_b)⟩

(vs Rung3's entangled Bell-pattern amp). The optimizer therefore
has **6 knobs** for the cancellation pair (`a.phi`, `b.phi`,
`b.theta_amp`, `b.psi_aux`, `b.theta_amp_b`, `b.psi_amp_b`) — two
more than Rung3 — and we expect strictly more headroom *if* the
extra product-amp dimensions matter for cancellation. The
calibrated `min_amp_overlap` constraint applies to the product
overlap `|⟨amp_a|amp_b⟩|² = |sq_a · sq_b|² · |sq_b · sq_b|²`.

Usage
-----

    python examples/rung4_viability_spike.py \\
        --sae-checkpoint scratch/real-sae/.../sae_weights.safetensors \\
        --output-dir examples/output/rung4_spike

Skip paths
----------

- SAE checkpoint missing → exit 0 with the canonical
  `hf download jbloom/GPT2-Small-SAEs-Reformatted` hint.
- `torch` / `transformers` missing → exit 0 after the analytic
  (criterion A) phase completes. The behavioural phase (B, C, D)
  requires the `[behavioural]` extra; the partial JSON is still
  written.

JSON schema (rung4_viability_spike.json)
----------------------------------------

```
{
  "selection": [int, ...],          # feature_ids
  "n_pairs": int,                   # 28
  "min_amp_overlap": float,         # non-degenerate-amp threshold (0 = off)
  "pairs": [
    {
      "i": int, "j": int,
      "mps_floor": float,
      "mps_pre_overlap": float,
      "mps_post_overlap": float,
      "mps_efficiency": float | null,
      "rung4_post_overlap": float,
      "rung4_theta_amp_optimum": float,    # feature B branch-A θ
      "rung4_psi_aux_optimum": float,      # feature B branch-A ψ
      "rung4_theta_amp_b_optimum": float,  # feature B branch-B θ (new vs Rung3)
      "rung4_psi_amp_b_optimum": float,    # feature B branch-B ψ (new vs Rung3)
      "rung4_residual_ratio": float,       # post_rung4 / mps_floor
      "behavioural": {
        "baseline_jaccard": float,
        "baseline_polygram_overlap": float,
        "baseline_gate_pass": bool,
        "rung4_jaccard": float,
        "rung4_polygram_overlap": float,
        "rung4_gate_pass": bool,
        "n_both_fire": int,
      }
    }, ...
  ],
  "criteria": {
    "A_floor_breaking": {"median_residual_ratio": float, "bucket": str},
    "B_gate_tpr": {"value": float, "bucket": str},
    "C_ranker_preservation": {"spearman": float, "bucket": str},
    "D_coverage": {"value": float, "bucket": str}
  },
  "decision_bucket": str   # "strong_pass" | "partial_pass" | "fail"
}
```
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from polygram import (
    Cancellation,
    Dictionary,
    MPSRung1,
    Rung4,
    from_sae_lens,
    load_sae_safetensors,
)


SELECTION_FEATURE_IDS: tuple[int, ...] = (
    12999, 19398, 4192, 23625, 8371, 2287, 68, 13737,
)

DEFAULT_SAE_PATH = Path(
    "./scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors"
)


def _build_dictionary(
    sae_path: Path, feature_ids: list[int], encoding, name: str
) -> Dictionary:
    records = load_sae_safetensors(sae_path, feature_ids=feature_ids)
    dictionary, _ = from_sae_lens(
        records, feature_ids, assign_gamma=True, name=name,
    )
    return replace(dictionary, encoding=encoding)


def _cancel_pair(
    dictionary: Dictionary, target_pair, *,
    encoding_str: str, min_amp_overlap: float = 0.0,
):
    # `grid_outer` is interpreted differently per encoding:
    #   - mps:   2D outer over (φ_a, φ_b) — (5, 5) = 25 cells.
    #   - rung4: 4D outer over (θ_amp_b, ψ_aux_b, θ_amp_b_b, ψ_amp_b_b) —
    #     (M, N) becomes M*N*M*N = (M*N)^2 cells. (3, 3) = 81 cells
    #     per pair (the docstring-suggested default for callers who
    #     find (5, 5) = 625 too slow).
    grid_outer = (3, 3) if encoding_str == "rung4" else (5, 5)
    canc = Cancellation(
        dictionary=dictionary,
        target_pair=target_pair,
        preserve_tiers=False,
        encoding=encoding_str,
        grid_outer=grid_outer,
        optimize={"method": "grid", "max_steps": 25},
        min_amp_overlap=min_amp_overlap if encoding_str == "rung4" else 0.0,
    )
    return canc.run()


def _apply_optimum(
    master: Dictionary, optimized_knobs: dict[str, float]
) -> Dictionary:
    out = master
    for path, value in optimized_knobs.items():
        out = out.with_knob(path, float(value))
    return out


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    a = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    b = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    a = a.astype(float)
    b = b.astype(float)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def _bucket_a(residual: float) -> str:
    if residual <= 0.3:
        return "strong"
    if residual <= 0.7:
        return "partial"
    return "fail"


def _bucket_b(tpr: float) -> str:
    if tpr >= 0.80:
        return "strong"
    if tpr >= 0.66:
        return "partial"
    return "fail"


def _bucket_c(spearman: float) -> str:
    if spearman >= 0.65:
        return "strong"
    if spearman >= 0.50:
        return "partial"
    return "fail"


def _bucket_d(coverage: float) -> str:
    if coverage >= 0.90:
        return "strong"
    if coverage >= 0.80:
        return "partial"
    return "fail"


def _decision(buckets: dict[str, str]) -> str:
    if buckets["D"] == "fail":
        return "fail"
    if buckets["A"] == "fail":
        return "fail"
    if buckets["A"] == "strong" and (
        buckets["B"] == "strong" or buckets["C"] == "strong"
    ):
        return "strong_pass"
    return "partial_pass"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-ids", type=int, nargs="+",
        default=list(SELECTION_FEATURE_IDS),
        help="feature IDs to probe (default: §4.4 8-feature panel)",
    )
    parser.add_argument(
        "--sae-checkpoint", type=Path, default=DEFAULT_SAE_PATH,
        help=f"path to the SAE checkpoint (default: {DEFAULT_SAE_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("examples/output/rung4_viability_spike"),
        help="where artifacts land (baseline/, rung4/, spike json)",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=12,
        help="how many prompts to forward (default: 12)",
    )
    parser.add_argument(
        "--min-amp-overlap", type=float, default=0.0,
        help=(
            "non-degenerate-amp constraint for the rung4 joint optimizer. "
            "When > 0, outer-grid cells and scipy candidates whose product "
            "amp factor |⟨amp_a|amp_b⟩|² falls below this threshold are "
            "marked infeasible. This blocks the trivial amp-zeroing "
            "solution and forces the optimizer to find an amp configuration "
            "that combines non-trivially with the MPS-side phase knobs. "
            "The Rung3 spike showed that a constrained re-run "
            "(ε = 0.5) was load-bearing for the verdict — same expected "
            "here. Suggested ε for the spike: 0.5."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress per-pair progress lines",
    )
    args = parser.parse_args(argv)

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        print(
            f"rung4_viability_spike: SAE checkpoint not found at "
            f"{sae_path}. Download with `hf download "
            f"jbloom/GPT2-Small-SAEs-Reformatted "
            f"--include='blocks.10.hook_resid_pre/"
            f"sae_weights.safetensors' --local-dir ./scratch/real-sae`. "
            f"Skipping.",
            file=sys.stderr,
        )
        return 0

    feature_ids = list(args.feature_ids)
    if len(feature_ids) < 2:
        print(
            "rung4_viability_spike: need at least 2 feature IDs",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = out_dir / "baseline"
    rung4_dir = out_dir / "rung4"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    rung4_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(
            f"rung4_viability_spike: building MPS + Rung4 dictionaries "
            f"on {len(feature_ids)} features ..."
        )
    mps_dict = _build_dictionary(
        sae_path, feature_ids, MPSRung1(), "Rung4SpikeMps"
    )
    rung4_dict = _build_dictionary(
        sae_path, feature_ids, Rung4(), "Rung4SpikeRung4"
    )

    pairs = list(itertools.combinations(range(len(feature_ids)), 2))
    if not args.quiet:
        print(
            f"rung4_viability_spike: cancelling {len(pairs)} pairs "
            f"through both encodings ..."
        )

    per_pair_records: list[dict] = []
    mps_master = mps_dict
    rung4_master = rung4_dict

    for pair_idx, (i_idx, j_idx) in enumerate(pairs):
        i_id = feature_ids[i_idx]
        j_id = feature_ids[j_idx]
        a_name = mps_dict.features[i_idx].name
        b_name = mps_dict.features[j_idx].name

        mps_result = _cancel_pair(
            mps_dict, (a_name, b_name), encoding_str="mps"
        )
        rung4_result = _cancel_pair(
            rung4_dict, (a_name, b_name), encoding_str="rung4",
            min_amp_overlap=float(args.min_amp_overlap),
        )

        floor = float(mps_result.structural_floor)
        residual = (
            float(rung4_result.after_overlap) / floor
            if floor > 1e-12 else float("nan")
        )

        # Extract Rung4-specific knobs from the optimized_knobs dict.
        # `theta_amp_optimum` and `psi_aux_optimum` are populated by
        # the joint optimizer (Rung3 + Rung4 branch-A knobs). The new
        # Rung4-only branch-B knobs live in `optimized_knobs` under
        # `<b_name>.theta_amp_b` / `<b_name>.psi_amp_b`.
        theta_amp_b_optimum = float(
            rung4_result.optimized_knobs.get(f"{b_name}.theta_amp_b", float("nan"))
        )
        psi_amp_b_optimum = float(
            rung4_result.optimized_knobs.get(f"{b_name}.psi_amp_b", float("nan"))
        )

        per_pair_records.append({
            "i": i_id, "j": j_id,
            "mps_floor": floor,
            "mps_pre_overlap": float(mps_result.before_overlap),
            "mps_post_overlap": float(mps_result.after_overlap),
            "mps_efficiency": (
                None if mps_result.cancellation_efficiency is None
                else float(mps_result.cancellation_efficiency)
            ),
            "rung4_post_overlap": float(rung4_result.after_overlap),
            "rung4_theta_amp_optimum": float(rung4_result.theta_amp_optimum),
            "rung4_psi_aux_optimum": float(rung4_result.psi_aux_optimum),
            "rung4_theta_amp_b_optimum": theta_amp_b_optimum,
            "rung4_psi_amp_b_optimum": psi_amp_b_optimum,
            "rung4_residual_ratio": residual,
        })

        mps_master = _apply_optimum(mps_master, mps_result.optimized_knobs)
        rung4_master = _apply_optimum(
            rung4_master, rung4_result.optimized_knobs
        )

        if not args.quiet:
            print(
                f"  [{pair_idx + 1:>2}/{len(pairs)}] "
                f"feat_{i_id:>5d}×feat_{j_id:>5d}  "
                f"floor={floor:.4f}  "
                f"mps_post={mps_result.after_overlap:.4f}  "
                f"r4_post={rung4_result.after_overlap:.4f}  "
                f"residual={residual:.3f}"
            )

    # Materialize both master dictionaries.
    from polygram.emit import write_qorca

    mps_master = replace(mps_master, name="Rung4SpikeMpsOptimized")
    rung4_master = replace(rung4_master, name="Rung4SpikeRung4Optimized")
    write_qorca(mps_master, baseline_dir / f"{mps_master.name}.q.orca.md")
    # The Rung4 master dict snapshot — q-orca's 5-qubit emit path is
    # the same MPS-substrate as Rung3 (the amp branch lives in the
    # analytic `Dictionary.gram()` path, not in q-orca), so we
    # snapshot the optimized knobs as JSON rather than emit a q-orca
    # machine that wouldn't reflect the amp factors.
    rung4_snapshot = rung4_dir / "rung4_master_knobs.json"
    rung4_snapshot.write_text(json.dumps({
        "name": rung4_master.name,
        "knobs": [
            {
                "name": f.name, "cluster": f.cluster,
                "alpha": f.alpha, "beta": f.beta,
                "gamma": f.gamma, "phi": f.phi,
                "theta_amp": f.theta_amp, "psi_aux": f.psi_aux,
                "theta_amp_b": f.theta_amp_b, "psi_amp_b": f.psi_amp_b,
            }
            for f in rung4_master.features
        ],
    }, indent=2))

    # Try to import BehaviouralValidator. If torch+transformers aren't
    # installed, we skip the behavioural phase and write a partial JSON
    # (A criterion only).
    try:
        from polygram import BehaviouralValidator
    except ImportError as exc:
        print(
            f"rung4_viability_spike: BehaviouralValidator import failed "
            f"({exc}); skipping behavioural phase. Install the "
            f"`[behavioural]` extra to run B/C/D criteria.",
            file=sys.stderr,
        )
        return _finish_partial(
            per_pair_records, feature_ids, pairs, out_dir,
            float(args.min_amp_overlap),
        )

    # Run BehaviouralValidator on each master dict.
    if not args.quiet:
        print("rung4_viability_spike: running BehaviouralValidator on baseline ...")

    from examples.behavioural_gram_scaleup import PROMPTS
    n_prompts = max(1, min(args.n_prompts, len(PROMPTS)))
    prompts = list(PROMPTS[:n_prompts])

    def _run_validator(d: Dictionary, label: str):
        validator = BehaviouralValidator(
            dictionary=d, sae_checkpoint=sae_path,
            feature_ids=feature_ids, prompts=prompts, layer=10,
        )
        try:
            return validator.run()
        except ImportError as exc:
            print(
                f"rung4_viability_spike: {label}: {exc}",
                file=sys.stderr,
            )
            return None

    baseline_report = _run_validator(mps_master, "baseline")
    if baseline_report is None:
        return _finish_partial(
            per_pair_records, feature_ids, pairs, out_dir,
            float(args.min_amp_overlap),
        )
    baseline_report.to_json(baseline_dir / "validation_report.json")

    if not args.quiet:
        print("rung4_viability_spike: running BehaviouralValidator on rung4 ...")
    rung4_report = _run_validator(rung4_master, "rung4")
    if rung4_report is None:
        return _finish_partial(
            per_pair_records, feature_ids, pairs, out_dir,
            float(args.min_amp_overlap),
        )
    rung4_report.to_json(rung4_dir / "validation_report.json")

    # Splice behavioural rows into per-pair records.
    baseline_pairs = {(p.i, p.j): p for p in baseline_report.pairs}
    rung4_pairs = {(p.i, p.j): p for p in rung4_report.pairs}
    for rec in per_pair_records:
        key = (rec["i"], rec["j"])
        bp = baseline_pairs.get(key)
        rp = rung4_pairs.get(key)
        rec["behavioural"] = {
            "baseline_jaccard": (
                float(bp.jaccard) if bp else float("nan")
            ),
            "baseline_polygram_overlap": (
                float(bp.polygram_overlap) if bp else float("nan")
            ),
            "baseline_gate_pass": bool(bp.gate_pass) if bp else False,
            "rung4_jaccard": (
                float(rp.jaccard) if rp else float("nan")
            ),
            "rung4_polygram_overlap": (
                float(rp.polygram_overlap) if rp else float("nan")
            ),
            "rung4_gate_pass": bool(rp.gate_pass) if rp else False,
            "n_both_fire": int(rp.n_both_fire) if rp else 0,
        }

    # Compute the four criteria over the rung4 cohort.
    residuals = [
        r["rung4_residual_ratio"] for r in per_pair_records
        if not math.isnan(r["rung4_residual_ratio"])
    ]
    median_residual = float(np.median(residuals)) if residuals else float("nan")

    # B: TPR among Polygram ≥ 0.7 pairs (rung4 cohort).
    high_overlap = [
        r for r in per_pair_records
        if r["behavioural"]["rung4_polygram_overlap"] >= 0.7
    ]
    if high_overlap:
        tpr = sum(
            1 for r in high_overlap if r["behavioural"]["rung4_gate_pass"]
        ) / len(high_overlap)
    else:
        tpr = float("nan")

    # C: Spearman(rung4_polygram_overlap, rung4_jaccard) over all 28 pairs.
    poly_xs, jacc_ys = [], []
    for r in per_pair_records:
        po = r["behavioural"]["rung4_polygram_overlap"]
        ja = r["behavioural"]["rung4_jaccard"]
        if not (math.isnan(po) or math.isnan(ja)):
            poly_xs.append(po)
            jacc_ys.append(ja)
    spearman = _spearman(poly_xs, jacc_ys)

    # D: coverage = fraction of jaccard ≥ 0.30 pairs caught by gate.
    redundant = [
        r for r in per_pair_records
        if r["behavioural"]["rung4_jaccard"] >= 0.30
    ]
    if redundant:
        coverage = sum(
            1 for r in redundant if r["behavioural"]["rung4_gate_pass"]
        ) / len(redundant)
    else:
        coverage = float("nan")

    buckets = {
        "A": _bucket_a(median_residual)
            if not math.isnan(median_residual) else "fail",
        "B": _bucket_b(tpr) if not math.isnan(tpr) else "fail",
        "C": _bucket_c(spearman) if not math.isnan(spearman) else "fail",
        "D": _bucket_d(coverage) if not math.isnan(coverage) else "fail",
    }
    decision_bucket = _decision(buckets)

    payload = {
        "selection": feature_ids,
        "n_pairs": len(pairs),
        "min_amp_overlap": float(args.min_amp_overlap),
        "pairs": per_pair_records,
        "criteria": {
            "A_floor_breaking": {
                "median_residual_ratio": median_residual,
                "bucket": buckets["A"],
            },
            "B_gate_tpr": {"value": tpr, "bucket": buckets["B"]},
            "C_ranker_preservation": {
                "spearman": spearman, "bucket": buckets["C"],
            },
            "D_coverage": {"value": coverage, "bucket": buckets["D"]},
        },
        "decision_bucket": decision_bucket,
    }
    json_path = out_dir / "rung4_viability_spike.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    print()
    print("=" * 78)
    print(
        f"RUNG4-VIABILITY-SPIKE @ blocks.{baseline_report.layer} — "
        f"{len(feature_ids)} features, {len(pairs)} pairs, "
        f"{baseline_report.n_tokens} tokens"
    )
    print("=" * 78)
    print(
        f"A. Floor-breaking — median residual    "
        f"{median_residual:.4f}  [{buckets['A']}]"
    )
    print(
        f"B. Gate true-positive rate              "
        f"{tpr:.4f}  [{buckets['B']}]"
    )
    print(
        f"C. Ranker preservation (Spearman)       "
        f"{spearman:+.4f}  [{buckets['C']}]"
    )
    print(
        f"D. Coverage                              "
        f"{coverage:.4f}  [{buckets['D']}]"
    )
    print(f"Decision bucket:                       {decision_bucket}")
    print()
    print(f"JSON written → {json_path}")
    return 0


def _finish_partial(
    per_pair_records: list[dict],
    feature_ids: list[int],
    pairs: list[tuple[int, int]],
    out_dir: Path,
    min_amp_overlap: float,
) -> int:
    """Write a partial JSON when the behavioural phase is unavailable.
    Includes criterion A only (B/C/D become NaN/`fail`)."""
    residuals = [
        r["rung4_residual_ratio"] for r in per_pair_records
        if not math.isnan(r["rung4_residual_ratio"])
    ]
    median_residual = float(np.median(residuals)) if residuals else float("nan")
    bucket_a = _bucket_a(median_residual) if not math.isnan(median_residual) else "fail"

    payload = {
        "selection": feature_ids,
        "n_pairs": len(pairs),
        "min_amp_overlap": min_amp_overlap,
        "pairs": per_pair_records,
        "criteria": {
            "A_floor_breaking": {
                "median_residual_ratio": median_residual,
                "bucket": bucket_a,
            },
            "B_gate_tpr": {
                "value": float("nan"),
                "bucket": "skipped_behavioural_unavailable",
            },
            "C_ranker_preservation": {
                "spearman": float("nan"),
                "bucket": "skipped_behavioural_unavailable",
            },
            "D_coverage": {
                "value": float("nan"),
                "bucket": "skipped_behavioural_unavailable",
            },
        },
        "decision_bucket": "partial_analytic_only",
    }
    json_path = out_dir / "rung4_viability_spike.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    print()
    print("=" * 78)
    print(
        f"RUNG4-VIABILITY-SPIKE (ANALYTIC ONLY) — "
        f"{len(feature_ids)} features, {len(pairs)} pairs"
    )
    print("=" * 78)
    print(
        f"A. Floor-breaking — median residual    "
        f"{median_residual:.4f}  [{bucket_a}]"
    )
    print(
        "B/C/D — skipped (BehaviouralValidator unavailable; install "
        "polygram[behavioural])"
    )
    print()
    print(f"Partial JSON written → {json_path}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
