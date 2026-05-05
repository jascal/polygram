"""Worked example — `add-rung3-encoding-mvp` §4.5 viability spike.

Runs all 28 pairs of the §4.4 8-feature panel through both
`Cancellation(encoding="mps")` and `Cancellation(encoding="rung3")`,
materializes the optimized dictionaries to disk, runs
`BehaviouralValidator` on each, and prints the four-criterion
(A, B, C, D) decision banner per the proposal's calibrated rule.

Usage
-----

    python examples/rung3_viability_spike.py \\
        --sae-checkpoint scratch/real-sae/.../sae_weights.safetensors \\
        --output-dir examples/output/rung3_spike

Skip paths
----------

- SAE checkpoint missing → exit 0 with the canonical
  `hf download jbloom/GPT2-Small-SAEs-Reformatted` hint.
- `torch` / `transformers` missing → exit 0 with the
  `pip install polygram[behavioural]` hint.

JSON schema (rung3_viability_spike.json)
----------------------------------------

```
{
  "selection": [int, ...],          # feature_ids
  "n_pairs": int,                   # 28
  "pairs": [
    {
      "i": int, "j": int,
      "mps_floor": float,
      "mps_pre_overlap": float,
      "mps_post_overlap": float,
      "mps_efficiency": float | null,
      "rung3_post_overlap": float,
      "rung3_theta_amp_optimum": float,
      "rung3_psi_aux_optimum": float,
      "rung3_residual_ratio": float,    # post_rung3 / mps_floor
      "behavioural": {
        "baseline_jaccard": float,
        "baseline_polygram_overlap": float,
        "baseline_gate_pass": bool,
        "rung3_jaccard": float,
        "rung3_polygram_overlap": float,
        "rung3_gate_pass": bool,
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
    BehaviouralValidator,
    Cancellation,
    Dictionary,
    MPSRung1,
    Rung3,
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
    dictionary: Dictionary, target_pair, *, encoding_str: str
) -> dict:
    canc = Cancellation(
        dictionary=dictionary,
        target_pair=target_pair,
        preserve_tiers=False,
        encoding=encoding_str,
        grid_outer=(5, 5) if encoding_str == "rung3" else (5, 5),
        optimize={"method": "grid", "max_steps": 25},
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
    if tpr >= 0.70:
        return "partial"
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
        default=Path("examples/output/rung3_viability_spike"),
        help="where artifacts land (baseline/, rung3/, spike json)",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=12,
        help="how many prompts to forward (default: 12)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress per-pair progress lines",
    )
    args = parser.parse_args(argv)

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        print(
            f"rung3_viability_spike: SAE checkpoint not found at "
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
            "rung3_viability_spike: need at least 2 feature IDs",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = out_dir / "baseline"
    rung3_dir = out_dir / "rung3"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    rung3_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(
            f"rung3_viability_spike: building MPS + Rung3 dictionaries "
            f"on {len(feature_ids)} features ..."
        )
    mps_dict = _build_dictionary(
        sae_path, feature_ids, MPSRung1(), "Rung3SpikeMps"
    )
    rung3_dict = _build_dictionary(
        sae_path, feature_ids, Rung3(), "Rung3SpikeRung3"
    )

    pairs = list(itertools.combinations(range(len(feature_ids)), 2))
    if not args.quiet:
        print(
            f"rung3_viability_spike: cancelling {len(pairs)} pairs "
            f"through both encodings ..."
        )

    per_pair_records: list[dict] = []
    mps_master = mps_dict
    rung3_master = rung3_dict

    for pair_idx, (i_idx, j_idx) in enumerate(pairs):
        i_id = feature_ids[i_idx]
        j_id = feature_ids[j_idx]
        a_name = mps_dict.features[i_idx].name
        b_name = mps_dict.features[j_idx].name

        mps_result = _cancel_pair(
            mps_dict, (a_name, b_name), encoding_str="mps"
        )
        rung3_result = _cancel_pair(
            rung3_dict, (a_name, b_name), encoding_str="rung3"
        )

        floor = float(mps_result.structural_floor)
        residual = (
            float(rung3_result.after_overlap) / floor
            if floor > 1e-12 else float("nan")
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
            "rung3_post_overlap": float(rung3_result.after_overlap),
            "rung3_theta_amp_optimum": float(rung3_result.theta_amp_optimum),
            "rung3_psi_aux_optimum": float(rung3_result.psi_aux_optimum),
            "rung3_residual_ratio": residual,
        })

        mps_master = _apply_optimum(mps_master, mps_result.optimized_knobs)
        rung3_master = _apply_optimum(
            rung3_master, rung3_result.optimized_knobs
        )

        if not args.quiet:
            print(
                f"  [{pair_idx + 1:>2}/{len(pairs)}] "
                f"feat_{i_id:>5d}×feat_{j_id:>5d}  "
                f"floor={floor:.4f}  "
                f"mps_post={mps_result.after_overlap:.4f}  "
                f"r3_post={rung3_result.after_overlap:.4f}  "
                f"residual={residual:.3f}"
            )

    # Materialize both master dictionaries.
    from polygram.emit import write_qorca

    mps_master = replace(mps_master, name="Rung3SpikeMpsOptimized")
    rung3_master = replace(rung3_master, name="Rung3SpikeRung3Optimized")
    write_qorca(mps_master, baseline_dir / f"{mps_master.name}.q.orca.md")
    # Q-orca emission for rung3 isn't wired in v0; skip materializing the
    # 5-qubit machine and let the validator consume the live Dictionary.
    # The rung3 master dict is still snapshotted as a JSON beta dump for
    # reproducibility of the (theta_amp, psi_aux) coordinates.
    rung3_snapshot = rung3_dir / "rung3_master_knobs.json"
    rung3_snapshot.write_text(json.dumps({
        "name": rung3_master.name,
        "knobs": [
            {
                "name": f.name, "cluster": f.cluster,
                "alpha": f.alpha, "beta": f.beta,
                "gamma": f.gamma, "phi": f.phi,
                "theta_amp": f.theta_amp, "psi_aux": f.psi_aux,
            }
            for f in rung3_master.features
        ],
    }, indent=2))

    # Run BehaviouralValidator on each master dict.
    if not args.quiet:
        print("rung3_viability_spike: running BehaviouralValidator on baseline ...")

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
                f"rung3_viability_spike: {label}: {exc}",
                file=sys.stderr,
            )
            return None

    baseline_report = _run_validator(mps_master, "baseline")
    if baseline_report is None:
        return 0
    baseline_report.to_json(baseline_dir / "validation_report.json")

    if not args.quiet:
        print("rung3_viability_spike: running BehaviouralValidator on rung3 ...")
    rung3_report = _run_validator(rung3_master, "rung3")
    if rung3_report is None:
        return 0
    rung3_report.to_json(rung3_dir / "validation_report.json")

    # Splice behavioural rows into per-pair records.
    baseline_pairs = {(p.i, p.j): p for p in baseline_report.pairs}
    rung3_pairs = {(p.i, p.j): p for p in rung3_report.pairs}
    for rec in per_pair_records:
        key = (rec["i"], rec["j"])
        bp = baseline_pairs.get(key)
        rp = rung3_pairs.get(key)
        rec["behavioural"] = {
            "baseline_jaccard": (
                float(bp.jaccard) if bp else float("nan")
            ),
            "baseline_polygram_overlap": (
                float(bp.polygram_overlap) if bp else float("nan")
            ),
            "baseline_gate_pass": bool(bp.gate_pass) if bp else False,
            "rung3_jaccard": (
                float(rp.jaccard) if rp else float("nan")
            ),
            "rung3_polygram_overlap": (
                float(rp.polygram_overlap) if rp else float("nan")
            ),
            "rung3_gate_pass": bool(rp.gate_pass) if rp else False,
            "n_both_fire": int(rp.n_both_fire) if rp else 0,
        }

    # Compute the four criteria over the rung3 cohort.
    residuals = [
        r["rung3_residual_ratio"] for r in per_pair_records
        if not math.isnan(r["rung3_residual_ratio"])
    ]
    median_residual = float(np.median(residuals)) if residuals else float("nan")

    # B: TPR among Polygram ≥ 0.7 pairs (rung3 cohort).
    high_overlap = [
        r for r in per_pair_records
        if r["behavioural"]["rung3_polygram_overlap"] >= 0.7
    ]
    if high_overlap:
        tpr = sum(
            1 for r in high_overlap if r["behavioural"]["rung3_gate_pass"]
        ) / len(high_overlap)
    else:
        tpr = float("nan")

    # C: Spearman(rung3_polygram_overlap, rung3_jaccard) over all 28 pairs.
    poly_xs, jacc_ys = [], []
    for r in per_pair_records:
        po = r["behavioural"]["rung3_polygram_overlap"]
        ja = r["behavioural"]["rung3_jaccard"]
        if not (math.isnan(po) or math.isnan(ja)):
            poly_xs.append(po)
            jacc_ys.append(ja)
    spearman = _spearman(poly_xs, jacc_ys)

    # D: coverage = fraction of jaccard ≥ 0.30 pairs caught by gate.
    redundant = [
        r for r in per_pair_records
        if r["behavioural"]["rung3_jaccard"] >= 0.30
    ]
    if redundant:
        coverage = sum(
            1 for r in redundant if r["behavioural"]["rung3_gate_pass"]
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
    json_path = out_dir / "rung3_viability_spike.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    print()
    print("=" * 78)
    print(
        f"RUNG3-VIABILITY-SPIKE @ blocks.{baseline_report.layer} — "
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


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
