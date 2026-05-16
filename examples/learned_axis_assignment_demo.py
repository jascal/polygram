"""Reproduces the headline of `docs/research/rung5-pareto-scans.md`
scan 4 (greedy axis-to-knob lift) using the production
``LearnedKnobAssignment`` strategy.

Builds a synthetic 64-feature clustered SAE (16 clusters of 4,
d_model=32), imports it under both the hardcoded baseline and the
learned strategy at k=3 and k=4, and reports the Spearman delta plus
gram condition number ratio. Writes JSON to
``docs/research/data/learned_axis_assignment_demo.json`` when
``--json-out`` is supplied.

Usage::

    python examples/learned_axis_assignment_demo.py
    python examples/learned_axis_assignment_demo.py \\
        --json-out docs/research/data/learned_axis_assignment_demo.json

CPU-only, no torch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from polygram import from_sae_lens
from polygram.encoding import Rung5
from polygram.geometry import LearnedKnobAssignment
from polygram.geometry.objectives import spearman_objective
from polygram.sae_import import SAEFeatureRecord


def _synth_records(
    n_clusters: int = 16,
    cluster_size: int = 4,
    d_model: int = 32,
    seed: int = 0,
    noise_sigma: float = 0.08,
) -> dict:
    """Tight-cluster synth — one centroid per cluster, isotropic noise
    siblings. Matches scan 4's fixture exactly."""
    rng = np.random.default_rng(seed)
    records: dict = {}
    for c in range(n_clusters):
        centroid = rng.standard_normal(d_model)
        centroid /= np.linalg.norm(centroid) + 1e-12
        for s in range(cluster_size):
            v = centroid + rng.standard_normal(d_model) * noise_sigma
            v /= np.linalg.norm(v) + 1e-12
            i = c * cluster_size + s
            records[i] = SAEFeatureRecord(
                feature_id=i,
                name=f"c{c:02d}_f{c:02d}_{s:02d}",
                label=None,
                projection=v.astype(np.float32),
                activation_mean=0.0,
                activation_std=1.0,
            )
    return records


def _decoder_cos_sq(records: dict) -> np.ndarray:
    """|cos(v_i, v_j)|² over the unit-normalised projections in
    record-id order."""
    projs = np.array([records[i].projection for i in sorted(records.keys())])
    norms = np.linalg.norm(projs, axis=1, keepdims=True) + 1e-12
    u = projs / norms
    cos = u @ u.T
    return cos ** 2


def _spearman_on_decoder(d, records: dict) -> float:
    """Spearman of the Dictionary's analytic gram against the decoder
    cos² matrix (off-diagonal upper-triangle)."""
    cos_sq = _decoder_cos_sq(records)
    g = d.gram()
    return spearman_objective(g, cos_sq)


def _cond(d) -> float:
    s = np.linalg.svd(d.gram(), compute_uv=False)
    sr = np.real(s)
    sr = sr[sr > 0]
    if sr.size == 0:
        return float("inf")
    return float(sr.max() / sr.min())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[3, 4],
        help="Rung5 n_amp_qubits values to demo (default: 3, 4).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Synth fixture seed (default: 0).",
    )
    args = parser.parse_args(argv)

    records = _synth_records(seed=args.seed)
    ids = sorted(records.keys())

    runs: list[dict] = []
    for k in args.k:
        encoding = Rung5(n_amp_qubits=k)

        # Hardcoded baseline.
        t0 = time.perf_counter()
        d_base, _ = from_sae_lens(records, ids, encoding=encoding)
        t_base = time.perf_counter() - t0
        s_base = _spearman_on_decoder(d_base, records)
        c_base = _cond(d_base)

        # Learned strategy.
        t0 = time.perf_counter()
        d_learn, report_learn = from_sae_lens(
            records, ids, encoding=encoding,
            learn_axis_assignment=LearnedKnobAssignment(),
        )
        t_learn = time.perf_counter() - t0
        s_learn = _spearman_on_decoder(d_learn, records)
        c_learn = _cond(d_learn)

        runs.append({
            "k": k,
            "spearman_baseline": s_base,
            "spearman_learned": s_learn,
            "delta_spearman": s_learn - s_base,
            "cond_baseline": c_base,
            "cond_learned": c_learn,
            "cond_ratio": c_base / c_learn if c_learn else float("inf"),
            "baseline_seconds": t_base,
            "learned_seconds": t_learn,
            "learned_assignment": report_learn.learned_axis_assignment[
                "axis_assignment"
            ],
        })
        print(
            f"k={k}  baseline Spearman={s_base:+.4f}  "
            f"learned Spearman={s_learn:+.4f}  "
            f"Δ={s_learn - s_base:+.4f}\n"
            f"      cond {c_base:.2e} → {c_learn:.2e}  "
            f"(ratio {c_base / c_learn:.2f}×)  "
            f"learned in {t_learn:.2f}s vs baseline {t_base:.2f}s"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "schema": "polygram.learned_axis_assignment_demo.v1",
            "config": {"seed": args.seed, "k": list(args.k)},
            "runs": runs,
        }, indent=2))
        print(f"\nJSON artifact written → {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
