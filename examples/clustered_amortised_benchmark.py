"""Downstream-amortised benchmark for `ClusteredDictionary`.

Companion to `examples/clustered_dictionary_walkthrough.py` (the §9
killer experiment) and `docs/research/clustered-amortised-benchmark.md`
(the writeup). Measures whether clustering's per-block O(K²) ops
amortise against flat's O(N²) once you reuse one `ClusteredDictionary`
for repeated operations.

See `polygram.benchmarks.clustered_amortised.run_amortised_benchmark`
for the framing of each op and the cross-over metric.

Usage::

    python examples/clustered_amortised_benchmark.py \\
        --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \\
        --n-features 24576 \\
        --encoding rung3 \\
        --op all \\
        --n-repeats 64 \\
        --output docs/research/data/clustered_amortised_benchmark.json

`--op all` runs `gram`, `cross_block_overlap`, and `redundancy` in one
invocation and emits a single combined JSON whose ``ops`` field is a
list of per-op `BenchmarkReport` dicts.

When `--sae` is omitted, falls back to the bundled toy SAE fixture so
the script runs in CI without external dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from polygram.benchmarks import run_amortised_benchmark
from polygram.dictionary import Dictionary, Feature
from polygram.encoding import HEA_Rung2, MPSRung1, Rung3, Rung4


ENCODING_REGISTRY: dict[str, tuple[callable, int]] = {
    "mps":   (lambda: MPSRung1(),                     MPSRung1.max_features),
    "rung3": (lambda: Rung3(),                        Rung3.max_features),
    "rung4": (lambda: Rung4(),                        Rung4.max_features),
    "hea":   (lambda: HEA_Rung2(depth=1, n_qubits=5), 32),
}

ALL_OPS = ("gram", "cross_block_overlap", "redundancy")

FIXTURE_TOY = (
    Path(__file__).parent.parent / "tests" / "fixtures" / "toy_sae.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sae", type=Path, default=None)
    p.add_argument("--n-features", type=int, default=None)
    p.add_argument(
        "--encoding", choices=sorted(ENCODING_REGISTRY), default="mps",
    )
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument(
        "--op", choices=(*ALL_OPS, "all"), default="all",
        help="Which op to benchmark, or 'all' for the full matrix.",
    )
    p.add_argument("--n-repeats", type=int, default=64)
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--cosine-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def load_decoder_vectors(
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
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


def main() -> None:
    args = parse_args()
    vectors_full, fixture_label = load_decoder_vectors(args)
    n_total = vectors_full.shape[0]

    if args.n_features is None:
        n_features = n_total
    else:
        n_features = min(args.n_features, n_total)
    rng = np.random.default_rng(args.seed)
    if n_features < n_total:
        idx = rng.choice(n_total, size=n_features, replace=False)
        idx.sort()
        vectors = vectors_full[idx]
    else:
        vectors = vectors_full

    encoding_factory, default_block_size = ENCODING_REGISTRY[args.encoding]
    encoding = encoding_factory()
    block_size = args.block_size if args.block_size is not None else default_block_size

    features = [
        Feature(name=f"f{i}", cluster="all", beta=0.0)
        for i in range(n_features)
    ]
    dictionary = Dictionary(
        name="amortised_benchmark",
        features=features,
        hierarchy={"all": [f.name for f in features]},
        encoding=encoding,
    )

    print(f"fixture: {fixture_label}")
    print(f"  total features: {n_total}, d_model: {vectors.shape[1]}")
    print(f"  subset for run: N={n_features} (seed={args.seed})")
    print(f"  encoding={args.encoding} (cap={encoding.max_features}), "
          f"block_size={block_size}")
    print(f"  cosine_threshold={args.cosine_threshold}, "
          f"n_repeats={args.n_repeats}, sample_size={args.sample_size}")
    print()

    ops_to_run = ALL_OPS if args.op == "all" else (args.op,)
    op_reports: list[dict[str, Any]] = []
    for op in ops_to_run:
        report = run_amortised_benchmark(
            dictionary,
            vectors,
            op=op,
            n_repeats=args.n_repeats,
            sample_size=args.sample_size,
            cosine_threshold=args.cosine_threshold,
            block_size_max=block_size,
            seed=args.seed,
        )
        print(f"== op={op} ==")
        print(f"  flat_per_op       = {report.flat_wall_seconds_per_op*1000:.2f} ms")
        print(f"  clustered_per_op  = {report.clustered_wall_seconds_per_op*1000:.2f} ms")
        print(f"  build_once        = {report.clustered_block_formation_seconds*1000:.1f} ms")
        if report.cross_over_n_repeats is None:
            print("  cross_over        = none (clustered ≥ flat per-op — worst case for clustering)")
        else:
            print(f"  cross_over        = {report.cross_over_n_repeats} ops")
        print(f"  rss_delta_kb      = {report.rss_delta_kb}")
        print(f"  object_delta      = {report.object_count_delta}")
        if op == "cross_block_overlap":
            print(f"  sample_size       = {report.sample_size_effective}/{report.sample_size_requested}")
        print()
        op_reports.append(report.to_dict())

    artifact = {
        "polygram_version": op_reports[0]["polygram_version"],
        "git_commit": op_reports[0]["git_commit"],
        "fixture": fixture_label,
        "n_total_features": int(n_total),
        "n_subset_features": int(n_features),
        "d_model": int(vectors.shape[1]),
        "seed": int(args.seed),
        "encoding": args.encoding,
        "block_size": int(block_size),
        "cosine_threshold": float(args.cosine_threshold),
        "n_repeats": int(args.n_repeats),
        "sample_size": int(args.sample_size),
        "ops": op_reports,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
