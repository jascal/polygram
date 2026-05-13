"""Clustered-dictionary walkthrough — §9 killer experiment.

Compares cosine-based redundancy detection between two pipelines on a
real GPT-2-small SAE checkpoint:

1. **Flat baseline** — full N² pairwise cosine; enumerate every
   feature pair with cosine ≥ `redundancy_threshold`.
2. **Clustered** — partition into ≤K-feature blocks via
   `build_clustered_dictionary`, then read out cross-block redundant
   pairs (intra-block pairs are caught by the per-block dense Gram
   and counted toward the recall set since they're a strict subset
   of the flat baseline at the same threshold).

Reports recall, precision, and wall-clock speedup. Targets per
`openspec/changes/clustered-dictionary-analysis/tasks.md` §9:

- Recall ≥ 0.95.
- Speedup ≥ 100× (clustered wall-clock vs flat).

Companion writeup: `docs/research/clustered-dictionary-recall-vs-flat.md`.

Usage:

    python examples/clustered_dictionary_walkthrough.py \\
        --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \\
        --n-features 2048 \\
        --block-size 32 \\
        --redundancy-threshold 0.7 \\
        --output docs/research/data/clustered_dictionary_recall.json

When `--sae` is omitted, falls back to the bundled toy SAE fixture
(16 features) so the script runs in CI without external dependencies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from polygram.clustered_dictionary import (
    BlockFormation,
    build_clustered_dictionary,
    compute_cosine_pair_graph,
)
from polygram.dictionary import Feature
from polygram.encoding import MPSRung1


FIXTURE_TOY = Path(__file__).parent.parent / "tests" / "fixtures" / "toy_sae.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--sae",
        type=Path,
        default=None,
        help="Path to an SAE safetensors file with W_dec (n_features, d_model). "
        "When omitted, falls back to the bundled toy fixture.",
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=512,
        help="Subset size to run the experiment on (default: 512). "
        "The flat baseline is O(N²), so keep N reasonable.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Per-block feature cap (default: 8, MPSRung1's cap).",
    )
    p.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.3,
        help="Block-formation cosine threshold (default: 0.3).",
    )
    p.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.7,
        help="Redundancy cosine threshold (default: 0.7). Pairs with "
        "cosine >= this are counted as redundant.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the result artifact.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_decoder_vectors(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """Return `(decoder_vectors, fixture_label)`."""
    if args.sae is None:
        # Fallback: toy fixture (16 features × 8 d_model).
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


def run_flat_baseline(
    vectors: np.ndarray, threshold: float
) -> tuple[set[tuple[int, int]], float]:
    """Compute the full pairwise cosine + threshold the result."""
    t0 = time.monotonic()
    pairs = compute_cosine_pair_graph(vectors, threshold=threshold)
    wall = time.monotonic() - t0
    return pairs, wall


def run_clustered(
    vectors: np.ndarray,
    *,
    block_size: int,
    cosine_threshold: float,
    redundancy_threshold: float,
) -> tuple[set[tuple[int, int]], float, dict]:
    """Build ClusteredDictionary, surface redundant pairs (intra-block
    via per-block dense gram of cosines, cross-block via the analytic
    primitive). Returns the discovered pair set + wall-clock + stats.
    """
    n = vectors.shape[0]
    features = [
        Feature(name=f"f{i}", cluster="all", beta=0.0) for i in range(n)
    ]
    t0 = time.monotonic()
    cd = build_clustered_dictionary(
        name="walkthrough",
        features=features,
        decoder_vectors=vectors,
        encoding=MPSRung1(),
        block_formation=BlockFormation(
            strategy="cosine",
            cosine_threshold=cosine_threshold,
            block_size_max=block_size,
        ),
    )
    pairs: set[tuple[int, int]] = set()
    # Intra-block: same-block pairs with cosine >= redundancy_threshold.
    # We avoid recomputing per-block grams (the analytic-quantum Gram
    # is in a different unit from the decoder cosine); use direct
    # decoder cosines on each block's feature subset for consistency
    # with the flat baseline.
    for block_idx, block in enumerate(cd.blocks):
        block_feat_global_ids = [
            int(f.name[1:]) for f in block.features  # "f<idx>" → idx
        ]
        block_vectors = vectors[block_feat_global_ids]
        intra_pairs_local = compute_cosine_pair_graph(
            block_vectors, threshold=redundancy_threshold
        )
        for li, lj in intra_pairs_local:
            gi = block_feat_global_ids[li]
            gj = block_feat_global_ids[lj]
            pairs.add((min(gi, gj), max(gi, gj)))
    # Cross-block: the analytic primitive.
    report = cd.cross_block_redundant_pairs(threshold=redundancy_threshold)
    for p in report.pairs:
        gi = int(p.feat_i_name[1:])
        gj = int(p.feat_j_name[1:])
        pairs.add((min(gi, gj), max(gi, gj)))
    wall = time.monotonic() - t0
    stats = {
        "n_blocks": cd.n_blocks,
        "mean_block_size": cd.mean_block_size,
        "n_cross_block_edges_total": cd.n_cross_block_edges,
        "n_cross_block_redundant": len(report.pairs),
    }
    return pairs, wall, stats


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    vectors_full, fixture_label = load_decoder_vectors(args)
    n_total = vectors_full.shape[0]
    n_features = min(args.n_features, n_total)
    selected = rng.choice(n_total, size=n_features, replace=False)
    selected.sort()
    vectors = vectors_full[selected]

    print(f"fixture: {fixture_label}")
    print(f"  total features: {n_total}, d_model: {vectors_full.shape[1]}")
    print(f"  subset for run: N={n_features} (seed={args.seed})")
    print(f"  block_size={args.block_size}, cosine_threshold={args.cosine_threshold}")
    print(f"  redundancy_threshold={args.redundancy_threshold}\n")

    print("== Flat baseline ==")
    flat_pairs, flat_wall = run_flat_baseline(
        vectors, threshold=args.redundancy_threshold
    )
    print(f"  pairs found: {len(flat_pairs)}")
    print(f"  wall: {flat_wall * 1000:.1f} ms\n")

    print("== Clustered ==")
    clustered_pairs, clustered_wall, stats = run_clustered(
        vectors,
        block_size=args.block_size,
        cosine_threshold=args.cosine_threshold,
        redundancy_threshold=args.redundancy_threshold,
    )
    print(f"  pairs found: {len(clustered_pairs)}")
    print(f"  blocks: {stats['n_blocks']} (mean size {stats['mean_block_size']:.1f})")
    print(f"  cross-block edges: {stats['n_cross_block_edges_total']}")
    print(f"  cross-block redundant: {stats['n_cross_block_redundant']}")
    print(f"  wall: {clustered_wall * 1000:.1f} ms\n")

    # Compute recall / precision / speedup.
    intersect = flat_pairs & clustered_pairs
    recall = len(intersect) / max(1, len(flat_pairs))
    precision = len(intersect) / max(1, len(clustered_pairs))
    speedup = flat_wall / clustered_wall if clustered_wall > 0 else float("inf")

    print("== Comparison ==")
    print(f"  recall    = {recall:.4f}  (target ≥ 0.95)")
    print(f"  precision = {precision:.4f}")
    print(f"  speedup   = {speedup:.1f}× (target ≥ 100× at SAE scale)")

    artifact = {
        "fixture": fixture_label,
        "n_total_features": int(n_total),
        "n_subset_features": int(n_features),
        "d_model": int(vectors_full.shape[1]),
        "seed": int(args.seed),
        "block_size": int(args.block_size),
        "cosine_threshold": float(args.cosine_threshold),
        "redundancy_threshold": float(args.redundancy_threshold),
        "flat": {
            "n_pairs": len(flat_pairs),
            "wall_seconds": float(flat_wall),
        },
        "clustered": {
            "n_pairs": len(clustered_pairs),
            "wall_seconds": float(clustered_wall),
            **stats,
        },
        "recall": float(recall),
        "precision": float(precision),
        "speedup": float(speedup),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
