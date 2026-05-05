"""Worked example — `polygram.compression.EpochCompressor` end-to-end.

Runs the multi-panel orchestrator on the §4.4 SAE checkpoint
(`blocks.10.hook_resid_pre`) over the §4.4 prompt set, scaling the
validate→compress loop to multiple panels with stable-cluster
fixed-point iteration.

Usage
-----

    python examples/compress_epoch_validated.py \\
        --output-dir examples/output/compress_epoch_validated

Skip paths
----------

- SAE checkpoint missing → exits 0 with the canonical
  `hf download jbloom/GPT2-Small-SAEs-Reformatted` hint.
- `torch` / `transformers` missing → exits 0 with the
  `pip install ".[behavioural]"` hint.

Both skip paths exit 0 — they are normal "fetch the data first"
states, not failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polygram import EpochCompressor


DEFAULT_SAE_PATH = Path(
    "./scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-checkpoint",
        type=Path,
        default=DEFAULT_SAE_PATH,
        help=f"path to the SAE checkpoint (default: {DEFAULT_SAE_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/compress_epoch_validated"),
        help="where the rewritten .safetensors and "
             "epoch_report.json land",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=None,
        help="how many prompts to forward (default: all)",
    )
    parser.add_argument(
        "--n-panels-max", type=int, default=20,
        help="cap on total panels per iteration (default: 20)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="cap on iteration count (default: 3)",
    )
    parser.add_argument(
        "--coverage-target", type=float, default=0.5,
        help="target fraction of cosine-similar pairs to cover "
             "(default: 0.5; lower than the spec default 0.95 to "
             "keep the example tractable)",
    )
    args = parser.parse_args(argv)

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        print(
            f"compress_epoch_validated: SAE checkpoint not found at "
            f"{sae_path}. Download with `hf download "
            f"jbloom/GPT2-Small-SAEs-Reformatted "
            f"--include='blocks.10.hook_resid_pre/"
            f"sae_weights.safetensors' --local-dir ./scratch/real-sae`. "
            f"Skipping.",
            file=sys.stderr,
        )
        return 0

    try:
        from examples.behavioural_gram_scaleup import PROMPTS
    except ImportError:
        print(
            "compress_epoch_validated: examples/behavioural_gram_scaleup "
            "is required for the prompt set. Skipping.",
            file=sys.stderr,
        )
        return 0
    n = len(PROMPTS) if args.n_prompts is None else max(1, min(args.n_prompts, len(PROMPTS)))
    prompts = list(PROMPTS[:n])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "sae_weights.epoch.safetensors"
    out_report = out_dir / "epoch_report.json"

    print(f"compress_epoch_validated: building EpochCompressor on "
          f"{sae_path} with {n} prompts ...")
    epoch = EpochCompressor(
        sae_checkpoint=sae_path,
        prompts=prompts,
        layer=10,
        n_panels_max=args.n_panels_max,
        max_iterations=args.max_iterations,
        coverage_target=args.coverage_target,
    )

    try:
        result = epoch.run(out_ckpt)
    except ImportError as exc:
        print(
            f"compress_epoch_validated: {exc}",
            file=sys.stderr,
        )
        return 0

    result.report.to_json(out_report)

    print()
    print("=" * 78)
    print(
        f"COMPRESS-EPOCH-VALIDATED — convergence={result.report.convergence_reason} "
        f"iterations={len(result.report.iterations)} "
        f"features_zeroed_total={result.report.n_features_zeroed_total} "
        f"panels_total={result.report.n_panels_total} "
        f"coverage={result.report.coverage_achieved:.3f}"
    )
    print("=" * 78)
    print(f"Source sha256:   {result.report.source_checkpoint_sha256[:16]}…")
    print(f"Output sha256:   {result.report.output_checkpoint_sha256[:16]}…")
    print(f"Wall seconds:    {result.report.wall_seconds:.1f}")
    print()
    for it in result.report.iterations:
        print(
            f"  iter {it.iteration}: panels={len(it.panels)} "
            f"confirmed={it.confirmed_pair_count} "
            f"clusters={it.clusters_compressed} "
            f"zeroed={len(it.features_zeroed_this_iteration)} "
            f"Δ={it.cross_entropy_delta:.3e} state={it.convergence_state}"
        )
    print()
    print(f"Checkpoint  → {out_ckpt}")
    print(f"Report JSON → {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
