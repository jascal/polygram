"""Rung-viability v2 — **Axis 1** (compression coverage).

Per `docs/research/rung-viability-methodology.md` Axis 1: run
`EpochCompressor(encoding=X)` on the same SAE checkpoint with the
same quality budget across encodings, then compare the compression
yield:

- `n_features_zeroed_total` — how many features actually got
  compressed
- `cumulative_cross_entropy_delta` — the quality cost incurred
- `n_iterations` — how many iterations were needed to converge

The hypothesis: higher rungs find more redundancies per iteration
because their `max_features` cap lets `_select_panels` group more
features per panel (and PR #57 confirmed the K=8 → K=32 lift drops
the block count 41% on the same fixture, which directly translates
to per-block work for `EpochCompressor`).

Usage
-----

    python examples/rung_compression_coverage.py \\
        --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \\
        --encoding rung4 \\
        --output docs/research/data/rung_compression_coverage_rung4.json

Skip paths
----------

- SAE checkpoint missing → exit 0 with the canonical
  `hf download jbloom/GPT2-Small-SAEs-Reformatted` hint.
- `torch` / `transformers` missing → exit 0 with the
  `pip install polygram[behavioural]` hint. The compressor's
  pre-pass needs a real LLM forward to compute firing rates +
  residuals; the analytic phase that runs without torch is
  insufficient for the compression-coverage probe.

JSON schema
-----------

```
{
  "encoding": "mps" | "rung3" | "rung4",
  "max_features": int,
  "sae_checkpoint": str,
  "model_name": str,
  "layer": int,
  "n_prompts": int,
  "epoch_kwargs": {...},
  "n_features_zeroed_total": int,
  "n_iterations": int,
  "final_iteration": {
    "iteration": int,
    "cumulative_cross_entropy_delta": float,
    "convergence_state": str
  },
  "per_iteration": [
    {"iteration": int, "features_zeroed_this_iteration_count": int,
     "cumulative_cross_entropy_delta": float}, ...
  ]
}
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polygram.encoding import HEA_Rung2, MPSRung1, Rung3, Rung4


DEFAULT_SAE_PATH = Path(
    "./scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors"
)

ENCODING_REGISTRY: dict[str, callable] = {
    "mps":   lambda: MPSRung1(),
    "rung3": lambda: Rung3(),
    "rung4": lambda: Rung4(),
    "hea":   lambda: HEA_Rung2(depth=1, n_qubits=5),
}


# A small, fixed prompt set so the pre-pass cost is bounded across
# rungs. The same prompts go through each rung's compression run.
CANONICAL_PROMPTS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, gradient descent is a fundamental optimization algorithm.",
    "The mitochondrion is the powerhouse of the cell.",
    "Distillation transfers knowledge from a teacher model to a student model.",
    "Sparse autoencoders decompose neural activations into interpretable features.",
    "Quantum computing exploits superposition and entanglement for computation.",
    "The transformer architecture has revolutionized natural language processing.",
    "Reinforcement learning agents learn through interaction with their environment.",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--sae", type=Path, default=DEFAULT_SAE_PATH,
        help=f"Path to the SAE checkpoint (default: {DEFAULT_SAE_PATH})",
    )
    p.add_argument(
        "--encoding", choices=sorted(ENCODING_REGISTRY.keys()), default="rung4",
        help="Encoding to probe (default: rung4).",
    )
    p.add_argument(
        "--model-name", default="gpt2",
        help="HuggingFace model name for the pre-pass forward (default: gpt2).",
    )
    p.add_argument(
        "--layer", type=int, default=10,
        help="Transformer layer for residuals (default: 10).",
    )
    p.add_argument(
        "--n-panels-max", type=int, default=200,
        help="EpochCompressor.n_panels_max (default: 200).",
    )
    p.add_argument(
        "--max-iterations", type=int, default=3,
        help="EpochCompressor.max_iterations (default: 3).",
    )
    p.add_argument(
        "--coverage-target", type=float, default=0.5,
        help="EpochCompressor.coverage_target (default: 0.5).",
    )
    p.add_argument(
        "--cosine-threshold", type=float, default=0.3,
        help="EpochCompressor.cosine_threshold (default: 0.3).",
    )
    p.add_argument(
        "--n-prompts", type=int, default=len(CANONICAL_PROMPTS),
        help=(
            f"How many prompts to forward (default: "
            f"{len(CANONICAL_PROMPTS)} — full canonical set)."
        ),
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Optional JSON path for the result artifact.",
    )
    p.add_argument(
        "--assign-amp-knobs",
        action="store_true",
        help=(
            "Populate higher-rung amp-branch knobs from decoder "
            "geometry (PCA-axis extension). Without this flag, "
            "Rung3/Rung4 dictionaries collapse to MPSRung1-equivalent "
            "gram in the validator — see encoding-aware-knob-assignment."
        ),
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-iteration progress lines.",
    )
    return p.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else _parse(argv)

    sae_path = Path(args.sae)
    if not sae_path.is_file():
        print(
            f"rung_compression_coverage: SAE checkpoint not found at "
            f"{sae_path}. Download with `hf download "
            f"jbloom/GPT2-Small-SAEs-Reformatted "
            f"--include='blocks.10.hook_resid_pre/"
            f"sae_weights.safetensors' --local-dir ./scratch/real-sae`. "
            f"Skipping.",
            file=sys.stderr,
        )
        return 0

    # EpochCompressor's pre-pass needs torch + transformers (real
    # GPT-2 forward to compute firing rates + residuals). Skip
    # cleanly when the [behavioural] extra is missing.
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        print(
            f"rung_compression_coverage: behavioural extras missing "
            f"({exc}). Install with `pip install polygram[behavioural]` "
            f"to run on a real SAE. Skipping.",
            file=sys.stderr,
        )
        return 0

    import tempfile

    from polygram import EpochCompressor

    encoding = ENCODING_REGISTRY[args.encoding]()
    prompts = list(CANONICAL_PROMPTS[: max(1, args.n_prompts)])

    epoch_kwargs = dict(
        layer=int(args.layer),
        n_panels_max=int(args.n_panels_max),
        max_iterations=int(args.max_iterations),
        coverage_target=float(args.coverage_target),
        cosine_threshold=float(args.cosine_threshold),
    )

    if not args.quiet:
        print(f"sae_checkpoint: {sae_path}")
        print(f"encoding: {args.encoding} (max_features={encoding.max_features})")
        print(f"assign_amp_knobs: {bool(args.assign_amp_knobs)}")
        print(f"model: {args.model_name}, layer: {args.layer}")
        print(f"prompts: {len(prompts)}")
        print(f"epoch_kwargs: {epoch_kwargs}")
        print()
        print("== running EpochCompressor ==")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "compressed.safetensors"
        epoch = EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=prompts,
            model_name=args.model_name,
            encoding=encoding,
            assign_amp_knobs=bool(args.assign_amp_knobs),
            **epoch_kwargs,
        )
        result = epoch.run(out_path)

    report = result.report
    final_iter = report.iterations[-1] if report.iterations else None
    cumulative_by_iter: list[float] = []
    running = 0.0
    for it in report.iterations:
        running += float(it.cross_entropy_delta)
        cumulative_by_iter.append(running)
    final_cumulative_ce_delta = cumulative_by_iter[-1] if cumulative_by_iter else 0.0

    if not args.quiet:
        print()
        print("== Compression coverage metrics ==")
        print(f"  n_features_zeroed_total:        {report.n_features_zeroed_total}")
        print(f"  n_iterations:                   {len(report.iterations)}")
        if final_iter is not None:
            print(
                f"  final cumulative CE delta:      "
                f"{final_cumulative_ce_delta:.6f}"
            )
            print(f"  final convergence state:        {final_iter.convergence_state}")
        print()
        if report.iterations:
            print("  per-iteration trajectory:")
            for it, cum in zip(report.iterations, cumulative_by_iter):
                print(
                    f"    iter {it.iteration}: zeroed_count="
                    f"{len(it.features_zeroed_this_iteration)}, "
                    f"CE_delta={it.cross_entropy_delta:.6f}, "
                    f"cumulative_CE_delta={cum:.6f}, "
                    f"state={it.convergence_state}"
                )

    payload = {
        "encoding": args.encoding,
        "max_features": encoding.max_features,
        "assign_amp_knobs": bool(args.assign_amp_knobs),
        "sae_checkpoint": str(sae_path),
        "model_name": args.model_name,
        "layer": int(args.layer),
        "n_prompts": len(prompts),
        "epoch_kwargs": epoch_kwargs,
        "n_features_zeroed_total": int(report.n_features_zeroed_total),
        "n_iterations": len(report.iterations),
        "final_iteration": (
            None if final_iter is None
            else {
                "iteration": int(final_iter.iteration),
                "cross_entropy_delta": float(final_iter.cross_entropy_delta),
                "cumulative_cross_entropy_delta": float(final_cumulative_ce_delta),
                "convergence_state": final_iter.convergence_state,
            }
        ),
        "per_iteration": [
            {
                "iteration": int(it.iteration),
                "features_zeroed_this_iteration_count":
                    len(it.features_zeroed_this_iteration),
                "cross_entropy_delta": float(it.cross_entropy_delta),
                "cumulative_cross_entropy_delta": float(cum),
                "convergence_state": it.convergence_state,
            }
            for it, cum in zip(report.iterations, cumulative_by_iter)
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output}")
    return 0


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sae", type=Path, default=DEFAULT_SAE_PATH)
    p.add_argument("--encoding", choices=sorted(ENCODING_REGISTRY.keys()), default="rung4")
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--n-panels-max", type=int, default=200)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--coverage-target", type=float, default=0.5)
    p.add_argument("--cosine-threshold", type=float, default=0.3)
    p.add_argument("--n-prompts", type=int, default=len(CANONICAL_PROMPTS))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--assign-amp-knobs", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
