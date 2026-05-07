"""Worked example — `polygram.compression.Regrower` end-to-end.

Reads the `CompressionReport` produced by
`examples/compress_validated.py`, captures residuals from the §4.4
12-prompt set, and runs `Regrower.from_compression_report` on the
already-compressed checkpoint to repopulate its zeroed slots with new
directions extracted from the activation residuals.

Usage
-----

    # First run the validator + compressor:
    python examples/behavioural_validate.py \\
        --output-dir examples/output/behavioural_validate
    python examples/compress_validated.py \\
        --validation-report examples/output/behavioural_validate/validation_report.json \\
        --output-dir examples/output/compress_validated

    # Then regrow the zeroed slots:
    python examples/regrow_validated.py \\
        --compression-report examples/output/compress_validated/compression_report.json \\
        --output-dir examples/output/regrow_validated

Skip paths
----------

- Compression report missing → exits 0 with a "run the compressor
  first" hint.
- Compressed SAE checkpoint (named in the compression report) missing
  → exits 0 with a clear hint.

Both skip paths exit 0 — they are normal "fetch the data first"
states, not failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polygram import CompressionReport, Regrower


DEFAULT_COMPRESSION_REPORT = Path(
    "examples/output/compress_validated/compression_report.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compression-report",
        type=Path,
        default=DEFAULT_COMPRESSION_REPORT,
        help=f"path to a CompressionReport JSON "
             f"(default: {DEFAULT_COMPRESSION_REPORT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/regrow_validated"),
        help="where the regrown .safetensors and "
             "regrow_report.json land",
    )
    parser.add_argument(
        "--strategy",
        choices=("residual_kmeans",),
        default="residual_kmeans",
        help="regrow strategy (initial release: residual_kmeans only)",
    )
    parser.add_argument(
        "--layer", type=int, default=10,
        help="transformer block whose forward_pre hook captures the "
             "residual stream (default: 10, matching the §4.4 hook)",
    )
    parser.add_argument(
        "--model-name", type=str, default="gpt2",
        help="HuggingFace model id for the residual-capture pass "
             "(default: gpt2). Required by Regrower.from_compression_report "
             "since polygram-tuning-config removed the silent GPT-2 default.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for k-means (default: 0)",
    )
    args = parser.parse_args(argv)

    cr_path = Path(args.compression_report)
    if not cr_path.is_file():
        print(
            f"regrow_validated: compression report not found at "
            f"{cr_path}. Run `python examples/compress_validated.py` "
            f"first. Skipping.",
            file=sys.stderr,
        )
        return 0

    print(f"regrow_validated: loading {cr_path} ...")
    compression_report = CompressionReport.from_json(cr_path)

    sae_path = Path(compression_report.output_checkpoint)
    if not sae_path.is_file():
        print(
            f"regrow_validated: compressed SAE checkpoint not found at "
            f"{sae_path}. Run the compress step first. Skipping.",
            file=sys.stderr,
        )
        return 0

    n_zeroed = sum(
        len(c.zeroed) for c in compression_report.plan.clusters
    )
    if n_zeroed == 0:
        print(
            "regrow_validated: compression report has no zeroed slots — "
            "nothing to regrow. Exiting cleanly.",
            file=sys.stderr,
        )
        return 0

    # Reuse the §4.4 prompt set lazily — only if compress_validated's
    # outputs are present.
    try:
        from examples.behavioural_gram_scaleup import PROMPTS
    except ImportError:
        print(
            "regrow_validated: examples/behavioural_gram_scaleup is "
            "required for the prompt set. Skipping.",
            file=sys.stderr,
        )
        return 0
    prompts = list(PROMPTS)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "sae_weights.regrown.safetensors"
    out_report = out_dir / "regrow_report.json"

    print(
        f"regrow_validated: constructing Regrower (chained from "
        f"CompressionReport, {n_zeroed} zeroed slots) ..."
    )
    regrower = Regrower.from_compression_report(
        compression_report,
        sae_checkpoint=sae_path,
        strategy=args.strategy,
        prompts=prompts,
        seed=args.seed,
        layer=args.layer,
        model_name=args.model_name,
    )

    print(
        "regrow_validated: capturing residuals + running k-means ..."
    )
    plan = regrower.plan()
    print(
        f"  n_residual_tokens: {plan.n_residual_tokens}  "
        f"slots: {len(plan.slots)}"
    )
    for slot in plan.slots:
        status = (
            f"populated (cluster_size={slot.cluster_size})"
            if slot.cluster_size > 0 else "left zero"
        )
        print(f"  feat_{slot.feature_id:>5d}: {status}")

    print(f"regrow_validated: rewriting weights → {out_ckpt} ...")
    result = regrower.apply(plan, output_checkpoint=out_ckpt)
    result.report.to_json(out_report)

    print()
    print("=" * 78)
    print(
        f"REGROW-VALIDATED — {result.report.n_slots_repopulated} slots "
        f"populated, {result.report.n_slots_left_zero} left zero"
    )
    print("=" * 78)
    print(f"Source sha256:   {result.report.source_checkpoint_sha256[:16]}…")
    print(f"Output sha256:   {result.report.output_checkpoint_sha256[:16]}…")
    print(f"Strategy:        {result.report.strategy}")
    print(
        f"Provenance:      compression_report sha "
        f"{result.report.provenance['compression_report_source_sha256'][:16]}… "
        f"({result.report.provenance['compression_report_dictionary_name']})"
    )
    print()
    print(f"Checkpoint  → {out_ckpt}")
    print(f"Report JSON → {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
