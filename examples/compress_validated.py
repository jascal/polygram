"""Worked example — `polygram.compression.Compressor` end-to-end.

Reads the `ValidationReport` produced by `examples/behavioural_validate.py`
(or any compatible JSON file), constructs a `Compressor` over the same
SAE checkpoint, runs `plan() + apply()`, and dumps the
`CompressionReport` next to the rewritten `.safetensors`.

Usage
-----

    # First run the validator to produce a report (or use an existing one):
    python examples/behavioural_validate.py \\
        --output-dir examples/output/behavioural_validate

    # Then compress the SAE against that report:
    python examples/compress_validated.py \\
        --validation-report examples/output/behavioural_validate/validation_report.json \\
        --output-dir examples/output/compress_validated

Skip paths
----------

- SAE checkpoint missing → exits 0 with a clear hint (mirrors the
  validator example's skip pattern).
- ValidationReport missing → exits 0 with a "run the validator first"
  hint.

Both skip paths exit 0 — they are normal "fetch the data first"
states, not failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polygram import Compressor, ValidationReport


DEFAULT_SAE_PATH = Path(
    "./scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors"
)
DEFAULT_VALIDATION_REPORT = Path(
    "examples/output/behavioural_validate/validation_report.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
        help=f"path to a ValidationReport JSON "
             f"(default: {DEFAULT_VALIDATION_REPORT})",
    )
    parser.add_argument(
        "--sae-checkpoint",
        type=Path,
        default=DEFAULT_SAE_PATH,
        help=f"source SAE checkpoint (default: {DEFAULT_SAE_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/compress_validated"),
        help="where the rewritten .safetensors and "
             "compression_report.json land",
    )
    parser.add_argument(
        "--strategy",
        choices=("zero",),
        default="zero",
        help="compression strategy (initial release: zero only)",
    )
    args = parser.parse_args(argv)

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        print(
            f"compress_validated: SAE checkpoint not found at "
            f"{sae_path}. Download with `hf download "
            f"jbloom/GPT2-Small-SAEs-Reformatted "
            f"--include='blocks.10.hook_resid_pre/"
            f"sae_weights.safetensors' --local-dir ./scratch/real-sae`. "
            f"Skipping.",
            file=sys.stderr,
        )
        return 0

    vreport_path = Path(args.validation_report)
    if not vreport_path.is_file():
        print(
            f"compress_validated: validation report not found at "
            f"{vreport_path}. Run `python examples/behavioural_validate.py` "
            f"first. Skipping.",
            file=sys.stderr,
        )
        return 0

    print(f"compress_validated: loading {vreport_path} ...")
    report = ValidationReport.from_json(vreport_path)
    print(
        f"  {len(report.confirmed)} confirmed pairs over "
        f"{len(report.feature_ids)} features at blocks.{report.layer}"
    )
    if not report.confirmed:
        print(
            "compress_validated: validation report has zero confirmed "
            "pairs — nothing to compress. Exiting cleanly.",
            file=sys.stderr,
        )
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "sae_weights.compressed.safetensors"
    out_report = out_dir / "compression_report.json"

    compressor = Compressor(
        validation_report=report,
        sae_checkpoint=sae_path,
        strategy=args.strategy,
    )

    print("compress_validated: building plan ...")
    plan = compressor.plan()
    for cluster in plan.clusters:
        print(
            f"  cluster {cluster.cluster_id}: members={list(cluster.members)} "
            f"rep={cluster.representative} zeroed={list(cluster.zeroed)}"
        )

    print(f"compress_validated: rewriting weights → {out_ckpt} ...")
    result = compressor.apply(plan, output_checkpoint=out_ckpt)
    result.report.to_json(out_report)

    print()
    print("=" * 78)
    print(
        f"COMPRESS-VALIDATED — {result.report.n_clusters} clusters, "
        f"{result.report.n_features_zeroed} features zeroed, "
        f"{result.report.n_features_kept} kept"
    )
    print("=" * 78)
    print(f"Source sha256:   {result.report.source_checkpoint_sha256[:16]}…")
    print(f"Output sha256:   {result.report.output_checkpoint_sha256[:16]}…")
    print(f"Strategy:        {result.report.strategy}")
    print()
    print(f"Checkpoint  → {out_ckpt}")
    print(f"Report JSON → {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
