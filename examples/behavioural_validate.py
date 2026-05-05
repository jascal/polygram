"""Worked example — `polygram.behavioural.BehaviouralValidator` on
the §4.4 GPT-2-small selection.

Runs the validator against the same 8-feature panel
`examples/behavioural_gram_scaleup.py` used to produce
`docs/research/data/scaleup_pairs.csv`, then dumps the JSON +
CSV outputs and prints the list of confirmed candidates.

Usage
-----

    python examples/behavioural_validate.py \\
        --output-dir examples/output

Skip paths
----------

- SAE checkpoint missing → exits with the canonical
  `hf download jbloom/GPT2-Small-SAEs-Reformatted` hint.
- `torch` / `transformers` missing → exits with the
  `pip install polygram[behavioural]` hint.

Both paths exit 0 — they are normal "you need to fetch the data first"
states, not failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from polygram import (
    BehaviouralValidator,
    from_sae_lens,
    load_sae_safetensors,
)


# Same 12-prompt set §4.2 / §4.3 / §4.4 used. Imported lazily from the
# scaleup script to avoid duplicating the literal.
from examples.behavioural_gram_scaleup import PROMPTS

# Same selection §4.4 produced: seed = 12999, near-cluster
# {19398, 4192, 23625}, far-cluster {8371, 2287, 68, 13737}.
SELECTION_FEATURE_IDS: tuple[int, ...] = (
    12999, 19398, 4192, 23625, 8371, 2287, 68, 13737,
)

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
        default=Path("examples/output/behavioural_validate"),
        help="where validation_report.json + validation_pairs.csv land",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=len(PROMPTS),
        help=f"how many prompts to forward (1..{len(PROMPTS)})",
    )
    args = parser.parse_args(argv)

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        print(
            f"behavioural_validate: SAE checkpoint not found at "
            f"{sae_path}. Download with `hf download "
            f"jbloom/GPT2-Small-SAEs-Reformatted "
            f"--include='blocks.10.hook_resid_pre/"
            f"sae_weights.safetensors' --local-dir ./scratch/real-sae`. "
            f"Skipping.",
            file=sys.stderr,
        )
        return 0

    feature_ids = list(SELECTION_FEATURE_IDS)
    n_prompts = max(1, min(args.n_prompts, len(PROMPTS)))
    prompts = list(PROMPTS[:n_prompts])

    print("behavioural_validate: building Dictionary via from_sae_lens ...")
    records = load_sae_safetensors(sae_path, feature_ids=feature_ids)
    dictionary, selection = from_sae_lens(
        records,
        feature_ids,
        assign_gamma=True,
        name="ScaleupBlocks10",
    )
    print(
        f"  cluster method: {selection.cluster_method} "
        f"(β var-explained {selection.beta_variance_explained:.3f})"
    )

    print("behavioural_validate: constructing BehaviouralValidator ...")
    validator = BehaviouralValidator(
        dictionary=dictionary,
        sae_checkpoint=sae_path,
        feature_ids=feature_ids,
        prompts=prompts,
        layer=10,
    )

    print("behavioural_validate: running predict() + validate() ...")
    try:
        report = validator.run()
    except ImportError as exc:
        print(
            f"behavioural_validate: {exc}",
            file=sys.stderr,
        )
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "validation_report.json"
    csv_path = out_dir / "validation_pairs.csv"
    report.to_json(json_path)
    report.to_csv(csv_path)

    print()
    print("=" * 78)
    print(
        f"BEHAVIOURAL-VALIDATOR @ blocks.{report.layer} — "
        f"{len(report.feature_ids)} features, {len(report.pairs)} pairs, "
        f"{report.n_tokens} tokens"
    )
    print("=" * 78)
    print(
        f"Spearman(Polygram, Jaccard):   "
        f"{report.summary.spearman_polygram_jaccard:+.4f}"
    )
    print(
        f"Pearson(Polygram, Jaccard):    "
        f"{report.summary.pearson_polygram_jaccard:+.4f}"
    )
    print(f"Outcome:                       {report.summary.outcome}")
    print()
    print("Confirmed candidates (gate_pass = True):")
    if not report.confirmed:
        print("  (none)")
    else:
        for i, j in report.confirmed:
            row = next(p for p in report.pairs if (p.i, p.j) == (i, j))
            print(
                f"  feat_{i:>5d} × feat_{j:>5d}  "
                f"polygram={row.polygram_overlap:.4f}  "
                f"jaccard={row.jaccard:.4f}  "
                f"n_both_fire={row.n_both_fire}"
            )
    print()
    print(f"JSON written → {json_path}")
    print(f"CSV  written → {csv_path}")
    return 0


# Suppress unused-import warnings — np stays imported for downstream
# follow-ups that want to slice the report's per-pair arrays.
_ = np


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
