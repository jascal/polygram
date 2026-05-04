"""Safetensors SAE walk-through.

Demonstrates the full chain from a `.safetensors` decoder file to a
verifying Polygram `.q.orca.md` artifact, all without pulling torch
or `sae_lens` into the dep tree.

Pipeline:

1. Synthesize a tiny `.safetensors` fixture under the example's
   output dir (16 features, 8-dim projections, deterministic).
2. Load via `polygram.load_sae_safetensors` (auto-detects the
   `W_dec` key).
3. Pick four features `[0, 1, 4, 5]` (≤ 8 — the rung-1 MPS cap).
4. Build a Polygram `Dictionary` via `from_sae_lens` using the
   k-means cluster path.
5. Emit a verifying `.q.orca.md` via `write_qorca` and parse it
   back through q-orca to assert it's well-formed.

Output layout under ``output_dir / "sae_safetensors"``:

- ``synthesized.safetensors``      — the fixture
- ``ImportedSafetensors.q.orca.md`` — the emitted Polygram dictionary
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polygram import (
    Dictionary,
    from_sae_lens,
    load_sae_safetensors,
    write_qorca,
)


def main(output_dir: str | Path = "examples/output") -> None:
    out_dir = Path(output_dir) / "sae_safetensors"
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_path = out_dir / "synthesized.safetensors"
    _synthesize_fixture(sae_path, n_features=16, d_model=8, seed=0)

    records = load_sae_safetensors(sae_path)
    print(f"loaded {len(records)} features from {sae_path}")

    feature_ids = [0, 1, 4, 5]
    dictionary, report = from_sae_lens(records, feature_ids)
    print(
        f"dictionary: {dictionary.name} "
        f"({len(dictionary.features)} features, "
        f"{len(dictionary.hierarchy)} clusters)"
    )
    print(
        f"selection report: cluster_method={report.cluster_method}, "
        f"beta_variance_explained={report.beta_variance_explained:.4f}"
    )

    machine_path = out_dir / "ImportedSafetensors.q.orca.md"
    renamed = Dictionary(
        name="ImportedSafetensors",
        features=dictionary.features,
        hierarchy=dictionary.hierarchy,
        encoding=dictionary.encoding,
    )
    write_qorca(renamed, machine_path)
    print(f"emitted: {machine_path}")

    # Sanity-check that the emitted markdown round-trips through the
    # q-orca parser without errors.
    from q_orca.parser.markdown_parser import parse_q_orca_markdown

    parsed = parse_q_orca_markdown(machine_path.read_text())
    if parsed.errors:
        raise SystemExit(f"q-orca parse errors: {parsed.errors}")
    print("q-orca parse: ok")


def _synthesize_fixture(
    path: Path, *, n_features: int, d_model: int, seed: int
) -> None:
    """Write a deterministic .safetensors fixture with two pseudo-clusters
    arranged so that k-means recovers the structure cleanly."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    half = n_features // 2
    cluster_a = np.zeros((half, d_model), dtype=np.float32)
    cluster_b = np.zeros((n_features - half, d_model), dtype=np.float32)
    cluster_a[:, 0] = 1.0
    cluster_b[:, 1] = 1.0
    cluster_a += rng.standard_normal(cluster_a.shape).astype(np.float32) * 0.05
    cluster_b += rng.standard_normal(cluster_b.shape).astype(np.float32) * 0.05
    weights = np.concatenate([cluster_a, cluster_b], axis=0)
    save_file({"W_dec": weights}, str(path))


if __name__ == "__main__":
    main()
