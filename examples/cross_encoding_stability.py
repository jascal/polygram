"""Cross-encoding stability spike — does the rung-1 closed-form
triage agree with HEA on the same feature set?

Both `add-sharing-graph-triage` and `add-batch-experiment` ride on the
rung-1 ``(M, V, structural_floor, cancellation_gap)`` decomposition,
which is exact for ``MPSRung1`` and only-approximate for
``HEA_Rung2`` (``Cancellation.structural_floor`` raises
``NotImplementedError`` outside the canonical 2-φ rung-1 shape — see
``polygram/cancellation.py``). The open question:

    Does a feature pair classified as "good sharing candidate" or
    "must separate" under MPSRung1 stay in that bucket under
    HEA_Rung2 on the same (β, α, γ, φ) configuration?

If classifications drift across encodings on real data, the
closed-form rung-1 predictions are telling us about the encoding,
not the SAE's intrinsic geometry — which would be a load-bearing
finding before any compression-pipeline work.

This script runs the comparison on three fixtures of increasing
realism:

1. Animals (4 hand-crafted features, 2 clusters of 2) — controlled,
   known geometry.
2. Toy SAE (4 features from ``tests/fixtures/toy_sae.json``) —
   small but real-ish projection vectors.
3. Real GPT-2 SAE (``./scratch/real-sae/.../sae_weights.safetensors``
   if present, ``blocks.0.hook_resid_pre``, 4 projection-similar
   features) — the same setup that surfaced the
   ``--assign-gamma`` finding.

For each fixture, it builds two `Dictionary` instances with the same
features but different encodings (MPSRung1 vs HEA_Rung2(depth=2)),
runs `triage_dictionary` and the sharing/separation graph builders
on both, and prints a side-by-side comparison.

The findings are interpreted in
``docs/research/cross-encoding-stability.md``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from polygram import (
    Dictionary,
    Feature,
    HEA_Rung2,
    MPSRung1,
    load_sae_safetensors,
    load_toy_sae,
)
from polygram.analysis import (
    FLOOR_BLOCK,
    build_separation_graph,
    build_sharing_graph,
    triage_dictionary,
)
from polygram.sae_import import from_sae_lens


def build_animals() -> Dictionary:
    return Dictionary(
        name="Animals",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.50, alpha=0.05, gamma=0.02),
            Feature("dog_beagle", "dogs", beta=-0.48, alpha=0.04, gamma=0.03),
            Feature("bird_hawk", "birds", beta=0.50, alpha=-0.04, gamma=0.02),
            Feature("bird_sparrow", "birds", beta=0.52, alpha=-0.03, gamma=0.01),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=MPSRung1(),
    )


def build_toy_sae() -> Dictionary:
    fixture = Path("tests/fixtures/toy_sae.json")
    records = load_toy_sae(fixture)
    dictionary, _ = from_sae_lens(records, [0, 1, 4, 5], assign_gamma=True)
    return replace(dictionary, name="ToySAE")


def build_real_sae() -> Dictionary | None:
    """Return None if the real SAE checkpoint isn't on disk — keeps
    the script runnable without the optional download."""
    path = Path(
        "./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors"
    )
    if not path.exists():
        return None
    feature_ids = [7836, 13953, 15796, 11978]
    records = load_sae_safetensors(path, feature_ids=feature_ids)
    dictionary, _ = from_sae_lens(
        records, feature_ids, assign_gamma=True, name="RealSAE",
    )
    return dictionary


def reencode_as_hea(dictionary: Dictionary, depth: int = 2) -> Dictionary:
    """Return a copy with HEA_Rung2 encoding, same features verbatim.

    Each feature's `(α, β, γ, φ)` is preserved; HEA's
    ``_default_hea_theta`` synthesizes a θ tensor from those knobs,
    matching the documented spike layout in
    ``polygram/dictionary.py``.
    """
    return Dictionary(
        name=dictionary.name + "Hea",
        features=dictionary.features,
        hierarchy=dictionary.hierarchy,
        encoding=HEA_Rung2(depth=depth),
    )


def compare(name: str, d_mps: Dictionary) -> None:
    d_hea = reencode_as_hea(d_mps)
    pred_mps = triage_dictionary(d_mps)
    pred_hea = triage_dictionary(d_hea)

    print()
    print("=" * 78)
    print(f"FIXTURE: {name}")
    print("=" * 78)
    print(f"  features: {[f.name for f in d_mps.features]}")
    print(
        f"  hierarchy: { {c: len(m) for c, m in d_mps.hierarchy.items()} }"
    )
    print()

    # Per-pair comparison.
    print(f"  {'pair':40s}  {'enc':3s}  {'curr':>7s}  {'floor':>7s}  "
          f"{'gap':>7s}  {'V':>8s}")
    print(f"  {'-' * 40}  {'---'}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
    delta_floors: list[float] = []
    delta_currents: list[float] = []
    for p_mps, p_hea in zip(pred_mps.pairs, pred_hea.pairs):
        assert p_mps.feature_a == p_hea.feature_a
        assert p_mps.feature_b == p_hea.feature_b
        pair_label = f"{p_mps.feature_a} ↔ {p_mps.feature_b}"
        print(f"  {pair_label:40s}  MPS  {p_mps.current_overlap:>7.4f}  "
              f"{p_mps.structural_floor:>7.4f}  {p_mps.cancellation_gap:>7.4f}  "
              f"{p_mps.V:+8.4f}")
        print(f"  {'':40s}  HEA  {p_hea.current_overlap:>7.4f}  "
              f"{p_hea.structural_floor:>7.4f}  {p_hea.cancellation_gap:>7.4f}  "
              f"{p_hea.V:+8.4f}")
        delta_currents.append(p_hea.current_overlap - p_mps.current_overlap)
        delta_floors.append(p_hea.structural_floor - p_mps.structural_floor)

    print()
    print(f"  per-pair Δ(current) — HEA minus MPS: "
          f"min={min(delta_currents):+.4f} "
          f"mean={float(np.mean(delta_currents)):+.4f} "
          f"max={max(delta_currents):+.4f}")
    print(f"  per-pair Δ(floor)   — HEA minus MPS: "
          f"min={min(delta_floors):+.4f} "
          f"mean={float(np.mean(delta_floors)):+.4f} "
          f"max={max(delta_floors):+.4f}")
    print()

    # Edge-set agreement: build sharing + separation graphs at the
    # default thresholds and compare kept-edge sets.
    for kind, builder, kw in [
        ("sharing", build_sharing_graph, {"allow_cross_cluster": True}),
        ("separation", build_separation_graph, {"include_within_cluster": True}),
    ]:
        g_mps = builder(pred_mps, threshold=0.0, **kw)
        g_hea = builder(pred_hea, threshold=0.0, **kw)
        edges_mps = {(e.source, e.target) for e in g_mps.edges}
        edges_hea = {(e.source, e.target) for e in g_hea.edges}
        common = edges_mps & edges_hea
        only_mps = edges_mps - edges_hea
        only_hea = edges_hea - edges_mps
        print(f"  {kind} graph (threshold=0.0): "
              f"MPS={len(edges_mps)} edges, HEA={len(edges_hea)} edges, "
              f"common={len(common)}")
        if only_mps:
            print(f"    only MPS: {sorted(only_mps)}")
        if only_hea:
            print(f"    only HEA: {sorted(only_hea)}")

        # Now test with the actual default thresholds (sharing=0.5,
        # separation=0.2) — this is what would actually drive
        # downstream BatchExperiment.
        default_th = 0.5 if kind == "sharing" else 0.2
        g_mps = builder(pred_mps, threshold=default_th, **kw)
        g_hea = builder(pred_hea, threshold=default_th, **kw)
        edges_mps = {(e.source, e.target) for e in g_mps.edges}
        edges_hea = {(e.source, e.target) for e in g_hea.edges}
        print(f"  {kind} graph (threshold={default_th}, default): "
              f"MPS={len(edges_mps)} edges, HEA={len(edges_hea)} edges, "
              f"common={len(edges_mps & edges_hea)}")

    # Suitability scores — single scalars; do they agree directionally?
    print()
    print(f"  encoding_suitability_score: "
          f"MPS={pred_mps.encoding_suitability_score:.4f} "
          f"HEA={pred_hea.encoding_suitability_score:.4f}")
    print(f"  FLOOR_BLOCK constant: {FLOOR_BLOCK}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-real-sae", action="store_true",
        help="skip the real-SAE fixture even if the checkpoint is present",
    )
    args = parser.parse_args(argv)

    compare("Animals (controlled, hand-crafted)", build_animals())
    compare("Toy SAE (tests/fixtures/toy_sae.json, ids 0,1,4,5, assign_gamma)",
            build_toy_sae())
    if not args.skip_real_sae:
        real = build_real_sae()
        if real is None:
            print()
            print("=" * 78)
            print("REAL SAE: skipped (./scratch/real-sae/... not on disk)")
            print("=" * 78)
        else:
            compare(
                "Real GPT-2 SAE (jbloom/...resid_pre, ids 7836,13953,15796,11978)",
                real,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
