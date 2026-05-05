"""Decoder-Gram validity spike — does Polygram's predicted Gram track
the actual SAE decoder geometry it claims to encode?

PR #16 (cross-encoding stability) closed the *internal* consistency
question: MPSRung1 and HEA_Rung2(depth=2) agree on which pairs cross
the kept-edge gates across three fixtures including a real GPT-2 SAE.
That note's closing caveat names the open *external* question:

    compares two encodings to each other, not either encoding to
    actual SAE behaviour on text.

This spike runs the smallest test that can falsify the load-bearing
assumption underlying every downstream prediction
(``BatchExperiment.cancellation_efficiency``, the
``build_separation_graph`` "must-separate" flagging, the deferred
disentanglement loop): does the encoded interference reflect real
decoder geometry?

Method (per `tech-debt-backlog/tasks.md` §4.1):

(a) For each fixture, take the projection vectors that
    `from_sae_lens` consumed.
(b) Compute the *raw* decoder squared-cosine Gram

        G_real[i,j] = (W_dec[:,i] · W_dec[:,j])^2
                      / (||W_dec[:,i]||^2 · ||W_dec[:,j]||^2)

    directly — no Polygram involved.
(c) Build a `Dictionary` via `from_sae_lens` (default knobs, φ=0)
    and compute Polygram's analytic Gram squared
    ``G_polygram[i,j] = |Dictionary.gram()[i,j]|^2`` under both
    `MPSRung1` and `HEA_Rung2(depth=2)`.
(d) Report: per-pair scatter, Pearson + Spearman correlation on
    off-diagonal entries, max absolute drift, and whether
    classifications (sharing-kept / separation-kept / floor-blocked)
    computed from `G_real` agree with those computed from
    `G_polygram`.

Three outcomes shape the next move:

- Spearman > 0.8 across both encodings: encoding tracks real
  geometry. The first of three blockers in
  `docs/research/spec-disentanglement-loop.md` (gradient signal
  exists) gets meaningful evidence.
- Spearman 0.3–0.8: encoding tracks real geometry on average but
  loses fine structure. Polygram is a useful *ranker*; quantitative
  claims need per-workload calibration.
- Spearman < 0.3: encoding reads geometry the SAE doesn't have.
  Disentanglement loop blocked indefinitely; `from_sae_lens` itself
  needs rethinking before any compression-pipeline work proceeds.

The findings are interpreted in
``docs/research/decoder-gram-validity.md``.

Reproduce::

    python examples/decoder_gram_validity.py

The Real SAE fixture is auto-skipped if
``./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors``
isn't on disk; see ``docs/research/cross-encoding-stability.md`` for
the download command.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from polygram import (
    Dictionary,
    HEA_Rung2,
    load_toy_sae,
)
from polygram.analysis import FLOOR_BLOCK
from polygram.sae_import import SAEFeatureRecord, from_sae_lens

# Threshold above which a pair is flagged "high overlap" — matches
# the FLOOR_BLOCK gate that build_sharing_graph uses to drop
# unreachable pairs from the sharing edge list.
HIGH_OVERLAP_THRESHOLD = FLOOR_BLOCK  # 0.5


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation in pure numpy. Equivalent to
    ``scipy.stats.spearmanr(a, b).statistic`` for 1D inputs without
    ties; with ties this differs slightly (we use simple argsort
    ranks rather than averaged ranks). Adequate for the spike since
    real-decoder cosine values are continuous and ties are vanishing.
    """
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _real_gram(records: list[SAEFeatureRecord]) -> np.ndarray:
    """Squared cosine Gram of decoder columns, ordered by record."""
    projs = np.stack([r.projection for r in records], axis=0)
    norms = np.linalg.norm(projs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = projs / norms
    return np.abs(unit @ unit.T) ** 2


def _polygram_gram(d: Dictionary) -> np.ndarray:
    return np.abs(d.gram()) ** 2


def _classify(off_diag: np.ndarray) -> np.ndarray:
    """Per-pair label by squared-overlap magnitude:
    - ``"high"``  if value >= HIGH_OVERLAP_THRESHOLD (would block a
      sharing edge in build_sharing_graph; would keep a separation
      edge above default threshold).
    - ``"low"``   otherwise.
    """
    return np.where(off_diag >= HIGH_OVERLAP_THRESHOLD, "high", "low")


def _contingency(real_label: np.ndarray, pred_label: np.ndarray) -> dict:
    """2x2 contingency: rows = real {high, low}, cols = predicted
    {high, low}. Returns counts plus a one-line miss/false-alarm
    summary.
    """
    n_pp = int(np.sum((real_label == "high") & (pred_label == "high")))
    n_nn = int(np.sum((real_label == "low") & (pred_label == "low")))
    n_miss = int(np.sum((real_label == "high") & (pred_label == "low")))
    n_fa = int(np.sum((real_label == "low") & (pred_label == "high")))
    total = n_pp + n_nn + n_miss + n_fa
    accuracy = (n_pp + n_nn) / total if total else float("nan")
    return {
        "true_positive": n_pp,
        "true_negative": n_nn,
        "miss": n_miss,
        "false_alarm": n_fa,
        "accuracy": accuracy,
    }


def _print_report(
    name: str,
    feature_names: list[str],
    g_real: np.ndarray,
    g_mps: np.ndarray,
    g_hea: np.ndarray,
) -> None:
    n = g_real.shape[0]
    iu = np.triu_indices(n, k=1)
    real_off = g_real[iu]
    mps_off = g_mps[iu]
    hea_off = g_hea[iu]

    print()
    print("=" * 78)
    print(f"FIXTURE: {name}")
    print("=" * 78)
    print(f"  features: {feature_names}")
    print(f"  pairs:    {n * (n - 1) // 2}")
    print()

    # Per-pair table.
    print(f"  {'pair':45s}  {'G_real':>7s}  {'G_mps':>7s}  {'G_hea':>7s}")
    print(f"  {'-' * 45}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    for k, (i, j) in enumerate(zip(*iu)):
        label = f"{feature_names[i]} ↔ {feature_names[j]}"
        print(
            f"  {label:45s}  {real_off[k]:>7.4f}  "
            f"{mps_off[k]:>7.4f}  {hea_off[k]:>7.4f}"
        )
    print()

    # Correlations.
    print("  correlations (off-diagonal squared overlaps):")
    print(
        f"    Pearson(G_real, G_mps)  = {_pearson(real_off, mps_off):+.4f}"
    )
    print(
        f"    Pearson(G_real, G_hea)  = {_pearson(real_off, hea_off):+.4f}"
    )
    print(
        f"    Spearman(G_real, G_mps) = {_spearman(real_off, mps_off):+.4f}"
    )
    print(
        f"    Spearman(G_real, G_hea) = {_spearman(real_off, hea_off):+.4f}"
    )
    print()

    # Drift magnitudes.
    print("  per-pair drift (squared overlap units):")
    print(
        f"    |G_mps − G_real|: "
        f"min={float(np.min(np.abs(mps_off - real_off))):.4f} "
        f"mean={float(np.mean(np.abs(mps_off - real_off))):.4f} "
        f"max={float(np.max(np.abs(mps_off - real_off))):.4f}"
    )
    print(
        f"    |G_hea − G_real|: "
        f"min={float(np.min(np.abs(hea_off - real_off))):.4f} "
        f"mean={float(np.mean(np.abs(hea_off - real_off))):.4f} "
        f"max={float(np.max(np.abs(hea_off - real_off))):.4f}"
    )
    print()

    # Classification agreement at HIGH_OVERLAP_THRESHOLD = 0.5.
    real_lbl = _classify(real_off)
    mps_lbl = _classify(mps_off)
    hea_lbl = _classify(hea_off)
    print(
        f"  classification agreement at threshold = {HIGH_OVERLAP_THRESHOLD} "
        "(high vs low squared overlap):"
    )
    for tag, pred_lbl in [("MPS", mps_lbl), ("HEA", hea_lbl)]:
        ct = _contingency(real_lbl, pred_lbl)
        print(
            f"    {tag}: TP={ct['true_positive']} TN={ct['true_negative']} "
            f"miss={ct['miss']} FA={ct['false_alarm']} "
            f"accuracy={ct['accuracy']:.2f}"
        )


def _build_toy_sae() -> tuple[str, list[SAEFeatureRecord], Dictionary, Dictionary]:
    fixture = Path("tests/fixtures/toy_sae.json")
    records_all = load_toy_sae(fixture)
    feature_ids = [0, 1, 4, 5]
    records = [records_all[i] for i in feature_ids]
    d_mps, _ = from_sae_lens(records_all, feature_ids, assign_gamma=True)
    d_mps = replace(d_mps, name="ToySAE")
    d_hea = Dictionary(
        name=d_mps.name + "Hea",
        features=d_mps.features,
        hierarchy=d_mps.hierarchy,
        encoding=HEA_Rung2(depth=2),
    )
    return "Toy SAE (tests/fixtures/toy_sae.json, ids 0,1,4,5)", records, d_mps, d_hea


def _build_real_sae() -> tuple[str, list[SAEFeatureRecord], Dictionary, Dictionary] | None:
    """Real GPT-2 SAE fixture; returns None if the checkpoint isn't
    on disk."""
    path = Path(
        "./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors"
    )
    if not path.exists():
        return None
    try:
        from polygram import load_sae_safetensors
    except ImportError:
        return None
    feature_ids = [7836, 13953, 15796, 11978]
    records_dict = load_sae_safetensors(path, feature_ids=feature_ids)
    records = [records_dict[i] for i in feature_ids]
    d_mps, _ = from_sae_lens(
        records_dict, feature_ids, assign_gamma=True, name="RealSAE",
    )
    d_hea = Dictionary(
        name=d_mps.name + "Hea",
        features=d_mps.features,
        hierarchy=d_mps.hierarchy,
        encoding=HEA_Rung2(depth=2),
    )
    return (
        "Real GPT-2 SAE (jbloom/...resid_pre, ids 7836,13953,15796,11978)",
        records,
        d_mps,
        d_hea,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-real-sae", action="store_true",
        help="skip the real-SAE fixture even if the checkpoint is present",
    )
    args = parser.parse_args(argv)

    print(__doc__.splitlines()[0])  # one-line banner
    print(f"HIGH_OVERLAP_THRESHOLD = {HIGH_OVERLAP_THRESHOLD}")

    name, records, d_mps, d_hea = _build_toy_sae()
    g_real = _real_gram(records)
    g_mps = _polygram_gram(d_mps)
    g_hea = _polygram_gram(d_hea)
    _print_report(name, [f.name for f in d_mps.features], g_real, g_mps, g_hea)

    if not args.skip_real_sae:
        result = _build_real_sae()
        if result is None:
            print()
            print("=" * 78)
            print(
                "REAL SAE: skipped "
                "(./scratch/real-sae/... not on disk or safetensors not installed)"
            )
            print("=" * 78)
        else:
            name, records, d_mps, d_hea = result
            g_real = _real_gram(records)
            g_mps = _polygram_gram(d_mps)
            g_hea = _polygram_gram(d_hea)
            _print_report(
                name, [f.name for f in d_mps.features], g_real, g_mps, g_hea,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
