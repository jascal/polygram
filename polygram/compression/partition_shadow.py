"""Emit a partition-block-ids shadow checkpoint for sae-forge.

Bridges polygram's clustering-based partition (BlockSpec / heaviness
scores from DecoderGeometryConfirmer) to sae-forge's basis-selection
contract. sae-forge's `_run_capability_cell` consumes an optional
``partition_block_ids`` int tensor in the SAE state dict; this module
generates that tensor from polygram's clustering machinery.

Output shape:

  - encoder.weight, encoder.bias, decoder.weight, decoder.bias —
    copied verbatim from the source SAE.
  - partition_block_ids: int64 tensor of shape (n_features,), with
    per-feature tier id assigned by heaviness-quantile clustering
    (deterministic; same SAE + same hyperparams → same partition).

Two partition strategies:

  - ``heaviness_quantile``: split features into ``n_tiers`` by
    heaviness quantile (heaviness = decoder_norm² × (1 + pair_count)
    where pair_count comes from polygram's
    DecoderGeometryConfirmer). Default; matches sae-forge's
    partition_q4 / partition_q8 shape.
  - ``top_k_heavy``: top ``k_heavy`` features go to tier 0 (heavy),
    rest to tier 1 (tail). Matches the original Wave C 2-block
    partition from polygram/runs/real_partition_experiment.py.

Why heaviness instead of just decoder_norm? Pair-count from
DecoderGeometryConfirmer adds semantic structure: features that
co-activate cluster together regardless of absolute norm. A
high-norm feature that doesn't pair-cluster with anything else gets
treated like a singleton; a mid-norm feature that pairs with many
others gets clustered. This is the load-bearing distinction vs the
row-norm-quantile heuristic in bio-sae's
materialize_partition_checkpoint.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


def _load_sae_state_dict(path: Path) -> dict:
    """Load an SAE state dict from either safetensors or PyTorch
    pickle format. Returns a dict with numpy arrays.

    bio-sae's SAEs are saved as PyTorch ``.pt`` files via
    ``torch.save``; sae-lens / polygram convention is ``.safetensors``.
    Try both; raise if neither parses.
    """
    suffix = path.suffix.lower()
    # Try safetensors first if extension suggests it; else try torch.
    if suffix in (".safetensors", ".st"):
        from safetensors.numpy import load_file as load_numpy
        return load_numpy(str(path))
    # PyTorch .pt / .pth / generic — use torch.load and convert.
    import torch
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


@dataclass(frozen=True)
class PartitionShadowResult:
    """Outcome of a partition-shadow emit run."""

    output_path: Path
    n_features: int
    n_tiers: int
    strategy: str
    tier_sizes: list[int]
    tier_heaviness_medians: list[float]
    n_confirmed_pairs: int
    threshold: float


def compute_heaviness_scores(
    sae_checkpoint: Path,
    *,
    threshold: float = 0.5,
    confirmer_thresholds_fallback: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1),
) -> tuple[np.ndarray, int, float]:
    """Compute per-feature heaviness scores via DecoderGeometryConfirmer.

    Returns ``(heaviness, n_confirmed_pairs, threshold_used)``.

    The confirmer threshold falls back from ``threshold`` through
    ``confirmer_thresholds_fallback`` until at least one confirmed
    pair is found. If no pairs are found at the lowest threshold,
    raises ValueError.

    heaviness[fid] = decoder_norm² × (1 + pair_count). The +1 ensures
    isolated high-norm features still get a score proportional to
    their norm; the multiplier amplifies features that participate in
    confirmed-pair clusters.
    """
    from polygram import SAEFeatureRecord
    from polygram.confirmation.decoder_geometry import DecoderGeometryConfirmer

    sae_state = _load_sae_state_dict(sae_checkpoint)
    # Support both W_dec (sae-lens / polygram convention) and
    # decoder.weight (PyTorch nn.Linear convention).
    if "W_dec" in sae_state:
        W_dec = np.asarray(sae_state["W_dec"], dtype=np.float64)
    elif "decoder.weight" in sae_state:
        # decoder.weight has shape (d_model, n_features); transpose
        # to (n_features, d_model) for row-norm computation.
        W_dec = np.asarray(sae_state["decoder.weight"], dtype=np.float64).T
    else:
        raise ValueError(
            f"compute_heaviness_scores: source SAE at "
            f"{sae_checkpoint} lacks 'W_dec' or 'decoder.weight' key; "
            f"got {list(sae_state.keys())!r}"
        )
    n_features = W_dec.shape[0]
    decoder_norms = np.linalg.norm(W_dec, axis=1)

    # Build SAEFeatureRecord dict for the confirmer.
    records: dict[int, SAEFeatureRecord] = {}
    for fid in range(n_features):
        records[fid] = SAEFeatureRecord(
            feature_id=fid,
            name=f"feat_{fid}",
            projection=W_dec[fid].astype(np.float64),
        )
    feature_ids = list(range(n_features))

    # Try the requested threshold first; fall back through
    # progressively-looser thresholds until at least one pair is
    # confirmed. Matches the pattern in
    # polygram/runs/real_partition_experiment.py.
    thresholds_to_try = (threshold,) + confirmer_thresholds_fallback
    confirmer = None
    vr = None
    threshold_used = threshold
    for thr in thresholds_to_try:
        confirmer = DecoderGeometryConfirmer(
            records=records,
            sae_checkpoint=sae_checkpoint,
            feature_ids=feature_ids,
            threshold=thr,
        )
        vr = confirmer.run()
        threshold_used = thr
        if len(vr.confirmed) > 0:
            break
    if vr is None or len(vr.confirmed) == 0:
        raise ValueError(
            f"compute_heaviness_scores: no confirmed pairs at any "
            f"threshold {thresholds_to_try!r} for SAE at "
            f"{sae_checkpoint}. Decoder geometry may be degenerate."
        )

    pair_count = np.zeros(n_features, dtype=int)
    for (i, j) in vr.confirmed:
        pair_count[i] += 1
        pair_count[j] += 1

    heaviness = (decoder_norms ** 2) * (1 + pair_count.astype(np.float64))
    return heaviness, int(len(vr.confirmed)), threshold_used


def heaviness_quantile_partition(
    heaviness: np.ndarray, *, n_tiers: int,
) -> np.ndarray:
    """Assign each feature to a tier by heaviness quantile.

    Tier 0 = top quantile (highest heaviness — most load-bearing).
    Tier n_tiers-1 = bottom quantile.

    Returns ``(n_features,)`` int64 tensor.
    """
    n = heaviness.shape[0]
    # Sort indices by descending heaviness so tier 0 captures the top.
    order = np.argsort(-heaviness, kind="stable")
    tier_size = n // n_tiers
    block_ids = np.empty(n, dtype=np.int64)
    for tier in range(n_tiers):
        start = tier * tier_size
        end = (tier + 1) * tier_size if tier < n_tiers - 1 else n
        block_ids[order[start:end]] = tier
    return block_ids


def top_k_heavy_partition(
    heaviness: np.ndarray, *, k_heavy: int,
) -> np.ndarray:
    """Two-tier partition: top-K features go to tier 0 (heavy),
    rest go to tier 1 (tail). Mirrors Wave C's
    polygram/runs/real_partition_experiment.py setup."""
    n = heaviness.shape[0]
    if k_heavy >= n:
        raise ValueError(
            f"top_k_heavy_partition: k_heavy ({k_heavy}) must be < "
            f"n_features ({n})"
        )
    block_ids = np.ones(n, dtype=np.int64)  # default tier 1 (tail)
    top_ids = np.argsort(-heaviness, kind="stable")[:k_heavy]
    block_ids[top_ids] = 0  # tier 0 (heavy)
    return block_ids


def emit_partition_shadow(
    sae_checkpoint: Path,
    output_path: Path,
    *,
    strategy: Literal["heaviness_quantile", "top_k_heavy"] = "heaviness_quantile",
    n_tiers: int = 4,
    k_heavy: int = 64,
    threshold: float = 0.5,
) -> PartitionShadowResult:
    """Top-level entry. Reads source SAE, computes clustering-based
    partition_block_ids, writes a shadow safetensors file in sae-
    forge's expected format.

    For ``strategy="heaviness_quantile"``: partitions into ``n_tiers``
    by heaviness quantile.

    For ``strategy="top_k_heavy"``: 2-tier partition with top
    ``k_heavy`` features as heavy.
    """
    import torch

    if not sae_checkpoint.exists():
        raise FileNotFoundError(
            f"emit_partition_shadow: source SAE not found: "
            f"{sae_checkpoint}"
        )

    heaviness, n_confirmed_pairs, threshold_used = compute_heaviness_scores(
        sae_checkpoint, threshold=threshold,
    )
    n_features = heaviness.shape[0]

    if strategy == "heaviness_quantile":
        block_ids = heaviness_quantile_partition(heaviness, n_tiers=n_tiers)
    elif strategy == "top_k_heavy":
        block_ids = top_k_heavy_partition(heaviness, k_heavy=k_heavy)
        n_tiers = 2
    else:
        raise ValueError(
            f"emit_partition_shadow: unknown strategy {strategy!r}; "
            f"expected 'heaviness_quantile' or 'top_k_heavy'"
        )

    # Per-tier diagnostics.
    tier_sizes = [
        int((block_ids == t).sum()) for t in range(n_tiers)
    ]
    tier_heaviness_medians = [
        float(np.median(heaviness[block_ids == t])) if (block_ids == t).any() else float("nan")
        for t in range(n_tiers)
    ]

    # Load source SAE state via the same format-agnostic loader
    # used in compute_heaviness_scores. Write the shadow in the
    # SAME format as the source (PyTorch .pt for bio-sae's SAEs;
    # safetensors otherwise). sae-forge's _run_capability_cell
    # uses torch.load with weights_only=True so PyTorch .pt is
    # the canonical format for shadow checkpoints that ship into
    # sae-forge.
    src_state_np = _load_sae_state_dict(sae_checkpoint)
    out_state: dict[str, "torch.Tensor"] = {}
    for k, v in src_state_np.items():
        out_state[k] = torch.from_numpy(np.asarray(v).copy())
    out_state["partition_block_ids"] = torch.from_numpy(block_ids).long()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write as torch .pt so sae-forge's torch.load(weights_only=True)
    # consumes it directly. Matches bio-sae's
    # materialize_partition_checkpoint.py output format.
    torch.save(out_state, str(output_path))

    # Write a JSON manifest alongside the shadow safetensors. Same
    # shape as bio-sae's existing
    # materialize_partition_checkpoint.manifest.json for cross-
    # comparison.
    manifest = {
        "source_sae": str(sae_checkpoint),
        "partition_strategy": strategy,
        "n_features": int(n_features),
        "n_tiers": int(n_tiers),
        "n_confirmed_pairs": int(n_confirmed_pairs),
        "decoder_geometry_threshold": float(threshold_used),
        "tier_sizes": {f"tier_{t}": s for t, s in enumerate(tier_sizes)},
        "tier_heaviness_medians": {
            f"tier_{t}": v for t, v in enumerate(tier_heaviness_medians)
        },
        "note": (
            "Polygram-generated partition shadow. Block ids derived "
            "from DecoderGeometryConfirmer + heaviness scoring "
            "(decoder_norm² × (1 + pair_count)). Drop into sae-forge's "
            "sweep_pareto_capability via encodings=[(label, "
            f"'{output_path}')] to consume."
        ),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return PartitionShadowResult(
        output_path=output_path,
        n_features=n_features,
        n_tiers=n_tiers,
        strategy=strategy,
        tier_sizes=tier_sizes,
        tier_heaviness_medians=tier_heaviness_medians,
        n_confirmed_pairs=n_confirmed_pairs,
        threshold=threshold_used,
    )
