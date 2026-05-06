## Context

`Compressor` currently has one representative-selection policy (highest total `n_fires`, tiebreak lowest fid) and one compression strategy (`zero` — silence all non-representative features). Both are adequate for proof-of-concept but discard useful signal:

- `n_fires` alone ignores decoder norm: a feature that fires a lot but has a tiny norm contributes little to reconstruction; it should yield to a lower-frequency feature with a larger, median-aligned norm.
- Hard zeroing discards every redundant feature's norm mass; for mid-sized clusters this measurably inflates reconstruction error relative to a norm-preserving merge.
- The report contains no scale diagnostics, making it impossible to know how much norm mass was lost.

`ValidationReport.pairs` already carries `kl_ablate_i/j` and `n_fires_i/j`. W_dec norms are trivially computable from the checkpoint. The data is there; we just need to use it.

## Goals / Non-Goals

**Goals:**
- `rep_selection="scale_aware"`: score each cluster candidate by (a) decoder norm closeness to cluster median, (b) KL-ablation functional importance, (c) log activation frequency. Graceful NaN fallback for geometry-only confirmer paths.
- `strategy="merge"`: new `apply_merge` in `strategies/merge.py` — rescales the surviving W_dec row to the freq-weighted (or simple-mean) average of the cluster's norms. Encoder/bias zeroing for non-representatives is unchanged.
- `merge_mode` param: `"freq_weighted"` (default) | `"simple_mean"`.
- Per-cluster scale stats in `CompressionReport`: `cluster_norm_mean`, `cluster_norm_std`, `merged_norm` (`None` for zero strategy), `scale_compression_ratio` aggregate across all clusters.
- All defaults unchanged (`strategy="zero"`, `rep_selection="n_fires"`).

**Non-Goals:**
- Vector direction averaging (merging W_dec directions, not just norms).
- Hybrid strategies (zero weak / merge strong) — deferred.
- Modifying `EpochCompressor` or `Regrower` in this change.
- CLI surface changes.

## Decisions

**Decision 1 — Load W_dec in `plan()` for scale-aware path (not `apply()`).**

The representative must be chosen before `apply()` so that overrides can be validated. Loading just W_dec in `plan()` when `rep_selection="scale_aware"` is acceptable; W_dec for a 4k-feature SAE at float32 is ~36 MB. Alternative (moving rep selection to `apply()`) would require refactoring the plan/apply boundary and break the `representatives` override validation. Chosen: load in plan(), gated by the `scale_aware` flag.

**Decision 2 — Scale-aware scoring is torch-free numpy.**

The compressor is already torch-free; keeping it that way avoids a heavy dependency and lets the plan step run on machines without CUDA. Numpy is sufficient for the norm/median/log arithmetic.

**Decision 3 — Scoring weights follow the advisor spec (0.4 / 0.4 / 0.2) but are not exposed as params yet.**

Exposing weights adds API surface with no empirical backing yet. They can be promoted to a `ScoreWeights` dataclass in a follow-on once we have ablation results on more SAEs.

**Decision 4 — `kl_ablate` fallback: if all values are NaN in a cluster, fall back to `n_fires` ranking.**

Geometry-only confirmer sets `kl_ablate_i/j = NaN`. Rather than silently producing NaN scores, detect this per-cluster and revert to the existing `n_fires` logic. This makes `scale_aware` safe to use without knowing how the upstream confirmer was configured.

**Decision 5 — `merge` strategy only rescales the representative's W_dec row; encoder weights for non-representatives are still zeroed.**

Merging encoder columns would risk activating the surviving feature at sites the original representative didn't fire. Restricting to W_dec norm rescaling is conservative and sufficient to preserve norm mass in the reconstruction direction. Encoder merging is a future option.

**Decision 6 — Scale stats are added to `ClusterPlan` (source norm stats) and `CompressionReport` (merged_norm, ratio).**

`ClusterPlan` is populated in `plan()` before any checkpoint writes, so norm stats computed then are stable. `merged_norm` is write-time (only known after merge arithmetic), so it lives in the report-level cluster entry. To avoid a parallel cluster structure in the report, add `merged_norm` as an optional field on `ClusterPlan` itself (populated by `apply()` for merge strategy, `None` for zero).

**Decision 7 — `scale_compression_ratio` = sum(surviving norms after) / sum(all norms before) over all cluster members.**

A value of 1.0 means perfect norm preservation (pure merge); 0.0 would mean all norms discarded (degenerate zero). This ratio is aggregate across the whole plan; per-cluster ratios are recoverable from `ClusterPlan.cluster_norm_mean` + `merged_norm`.

## Risks / Trade-offs

- **Norm loading in plan() increases its cost** for scale_aware path → Mitigation: document the trade-off; keep it opt-in. plan() remains cheap for the default `n_fires` path.
- **kl_ablate NaN fallback silently changes behaviour** if a user enables scale_aware on geometry-only output → Mitigation: emit a warning when the fallback triggers.
- **merge may produce W_dec rows with unusual norms** if cluster norms are highly skewed → Mitigation: `scale_compression_ratio` in the report makes this visible; users can inspect before committing.
- **ClusterPlan is frozen; adding merged_norm changes its fields** → No existing serialized ClusterPlan objects are expected to be loaded from disk (CompressionReport.from_json reconstructs them); treat as additive.

## Open Questions

- Should `scale_compression_ratio` exclude singleton features (non-clustered, norm preserved exactly)? Currently only cluster members count.
- Is `"ablation_first"` and `"frequency"` rep_selection needed now, or is `"scale_aware"` sufficient for the advisor's use case?
