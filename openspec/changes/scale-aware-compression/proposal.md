## Why

The current `Compressor` picks cluster representatives by raw activation count and always zeros redundant features — discarding their decoder norm entirely. This loses information about each feature's relative contribution to reconstruction, and is less principled than preserving total "feature mass" across a cluster.

## What Changes

- Add `rep_selection` parameter to `Compressor`: `"n_fires"` (current default, no change) and `"scale_aware"` (norm-closeness to cluster median + ablation importance + log activation frequency).
- Add `strategy="merge"` alongside `"zero"`: instead of zeroing a redundant feature's decoder row, freq-weight-average the cluster's norms and rescale the surviving W_dec row accordingly.
- Add `merge_mode` parameter: `"freq_weighted"` (default) | `"simple_mean"`.
- Extend `CompressionReport` with per-cluster scale statistics: `cluster_norm_mean`, `cluster_norm_std`, `merged_norm` (or `None` for zero strategy), and a roll-up `scale_compression_ratio`.

## Capabilities

### New Capabilities

- `scale-aware-rep-selection`: Scoring function that combines decoder norm closeness, KL-ablation functional importance, and log firing frequency to choose the cluster representative; falls back gracefully to `n_fires` when `kl_ablate` is NaN (geometry-only confirmer path).
- `merge-compression-strategy`: `apply_merge` strategy in `strategies/merge.py` that freq-weighted-averages cluster norms and rescales the surviving W_dec row to preserve total norm mass; `merge_mode` selects the averaging formula.
- `compression-scale-stats`: Cluster-level norm statistics (`cluster_norm_mean`, `cluster_norm_std`, `merged_norm`, `scale_compression_ratio`) surfaced in `CompressionReport` and `ClusterPlan`.

### Modified Capabilities

- `sae`: `Compressor` gains `rep_selection` and `merge_mode` fields; `CompressionReport` gains scale-stat fields. JSON schema for the report changes (additive, not breaking).

## Impact

- `polygram/compression/compressor.py` — new params, norm loading in `plan()` for scale-aware path, updated `_pick_representative`.
- `polygram/compression/strategies/merge.py` — new file.
- `polygram/compression/report.py` — `ClusterPlan` and `CompressionReport` extended with scale fields.
- `tests/test_compression*.py` — new test cases for `scale_aware` rep selection, `merge` strategy, and scale stats.
- No breaking changes; `strategy="zero"` and `rep_selection="n_fires"` remain the defaults.
