## 1. Extend report dataclasses

- [x] 1.1 Add `cluster_norm_mean: float`, `cluster_norm_std: float`, `merged_norm: float | None` to `ClusterPlan` in `report.py`; update `_cluster_to_dict` / `_cluster_from_dict` for JSON round-trip
- [x] 1.2 Add `scale_compression_ratio: float` to `CompressionReport` in `report.py`; update `_serialize` and `from_json`

## 2. Scale-aware representative selection

- [x] 2.1 Add `rep_selection: str = "n_fires"` param to `Compressor`; validate in `__post_init__` (`{"n_fires", "scale_aware"}`)
- [x] 2.2 Implement `_score_scale_aware(cluster, pair_lookup, w_dec)` helper in `compressor.py`: norm proximity (0.4), normalised KL-ablation (0.4), log n_fires (0.2); NaN fallback with `UserWarning` when all kl_ablate are NaN
- [x] 2.3 Wire `_score_scale_aware` into `_pick_representative`; load W_dec from checkpoint in `_build_plan` when `rep_selection="scale_aware"` (one load, passed down)
- [x] 2.4 Write unit tests for `scale_aware` rep selection: median-norm preference, NaN fallback fires a warning, `n_fires` default unchanged

## 3. Merge compression strategy

- [x] 3.1 Create `polygram/compression/strategies/merge.py` with `apply_merge(state_dict, plan, merge_mode)` — freq-weighted and simple-mean paths; zero-norm guard; populates `ClusterPlan.merged_norm`
- [x] 3.2 Add `merge_mode: str = "freq_weighted"` param to `Compressor`; validate (`{"freq_weighted", "simple_mean"}`); update `_SUPPORTED_STRATEGIES` to include `"merge"`; wire `_dispatch_strategy`
- [x] 3.3 Write unit tests for `apply_merge`: freq_weighted arithmetic, simple_mean arithmetic, non-representative rows zeroed, zero-norm representative guard, merge_mode ignored for zero strategy

## 4. Scale stats in apply()

- [x] 4.1 Compute `cluster_norm_mean`, `cluster_norm_std`, `merged_norm` per cluster during `apply()` and patch them onto the `ClusterPlan` objects in the result
- [x] 4.2 Compute `scale_compression_ratio` in `apply()` and include in `CompressionReport`
- [x] 4.3 Write unit tests: zero strategy `merged_norm` is None, merge strategy `merged_norm` is positive, `scale_compression_ratio` < 1 for zero strategy, ratio ≈ 1 for merge+simple_mean

## 5. Integration and regression

- [x] 5.1 Verify existing `test_compression*.py` tests still pass (no regressions from dataclass changes)
- [x] 5.2 Add an end-to-end integration test: synthetic 4-feature SAE, two confirmed pairs, run `strategy="merge"` + `rep_selection="scale_aware"`, assert rep chosen correctly and output W_dec row has the expected merged norm
