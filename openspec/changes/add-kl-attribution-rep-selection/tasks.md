## Phase 1 — Constant + config validation

### 1. `_SUPPORTED_REP_SELECTIONS` extension

- [ ] 1.1 In `polygram/compression/compressor.py`, extend the module-level `_SUPPORTED_REP_SELECTIONS` tuple from `("n_fires", "scale_aware")` to `("n_fires", "scale_aware", "kl_attribution")`.
- [ ] 1.2 In `polygram/config.py`, extend the matching constant referenced by `CompressionConfig.__post_init__` to allow `"kl_attribution"`. (If polygram already centralises the constant, just update the one location.)
- [ ] 1.3 No test changes needed for the constant itself; coverage comes from the rep-selection tests below.
- [ ] 1.4 Extend the `CompressionConfig.rep_selection` docstring (in `polygram/config.py`) with a "when to use" guidance block. Required content: (a) `kl_attribution` requires a behaviourally-rich `ValidationReport` — pairs emitted by `DecoderGeometryConfirmer` / `ClusterConfirmer` have NaN `kl_ablate_*` and an all-NaN cluster raises `ValueError`; (b) the choice between `kl_attribution` and `scale_aware` is most meaningful in structurally-feasible forge regimes (sae-forge's `quality_tier in {"good", "saturated"}` rows from `add-forge-quality-diagnostics`); in degenerate regimes the rep choice is curiosity-level noise; (c) the default remains `scale_aware` until empirical evidence on real forge sweeps establishes a Pareto-dominant choice. Mirror this guidance into any `polygram compress` CLI help text that documents `--rep-selection` once the impl exposes that flag.

## Phase 2 — Algorithm

### 2. `_score_kl_attribution` helper

- [ ] 2.1 Add a private function `_score_kl_attribution(cluster: set[int], pair_lookup: dict[tuple[int, int], CandidatePair], w_dec_cache: np.ndarray | None) -> int` in `polygram/compression/compressor.py`. Returns the chosen rep feature id.
- [ ] 2.2 The function SHALL:
  - For each `f ∈ cluster`, collect the per-pair `kl_ablate_f` values: for every `(i, j)` pair in `pair_lookup` where `i == f or j == f`, take `pair.kl_ablate_i if pair.i == f else pair.kl_ablate_j`. Aggregate as the arithmetic mean over non-NaN values.
  - Maintain a parallel `n_fires_total(f)` count for tiebreaking and per-feature NaN fallback.
  - If a feature's mean `kl_ablate` is NaN AND there is at least one non-NaN value elsewhere in the cluster, fall back to `_score_scale_aware` for that single feature only (so it competes on a comparable axis with the KL-scored features). Requires `w_dec_cache`.
  - If every feature in the cluster has all-NaN `kl_ablate`, raise `ValueError("Compressor: rep_selection='kl_attribution' requires behavioural confirmation; this cluster's kl_ablate values are all NaN (likely came through DecoderGeometryConfirmer or ClusterConfirmer). Use rep_selection='scale_aware' or 'n_fires' for geometry-only reports.")`.
- [ ] 2.3 Pick rep via deterministic tiebreak: `min(cluster, key=lambda f: (-score(f), -n_fires_total(f), f))`. Higher `score` wins; ties broken by higher `n_fires_total`, then lower feature_id.

### 3. `_pick_representative` dispatch

- [ ] 3.1 In `Compressor._pick_representative`, add a branch: `if self.rep_selection == "kl_attribution": return _score_kl_attribution(cluster, pair_lookup, self._cached_w_dec)`. Branch is mutually exclusive with the existing `scale_aware` and the default `n_fires` paths.
- [ ] 3.2 Ensure `self._cached_w_dec` is loaded eagerly for `kl_attribution` callers (currently it's loaded lazily only when `rep_selection == "scale_aware"`). Extend the `if self.rep_selection == "scale_aware"` lazy-load check in `_build_plan` to also trigger when `rep_selection == "kl_attribution"`, so the fallback path has `w_dec` available.
- [ ] 3.3 Document the new branch in `Compressor.__doc__` alongside the existing rep_selection options.

## Phase 3 — Tests

### 4. Per-feature scoring

- [ ] 4.1 `tests/test_compression_rep_selection.py::test_kl_attribution_picks_highest_kl_ablate`: build a synthetic 3-feature cluster with `kl_ablate` values [0.1, 0.5, 0.2]; assert rep is feature 1 (the one with 0.5).
- [ ] 4.2 `test_kl_attribution_tiebreaks_on_n_fires`: two features tied on `kl_ablate`; assert the one with higher `n_fires_total` wins.
- [ ] 4.3 `test_kl_attribution_tiebreaks_on_feature_id`: two features tied on both `kl_ablate` and `n_fires_total`; assert the lower feature_id wins.

### 5. NaN handling

- [ ] 5.1 `test_kl_attribution_per_feature_nan_fallback_scale_aware`: 3-feature cluster where feature 1 has NaN `kl_ablate` but features 0 and 2 have finite values; assert feature 1 is scored via `_score_scale_aware` for that single feature and the final rep is whichever of {0, 1, 2} has the highest combined score. (Construct the fixture so the fallback feature's `scale_aware` score is the largest, proving fallback actually changes the outcome.)
- [ ] 5.2 `test_kl_attribution_all_nan_cluster_raises`: 3-feature cluster where every member's `kl_ablate` is NaN; assert `_score_kl_attribution` raises `ValueError` whose message names `kl_attribution`, `behavioural confirmation`, and `DecoderGeometryConfirmer`.
- [ ] 5.3 `test_kl_attribution_mean_robust_to_per_pair_noise`: a feature appearing in 3 pairs with `kl_ablate` values [0.3, 0.31, 0.29] should score as 0.30 (mean), not 0.31 (max) or 0.29 (min); confirms aggregation rule.

### 6. CompressionConfig surface

- [ ] 6.1 `test_compression_config_accepts_kl_attribution`: `CompressionConfig(rep_selection="kl_attribution")` constructs successfully.
- [ ] 6.2 `test_compression_config_rejects_unknown_rep_selection`: `CompressionConfig(rep_selection="bogus")` raises `ValueError` listing all three supported values.
- [ ] 6.3 `test_compression_config_to_dict_round_trip_kl_attribution`: round-trips via `to_dict` / `from_dict`.

### 7. Byte-identity preservation

- [ ] 7.1 `test_default_rep_selection_byte_identical`: `Compressor(report, ckpt)` with no `rep_selection` kwarg AND no `config` produces a `CompressionPlan` that is byte-identical (via `CompressionPlan.to_dict()`) to the pre-change reference for the existing toy fixture.
- [ ] 7.2 `test_explicit_scale_aware_byte_identical`: `Compressor(report, ckpt, rep_selection="scale_aware")` byte-identical to the pre-change reference.
- [ ] 7.3 `test_explicit_n_fires_byte_identical`: `Compressor(report, ckpt, rep_selection="n_fires")` byte-identical to the pre-change reference.

### 8. End-to-end smoke

- [ ] 8.1 `test_kl_attribution_end_to_end`: synthetic SAE + synthetic `ValidationReport` with non-degenerate `kl_ablate` values across pairs; `Compressor(rep_selection="kl_attribution").plan().apply()` produces a reasonable compressed checkpoint; assert the rep selected for each cluster has the maximum `kl_ablate` within that cluster.

## Phase 4 — Spec + Release

### 9. Spec deltas

- [ ] 9.1 Author `openspec/changes/add-kl-attribution-rep-selection/specs/tuning-config/spec.md` MODIFIED requirement that extends the `rep_selection` supported value list to include `"kl_attribution"`.
- [ ] 9.2 Author `openspec/changes/add-kl-attribution-rep-selection/specs/recon-aware-rep-selection/spec.md` (NEW capability) defining the algorithm, NaN handling, tiebreak rule, and the relationship to `CandidatePair.kl_ablate_*`.

### 10. Validation + release

- [ ] 10.1 `openspec validate add-kl-attribution-rep-selection --strict` is green.
- [ ] 10.2 Full `pytest` suite passes; new tests at least cover §4 through §8.
- [ ] 10.3 `ruff check` clean on touched files.
- [ ] 10.4 Bump `polygram.__version__` to `0.5.0` (minor — additive feature).
- [ ] 10.5 `CHANGELOG.md` entry under a new `0.5.0` heading.
- [ ] 10.6 `openspec archive add-kl-attribution-rep-selection` after merge.

## What this change explicitly defers

- [ ] 11.1 `rep_selection="recon_loss_direct"` that consumes a held-out activation tensor for direct reconstruction loss — requires Compressor interface widening. Wait for empirical evidence that `kl_attribution` is insufficient.
- [ ] 11.2 Joint cluster optimisation (greedy minimisation of total cluster reconstruction loss across the rep choice). Per-feature scoring is the simpler first step.
- [ ] 11.3 Auto-selection of rep_selection based on the report's behavioural-field availability. Today the caller picks; auto-selection would add magic that's hard to debug.
- [ ] 11.4 Making `kl_attribution` the default. Wait for cross-encoding empirical evidence that it Pareto-dominates `scale_aware`.
- [ ] 11.5 WARNING-level logging when the per-feature NaN fallback to `scale_aware` fires. Polish; not required for correctness.
- [ ] 11.6 Extending `kl_attribution` to `EpochCompressor` (which currently drives `Compressor` with threshold-mode `plan()` — would inherit `kl_attribution` automatically through the `config` kwarg, but explicit test coverage there is a follow-up).
