# recon-aware-rep-selection Specification

## Purpose

Defines the `kl_attribution` rep_selection algorithm: an opt-in `CompressionConfig.rep_selection` mode that picks cluster representatives by **behavioural ablation importance** (the existing `CandidatePair.kl_ablate_i` / `kl_ablate_j` measurements from `BehaviouralValidator`) rather than by geometric proxies (`n_fires`, `scale_aware`). Addresses the [[project_kl_nonmonotonic]] regime where existing geometric rep_selection picks cluster reps that don't preserve downstream model behaviour as well as a different cluster member would.

The capability adds **no new fields** to `ValidationReport`, `CandidatePair`, or `Compressor`'s public interface — only a new value for the existing `rep_selection` enum and a new internal scoring function.

## ADDED Requirements

### Requirement: Compressor supports `rep_selection="kl_attribution"`

`Compressor` SHALL accept `rep_selection="kl_attribution"` either via the `rep_selection=` constructor kwarg or via `CompressionConfig(rep_selection="kl_attribution")`. When set, `_pick_representative` SHALL dispatch to a `_score_kl_attribution` helper that picks the cluster member with the maximum mean `kl_ablate` value across the pairs containing it.

The algorithm SHALL produce a deterministic result on the same input report — same chosen rep across runs, same order across repeated calls.

#### Scenario: Compressor accepts kl_attribution via config

- **GIVEN** `cfg = CompressionConfig(rep_selection="kl_attribution")`
- **WHEN** `Compressor(validation_report=..., sae_checkpoint=..., config=cfg)` is constructed
- **THEN** `compressor.rep_selection == "kl_attribution"` and `compressor.plan().apply()` runs without raising

#### Scenario: Compressor accepts kl_attribution via kwarg

- **WHEN** `Compressor(validation_report=..., sae_checkpoint=..., rep_selection="kl_attribution")` is constructed (no config)
- **THEN** `compressor.rep_selection == "kl_attribution"`

#### Scenario: Compressor rejects unknown rep_selection

- **WHEN** `Compressor(..., rep_selection="bogus")` is constructed
- **THEN** `__post_init__` raises `ValueError` listing the three supported values

### Requirement: kl_attribution scores features by mean kl_ablate across pairs

For each feature `f` in a cluster, the per-feature score `kl_ablate(f)` SHALL be computed as the arithmetic mean of `pair.kl_ablate_i if pair.i == f else pair.kl_ablate_j` across every `pair` in `validation_report.pairs` where `f ∈ {pair.i, pair.j}`. NaN values are excluded from the mean. The mean smooths over per-pair measurement noise.

#### Scenario: Mean aggregation smooths per-pair noise

- **GIVEN** feature `f` appears in three pairs with `kl_ablate_f` values `[0.3, 0.31, 0.29]`
- **WHEN** `_score_kl_attribution` computes `kl_ablate(f)`
- **THEN** the value is `0.30` (the mean), not `0.31` (max) or `0.29` (min)

#### Scenario: Highest kl_ablate wins the rep selection

- **GIVEN** a 3-feature cluster `{0, 1, 2}` with mean `kl_ablate` values `[0.1, 0.5, 0.2]`
- **WHEN** `_score_kl_attribution` runs
- **THEN** the returned rep is feature `1` (the one with the highest `kl_ablate`)

### Requirement: Tiebreak on n_fires then feature_id

When two or more cluster members have identical `kl_ablate` mean scores, the rep selection SHALL break ties first by `n_fires_total` descending (more-fired features have more reliable behavioural measurements), then by feature_id ascending (deterministic cross-platform).

#### Scenario: Tiebreak prefers higher n_fires_total

- **GIVEN** two cluster members tied on `kl_ablate` but with `n_fires_total` `[100, 50]`
- **WHEN** `_score_kl_attribution` runs
- **THEN** the feature with `n_fires_total == 100` is the rep

#### Scenario: Tiebreak prefers lower feature_id when both kl_ablate and n_fires tie

- **GIVEN** two cluster members tied on both `kl_ablate` and `n_fires_total`, with feature ids `[5, 7]`
- **WHEN** `_score_kl_attribution` runs
- **THEN** feature `5` is the rep

### Requirement: Per-feature NaN falls back to scale_aware scoring

If a single cluster member's mean `kl_ablate` is NaN (the feature fires too rarely for behavioural KL to be statistically meaningful, or the pair was emitted by a geometry-only confirmer for that one feature) while at least one other cluster member has a non-NaN value, the NaN feature SHALL be scored via `_score_scale_aware` for that single feature only. The fallback score is normalised onto a comparable scale so the NaN feature competes on the same axis as the KL-scored features.

The fallback is silent (no log message in this change; see deferred work for WARNING-level logging).

#### Scenario: Per-feature NaN does not abort the cluster

- **GIVEN** a 3-feature cluster where feature `1` has NaN `kl_ablate` mean but features `0` and `2` have finite values
- **WHEN** `_score_kl_attribution` runs
- **THEN** feature `1` is scored via `_score_scale_aware` (using its `W_dec` row and `n_fires` count); the chosen rep is the cluster member with the highest combined score; no `ValueError` is raised

### Requirement: All-cluster NaN raises ValueError

When every member of a cluster has all-NaN `kl_ablate` (the cluster came through a non-behavioural confirmer such as `DecoderGeometryConfirmer` or `ClusterConfirmer`), `_score_kl_attribution` SHALL raise `ValueError` rather than silently fall back. The error message SHALL name:

- the rep_selection mode that triggered the error (`'kl_attribution'`)
- the cause (behavioural fields are NaN; likely came through a geometry-only confirmer)
- the supported alternatives (`'scale_aware'`, `'n_fires'`)

This surfaces the caller's mis-configuration loudly rather than silently degrading to a different criterion.

#### Scenario: All-NaN cluster raises with actionable message

- **GIVEN** a cluster whose every member's `kl_ablate` is NaN (e.g. a `DecoderGeometryConfirmer`-produced report)
- **WHEN** `_score_kl_attribution` runs
- **THEN** `ValueError` is raised; the message contains the substrings `kl_attribution`, `behavioural confirmation`, and `DecoderGeometryConfirmer`; `scale_aware` and `n_fires` are named as alternatives

### Requirement: kl_attribution preserves byte-identity for non-opt-in callers

When `rep_selection` is unset, set to `"scale_aware"`, or set to `"n_fires"`, the Compressor's `plan()` → `apply()` output SHALL be byte-identical to the pre-change reference. The `kl_attribution` code path is not entered, no `_score_kl_attribution` calls are made, and no `kl_ablate` lookups are performed.

#### Scenario: Default rep_selection unchanged

- **WHEN** `Compressor(report, ckpt)` is constructed with no `rep_selection` kwarg AND no `config`
- **THEN** `compressor.rep_selection == "scale_aware"`; `compressor.plan().apply()` produces a `CompressionReport.to_json()` output bit-equal to the pre-change reference for the existing toy fixture

#### Scenario: Explicit scale_aware byte-identical

- **WHEN** `Compressor(report, ckpt, rep_selection="scale_aware")` is constructed and applied
- **THEN** output is bit-equal to the pre-change reference

#### Scenario: Explicit n_fires byte-identical

- **WHEN** `Compressor(report, ckpt, rep_selection="n_fires")` is constructed and applied
- **THEN** output is bit-equal to the pre-change reference

### Requirement: kl_attribution flows through plan_with_target and plan_pareto

`kl_attribution` is a rep_selection mode, not a planning mode. It SHALL flow transparently through `Compressor.plan_with_target()` and `Compressor.plan_pareto()` — every cluster produced by either path uses `_score_kl_attribution` when `rep_selection == "kl_attribution"`.

#### Scenario: plan_with_target uses kl_attribution rep selection

- **GIVEN** `Compressor(report, ckpt, config=CompressionConfig(rep_selection="kl_attribution", target_n_features_kept=200))`
- **WHEN** `plan_with_target()` runs
- **THEN** every cluster in the returned plan has its rep chosen via `_score_kl_attribution` (the chosen feature has the highest mean `kl_ablate` within its cluster)

#### Scenario: plan_pareto uses kl_attribution rep selection across all K

- **GIVEN** `Compressor(report, ckpt, config=CompressionConfig(rep_selection="kl_attribution"))`
- **WHEN** `plan_pareto([2, 4, 8])` runs
- **THEN** each of the three outcomes' clusters has reps chosen via `_score_kl_attribution`; no clusters fall back to `scale_aware` (unless individual features are NaN per the fallback rule above)
