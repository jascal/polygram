## Why

Two diagnostic gaps surfaced during external evaluation of polygram
0.9.0 against real Gemma-2 SAEs (and earlier on a GPT-2 K=211
prototype in #59). Both are pure observability fixes — no
algorithmic change — but both gate the bet's "smaller + cheaper"
sub-claim from being measurable in production usage:

### 1. `EpochReport.redundancy_ratio` is missing

`EpochReport` carries `n_features_zeroed_total` and
`n_panels_total` but not the **ratio**. In the field, the ratio
landed between 60% and 74% across runs and **correlated strongly
with downstream KL outcomes** — helpful on sparse SAEs,
catastrophic on dense ones. The ratio is the load-bearing
diagnostic for "did this compression help or hurt?", but every
downstream consumer has to compute it themselves from
`n_features_zeroed_total / <something the report doesn't carry>`.

Promoting `redundancy_ratio` to a first-class field on
`EpochReport` (with the divisor explicit) closes a real foot-gun:
the run log shows the number that actually predicts behaviour
instead of leaving it as a downstream derivation.

### 2. `n_clusters = 0` is a silent failure

`BlockFormation.cosine` builds the cosine pair graph at threshold
0.3 (the default), then greedy-seeds blocks. On Gemma-2 SAEs in
recent runs, **every cell returned `n_clusters = 0`**: no decoder
pair cleared 0.3, so the greedy partition produced only singletons
(or skipped entirely depending on the consumer). The same pattern
appeared in the K=211 GPT-2 prototype (#59).

Today the consumer silently inherits zero clusters, downstream
analyses degenerate, and the user has no signal that BlockFormation
gave up. Two paths to fix this — either is acceptable, this
proposal picks **loud-warning** as the safe default:

- **Loud warning** when post-formation `n_blocks_with_multi_features
  == 0` (no real clustering happened). The warning names the
  threshold, the observed max pairwise cosine, and the
  recommended fallback value.
- **Auto-fallback** to a lower threshold (e.g., 0.15) with the same
  warning attached. Convenient but changes behaviour silently —
  rejected here in favour of explicit opt-in to relaxation.

This proposal lands the loud warning; the auto-fallback can be a
follow-up if usage shows the warning is consistently ignored.

### Explicitly out of scope

- `polygram.recommend_feature_selection(sae)` helper. The note from
  the external evaluation flags this as lower-priority and possibly
  out-of-scope for polygram (argues it belongs in sae-forge as a
  feature-selection auto-mode). This proposal does not include it.
- Any sae-forge-side change. The external note included separate
  recommendations for sae-forge (RoPE in Llama-family adapters, etc.) —
  filed against that repo, not here.

## What Changes

- `polygram.compression.epoch_report.EpochReport` gets a new field
  `redundancy_ratio: float` plus a sibling `n_features_input: int`
  (the divisor) so the ratio is computable without external
  context. Schema version bumps; `to_dict`/`from_json` round-trips
  the new fields. Existing serialised reports remain readable via
  a `from_dict` migration shim (older payloads compute
  `redundancy_ratio` lazily from the available data, or leave it
  `None` if the divisor isn't recoverable).
- `polygram.clustered_dictionary.build_clustered_dictionary` emits
  a `UserWarning` (and a structured `selection_report.warnings`
  entry when invoked via `from_sae_lens(..., clustered=True)`)
  whenever the post-formation partition has *zero* multi-feature
  blocks. Warning text names the configured threshold, the
  observed max pairwise cosine (so the user can see how far off
  they are), and a recommended fallback.
- CLI: `polygram analyze` surfaces `redundancy_ratio` in the
  human-readable report block (line item alongside
  `n_features_zeroed_total`).
- Tests: regression tests cover both diagnostic surfaces (ratio
  populated correctly across the existing fixtures; warning fires
  on a synthetic decoder-set where every pairwise cosine is below
  threshold).

## Target

**polygram 0.10.** Bundled with whatever else lands in the 0.10
release cycle.

## Impact

- **Affected specs**: `compression` (new diagnostic field on
  EpochReport); `dictionary` (BlockFormation degenerate-partition
  warning).
- **Affected code**: `polygram/compression/epoch_report.py`
  (field + serialisation); `polygram/compression/epoch.py` (compute
  ratio on report construction); `polygram/clustered_dictionary.py`
  (warning emission in `build_clustered_dictionary`);
  `polygram/cli.py` (surface ratio in the analyze report); two new
  test sites.
- **Risk**: low. Strictly additive on the report; warning is
  default-on but doesn't change return values. The schema bump
  needs the same `from_dict` forward/backward-compat handling
  already used by the config tuning system.
