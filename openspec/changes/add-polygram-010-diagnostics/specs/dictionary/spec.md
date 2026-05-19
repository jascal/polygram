## ADDED Requirements

### Requirement: `build_clustered_dictionary` warns when the cosine partition is degenerate

`polygram.clustered_dictionary.build_clustered_dictionary` SHALL emit a `UserWarning` whenever `block_formation.strategy == "cosine"` AND the resulting partition has *zero* multi-feature blocks. The condition `n_multi_feature_blocks == 0` is the load-bearing signal: it means no pair of decoder vectors cleared the configured `cosine_threshold`, so every feature lands in its own singleton block and downstream analyses degenerate silently.

The warning message SHALL name:

- The configured `cosine_threshold`.
- The observed maximum off-diagonal pairwise cosine across the decoder matrix (so the user can see how far the threshold is from anything that would have clustered).
- A recommended fallback threshold (`max_observed_cosine * 0.8`, clipped to `[0.05, 0.5]`).

When `build_clustered_dictionary` is invoked via `from_sae_lens(..., clustered=True)`, the same condition SHALL append a structured entry to `SelectionReport.warnings` so callers consuming the loader-side report observe the degeneracy without inspecting `warnings.catch_warnings()`. The text matches the format of the existing auto-promote warning entry.

The warning SHALL NOT fire for `block_formation.strategy != "cosine"` — co_firing and user_declared have different failure modes; this diagnostic is cosine-specific.

#### Scenario: all-orthogonal decoder vectors trigger the warning

- **WHEN** `build_clustered_dictionary(..., block_formation=BlockFormation(strategy="cosine", cosine_threshold=0.3))` is called on a decoder matrix whose maximum off-diagonal cosine is below 0.3
- **THEN** a `UserWarning` is emitted whose message contains the string `"cosine_threshold=0.3"`, the observed max cosine value, and a recommended fallback threshold

#### Scenario: user_declared strategy is exempt

- **WHEN** `build_clustered_dictionary(..., block_formation=BlockFormation(strategy="user_declared"), hierarchy={...})` is called and the resulting partition has zero multi-feature blocks
- **THEN** no degenerate-partition warning is emitted (the user_declared path has its own failure modes; cosine-specific warnings would be noise here)

#### Scenario: SelectionReport surfaces the warning via the loader path

- **WHEN** `from_sae_lens(records, ids, clustered=True)` builds a `ClusteredDictionary` whose cosine partition is degenerate
- **THEN** `report.warnings` contains an entry whose text begins with the documented degenerate-partition prefix
