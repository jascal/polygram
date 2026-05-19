## 1. `EpochReport.redundancy_ratio`

- [ ] 1.1 Add `redundancy_ratio: float` and `n_features_input: int` fields to `polygram.compression.epoch_report.EpochReport`. Bump the report's `schema_version` constant.
- [ ] 1.2 Update `polygram.compression.epoch.EpochCompressor._build_report` (or equivalent — find the construction site via `git grep "EpochReport("`) to plumb the SAE's input feature count through and compute `redundancy_ratio = n_features_zeroed_total / max(1, n_features_input)`. Defensive sentinel: when `n_features_input == 0` (degenerate; should never happen on a real SAE but worth guarding) the ratio SHALL be `0.0`, not `nan` — downstream consumers and reviewers consistently misread `nan` as a real measurement.
- [ ] 1.3 `EpochReport.to_dict` / `to_json` SHALL emit the two new fields. `EpochReport.from_dict` / `from_json` SHALL accept older payloads (without the fields) by computing `redundancy_ratio` from any available divisor or setting it to `None` if no recovery path exists. Bump `schema_version` and add the migration to the existing `_migrate_payload` flow.
- [ ] 1.4 Update `polygram.cli.analyze` (CLI surface) to include `redundancy_ratio` on its human-readable report block alongside `n_features_zeroed_total`.

## 2. `BlockFormation` degenerate-partition warning

- [ ] 2.1 In `polygram.clustered_dictionary.build_clustered_dictionary`, after the cosine-strategy partition is built, compute `n_multi_feature_blocks = sum(1 for b in blocks if len(b.features) > 1)`. If `n_multi_feature_blocks == 0` AND `block_formation.strategy == "cosine"`, emit a `UserWarning` whose message names: (a) the configured `cosine_threshold`, (b) the observed maximum off-diagonal cosine from the precomputed pair graph (so the user sees how far the threshold is from anything that would have clustered), (c) a recommended fallback threshold (heuristic: `max_observed_cosine * 0.8`, clipped to `[0.05, 0.5]`). Canonical message template: `f"Degenerate cosine partition: n_multi_feature_blocks=0 at cosine_threshold={threshold}. Max observed off-diagonal cosine: {max_cosine:.4f}. Recommended fallback: {recommended:.3f}"`. The text reads cleanly in both console output and the structured `SelectionReport.warnings` slot.
- [ ] 2.2 When invoked via `from_sae_lens(..., clustered=True)`, the same condition appends a structured entry to `SelectionReport.warnings` matching the text format used by the existing auto-promote warning.
- [ ] 2.3 Suppress the warning when `block_formation.strategy != "cosine"` — co_firing / user_declared have different failure modes; this diagnostic is cosine-specific.

## 3. Tests

- [ ] 3.1 `tests/compression/test_epoch_report.py::test_redundancy_ratio_populated_on_real_fixture` — exercises the existing toy compression fixture and asserts the ratio matches the computed `n_features_zeroed_total / n_features_input`.
- [ ] 3.2 `tests/compression/test_epoch_report.py::test_redundancy_ratio_round_trips_through_json` — `to_json` → `from_json` preserves the new field.
- [ ] 3.3 `tests/compression/test_epoch_report.py::test_legacy_payload_loads_without_new_fields` — older serialised reports load with `redundancy_ratio` either computed-on-load or `None`, no crash.
- [ ] 3.4 `tests/test_clustered.py::test_degenerate_partition_warns` — synthetic SAE with all-orthogonal decoder vectors against `cosine_threshold=0.3`. Asserts a `UserWarning` is emitted naming the threshold and the observed max cosine.
- [ ] 3.5 `tests/test_clustered.py::test_degenerate_partition_no_warning_for_user_declared` — pin the cosine-only scope of the warning.
- [ ] 3.6 `tests/test_sae_import.py::test_from_sae_lens_surfaces_degenerate_partition_warning_in_selection_report` — checks the `SelectionReport.warnings` plumbing for the loader path.
- [ ] 3.7 `tests/test_sae_import.py::test_degenerate_partition_warning_round_trips_through_selection_report_serialisation` — exercises `SelectionReport.to_dict()` / `from_dict()` on a degenerate run; the warning entry text round-trips byte-identical. Closes the "is the structured warning observable downstream after persistence?" loop.
- [ ] 3.8 `tests/test_clustered.py::test_borderline_cosine_partition_does_not_warn` — synthetic decoder set with a *small* set of pairs above `cosine_threshold` (so the partition has at least one multi-feature block). Asserts NO `UserWarning` is emitted. Pins the warning to the actually-degenerate case and prevents over-firing on healthy-but-sparse fixtures.

## 4. Documentation

- [ ] 4.1 Update `polygram/compression/epoch_report.py` class docstring to describe `redundancy_ratio`'s correlation with downstream KL outcomes (cite the 60–74% range observed across recent Gemma-2 runs — helpful on sparse SAEs, catastrophic on dense ones). The range and direction belong on the field-level docstring, not buried in commit history.
- [ ] 4.2 Update `polygram/clustered_dictionary.BlockFormation` docstring to mention the degenerate-partition warning and the recommended threshold-tuning workflow when it fires.
- [ ] 4.3 Add a one-line CHANGELOG entry under `## 0.10.0` (or `## Unreleased` if 0.10 hasn't started) flagging both diagnostics.

## 5. Validation

- [ ] 5.1 `openspec validate add-polygram-010-diagnostics --strict`.
- [ ] 5.2 Full `pytest` non-sklearn pass.
- [ ] 5.3 Manual re-run of `polygram analyze` on the toy fixture to confirm `redundancy_ratio` surfaces in the CLI output.

## 6. Closing

- [ ] 6.1 Commit + PR. Tag for 0.10 milestone.
