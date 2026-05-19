## 1. Loader auto-promote

- [x] 1.1 In `polygram/sae_import.py`, change the `from_sae_lens` signature `clustered: bool = False` to `clustered: bool | None = None`.
- [x] 1.2 Where the current code raises `ValueError` for `not clustered and len(feature_ids) > encoding_cap`, branch on the new tri-state: `clustered is False` keeps the raise; `clustered is True` continues to clustered; `clustered is None and N > cap` sets a local `effective_clustered = True` and queues a warning string `"auto-promoted to clustered: N=<N> exceeds <Encoding>.max_features=<cap>"`.
- [x] 1.3 When auto-promoted, append the warning to the `warnings` list that is later assembled into `SelectionReport`.
- [x] 1.4 Update the docstring's "Refuses subsets larger than 8 features" line — it's been wrong since `per-encoding-feature-cap`, and is now further outdated by this change. Replace with a description of the tri-state.
- [x] 1.5 Update the `ValueError` message (still used on explicit `clustered=False`) so the suggestion text reads `"omit \`clustered=False\` to auto-promote, or pass \`clustered=True\` explicitly"`.

## 2. Dead-code removal

- [x] 2.1 In `polygram/clustered_dictionary.py`, remove the `_LEGACY_MAX_FEATURES = 8` constant and the `_encoding_max_features` helper's `getattr` fallback. The function becomes `return int(encoding.max_features)` (or inlined at each callsite, depending on what's tidier).
- [x] 2.2 If the helper is only used in one or two places, inline it and drop the helper entirely.

## 3. Tests

- [x] 3.1 In `tests/test_sae_import.py`, update the five tests that assert oversized-N raises with the implicit default (`test_select_too_many_features_rejected`, `test_clustered_true_skips_8_cap`, `test_clustered_error_message_points_to_clustered`, `test_rung3_seventeen_features_raises_with_encoding_name`, `test_mpsrung1_nine_features_raises_with_encoding_name`, `test_error_message_suggests_clustered_path`) to pass `clustered=False` explicitly. The assertions about the error message stay the same.
- [x] 3.2 Add a new test `test_oversized_n_auto_promotes_to_clustered` covering the new default: `from_sae_lens(records, list(range(16)))` (no `clustered` kwarg, against `MPSRung1` cap=8) returns a `ClusteredDictionary` and `report.warnings` contains the auto-promote string.
- [x] 3.3 Add `test_explicit_clustered_false_still_raises` covering the strict mode.

## 4. Validation

- [x] 4.1 Run `openspec validate auto-promote-clustered-on-cap-overflow --strict`.
- [x] 4.2 Run the full `pytest` suite. Verify no regressions.
