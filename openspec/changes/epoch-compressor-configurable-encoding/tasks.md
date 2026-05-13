## 1. Constructor parameter

- [ ] 1.1 Add `encoding: MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | None = None` field to `EpochCompressor` in `polygram/compression/epoch.py`. Place it next to the other tuning fields (after `config`).
- [ ] 1.2 In `__post_init__`, resolve `None` to `MPSRung1()` via a local import (matches the existing import-locality pattern at the `run()`-internal import line).
- [ ] 1.3 Validate the resolved encoding has a positive integer `max_features` attribute (cheap quack-duck check; defensive against future encoding misimplementations).
- [ ] 1.4 Update `EpochCompressor.fast` and `EpochCompressor.thorough` (if needed) to thread the encoding kwarg through `_from_preset`. Verify the preset paths preserve the existing default-resolution semantics.

## 2. `EpochCompressor.run` uses `self.encoding`

- [ ] 2.1 Remove the hardcoded `from polygram.encoding import MPSRung1 as _MPSRung1` and `encoding=_MPSRung1()` block in the per-iteration `ClusteredDictionary.from_compression_panels` call. Replace with `encoding=self.encoding`.
- [ ] 2.2 Remove the `TODO(issue #48)` comment.
- [ ] 2.3 Verify the panel↔block ordering assertion still fires correctly (the existing defense-in-depth `assert len(clustered.blocks) == len(panels)`).

## 3. `_select_panels` scales neighbour cap

- [ ] 3.1 Add `max_panel_size: int` as a required kwarg on `_select_panels`. Place after `coverage_target`.
- [ ] 3.2 Replace `if len(neighbours) >= 7:` with `if len(neighbours) >= max_panel_size - 1:`.
- [ ] 3.3 In `EpochCompressor.run`, pass `max_panel_size=self.encoding.max_features` to the `_select_panels` call.
- [ ] 3.4 No other callers of `_select_panels` need updating (private function; only one caller).

## 4. Differential regression — byte-identical default

- [ ] 4.1 Re-run the existing `test_byte_identical_epoch_result_against_frozen_reference` test in `tests/compression/test_epoch_clustered_consume.py`. It must pass unchanged — the `encoding=None` default path resolves to `MPSRung1()` and produces the same frozen-reference output.
- [ ] 4.2 Add `test_explicit_mpsrung1_byte_identical_against_frozen_reference` to the same file. Construct `EpochCompressor` with `encoding=MPSRung1()` explicitly; assert byte-identity vs the same frozen reference. This locks the default-resolution path.

## 5. Rung3 path smoke test

- [ ] 5.1 Create `tests/compression/_rung3_fixture.py` (sibling to `_clustered_fixture.py`): a deterministic 32-feature synthetic SAE with two engineered redundancy clusters of 10 features each, plus the monkeypatch-able pre-pass helper.
- [ ] 5.2 Create `tests/compression/test_epoch_encoding_configurable.py`:
  - `test_rung3_run_produces_large_panels`: assert at least one panel has `len(features) > 8`.
  - `test_rung3_run_respects_max_features_cap`: assert all panels have `len(features) <= Rung3.max_features` (16).
  - `test_rung3_run_zeros_features`: assert `result.n_features_zeroed_total > 0` (sanity — run did something).
- [ ] 5.3 (Optional, if Rung4 path is trivial to add) `test_rung4_run_respects_max_features_cap`: assert all panels have `len(features) <= Rung4.max_features` (32) on a fixture with enough engineered redundancy.

## 6. Validation

- [ ] 6.1 In `__post_init__`, reject encodings whose `max_features < 2` with a clear error (panels of size 1 are degenerate; current code path emits anchor-only panels under specific conditions but those aren't a primary mode).
- [ ] 6.2 Test: passing `encoding=ObjectWithMaxFeaturesZero()` raises `ValueError`.

## 7. CHANGELOG + closing

- [ ] 7.1 Add an entry under unreleased: "**`EpochCompressor`** — new `encoding=` constructor parameter (defaults to `MPSRung1()`); compression runs can opt into Rung3/Rung4/HEA_Rung2 to exploit larger per-encoding feature caps. The `_select_panels` neighbour cap now scales as `encoding.max_features - 1` (was hardcoded 7)."
- [ ] 7.2 Run `openspec validate epoch-compressor-configurable-encoding --strict`.
- [ ] 7.3 Verify `pytest` runs green (full suite + ruff check).
- [ ] 7.4 Close issue #48 with a pointer to the merged PR.

## 8. What this change explicitly defers

- [ ] 8.1 Encoding-aware tuning of `n_visits_per_feature`, `n_panels_max`, `coverage_target` — they remain user-tunable constructor kwargs with the existing defaults regardless of encoding.
- [ ] 8.2 A `BaseEncoding` protocol — the encoding parameter is typed `MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | None`; tightening this requires a protocol that all four encodings satisfy structurally, which is its own change.
- [ ] 8.3 A frozen Rung3 / Rung4 regression reference — premature. The Rung3+ compression behaviour is itself a research question (Rung4 viability spike). Freezing now would lock in arbitrary results.
- [ ] 8.4 Encoding-aware panel-selection criteria — `_select_panels` remains purely W_dec-cosine + priority + visit budget driven. If a future encoding needs different selection (e.g., HEA topology-aware), that's a separate change.
