## 1. Capture differential regression reference

- [ ] 1.1 Run the existing `EpochCompressor.run` on the bundled `tests/fixtures/toy_sae.json` fixture at seed 0 with shipped defaults. Capture the resulting `EpochResult` as JSON.
- [ ] 1.2 Commit the captured reference to `tests/compression/data/epoch_result_reference.json`.
- [ ] 1.3 Add `tests/compression/data/README.md` documenting how to regenerate the reference (the exact command, the toolchain pins, when regeneration is appropriate).
- [ ] 1.4 Run the reference once on the pre-refactor code path to confirm reproducibility — the captured JSON loads back into an equivalent `EpochResult`.

## 2. `_validate_panels` consumes `ClusteredDictionary`

- [ ] 2.1 Change `_validate_panels` in `polygram/compression/epoch.py` signature from `(panels: list[Panel], ...)` to `(clustered: ClusteredDictionary, ...)`. Body iterates `clustered.blocks` instead of `panels`. Per-block validation logic unchanged.
- [ ] 2.2 The order of `clustered.blocks` MUST match the order of the source `panels` list. `from_compression_panels` already preserves this; assert in `_validate_panels` (debug-only `assert` or add a property invariant test).
- [ ] 2.3 The return type stays `list[ValidationReport]` (one entry per block).

## 3. `_synthesize_validation_report` consumes `ClusteredDictionary` + per-block reports

- [ ] 3.1 Change signature from `(panels: list[Panel], per_panel_reports: list[ValidationReport], sae_checkpoint: Path)` to `(clustered: ClusteredDictionary, block_reports: list[ValidationReport], sae_checkpoint: Path)`. The `sae_checkpoint` arg is unchanged.
- [ ] 3.2 Internal logic translated to iterate `clustered.blocks` instead of `panels`. Cross-block evidence aggregation (which confirmed-pairs span multiple blocks, which features are representatives) preserves output shape exactly.
- [ ] 3.3 Returns the same `ValidationReport` (cross-block synthesis) as before.

## 4. `EpochCompressor.run` builds `ClusteredDictionary` per iteration

- [ ] 4.1 After the `_select_panels` call in `EpochCompressor.run`, construct `ClusteredDictionary.from_compression_panels(panels=panels, state_dict=current_state, encoding=MPSRung1(), name=f"{stem}_iter{iteration}")`.
- [ ] 4.2 Thread the resulting `ClusteredDictionary` into the `_validate_panels` and `_synthesize_validation_report` calls.
- [ ] 4.3 The early-exit path (`if not panels: ... break`) is unchanged.
- [ ] 4.4 The compression delegation to `Compressor.run` is unchanged.
- [ ] 4.5 The convergence-state determination (`_REASON_STABLE_CLUSTERS` etc.) is unchanged.

## 5. Differential regression test

- [ ] 5.1 Add `tests/compression/test_epoch_clustered_consume.py` with one test: load the frozen reference from `tests/compression/data/epoch_result_reference.json`, run the post-refactor `EpochCompressor.run` on the bundled fixture at seed 0, assert byte-identical `EpochResult` field-by-field.
- [ ] 5.2 Equality semantics: numeric fields (floats) compared bit-exact (no tolerance); string fields compared via `==`; collection fields (sets, frozensets, tuples) compared via element equality; nested dataclasses recursively.
- [ ] 5.3 Test runs on every CI build (no special markers or skips).

## 6. Existing compression test suite passes unchanged

- [ ] 6.1 Run `tests/test_compression*.py` and `tests/compression/` end-to-end on the refactored code. Every test passes without modification.
- [ ] 6.2 Run `polygram analyze` CLI on the bundled fixture; the output (logs, written checkpoints) is byte-identical to pre-refactor.

## 7. `ClusteredDictionary` shape invariant tests

- [ ] 7.1 Per-iteration `ClusteredDictionary` invariants: every block has ≤ `MPSRung1.max_features` features; every block's `Dictionary` is validly constructible; `clustered.n_blocks == len(panels)`; the cross-block adjacency uses the canonical `bi < bj` ordering.
- [ ] 7.2 Smoke test: construct a `ClusteredDictionary` from a manually-built panel list, assert structural fields.

## 8. Closing

- [ ] 8.1 Update `CHANGELOG.md` under unreleased: "EpochCompressor now consumes `ClusteredDictionary` as its internal panel data type. External API + output unchanged; gated by differential regression test on the bundled fixture."
- [ ] 8.2 Run `openspec validate compression-consumes-clustered-dictionary --strict`.
- [ ] 8.3 Verify the `clustered-dictionary-analysis` change's tasks.md §7 deferral note now points at this change as the resolution.

## 9. What this change explicitly defers (documented permanently)

- [x] 9.1 **Extracting `_select_panels` into a `BlockFormation` strategy is deferred indefinitely** and documented in the proposal's "pivot" section as the wrong shape of integration. The priority-driven seeded coverage and the geometric BFS cosine clustering solve different problems; they should remain separate algorithms.
- [x] 9.2 **Configurable encoding through `EpochCompressor.run`** (e.g., supporting Rung3 / Rung4 / HEA at >8 features in compression) is deferred to a future change. Compression currently uses `MPSRung1()` implicitly; this refactor preserves that.
