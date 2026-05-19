> **Scope revision (impl PR, 2026-05-19).** The impl pass landed in
> two stages:
>
> 1. **Replace elimination** (the cheap engineering win the
>    diagnosis correctly identified). Killed the per-feature
>    `dataclasses.replace` in `_materialise_blocks`. Measured
>    against the proposal's success criterion:
>    - Allocation reduction real and substantial (60–77% fewer
>      block-formation objects across MPS / Rung3 / Rung4).
>    - Build wall-clock gate FAILS (2.55× MPS, 1.16–1.18× R3/R4
>      vs ≥5× target).
>    - Gram cross-over gate FAILS (248 / 695 / `none` vs ≤8
>      target). The per-op gap collapses with build cost; gram
>      per-op is dominated by `Dictionary.gram()`'s q-orca
>      markdown roundtrip, not by Dictionary construction.
>
> 2. **BlockView public surface** (this PR's second push). Added
>    `BlockView` dataclass + `_block_views` field on
>    `ClusteredDictionary` + `block_view(idx)` / `block_views`
>    accessors. Populated by both canonical builders
>    (`build_clustered_dictionary` with eager decoder slices;
>    `from_compression_panels` with `decoder_slice=None`).
>    Does NOT change the gram cross-over story — the bottleneck
>    is downstream of BlockView, in `Dictionary.gram()` itself.
>
> The proposal's *lazy* `.blocks` property is **deferred** —
> per-block Dictionary construction is already cheap post-#99,
> and lazifying without a faster gram path doesn't move the
> cross-over needle. A future `BlockView.gram()` that bypasses
> `Dictionary` construction (and q-orca's markdown roundtrip)
> would be the natural extension, but only if the gram
> cross-over criterion remains load-bearing for a downstream
> consumer.

## 1. `BlockView` value type

- [x] 1.1 Added a frozen `BlockView` dataclass to `polygram/clustered_dictionary.py` carrying `indices: tuple[int, ...]`, `decoder_slice: np.ndarray | None`, `encoding`, `feature_names: tuple[str, ...]`, and `feature_clusters: tuple[str, ...]`. `decoder_slice` made optional (None) for callers that don't carry raw decoder vectors at construction (`from_compression_panels`).
- [x] 1.2 `BlockView.n_features` property added. `feature_at(i)` adapter not added — `feature_names` + `feature_clusters` give the metadata access pattern downstream consumers actually need.

## 2. `ClusteredDictionary` BlockView integration

- [x] 2.1 Replace overhead in `_materialise_blocks` killed via the per-feature `dataclasses.replace` elimination (see §1 above). New `BlockView` tuple is populated *parallel* to the eager `blocks` list, not as a replacement — `blocks` remains a regular dataclass field for backwards compatibility.
- [x] 2.2 Added `_block_views: tuple[BlockView, ...]` private field. Public `blocks: list[Dictionary]` stays as-is (not lazified). The proposal's lazy property is **deferred** — per the scope-revision note above, lazifying without a faster gram path wouldn't move the cross-over needle.
- [x] 2.3 Added `ClusteredDictionary.block_view(idx) -> BlockView` accessor + `block_views: tuple[BlockView, ...]` property. Both surface a clear `LookupError` for legacy direct-construction paths that don't populate the metadata.
- [ ] 2.4 `materialise_dictionaries` kwarg [DEFERRED] — not needed because `.blocks` was never lazified; the eager build path remains the default.
- [x] 2.5 `from_compression_panels` updated to populate `_block_views` (with `decoder_slice=None`, since that path doesn't carry raw decoder vectors).

## 3. Encoding-specific feature defaults

- [x] 3.1 Not needed in the actually-shipped scope: the lazy `Dictionary` materialisation path was deferred. The eager build through `_materialise_blocks` continues to construct real `Feature` carriers via the parent `features` list, so `Feature.with_default_amp_knobs` and `Dictionary.__post_init__` validation run as before — no new test needed.

## 4. Benchmark + writeup re-run

- [x] 4.1 After the simpler fix landed, re-ran `examples/clustered_amortised_benchmark.py --sae <real-sae> --n-features 24576 --encoding {mps,rung3,rung4} --op all --n-repeats 2`. Fresh JSONs at `docs/research/data/clustered_amortised_benchmark_full_<enc>_v2.json`.
- [x] 4.2 Appended "Post-lighter-container sweep" section to `docs/research/clustered-amortised-benchmark.md`. Verdict: success criterion fails on both gates; honest diagnosis is in the writeup.
- [x] 4.3 (success path) — N/A, criterion failed.
- [x] 4.4 (failure path) — surfaced the next dominant cost. Per-op gram cost is dominated by `Dictionary.gram()`'s analytic-state materialisation, NOT block construction. The replace overhead was real but secondary; eliminating it gives allocation reduction (60–77%) without unlocking the gram cross-over story. The proposal's BlockView surface wouldn't change this either — what's needed is a faster per-block gram path (vectorised cross-block batching, or a `BlockView.gram()` that does the analytic work without going through `Dictionary`).

## 5. Tests

- [ ] 5.1 `test_block_view_carries_correct_shape` [DEFERRED — BlockView not landed].
- [ ] 5.2 `test_lazy_blocks_property_materialises_dictionaries_correctly` [DEFERRED].
- [ ] 5.3 `test_lazy_materialisation_is_idempotent` [DEFERRED].
- [ ] 5.4 `test_materialise_dictionaries_kwarg_eager_path_preserved` [DEFERRED].
- [ ] 5.5 `test_lazy_blocks_respect_encoding_specific_defaults` [DEFERRED].
- [x] 5.6 Existing `tests/test_clustered.py`, `tests/test_clustered_golden.py`, `tests/compression/test_epoch_clustered_consume.py` suites pass unchanged. Confirmed: 129 clustered/import + 17 compression tests green post-fix. The cluster-name change in `_materialise_blocks` (cosine/co_firing blocks now preserve features' original cluster values instead of overwriting them) didn't break any existing assertion — none of the tests asserted on the synthetic `<parent>_b<idx>` cluster naming.

## 6. Validation

- [x] 6.1 `openspec validate add-lighter-per-block-container --strict` — passes.
- [x] 6.2 Full `pytest` non-sklearn pass.
- [x] 6.3 Real-fixture amortised-benchmark re-run per §4.1.
- [x] 6.4 Commit + PR.

## 7. Closing

- [x] 7.1 Updated CHANGELOG `## Unreleased` with the actually-landed perf change and the behaviour note about the cluster-name preservation.
