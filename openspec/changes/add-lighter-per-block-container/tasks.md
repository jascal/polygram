> **Scope revision (impl PR).** §1–§3 below scoped the full
> `BlockView` lazy-materialisation surface. The impl pass first
> tried the *smaller* fix the proposal's diagnosis actually points
> at — eliminating the per-feature `dataclasses.replace` in
> `_materialise_blocks` — and measured it against the proposal's
> success criterion. Result, captured in
> `docs/research/clustered-amortised-benchmark.md`'s
> post-lighter-container sweep section:
>
> - **Allocation reduction is real and substantial** (60–77% fewer
>   block-formation objects across MPS / Rung3 / Rung4).
> - **Build wall-clock gate FAILS** (2.55× MPS, 1.16–1.18× R3/R4
>   vs ≥5× target).
> - **Gram cross-over gate FAILS** (now 248 / 695 / `none` vs ≤8
>   target). The per-op gap collapses with build cost; gram
>   per-op is dominated by `Dictionary.gram()`'s analytic-state
>   materialisation, not by Dictionary construction.
>
> The full BlockView surface (§1, §2, §3) is therefore **deferred**.
> The replace-elimination ships on its own as the cheap engineering
> win the proposal correctly identified; the rest is parked until
> evidence justifies it (a future `BlockView.gram()` that bypasses
> Dictionary construction entirely is the natural next step, but
> only if the gram cross-over criterion remains load-bearing for
> a downstream consumer).

## 1. `BlockView` value type [DEFERRED]

- [ ] 1.1 Add a frozen `BlockView` dataclass to `polygram/clustered_dictionary.py` carrying `indices: tuple[int, ...]`, `decoder_slice: np.ndarray`, `encoding: MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | Rung5`, `feature_names: tuple[str, ...]`. The decoder slice SHALL be a non-copying numpy view when possible; the field's type allows arrays but the construction site uses `decoder_vectors[indices]` (numpy returns a view for contiguous-index slices and a copy otherwise — both are correct, just measure RSS once at impl time).
- [ ] 1.2 `BlockView.n_features` property; `BlockView.feature_at(i)` accessor that synthesises a minimal `Feature` carrier on demand (used by the lazy `Dictionary` materialisation path in §2). The synthesised `Feature` reuses the parent block-formation cluster name; knobs default to encoding defaults (per `Feature.with_default_amp_knobs`).

## 2. Lazy `ClusteredDictionary` materialisation [DEFERRED]

- [ ] 2.1 Replace `_materialise_blocks` with `_materialise_block_views`. The new helper returns `list[BlockView]` (or `tuple[BlockView, ...]`) rather than `list[Dictionary]`. The expensive `dataclasses.replace(f, cluster=cluster) for f in block_feats` loop is the line that dies. **— landed via a simpler edit in `_materialise_blocks` itself; see scope note above.**
- [ ] 2.2 Add an internal `ClusteredDictionary._block_views: tuple[BlockView, ...]` field, populated at construction time. The existing public `blocks: list[Dictionary]` field is replaced by a `@property` that lazily materialises Dictionary instances from `_block_views` on first access and caches them (via the same `object.__setattr__` pattern the cross-block-edges cache uses).
- [ ] 2.3 New method `ClusteredDictionary.block_view(idx) -> BlockView` for callers that want the lightweight container directly (the amortised-benchmark's gram path is the first known consumer).
- [ ] 2.4 New kwarg `materialise_dictionaries: bool = False` on `build_clustered_dictionary` and the `ClusteredDictionary` constructor. Default `False` skips the eager dictionary build; `True` preserves the pre-change behaviour for callers (compression panel construction is the only known one) that need the full per-block Dictionary upfront.
- [ ] 2.5 Update `from_compression_panels` and any other callsite that passes `blocks=` directly into the `ClusteredDictionary` constructor to either pass `_block_views` instead or accept the eager-materialisation flag.

## 3. Encoding-specific feature defaults [DEFERRED]

- [ ] 3.1 The lazy-materialisation path's synthesised `Feature` carriers MUST satisfy `Dictionary.__post_init__`'s validation (knob ranges, amp-knob length for Rung5). Re-use `Feature.with_default_amp_knobs` so the same defaults the loader uses today fall out automatically. Pin via a dedicated test (§5.5).

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
