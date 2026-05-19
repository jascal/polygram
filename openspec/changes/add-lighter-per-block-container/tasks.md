## 1. `BlockView` value type

- [ ] 1.1 Add a frozen `BlockView` dataclass to `polygram/clustered_dictionary.py` carrying `indices: tuple[int, ...]`, `decoder_slice: np.ndarray`, `encoding: MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | Rung5`, `feature_names: tuple[str, ...]`. The decoder slice SHALL be a non-copying numpy view when possible; the field's type allows arrays but the construction site uses `decoder_vectors[indices]` (numpy returns a view for contiguous-index slices and a copy otherwise — both are correct, just measure RSS once at impl time).
- [ ] 1.2 `BlockView.n_features` property; `BlockView.feature_at(i)` accessor that synthesises a minimal `Feature` carrier on demand (used by the lazy `Dictionary` materialisation path in §2). The synthesised `Feature` reuses the parent block-formation cluster name; knobs default to encoding defaults (per `Feature.with_default_amp_knobs`).

## 2. Lazy `ClusteredDictionary` materialisation

- [ ] 2.1 Replace `_materialise_blocks` with `_materialise_block_views`. The new helper returns `list[BlockView]` (or `tuple[BlockView, ...]`) rather than `list[Dictionary]`. The expensive `dataclasses.replace(f, cluster=cluster) for f in block_feats` loop is the line that dies.
- [ ] 2.2 Add an internal `ClusteredDictionary._block_views: tuple[BlockView, ...]` field, populated at construction time. The existing public `blocks: list[Dictionary]` field is replaced by a `@property` that lazily materialises Dictionary instances from `_block_views` on first access and caches them (via the same `object.__setattr__` pattern the cross-block-edges cache uses).
- [ ] 2.3 New method `ClusteredDictionary.block_view(idx) -> BlockView` for callers that want the lightweight container directly (the amortised-benchmark's gram path is the first known consumer).
- [ ] 2.4 New kwarg `materialise_dictionaries: bool = False` on `build_clustered_dictionary` and the `ClusteredDictionary` constructor. Default `False` skips the eager dictionary build; `True` preserves the pre-change behaviour for callers (compression panel construction is the only known one) that need the full per-block Dictionary upfront.
- [ ] 2.5 Update `from_compression_panels` and any other callsite that passes `blocks=` directly into the `ClusteredDictionary` constructor to either pass `_block_views` instead or accept the eager-materialisation flag.

## 3. Encoding-specific feature defaults

- [ ] 3.1 The lazy-materialisation path's synthesised `Feature` carriers MUST satisfy `Dictionary.__post_init__`'s validation (knob ranges, amp-knob length for Rung5). Re-use `Feature.with_default_amp_knobs` so the same defaults the loader uses today fall out automatically. Pin via a dedicated test (§5.5).

## 4. Benchmark + writeup re-run

- [ ] 4.1 After §1–§3 land, re-run `examples/clustered_amortised_benchmark.py --sae <real-sae> --n-features 24576 --encoding {mps,rung3,rung4} --op all --n-repeats 2`. Emit fresh JSONs at `docs/research/data/clustered_amortised_benchmark_full_<enc>_v2.json` (`_v2` suffix so the original numbers stay reproducible for diffing).
- [ ] 4.2 Append a "Post-lighter-container sweep" section to `docs/research/clustered-amortised-benchmark.md` documenting the new numbers vs the originals. Decision rule: pass / fail vs the success criterion in the proposal.
- [ ] 4.3 If the success criterion passes, update `[[project-interpretability-bet]]` memory: the primary "clustering ⇒ faster analytic ops" claim moves from "falsified at current impl" to "validated at the lighter-container impl level."
- [ ] 4.4 If the success criterion fails (either gate), surface the next dominant cost from RSS / object-count deltas in the same writeup — that's the next follow-up's load-bearing diagnosis.

## 5. Tests

- [ ] 5.1 `tests/test_clustered.py::test_block_view_carries_correct_shape` — a 16-feature synthetic fixture; assert every `BlockView` has the right `indices`, `decoder_slice.shape`, and `feature_names`.
- [ ] 5.2 `tests/test_clustered.py::test_lazy_blocks_property_materialises_dictionaries_correctly` — call `cd.blocks[0]`, assert it's a `Dictionary` whose features match the planted ones (name, cluster, encoding defaults).
- [ ] 5.3 `tests/test_clustered.py::test_lazy_materialisation_is_idempotent` — calling `cd.blocks[0]` twice returns the same `Dictionary` instance (cached, not re-materialised).
- [ ] 5.4 `tests/test_clustered.py::test_materialise_dictionaries_kwarg_eager_path_preserved` — when called with `materialise_dictionaries=True`, the result is byte-identical to the pre-change build path on the existing fixtures.
- [ ] 5.5 `tests/test_clustered.py::test_lazy_blocks_respect_encoding_specific_defaults` — Rung5 encoding produces blocks whose features pass `Dictionary.__post_init__`'s amp-knob length check; the lazy path didn't bypass validation.
- [ ] 5.6 Run the existing `tests/test_clustered.py`, `tests/test_clustered_golden.py`, `tests/compression/test_epoch_clustered_consume.py` suites unchanged. They're the regression net for the user-facing shape.

## 6. Validation

- [ ] 6.1 `openspec validate add-lighter-per-block-container --strict`.
- [ ] 6.2 Full `pytest` non-sklearn pass.
- [ ] 6.3 Real-fixture amortised-benchmark re-run per §4.1.
- [ ] 6.4 Commit + PR.

## 7. Closing

- [ ] 7.1 Update the proposal-track CHANGELOG `## Unreleased` entry once the impl lands (this proposal's section).
