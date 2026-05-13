## 1. `ClusteredDictionary` primitive

- [ ] 1.1 Create `polygram/clustered.py` (new module). Add `ClusteredDictionary` frozen dataclass with fields `blocks: list[Dictionary]`, `cross_block_pairs: Mapping`, `block_topology: Graph | None`, `block_formation: BlockFormation`.
- [ ] 1.2 Add `BlockFormation` config dataclass: `strategy: Literal["cosine", "co_firing", "user_declared"] = "cosine"`, `cosine_threshold: float = 0.3`, `block_size_max: int | None = None` (default from encoding's `max_features`), `firing_corpus: Sequence[str] | None = None` (required when `strategy="co_firing"`).
- [ ] 1.3 Add construction-time validation: every block ≤ `encoding.max_features`; every cross-block edge references valid (block, feature) coordinates; every feature appears in exactly one block (hard partition invariant in v1).
- [ ] 1.4 Add `ClusteredDictionary.n_features` (total features summed across blocks), `.n_blocks`, `.encoding` (shared across all blocks; validated in `__post_init__`).
- [ ] 1.5 Unit tests: construction with 0 blocks raises; mismatched encodings across blocks raises; duplicate (block, feature) raises; valid constructions round-trip through `to_dict` / `from_dict`.

## 2. Block-formation strategies

- [ ] 2.1 Refactor `polygram/compression/epoch.py:_compute_cosine_graph` into a shared helper at `polygram/clustered.py` (or `polygram/_block_formation.py`). `EpochCompressor` continues to call it through its existing call site. No behaviour change.
- [ ] 2.2 Implement `BlockFormation.cosine(...)`: greedy seeded coverage over the cosine pair graph, producing ≤K-feature blocks. Reuses the panel-selection logic from `_select_panels` but generalised to arbitrary K (not pinned at 8). Returns `(blocks, cross_block_edges)`.
- [ ] 2.3 Implement `BlockFormation.co_firing(...)`: clusters features by their activation patterns across the firing corpus. Reuses `_compute_firing_rates_and_residuals`. Returns the same shape as cosine.
- [ ] 2.4 Implement `BlockFormation.user_declared(hierarchy, encoding)`: takes an existing `hierarchy: dict[cluster -> list[feature]]` and produces blocks, splitting any cluster > K into multiple blocks if needed.
- [ ] 2.5 Unit tests for each strategy on synthetic SAE fixtures with known cluster structure: cosine recovers planted antipodal clusters; co_firing recovers planted co-firing clusters; user_declared respects the supplied hierarchy.

## 3. `BlockSparseGram` value type

- [ ] 3.1 Add `BlockSparseGram` frozen dataclass to `polygram/clustered.py`. Fields: `block_grams: list[np.ndarray]` (per-block dense complex grams, each `K_i × K_i`), `cross_block_edges: dict[tuple[int, int, int, int], complex]` keyed by `(block_i_idx, feat_i_idx, block_j_idx, feat_j_idx)`.
- [ ] 3.2 Add `BlockSparseGram.entries() -> Iterator[(global_i, global_j, value)]` yielding non-zero Gram entries lazily.
- [ ] 3.3 Add `BlockSparseGram.block_diagonal() -> list[np.ndarray]` returning the per-block grams.
- [ ] 3.4 Add `BlockSparseGram.cross_block_entries() -> Iterator[(global_i, global_j, value)]` yielding cross-block edges only.
- [ ] 3.5 Add `BlockSparseGram.to_dense() -> np.ndarray` escape hatch — materialises the full `N × N` complex matrix. Useful for small clustered dictionaries where dense fits in memory.
- [ ] 3.6 Add `BlockSparseGram.shape -> (int, int)`, `.density -> float` (fraction of non-zero off-block entries).
- [ ] 3.7 Unit tests: 4-block × 4-feature toy clustered dictionary, verify `.to_dense()` matches the analytic dense Gram for the same features.

## 4. `ClusteredDictionary.gram()`

- [ ] 4.1 Add `ClusteredDictionary.gram() -> BlockSparseGram`. For each block, delegate to `Dictionary.gram()` (existing quantum-encoding analytic path). For each cross-block edge, compute the direct decoder-vector inner product (no quantum encoding required — encoding-agnostic).
- [ ] 4.2 Unit test: dense Gram round-trip. Construct a 12-feature SAE (small enough for flat path), build a `ClusteredDictionary` with 3 blocks of 4, compute `clustered.gram().to_dense()`, and assert equality with the flat `Dictionary(... 12 features).gram()` modulo the cross-block-threshold sparsification.
- [ ] 4.3 Unit test: per-block tier-separation invariants hold (each block's diagonal sub-gram passes the existing tier-separation check).

## 5. Cross-block redundancy detection

- [ ] 5.1 Add `ClusteredDictionary.cross_block_redundant_pairs(threshold: float = 0.7) -> list[tuple[GlobalFeatId, GlobalFeatId, float]]`. Returns feature pairs where the cross-block decoder-vector cosine ≥ threshold.
- [ ] 5.2 Add a `CrossBlockRedundancyReport` dataclass to carry the result with metadata (threshold used, total pairs evaluated, pairs above threshold, pair-coverage by domain if domain tags are present).
- [ ] 5.3 Unit test: planted cross-cluster duplicates (two features with identical decoder vectors placed in different blocks) are caught and ranked first.
- [ ] 5.4 Unit test: threshold filtering — lowering threshold from 0.9 to 0.5 monotonically grows the result list.

## 6. `from_sae_lens` clustered path

- [ ] 6.1 Add `clustered: bool = False` and `block_formation: BlockFormation | None = None` kwargs to `from_sae_lens` in `polygram/sae_import.py`. Default `False` preserves all existing callers.
- [ ] 6.2 When `clustered=True`, the loader builds a `ClusteredDictionary` from the imported SAE rather than a single `Dictionary`. Block formation defaults to cosine with `block_size_max = encoding.max_features`.
- [ ] 6.3 `SelectionReport` extended to carry per-block coverage stats when the clustered path is used: `n_blocks`, `mean_block_size`, `n_cross_block_edges`.
- [ ] 6.4 Unit test: `from_sae_lens(... , clustered=True)` on a bundled JSON fixture produces a `ClusteredDictionary` whose blocks all satisfy the cap and whose cross-block edges respect the threshold.

## 7. `EpochCompressor` refactor (behaviour-preserving)

- [ ] 7.1 Reframe `_select_panels` as a thin wrapper over `ClusteredDictionary.from_sae_state(... , block_formation=cosine_seeded)`. The output panels are the new `ClusteredDictionary.blocks`. No behaviour change in the panels returned.
- [ ] 7.2 Reframe `_validate_panels` to iterate over `ClusteredDictionary.blocks`. No behaviour change in per-panel validation.
- [ ] 7.3 Reframe `_synthesize_validation_report` to consume the `ClusteredDictionary` plus per-panel `ValidationReport`s. No behaviour change in the synthesized cross-panel report.
- [ ] 7.4 Differential regression test: capture `EpochResult` from the pre-refactor implementation on the bundled GPT-2-small SAE fixture, freeze it as a reference, and assert the post-refactor implementation produces a byte-identical result on the same fixture + seeds. Differential test runs on every CI build.
- [ ] 7.5 Run the full `tests/test_compression*.py` suite — every existing test passes without modification.

## 8. Per-block Q-OrCA emission

- [ ] 8.1 Add `ClusteredDictionary.emit_qorca(output_dir: Path) -> dict[str, Path]`. Writes one `<block_id>.q.orca.md` per block plus a `manifest.json` describing block IDs, per-block feature lists, and cross-block adjacency.
- [ ] 8.2 Each per-block `.q.orca.md` is independently round-trippable through `q-orca verify` (the existing Q-OrCA verification pipeline). Integration test asserts this.
- [ ] 8.3 `manifest.json` schema: `{ "blocks": [{ "id": str, "machine": str (path), "features": [{ "name": str, "cluster": str | null }] }], "cross_block_edges": [{ "from": [block_id, feat_name], "to": [block_id, feat_name], "cosine": float }], "block_formation": { ... } }`.

## 9. Killer experiment + research note

- [ ] 9.1 Implement `examples/clustered_dictionary_walkthrough.py`. Loads a real GPT-2-small SAE fixture (the same one used in existing examples), builds a `ClusteredDictionary` with cosine block formation at K=8 (initially; bumped to K=16 if Rung3 is available, K=32 if Rung4 is shipped).
- [ ] 9.2 Compute flat baseline: full pairwise cosine + tier-separation analysis on the same 512-feature subset. Capture wall-clock and the set of redundant pairs (by tier-separation / behavioural-validation verdict).
- [ ] 9.3 Compute clustered version: same operations through `ClusteredDictionary`. Capture wall-clock and the redundant-pair set.
- [ ] 9.4 Compare: recall (clustered ∩ flat / flat), precision (clustered ∩ flat / clustered), speedup (flat_walltime / clustered_walltime). Target: recall ≥ 0.95, speedup ≥ 100.
- [ ] 9.5 Emit `docs/research/data/clustered_dictionary_recall.json` with the raw numbers.
- [ ] 9.6 Write `docs/research/clustered-dictionary-recall-vs-flat.md` documenting the methodology, results, and decision rule (does cosine clustering meet the recall target, or does v2 need co-firing).

## 10. Tests

- [ ] 10.1 `tests/test_clustered.py` covering all of §1–§5: construction, block formation, BlockSparseGram, cross-block redundancy detection.
- [ ] 10.2 `tests/test_sae_import.py::test_from_sae_lens_clustered_path` covering §6.
- [ ] 10.3 `tests/test_compression*.py` differential test from §7.4.
- [ ] 10.4 `tests/test_qorca_emit.py::test_clustered_emit_per_block` and `::test_clustered_emit_manifest` covering §8.
- [ ] 10.5 `tests/test_examples.py::test_clustered_dictionary_walkthrough_smoke` covering §9.

## 11. Closing

- [ ] 11.1 Run `pytest` full suite; verify no regressions.
- [ ] 11.2 Run `openspec validate clustered-dictionary-analysis --strict`.
- [ ] 11.3 Update `README.md`: "Scale" section mentioning the clustered primitive as the path to real-SAE-sized analyses.
- [ ] 11.4 Update `CHANGELOG.md` under unreleased: "ClusteredDictionary primitive: block-decomposition + sparse cross-block adjacency for SAE-scale analyses. See `docs/research/clustered-dictionary-recall-vs-flat.md`."

## 12. Findings PR (follows this change)

- [ ] 12.1 If the killer experiment's recall is below 0.95 on cosine clustering, open a follow-up `clustered-dictionary-cofiring-default` proposal switching the default `BlockFormation` strategy to `co_firing`.
- [ ] 12.2 If the wall-clock speedup is below 100×, audit the per-block Gram path for unnecessary statevector materialisation; potential follow-up: integrate with the q-orca MPS-transfer-matrix-contraction upgrade.
