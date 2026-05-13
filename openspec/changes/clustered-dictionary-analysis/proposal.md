## Why

Polygram's analytic primitives (`Dictionary.gram()`, `Cancellation`,
`BehaviouralValidator`, Q-OrCA emission) cap a dictionary at the
encoding's Hilbert-space dimension — 8 features for `MPSRung1`, 16
for `Rung3`, 32 for the proposed `Rung4`, and `2**n_qubits` for
`HEA_Rung2`. Even with the most aggressive single-encoding extension
the codebase can absorb (32-qubit `HEA_Rung2` → ~4×10⁹ feature
cap), the **pairwise N² Gram cost** dominates long before the
Hilbert-space cap matters:

| Features (N) | Pairs (N²) | Time @ 1 µs/pair | Time @ 1 ms/pair |
|---|---|---|---|
| 1k | 1M | 1 s | 17 min |
| 16k | 256M | 4 min | 3 days |
| 1M | 10¹² | 11 days | 32 years |

Real SAEs ship in the 16k–1M range (Gemma Scope, Llama Scope,
Qwen Scope). Today polygram works around this by **preselecting**
≤8-feature panels via `from_sae_lens(..., feature_ids=[...])` and
running per-panel analyses. That preselection step is itself a
form of clustering, but it's:

1. **Manual** — the user has to specify which features to study.
2. **Disconnected** — there's no single object that holds
   "selected feature *blocks* and their inter-block relationships".
3. **Duplicated inside compression** —
   `polygram/compression/epoch.py` already implements panel
   selection (`_select_panels`), per-panel validation
   (`_validate_panels`), and cross-panel evidence aggregation
   (`_synthesize_validation_report`). The same logic that
   compression needs for scale is exactly what every other
   primitive would need too — it's locked inside
   `EpochCompressor`.

This change extracts that panel logic into a shared
`ClusteredDictionary` abstraction, exposes the block-diagonal +
sparse off-diagonal Gram as a first-class primitive, and lets
compression / cancellation analysis / Q-OrCA emission all
consume the same clustered representation.

Empirical motivation: the `sae-geometry-regimes` change
(merged 2026-05-10) documented the "quasi-uniform sphere"
geometry across five SAEs (Whisper-tiny, Whisper-large-v1,
Qwen-Scope L14, Llama-Scope L0R, Llama-Scope L12R) — mean
off-diagonal cosine ≈ 0, std 0.016–0.056. The off-diagonal
Gram entries that *matter* (cosine > 0.5) are a sparse subset
of N². That sparsity is the lever this change exploits.

## What Changes

- **New `ClusteredDictionary` primitive** in
  `polygram/clustered.py` (new module). Holds `blocks: list[Dictionary]`
  (each ≤ `encoding.max_features` features), a sparse
  `cross_block_pairs` adjacency, and a `block_topology` graph.
- **Block-formation strategies** that turn an SAE checkpoint
  into a `ClusteredDictionary`: cosine clustering (reuses
  `_compute_cosine_graph` from `EpochCompressor`), co-firing
  clustering (uses firing-rate / activation traces), and
  user-declared partitioning. Strategy is selected by a
  `BlockFormation` config; defaults to cosine.
- **`BlockSparseGram` value type** as the return of
  `ClusteredDictionary.gram()`. Carries the per-block dense
  Gram matrices plus the sparse cross-block edge list. Exposes
  iteration helpers (`.entries()`, `.block_diagonal()`,
  `.cross_block_edges()`) and a `.to_dense()` escape hatch for
  small N where the dense form fits.
- **Cross-block redundancy detection primitive**:
  `ClusteredDictionary.cross_block_redundant_pairs(threshold)`
  returns the feature pairs whose direct decoder-vector cosine
  exceeds threshold. This is the headline new analytic
  capability — finds redundancies that span cluster boundaries
  without any quantum-encoding round-trip.
- **`EpochCompressor` refactor**: `_select_panels`,
  `_validate_panels`, `_synthesize_validation_report` are
  re-expressed as consumers of `ClusteredDictionary`.
  Compression behaviour is byte-identical to current behaviour
  on shipped fixtures — verified by an explicit regression test.
- **Per-block Q-OrCA emission**:
  `ClusteredDictionary.emit_qorca(output_dir)` writes one
  `.q.orca.md` per block plus a `manifest.json` listing the
  blocks and their cross-block adjacency. (No q-orca multi-machine
  composition in v1 — manifest-only.)

## Capabilities

### New Capabilities

- `clustered-dictionary`: the `ClusteredDictionary` primitive
  itself — construction from an SAE checkpoint via a block-
  formation strategy, sparse Gram computation, cross-block
  redundancy detection, per-block manifest emission.

### Modified Capabilities

- `sae`: `from_sae_lens` gains a `clustered: bool = False` flag
  (and a corresponding `block_formation: BlockFormation | None`)
  that, when set, returns a `ClusteredDictionary` instead of a
  single `Dictionary`. Default `False` preserves all existing
  callers byte-identically.
- `compression`: `EpochCompressor` internally consumes
  `ClusteredDictionary` for its panel decomposition. External
  API and reports unchanged; this is a refactor, not a behaviour
  change. Existing `EpochCompressor` tests pass without
  modification.

## Impact

- `polygram/clustered.py` — new module (`ClusteredDictionary`,
  `BlockFormation`, `BlockSparseGram`).
- `polygram/sae_import.py` — `from_sae_lens` gains the
  `clustered` flag and `block_formation` arg.
- `polygram/compression/epoch.py` — refactor of `_select_panels`,
  `_validate_panels`, `_synthesize_validation_report` to
  delegate to `ClusteredDictionary`.
- `tests/test_clustered.py` — new module, ~400 LOC of unit +
  integration tests including the "clustered vs flat baseline
  recall" experiment.
- `tests/test_compression*.py` — extended with explicit
  byte-identical regression tests confirming the refactor
  preserves shipped behaviour.
- `examples/clustered_dictionary_walkthrough.py` — new worked
  example on a small real SAE fixture (~512 features in 16-32
  blocks).
- `docs/research/clustered-dictionary-recall-vs-flat.md` — new
  research note documenting the >95% recall claim on a real
  GPT-2-small SAE at ~250× speedup, with reproducible artifacts
  in `docs/research/data/`.

**Sequencing dependencies:**
- Depends on `per-encoding-feature-cap` (in flight, PR #42)
  for per-block cap to be encoding-driven rather than a module
  constant.
- Does NOT depend on `add-rung4-encoding-mvp`; works with any
  shipped encoding. Larger per-block sizes (16-32 features)
  become available when Rung3 / Rung4 ship.
- Does NOT depend on the q-orca MPS-transfer-matrix-contraction
  upgrade; per-block Gram stays small enough for the
  statevector path.

**No breaking changes.** Every existing API path (single
`Dictionary` workflow, `from_sae_lens` without `clustered=True`,
`EpochCompressor` external surface) is preserved bit-for-bit.
The clustered primitive is opt-in.
