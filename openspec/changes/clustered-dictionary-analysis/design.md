## Context

`Dictionary` and its primitives (`gram`, `cancellation`, Q-OrCA
emit, behavioural validation) all assume a single dictionary that
fits inside one encoding's Hilbert space — at most 8 features on
MPSRung1, 16 on Rung3, 32 on the proposed Rung4, or `2**n_qubits`
on HEA. Real SAEs ship at 16k–1M features per layer, three to five
orders of magnitude past any single-encoding cap.

Two distinct walls separate today's polygram from real-SAE-scale
analysis:

1. **Per-encoding feature cap** — each encoding's Hilbert space
   limits how many features fit in one `Dictionary`. The
   `per-encoding-feature-cap` change (in flight) corrects today's
   uniform 8-cap to the actual per-encoding values. Combined with
   `Rung4` (also in flight) the practical cap reaches 32 per
   dictionary, or `2**n_qubits` for HEA — far below 16k.

2. **N² Gram explosion** — even with infinite Hilbert space, a
   1M-feature SAE has 10¹² pairs. At 1 µs per overlap that's
   11 days; at 1 ms it's 32 years. Bond-dimension or qubit-count
   upgrades cannot fix N² — only structured sparsity can.

This change addresses wall #2. Wall #1 is independent and
proceeds in parallel.

The N² problem is dischargeable because real SAEs are
structurally sparse:

- The `sae-geometry-regimes` change (merged 2026-05-10)
  measured five SAEs across audio + text + 1.7B + 8B scales
  and JumpReLU + TopK training recipes. All five sit on a
  quasi-uniform sphere: mean off-diagonal cosine ≈ 0, std
  0.016–0.056. **Most pairs have negligible overlap.**
- The pairs that matter for redundancy / interference analysis
  are the sparse minority with cosine above some threshold
  (typically 0.3–0.7 depending on application).
- The full `N × N` Gram is therefore wasteful: a block-diagonal
  + sparse off-diagonal representation captures the same
  signal at ~10⁴–10⁵× less compute.

A latent insight: polygram already implements much of this in
`polygram/compression/epoch.py`. `_select_panels` greedily forms
≤8-feature panels from the cosine pair graph;
`_validate_panels` runs per-panel behavioural validation;
`_synthesize_validation_report` aggregates cross-panel evidence
into a single `ValidationReport`. The pattern is **block
decomposition → per-block analysis → cross-block aggregation**.
It's locked inside compression and is duplicated in spirit by
every other primitive that wants to scale.

This change extracts the pattern into a reusable
`ClusteredDictionary` primitive that compression, cancellation,
Gram-export, and Q-OrCA emission can all consume.

## Goals / Non-Goals

**Goals:**

- A first-class `ClusteredDictionary` primitive that holds N
  features as a list of ≤K-feature blocks plus a sparse
  cross-block adjacency.
- Block-formation strategies grounded in the existing polygram
  signals (cosine on decoder vectors, co-firing on activation
  traces, user-declared via `hierarchy`).
- Block-diagonal + sparse off-diagonal Gram representation
  (`BlockSparseGram`) that doesn't require materialising the
  full N × N dense matrix.
- One headline cross-block analytic primitive
  (`cross_block_redundant_pairs`) demonstrating value beyond
  pure decomposition.
- Refactor `EpochCompressor`'s panel logic to consume
  `ClusteredDictionary`. Byte-identical compression output on
  shipped fixtures.
- Per-block Q-OrCA emission via a manifest.
- Empirical demonstration: >95% redundancy-pair recall against
  a flat-baseline at ~250× speedup on a real GPT-2-small SAE
  fixture (~512 features).

**Non-Goals:**

- **Cross-block phase cancellation.** Features in different
  blocks live in different Hilbert spaces. There is no shared
  δ to optimise. Cross-block "cancellation" reduces to
  compression (drop one of the pair) and that path already
  exists in `EpochCompressor`.
- **Q-OrCA multi-machine composition.** The orca-lang
  multi-machine syntax (`invoke: ChildMachine`) could in
  principle compose blocks into one logical machine; v1 ships
  a flat manifest (JSON + one .q.orca.md per block) instead.
  Multi-machine emission is deferred until there's a concrete
  consumer that wants it.
- **Soft / overlap clustering.** v1 ships hard partitioning
  (each feature in exactly one block). Overlap clustering
  eliminates boundary-artifact pairs but complicates Gram
  ownership, compression semantics, and the cross-block edge
  list. Defer until the boundary problem proves real on the
  killer-experiment fixture.
- **Reblocking under iterative compression.** When
  `EpochCompressor` zeros features mid-run, the partition may
  no longer reflect the post-compression geometry. v1 keeps
  the initial partition fixed across all iterations; the
  trade-off is documented in the research note. Dynamic
  reblocking is a follow-up.
- **A new compression strategy.** The compression-side change
  is a pure refactor: `EpochCompressor`'s output is
  byte-identical to today on shipped fixtures.
- **A `clustered-cancellation` primitive.** Cross-block
  cancellation isn't well-defined (see above). Per-block
  cancellation works today via the existing `Cancellation`
  primitive on a single block's `Dictionary` — no new API
  needed.

## Decisions

**Decision 1 — `ClusteredDictionary` is a new primitive, not a
subclass of `Dictionary`.**

A subclass would inherit `.gram()` semantics (dense N × N) that
clustered-Gram explicitly violates. The two types should be
sibling primitives, not parent/child. `Dictionary` stays the
"fits in one Hilbert space" primitive; `ClusteredDictionary` is
the "doesn't fit, decomposed" primitive.

**Decision 2 — Block formation defaults to cosine clustering.**

Cosine clustering reuses `_compute_cosine_graph` from
`EpochCompressor` and is the cheapest signal to compute. It
captures the geometric structure that drives most SAE
redundancies. Co-firing clustering is more principled
functionally but requires a forward-pass corpus, so it's the
second-cheapest option. User-declared (via `hierarchy`) is the
escape hatch for callers with external clustering signal.

Default is cosine. The two alternatives are selectable via the
`BlockFormation` config.

**Decision 3 — Block size defaults to the encoding's
`max_features`.**

A `ClusteredDictionary` constructed with `encoding=MPSRung1()`
gets K=8 per block; with `Rung3()` it gets K=16; with `Rung4()`
or HEA at suitable `n_qubits` it gets K=32 or higher. The cap
flows from `encoding.max_features`, which the
`per-encoding-feature-cap` change establishes. Callers can
override with `block_size_max` if they want smaller blocks for
faster per-block analysis at the cost of more cross-block
edges.

**Decision 4 — Cross-block adjacency is threshold-based, not
top-k.**

A cosine threshold (default 0.3) admits or rejects each
candidate cross-block pair. Top-k would have a fixed memory
budget but bury rare strong outliers below the cutoff.
Threshold-based scales with the SAE's geometry: uniform-sphere
SAEs produce few edges; concentrated SAEs produce more. The
threshold is exposed for tuning.

**Decision 5 — `BlockSparseGram` is a custom value type, not a
`scipy.sparse` matrix.**

`scipy.sparse` packs everything into a flat (i, j, value) form,
losing the block structure that downstream consumers want to
exploit (e.g., per-block tier separation analysis). The custom
type stores `block_grams: list[np.ndarray]` plus
`cross_block_edges: dict[(b_i, f_i, b_j, f_j) → complex]`.
A `.to_dense()` escape hatch materialises the full N × N matrix
for small-N callers who need it.

**Decision 6 — `EpochCompressor` refactor is behaviour-preserving.**

`_select_panels`, `_validate_panels`, and
`_synthesize_validation_report` are reimplemented as wrappers
over `ClusteredDictionary` methods. The `EpochResult` and
`EpochReport` shapes are unchanged. Existing tests pass without
modification — this is the load-bearing invariant that
de-risks the refactor.

**Decision 7 — Per-block Q-OrCA emission uses a manifest, not
multi-machine composition.**

Each block emits a `<block_id>.q.orca.md` and a top-level
`manifest.json` lists the blocks plus their cross-block
adjacency. This makes the emission verifiable per-block (each
machine is independently runnable through Q-OrCA) without
introducing a Q-OrCA multi-machine dependency.

When the orca-lang multi-machine feature stabilises and there's
a concrete consumer for cross-block composition, a `v2`
emission path can layer on top.

**Decision 8 — The killer experiment is recall vs flat baseline
on real GPT-2-small SAE features.**

Concrete fixture: ~512 GPT-2-small SAE features chosen via the
existing `from_sae_lens` import path. Two pipelines:

1. **Flat baseline**: full pairwise cosine + behavioural
   validation across all 512² pairs.
2. **Clustered**: 16-32 blocks of 16-32 features each
   (depending on encoding choice); per-block validation +
   cross-block adjacency above cosine 0.3.

Metric: fraction of redundant pairs (by behavioural-validation
verdict) caught by the clustered path. Target: ≥95%. Secondary
metric: wall-clock ratio (target: ≥100× speedup).

If recall is below 95%, the clustering strategy needs work
(co-firing instead of cosine, different threshold, finer
blocks). The experiment is the verdict on whether v1 ships
with cosine defaults or needs a tuning pass.

## Risks / Trade-offs

**Risk:** clustering quality is data-dependent.

If the SAE's redundancy structure doesn't align with cosine
similarity (e.g., functionally-equivalent features with
orthogonal decoders), cosine clustering will miss
cross-cluster redundancies. The cross-block adjacency catches
some of these (threshold-based on direct cosine, which is the
same signal), but anything below the cosine threshold is
invisible.

Mitigation: co-firing clustering is the second strategy and
captures functional similarity. The killer experiment
quantifies recall; if cosine underperforms, the research note
documents which strategy works on which SAE family.

**Risk:** `EpochCompressor` refactor regresses on edge cases.

Compression has been in production for several PRs and has
accumulated subtle behaviour around panel selection, panel
deduplication, and cross-panel evidence aggregation.

Mitigation: a "compression byte-identical regression suite"
task in §5 of `tasks.md` runs the existing
`EpochCompressor` tests *plus* a new differential test that
compares the new path's `EpochResult` against a frozen
reference checkpoint produced by the old path on the bundled
SAE fixture. The refactor lands only when this differential
test passes.

**Risk:** `BlockSparseGram` as a custom type creates new API
surface.

Callers who use `dictionary.gram()` today get a `np.ndarray`.
Callers who use `clustered_dictionary.gram()` get a
`BlockSparseGram`. The two are not interchangeable.

Mitigation: `.to_dense()` provides an escape hatch for small
clustered dictionaries. The two APIs are explicitly different
because the underlying object IS different — the type-system
distinction reflects the semantic distinction.

**Risk:** v1's hard partitioning misses redundancies at cluster
boundaries.

If features A and B are *just* below the cosine threshold to
land in the same cluster but *just* above the cross-block edge
threshold, they end up in different blocks AND get a cross-
block edge — that's fine. But if A and B are below both
thresholds yet functionally redundant, they're invisible.

Mitigation: the killer experiment's recall metric quantifies
this directly. If recall is below target, the proposal
revisits soft clustering as a follow-up change. v1 ships hard.

**Risk:** scope creep into compression-strategy changes.

The refactor temptation: "while we're in `EpochCompressor`, why
not also add overlap clustering / change panel ordering /
generalise the priority function?"

Mitigation: explicit Non-Goal. The refactor is
behaviour-preserving. Anything more goes in a follow-up
change.

## Sequencing

**Phase 0** — Prerequisite: `per-encoding-feature-cap`
(PR #42, in flight) merges. This change consumes
`encoding.max_features` for the per-block cap default.

**Phase 1** — `ClusteredDictionary` + `BlockSparseGram` types,
construction from an SAE checkpoint, cosine block formation.

**Phase 2** — `cross_block_redundant_pairs` analytic primitive.

**Phase 3** — `EpochCompressor` refactor (behaviour-preserving).
The byte-identical differential test is the gate.

**Phase 4** — Per-block Q-OrCA emission + manifest.

**Phase 5** — Killer experiment on real SAE fixture; research
note with recall/speedup numbers.

Each phase is mergeable independently if the change wants to
ship in slices. v1 closes when all five phases are landed.

## Migration Notes

No migration required for existing callers. Every existing path
(single `Dictionary`, `from_sae_lens` without `clustered=True`,
`EpochCompressor.run()`) preserves byte-identical output.

New workflows opt in by constructing `ClusteredDictionary`
directly or by setting `clustered=True` on `from_sae_lens`.
