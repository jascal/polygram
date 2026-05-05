# Compression epoch — design notes

> Engineering note pointing at the
> [`add-compression-epoch`](../../openspec/changes/add-compression-epoch/)
> spec. Captures one observation about why the orchestrator works the
> way it does.

## Pointer

The full spec for the multi-panel orchestrator lives at
[`openspec/changes/add-compression-epoch/`](../../openspec/changes/add-compression-epoch/).
That directory carries `proposal.md`, `design.md` (the eight-decision
table), `tasks.md` (the implementation checklist), and the
capability-spec deltas under `specs/`.

## Why coverage over the cosine graph, not the redundancy graph

The redundancy graph (which pairs are "really" redundant) is exactly
what the validator computes — it requires forward passes. The cosine
graph (which pairs share decoder direction) is computable from the
checkpoint alone, in milliseconds. PR #18 (§4.1) measured Spearman
0.94 between Polygram-predicted overlap and the real-decoder
squared-cosine Gram on the §4.4-class SAE — i.e., decoder cosine is a
strong proxy for which pairs the validator will eventually flag.

The coverage metric is therefore: every pair in the cosine-similar
graph (`cos ≥ cosine_threshold`) appears in at least one panel. This
is a cheap, observable, well-defined target that bounds how many
panels we need without forcing the orchestrator to predict its own
ground truth.

## Why the GC / defrag analogy holds, and where it breaks

Bounded examination window (heap walk, panel size), expensive full
pass (graph walk, C(N, k) panels), non-local mutation (zeroing one
member changes the redundancy structure for survivors), fixed-point
semantics (mark-sweep until allocation stabilizes vs. iterate
until cluster set stabilizes) — all of these map cleanly between
GC/defrag and the multi-panel compression problem.

The analogy breaks at one important point: GC has perfect knowledge
of references (the heap is a graph with deterministic edges). The
SAE has only **samples** from a behavioural distribution over a
held-out prompt set. So the right reference frame is *statistical
clustering under a per-query budget*, not mark-and-sweep — and the
budget guard (the cross-entropy delta multiplier) is what keeps the
loop honest about the sampling-error origin of its data.

## Why the orchestrator runs panels sequentially

The validator's per-panel cost is dominated by GPT-2 forward passes.
GPT-2 small fits comfortably in a single process's memory; running
panels sequentially against a single in-process model amortizes the
weights across all panels. Process-pool parallelism would force
re-loading the model in each worker — for a 1000-panel run with
~500MB GPT-2 small, that's ~500GB of redundant model loading. The
math gets worse at Gemma-2-2B (~10GB per worker).

The honest concurrency unit is therefore *the CLI invocation*. Users
wanting concurrency shard at that layer: invocation 1 handles
anchors with priority rank 0–999, invocation 2 handles 1000–1999,
etc. Their per-panel `ValidationReport`s land on disk; a separate
aggregation utility (deferred until a real workload demands it)
merges them into a final compressed checkpoint.

## See also

- [`compression-action-design.md`](compression-action-design.md) —
  the per-panel half (component-first compression).
- [`compression-regrow-design.md`](compression-regrow-design.md) —
  the post-compression repopulation primitive.
- [`add-compression-epoch/proposal.md`](../../openspec/changes/add-compression-epoch/proposal.md) —
  what changes and why.
- [`add-compression-epoch/design.md`](../../openspec/changes/add-compression-epoch/design.md) —
  the eight-decision table.
- [`decoder-gram-validity.md`](decoder-gram-validity.md) — the §4.1
  evidence (Spearman 0.94) that the cosine graph is a strong proxy
  for the redundancy graph.
