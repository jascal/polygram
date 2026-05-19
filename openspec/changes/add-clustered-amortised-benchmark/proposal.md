## Why

PR #91 ran the §12 killer experiment from `clustered-dictionary-analysis`
at full SAE width (N=24,576) and produced a load-bearing mixed verdict:

- **Recall = 1.000** across MPSRung1 / Rung3 / Rung4 — clustering
  catches every redundant pair the flat O(N²) sweep finds.
- **Speedup ≈ 0.9×** — clustered is *slightly slower* than flat for
  one-shot redundancy detection, because `_form_blocks_cosine` itself
  does the same O(N²) cosine work the flat baseline does, then more
  block-formation / cross-block-edge work on top.

The honest framing in
[`docs/research/clustered-dictionary-recall-vs-flat.md`](../../../docs/research/clustered-dictionary-recall-vs-flat.md):
clustering's speedup story lives **downstream** of detection — every
later analytic operation (per-block Gram, Cancellation, Q-OrCA emit)
costs `O(K²)` per block instead of `O(N²)` globally, so the savings
amortise across repeated operations on a fixed
`ClusteredDictionary`. The current walkthrough script measures the
wrong workload for clustering to win on.

This change scopes the **downstream-amortised benchmark** that does
measure the right workload: pay block formation once, then drive a
volume of per-block analytic operations (Gram, Cancellation, or
both) and report throughput vs the flat equivalent. If clustering's
asymptotic advantage is real, this benchmark surfaces it.

The benchmark is also the missing empirical leg of the
[core interpretability bet](../../../../../.claude/projects/-Users-allans-code-polygram/memory/project_interpretability_bet.md)'s
"smaller + cheaper at inference" sub-claim. We don't need an LLM
inference-cost framework yet; per-block analytic throughput at SAE
scale is the right intermediate signal.

## What Changes

- New module `polygram/benchmarks/clustered_amortised.py` exposing
  `run_amortised_benchmark(clustered, *, ops, n_repeats) -> BenchmarkReport`
  and a flat-equivalent comparison path.
- `BenchmarkReport` dataclass carrying per-op wall-clock distributions,
  total throughput (ops/sec), memory peak (RSS delta), and the
  flat-vs-clustered ratio at every op count from 1 → `n_repeats`.
- Built-in `ops`: `"gram"` (per-block `Dictionary.gram()`),
  `"cross_block_overlap"` (sample cross-block-edge cosines),
  `"redundancy"` (the existing single-shot detection — pinned as
  the worst-case baseline for clustering, established as ≈0.9× by
  PR #91).
- New example `examples/clustered_amortised_benchmark.py` mirroring
  the structure of `clustered_dictionary_walkthrough.py`. Reuses the
  same SAE fixture path so callers can compare against the existing
  recall-vs-flat JSONs.
- New research note
  `docs/research/clustered-amortised-benchmark.md` capturing the
  cross-over point: at what `n_repeats` does clustered beat flat?
- Raw JSON output `docs/research/data/clustered_amortised_benchmark.json`
  with `polygram_version` + `git_commit` for traceability
  (matching the PR #91 nit fix).

## Impact

- **Affected specs**: new capability `clustered-benchmark`.
- **Affected code**: new `polygram/benchmarks/` subpackage; one new
  example; one research note + JSON.
- **No new runtime deps.** Numpy + the existing `clustered_dictionary`
  + `dictionary.gram()` machinery.
- **Risk**: low. Strictly additive — no existing API touched.

## Hypothesis to test

Clustered's per-op cost is `O(K²)` per block × `n_blocks ≈ N/K` blocks
= `O(N·K)` total, vs flat's `O(N²)`. At `K=8, N=24,576` that's a
~3,072× per-op advantage — *after* the one-time `O(N²)` block
formation. So clustered wins amortised whenever
`n_repeats > flat_cost / (clustered_per_op_cost − flat_per_op_cost)` ≈
1 (block formation is ~1× a single flat sweep). **Predicted cross-over:
~2 ops.** If the measured cross-over is dramatically higher,
something is dominating clustered's per-op path that shouldn't be
(e.g., per-feature `dataclasses.replace` round-trip — the caveat
already flagged in `clustered-dictionary-recall-vs-flat.md`).

## Explicitly out of scope

- LLM-level inference-cost metrics (per-token FLOPs, activation
  sparsity from routing). That's a layer up; needs sae-forge.
- Sub-O(N²) approximate block formation (LSH / HNSW). That's the
  *other* §12.2 follow-up — different problem, separate proposal.
- A CLI wrapper. The example is the surface; CLI can come later if
  the benchmark sees recurring use.
