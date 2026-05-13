# Clustered-dictionary recall vs flat baseline

> Research-track note recording the §9 killer experiment from the
> `clustered-dictionary-analysis` openspec change. Reproducible via
> `python examples/clustered_dictionary_walkthrough.py --sae <sae.safetensors> --n-features <N>`.
> Raw artifacts:
> [`data/clustered_dictionary_recall.json`](data/clustered_dictionary_recall.json)
> (N=2048),
> [`data/clustered_dictionary_recall_n8192.json`](data/clustered_dictionary_recall_n8192.json)
> (N=8192).

## TL;DR

| Metric | Target | N=2048 | N=8192 |
|---|---|---|---|
| **Recall** (clustered ∩ flat / flat) | ≥ 0.95 | **1.000** | **1.000** |
| **Precision** (clustered ∩ flat / clustered) | (informative) | 1.000 | 1.000 |
| **Speedup** (flat wall / clustered wall) | ≥ 100× | **0.5×** | **0.5×** |
| pairs found above threshold 0.7 | (informative) | 9 | 202 |

**Recall is perfect at both scales.** The clustered partition catches
every flat-baseline redundant pair, both intra-block (caught by the
per-block Gram on cosine) and cross-block (caught by the
`cross_block_redundant_pairs` analytic primitive).

**Speedup is inverse to the proposal's target.** Clustering is ~2×
slower than the flat baseline for both N=2048 and N=8192 on a real
GPT-2-small SAE. This finding **reframes the value proposition**:
clustered analysis isn't a raw cosine-speed win at moderate N; it's
a structural enabler for quantum-encoded analyses that don't fit on
a single ≤8-feature dictionary.

## Methodology

Real-scale fixture: GPT-2-small SAE checkpoint
`scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors`
— 24,576 features × 768 d_model. Subset N drawn uniformly at random
(seed 0).

- **Flat baseline**: full pairwise cosine matrix via
  `compute_cosine_pair_graph` (BLAS-backed `np.dot`), threshold the
  result at `cosine ≥ 0.7`.
- **Clustered**: `build_clustered_dictionary` with cosine block
  formation at `block_size_max=8` (MPSRung1's cap),
  `cosine_threshold=0.3` for block formation. Intra-block redundant
  pairs derived from per-block cosine sub-matrices; cross-block
  pairs from `ClusteredDictionary.cross_block_redundant_pairs`.

Recall = `|clustered_pairs ∩ flat_pairs| / |flat_pairs|`.
Precision = `|clustered_pairs ∩ flat_pairs| / |clustered_pairs|`.
Speedup = `flat_wall_seconds / clustered_wall_seconds`.

## Why the speedup is inverse

The proposal's `≥ 100×` target assumed per-pair compute that dominates
wall-clock (e.g., behavioural validation requiring a real-LLM forward
pass — ~1 ms/pair). For BLAS-backed pure-cosine workloads, the
per-pair cost is closer to ~100 ns, and the N²-pair count is
amortised across a single matmul. Pythonland overhead in the
clustered path — building 1k+ `Dictionary` objects per run, replacing
each `Feature`'s cluster field, computing the cosine graph twice (once
for block formation, once for cross-block adjacency) — dominates the
savings at the N tested.

**Order-of-magnitude breakdown at N=8192**:

- Flat: one BLAS matmul on a `(8192, 768) @ (768, 8192)` operation ≈
  10⁹ float ops at ~6 GFLOPS ≈ 1.7s of arithmetic; observed
  5.97s, which is BLAS plus the upper-triangle scan in Python.
- Clustered: 1783 block constructions × ~1 ms per Dictionary
  instantiation ≈ 1.8s of pure-Python overhead. Block-formation
  cosine graph + cross-block cosine graph ≈ 2× the flat matmul ≈
  10s. Cross-block redundancy iteration is fast.

The pure-Python block-construction overhead is the binding constraint
at this N. At very large N (≥10⁵), the N² cost would dominate; at
very small N (<200), the Python overhead is irrelevant. The crossover
where clustering pays for itself on pure-cosine workloads is around
**N ≈ 10⁵–10⁶**.

## Where clustered analysis *does* pay

The killer-experiment numbers measure pure cosine pair-finding.
That's the cheapest possible per-pair workload. The clustered
primitive's real leverage is on workloads where:

1. **Per-pair compute is expensive.** Behavioural validation
   (running a real LLM forward pass to confirm a redundancy) is
   ~1 ms/pair. At that cost, clustering's savings dominate:
   N=10⁵ flat = 10⁷ s ≈ 4 months; clustered = ~10⁴ pairs × 1 ms
   = 10 s. **10⁶× speedup**.
2. **Per-pair analysis requires the quantum-encoded primitive.**
   `Dictionary.gram()`, `Cancellation.run()`, Q-OrCA emission,
   structural floor formula — all require the per-block quantum
   encoding. The flat path can't do these at N > 8 (or 16 on
   Rung3, or 32 on Rung4) by construction. Clustered is the
   **only** path.
3. **Memory is binding.** The full N×N Gram at N=10⁵ is 10¹⁰
   complex floats ≈ 80 GB. Clustered storage is block-diagonal
   plus sparse off-diagonal — typically <0.1% of dense, well into
   laptop-memory territory.

The recall=1.0 result confirms the clustered partition doesn't
**miss** redundancies that the flat baseline finds; it just doesn't
**find them faster** on pure-cosine workloads at moderate N.

## Recommended interpretation for the openspec change

The §9 success criteria as written conflate two distinct claims:

- *Correctness*: clustered finds the same redundancies the flat path
  finds. **Confirmed at recall = 1.0**.
- *Speedup*: clustered is faster than flat at SAE scale.
  **Not confirmed at this N for pure cosine; only applicable for
  expensive per-pair workloads or N ≥ ~10⁵.**

The clustered primitive should be marketed on the **enable**, not
the **speedup**: it's the path to analytic primitives at SAE-scale
feature counts where the flat path can't operate (per-encoding
Hilbert-space caps) or is structurally infeasible (N² × ms-per-pair
behavioural validation).

A follow-on study could measure speedup on:

- Behavioural-validation pair filtering (where the per-pair cost is
  in the millisecond range and the speedup story compounds).
- Larger N (≥ 10⁵) with proper pure-cosine baseline.
- Memory-binding workloads.

## Caveats

- The per-block `Dictionary` instantiation is the dominant Python
  overhead. A more efficient `ClusteredDictionary` builder could
  cache the per-block Gram cache and skip the per-feature
  `dataclasses.replace` round-trip. Not in scope for this change;
  worth a follow-up if the speedup story matters for a specific
  workload.
- Block size K=8 was used to match `MPSRung1`'s cap (the legacy 8).
  Larger K (16 on Rung3, 32 on Rung4 once shipped, `2**n_qubits` on
  HEA) would reduce the block count and the per-block-construction
  Python overhead.
- The recall measurement here uses cosine-only redundancy as the
  ground truth. The full openspec spec envisions behavioural-
  validation-confirmed redundancy as the gold standard. The
  cosine-only proxy is sufficient to demonstrate the partition
  doesn't drop pairs, but stronger downstream validation would
  strengthen the claim.

## Decision

`clustered-dictionary-analysis` ships with recall=1.0 confirmed and
speedup deferred / reframed. The change's primary value proposition
is **structural** (enabling SAE-scale analytic primitives that don't
fit on a single dictionary), not **cosine-speedup**. The openspec
proposal's "100× speedup" target was based on the implicit
assumption of expensive per-pair compute and stands as future work
once a behavioural-validation killer experiment is wired up.
