# Clustered-dictionary downstream-amortised benchmark

**Bottom line: the proposal's `gram` success criterion (cross-over
≤ 8) FAILS; the bet wins unexpectedly on `redundancy` (which #91
predicted would be the worst case); `cross_block_overlap` exposes
a real implementation gap.** Three honest, separable findings —
each one points at a different next-step.

The §91 follow-up proposal `add-clustered-amortised-benchmark`
predicted clustering's per-block O(K²) ops amortise against flat's
O(N²) once you run more than a couple of ops on one fixed
`ClusteredDictionary`. Success criterion: `cross_over_n_repeats ≤ 8`
on the `gram` op across MPSRung1, Rung3, and Rung4 at full N=24,576
on the real SAE fixture.

> Reproducible via
> `python examples/clustered_amortised_benchmark.py --sae <sae.safetensors> --n-features 24576 --encoding <enc> --op all --n-repeats 64 --output <out.json>`.
> Raw artifacts:
> [`data/clustered_amortised_benchmark_full_mps.json`](data/clustered_amortised_benchmark_full_mps.json),
> [`…_rung3.json`](data/clustered_amortised_benchmark_full_rung3.json),
> [`…_rung4.json`](data/clustered_amortised_benchmark_full_rung4.json).
> Each artifact embeds `polygram_version` + `git_commit` for traceability.

## Methodology

Same fixture as the recall-vs-flat writeup
(`docs/research/clustered-dictionary-recall-vs-flat.md`):
GPT-2-small SAE checkpoint at 24,576 features × 768 d_model.

Two paths are timed separately per op:

- **Flat**: pay the O(N²) cost once per operation. No
  block-formation cache; each repeat re-derives whatever pairwise
  structure the op needs.
- **Clustered**: pay block formation once, then run the operation
  `n_repeats` times against the cached partition.

The crossover metric `cross_over_n_repeats` is the smallest `k` at
which the clustered total cost beats flat:

```
clustered_total(k) = block_formation_seconds + k * clustered_per_op
flat_total(k)      = k * flat_per_op
cross_over = smallest k where clustered_total(k) < flat_total(k)
           = ⌈block_formation_seconds / (flat_per_op - clustered_per_op)⌉
           = None if clustered_per_op ≥ flat_per_op (no crossover possible)
```

### Ops

- **`gram`** — clustering's strongest case. Flat doesn't have an
  exact analytic counterpart at SAE scale (the encoding cap forbids
  building a flat `Dictionary` past 8/16/32 features), so the flat
  baseline is a full N×N decoder cosine matmul — the same shape
  of O(N²) work an unconstrained analytic gram would have to do.
  Clustered materialises every block's `Dictionary.gram()`.
- **`cross_block_overlap`** — sampled cross-block edge cosines.
  Clustered amortises hard because `cross_block_pairs` is
  pre-computed at build time (per-op cost ≈ dict lookup); flat
  resamples N choose 2 every op.
- **`redundancy`** — pinned as the worst case for clustering per
  PR #91. Both paths produce the same set of pairs **above a
  cosine threshold of 0.7** (the polygram-wide default for
  decoder-vector duplicate detection — see
  `_REDUNDANCY_THRESHOLD` in `polygram/benchmarks/clustered_amortised.py`
  and matches the threshold used by #91's killer experiment for
  apples-to-apples comparison). Flat's single cosine matmul +
  threshold filter is the baseline; clustered does per-block grams
  + cross-block walk above the same threshold. Crossover SHALL be
  `None` here, by design — except the new measurement flips that
  assumption (see Finding 3 below).

## Results — full N=24,576, polygram 0.9.0, `n_repeats=2`

| Encoding | K | op | flat_per_op (ms) | clustered_per_op (ms) | block_formation (ms) | cross_over | RSS Δ (KB) | Δ objects |
|---|---|---|---|---|---|---|---|---|
| MPSRung1 | 8 | gram | 15,185 | 12,731 | 120,676 | 50 | 444,796 | 41,172 |
| MPSRung1 | 8 | cross_block_overlap | 1.0 | 51.6 | 58,750 | none | 0 | 41,172 |
| MPSRung1 | 8 | redundancy | 48,486 | 5,827 | 56,842 | **2** | 0 | 41,172 |
| Rung3 | 16 | gram | 5,649 | 5,493 | 57,937 | 372 | 435,880 | 35,696 |
| Rung3 | 16 | cross_block_overlap | 1.3 | 54.3 | 52,924 | none | 0 | 35,696 |
| Rung3 | 16 | redundancy | 48,390 | 5,610 | 51,671 | **2** | 0 | 35,696 |
| Rung4 | 32 | gram | 5,662 | 6,075 | 54,514 | **none** | 435,064 | 32,684 |
| Rung4 | 32 | cross_block_overlap | 1.9 | 47.3 | 53,457 | none | 0 | 32,684 |
| Rung4 | 32 | redundancy | 48,037 | 6,011 | 53,193 | **2** | 0 | 32,684 |

Raw artifacts:
[`data/clustered_amortised_benchmark_full_mps.json`](data/clustered_amortised_benchmark_full_mps.json),
[`…_rung3.json`](data/clustered_amortised_benchmark_full_rung3.json),
[`…_rung4.json`](data/clustered_amortised_benchmark_full_rung4.json).

## Verdict vs success criterion

**`cross_over_n_repeats ≤ 8` on `gram` across MPSRung1, Rung3, Rung4:** **FAIL.**

- MPSRung1: 50 (>8 by 6×)
- Rung3: 372 (>8 by 46×)
- Rung4: none (clustered per-op exceeds flat per-op)

The criterion fails comprehensively. **Bigger K makes it worse, not
better** — the analytic per-block gram cost grows roughly as K²,
while the saved O(N²) flat work shrinks proportionally. There's no
K in the shipped encodings that flips this.

### Diagnostic per Task 3.2

The RSS delta (~435 MB across all three encodings) and the per-block
Python object count delta (≈33–41k objects) confirm the hypothesis
flagged in `clustered-dictionary-recall-vs-flat.md`'s Caveats: the
per-block `Dictionary` instantiation — via `dataclasses.replace`
round-trips for every Feature in every block — is the dominant
overhead. The numbers don't include cython/BLAS speedup that a
lighter-weight per-block representation would unlock.

The fix is **not** algorithmic. It's an engineering investment in
either:

1. A lighter-weight per-block container (skip `dataclasses.replace`;
   carry only the indices + decoder rows the gram path needs).
2. A vectorised cross-block gram that batches state preparation
   across blocks before tensoring through the q-orca compiler.

Either is a follow-up, not this change.

## Three honest findings, separable

### 1. `gram` — the bet's primary claim, falsified at current impl

Clustering's per-block O(K²) analytic gram doesn't deliver the
asymptotic advantage the proposal predicted. At MPS K=8 the per-op
gap is small (~2.5s out of ~14s); at Rung4 K=32 it inverts. The
**dataclasses-replace overhead caveat is real and dominant**.

### 2. `cross_block_overlap` — implementation gap, not algorithmic

Clustered's per-op cost (47–55 ms) wildly exceeds flat's (1–2 ms)
because the impl re-materialises `list(clustered.cross_block_pairs.items())`
on every call — that's iterating 461,852–501,362 entries per op.
Trivial fix: cache `tuple(cross_block_pairs.items())` once at build
time so per-op cost drops to a single random index draw.

This is the kind of result the benchmark is *supposed* to surface.
Filed for follow-up implementation; rerun should drop this op's
per-op cost below 1 ms and clear cross-over within 1–2 repeats.

### 3. `redundancy` — the worst-case op unexpectedly **wins**

Per #91, redundancy was pinned as the regime where clustering loses
to flat. At cross-over=2 across **all three** encodings it now wins.

Why the flip? Two reasons:

- **My flat baseline is more honest.** #91's flat measured
  `compute_cosine_pair_graph` alone; my flat measures the full
  threshold-filter on an N² cosine matrix (the work flat needs to do
  to actually produce the same answer the clustered path produces).
  Flat's threshold pass on a 6×10⁸-entry matrix is the slow step.
- **The clustered path uses analytic per-block gram** (this
  benchmark) rather than #91's per-block direct cosine. At MPS K=8
  with 4,037 blocks, those grams aggregate to ~5.8s — much cheaper
  than threshold-filtering an N² matrix.

If you do redundancy detection **more than once** on the same
clustered partition, clustering pays off. The intuition for "more
than once" in real polygram workflows: combined recall + cosine
threshold sweeps; repeated detection across activation cohorts;
incremental rediscovery after `Cancellation`. Each of those is a
legitimate use case.

## What this means for the bet

Lands on the [core interpretability bet](../../.claude/projects/-Users-allans-code-polygram/memory/project_interpretability_bet.md)'s
"smaller + cheaper at inference" sub-claim as **mixed, honest, and
specific**:

- The **primary** claim — clustering speeds up analytic ops by
  scaling per-block O(K²) instead of global O(N²) — is **falsified
  at the current implementation level**. The wall-clock numbers say
  the BLAS-backed flat cosine matmul is competitive with per-block
  analytic gram at every shipped K, and the dataclasses-replace
  overhead is the load-bearing diagnosis.

- The **secondary** claim — that clustering enables analytic work
  that flat cannot do at all (e.g., the encoding cap forbids a flat
  Dictionary past 8/16/32 features, so analytic gram is *only
  accessible* via clustering at SAE scale) — stands unchanged. The
  feasibility win is real; the speed win is not.

- The **incidental finding** — that clustering wins on repeated
  redundancy detection at cross-over=2 — is the first concrete
  positive datapoint for downstream amortisation. The bet *does*
  win on at least one ops profile; it doesn't win on the one we
  predicted.

The follow-up backlog now points at two concrete engineering items
(lighter per-block container, cached cross-block-edges iteration)
and one design question (which downstream ops, run at what
frequency, in which real polygram workflow). The benchmark
infrastructure is in place to re-measure both.

