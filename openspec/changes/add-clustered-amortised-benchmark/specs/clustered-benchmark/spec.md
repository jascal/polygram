## ADDED Requirements

### Requirement: `run_amortised_benchmark` measures clustered vs flat throughput across repeated operations

`polygram.benchmarks.clustered_amortised.run_amortised_benchmark(dictionary, decoder_vectors, *, op, n_repeats, encoding)` SHALL be the entry point for measuring how clustering's `O(K²)`-per-block per-op cost amortises against flat's `O(N²)` per-op cost as the number of operations grows.

The benchmark SHALL time two paths separately:

- **Flat**: build a flat `Dictionary`, call `op` `n_repeats` times.
- **Clustered**: call `build_clustered_dictionary` once (the amortised one-time cost), then call `op` `n_repeats` times across blocks.

The returned `BenchmarkReport` SHALL carry per-op wall-clock seconds for both paths, the clustered one-time block-formation cost, the smallest `n_repeats` at which clustered total cost beats flat total cost (`cross_over_n_repeats`, `None` if no cross-over observed), and provenance fields (`polygram_version`, `git_commit`) matching the PR #91 traceability nit.

#### Scenario: gram-op cross-over is observable

- **WHEN** `run_amortised_benchmark(dictionary, decoder_vectors,
  op="gram", n_repeats=64)` is called on a real-scale SAE
  (N ≥ 1,024 features, K=8)
- **THEN** the returned `BenchmarkReport.cross_over_n_repeats` is a
  positive integer less than `n_repeats`, demonstrating that
  clustering's per-op advantage compounds beyond the one-time
  block-formation overhead

#### Scenario: redundancy-op pins the worst case

- **WHEN** `run_amortised_benchmark(..., op="redundancy",
  n_repeats=1)` is called
- **THEN** the report reproduces the PR #91 result —
  `clustered_wall_seconds_per_op >= flat_wall_seconds_per_op`
  within 10% — explicitly recording the regime where clustering
  does not pay off

#### Scenario: cross_block_overlap sample size is reported and representative

- **WHEN** `run_amortised_benchmark(..., op="cross_block_overlap",
  n_repeats=k)` is called on a `ClusteredDictionary` whose
  cross-block edge set has fewer than `64 * k` entries
- **THEN** the per-op sample size SHALL be capped at the available
  edge count and the report SHALL record both the requested and
  effective sample size, so reviewers can tell whether the
  measurement reflects a representative slice or a saturated walk
  of every edge

### Requirement: built-in operations cover the analytic surface

`run_amortised_benchmark` SHALL support at least three built-in `op` values:

- `"gram"` — per-block `Dictionary.gram()` materialisation across every block in the clustered partition.
- `"cross_block_overlap"` — sample-based direct decoder-vector cosines on the cross-block edge set.
- `"redundancy"` — the single-shot `cross_block_redundant_pairs(0.7)` baseline.

Unknown `op` values SHALL raise `ValueError` naming the supported set.

#### Scenario: unknown op raises

- **WHEN** `run_amortised_benchmark(..., op="cancellation")` is
  called (a plausible-but-not-yet-implemented op name)
- **THEN** a `ValueError` is raised naming the supported op set
