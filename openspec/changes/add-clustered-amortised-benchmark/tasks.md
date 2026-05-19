## 1. Benchmark module

- [ ] 1.1 Create `polygram/benchmarks/__init__.py` (empty namespace).
- [ ] 1.2 Create `polygram/benchmarks/clustered_amortised.py` with `BenchmarkReport` frozen dataclass: `op_name: str`, `n_repeats: int`, `flat_wall_seconds_per_op: float`, `clustered_wall_seconds_per_op: float`, `clustered_block_formation_seconds: float`, `cross_over_n_repeats: int | None`, `polygram_version: str`, `git_commit: str`.
- [ ] 1.3 Add `run_amortised_benchmark(dictionary, decoder_vectors, *, op="gram", n_repeats=64, encoding=None) -> BenchmarkReport`. Time flat-equivalent: build a flat `Dictionary` once, call `op` `n_repeats` times. Time clustered: call `build_clustered_dictionary` once (the amortised cost), then call `op` `n_repeats` times across blocks.
- [ ] 1.4 Built-in op `"gram"`: each call materialises every block's `Dictionary.gram()` (analytic path).
- [ ] 1.5 Built-in op `"cross_block_overlap"`: sample 64 cross-block edges and compute direct decoder-vector cosines.
- [ ] 1.6 Built-in op `"redundancy"`: the single-shot `cross_block_redundant_pairs(0.7)` — the worst-case baseline established as ≈0.9× by PR #91. Pinned so the report explicitly shows the regime where clustering loses.
- [ ] 1.7 Compute `cross_over_n_repeats`: the smallest `k` where `clustered_block_formation_seconds + k * clustered_wall_seconds_per_op < (k+1) * flat_wall_seconds_per_op` (flat pays for one "build" + k ops). `None` if no cross-over within the measured range.

## 2. Example script

- [ ] 2.1 `examples/clustered_amortised_benchmark.py` mirroring `clustered_dictionary_walkthrough.py` style: `--sae`, `--n-features`, `--encoding`, `--op`, `--n-repeats`, `--output` args. `--op all` SHALL run the full op matrix (`gram`, `cross_block_overlap`, `redundancy`) in one invocation and emit a single combined JSON — keeps the "reproduce the table from the research note" path one command long.
- [ ] 2.2 Falls back to the bundled toy SAE fixture when `--sae` is omitted, so the script runs in CI.
- [ ] 2.3 Stamps `polygram_version` + `git_commit` at the top of the JSON output (matching the PR #91 nit fix).

## 3. Research note

- [ ] 3.1 `docs/research/clustered-amortised-benchmark.md` documenting methodology and cross-over result per encoding (MPSRung1, Rung3, Rung4).
- [ ] 3.2 Test the hypothesis from the proposal against the proposal's success criterion (`cross_over_n_repeats ≤ 8` on `gram` across MPS/Rung3/Rung4). If the measured cross-over fails the criterion, surface the dominant cost. Measure **RSS delta** around block formation (`resource.getrusage(RUSAGE_SELF).ru_maxrss` before/after) and **per-block Python object count** (`gc.get_count()` deltas or a `len(gc.get_objects())` snapshot at each phase). If RSS climbs disproportionately during block formation but the per-op path is also slow, `dataclasses.replace` round-trip is the prime suspect per `clustered-dictionary-recall-vs-flat.md`'s "Caveats". Record both numbers in the research note regardless of pass/fail — they're the diagnostic baseline future work needs.
- [ ] 3.3 Emit `docs/research/data/clustered_amortised_benchmark.json` with the raw numbers (one record per encoding × op combination).
- [ ] 3.4 Cross-link the new note from `clustered-dictionary-recall-vs-flat.md`'s "follow-up status" subsection — replacing the "filing pending" placeholder.

## 4. Tests

- [ ] 4.1 `tests/benchmarks/test_clustered_amortised.py` covering: smoke test on the toy fixture (small `n_repeats`, all three ops); `cross_over_n_repeats` is non-None and ≥ 1 on a synthetic high-block-count fixture; `BenchmarkReport` round-trips through `to_dict()` / `from_dict()`.
- [ ] 4.2 `tests/test_examples.py::test_clustered_amortised_benchmark_smoke` covering the example script.

## 5. Validation

- [ ] 5.1 `openspec validate add-clustered-amortised-benchmark --strict`.
- [ ] 5.2 Full pytest pass.
- [ ] 5.3 Run the benchmark at full N=24,576 on the real SAE fixture; record results in `clustered-amortised-benchmark.md`.
- [ ] 5.4 Update `[[project-interpretability-bet]]` memory entry with the measured cross-over numbers and a direct link to the new `docs/research/clustered-amortised-benchmark.md` note — the "smaller + cheaper" sub-claim gets its first downstream throughput data plus a navigable pointer from the memory index. Mirror the cross-link inside the memory entry's body so future sessions land on the writeup in one click.

## 6. Closing

- [ ] 6.1 Commit + PR.
