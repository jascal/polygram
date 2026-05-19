## 1. Benchmark module

- [ ] 1.1 Create `polygram/benchmarks/__init__.py` (empty namespace).
- [ ] 1.2 Create `polygram/benchmarks/clustered_amortised.py` with `BenchmarkReport` frozen dataclass: `op_name: str`, `n_repeats: int`, `flat_wall_seconds_per_op: float`, `clustered_wall_seconds_per_op: float`, `clustered_block_formation_seconds: float`, `cross_over_n_repeats: int | None`, `polygram_version: str`, `git_commit: str`.
- [ ] 1.3 Add `run_amortised_benchmark(dictionary, decoder_vectors, *, op="gram", n_repeats=64, encoding=None) -> BenchmarkReport`. Time flat-equivalent: build a flat `Dictionary` once, call `op` `n_repeats` times. Time clustered: call `build_clustered_dictionary` once (the amortised cost), then call `op` `n_repeats` times across blocks.
- [ ] 1.4 Built-in op `"gram"`: each call materialises every block's `Dictionary.gram()` (analytic path).
- [ ] 1.5 Built-in op `"cross_block_overlap"`: sample 64 cross-block edges and compute direct decoder-vector cosines.
- [ ] 1.6 Built-in op `"redundancy"`: the single-shot `cross_block_redundant_pairs(0.7)` — the worst-case baseline established as ≈0.9× by PR #91. Pinned so the report explicitly shows the regime where clustering loses.
- [ ] 1.7 Compute `cross_over_n_repeats`: the smallest `k` where `clustered_block_formation_seconds + k * clustered_wall_seconds_per_op < (k+1) * flat_wall_seconds_per_op` (flat pays for one "build" + k ops). `None` if no cross-over within the measured range.

## 2. Example script

- [ ] 2.1 `examples/clustered_amortised_benchmark.py` mirroring `clustered_dictionary_walkthrough.py` style: `--sae`, `--n-features`, `--encoding`, `--op`, `--n-repeats`, `--output` args.
- [ ] 2.2 Falls back to the bundled toy SAE fixture when `--sae` is omitted, so the script runs in CI.
- [ ] 2.3 Stamps `polygram_version` + `git_commit` at the top of the JSON output (matching the PR #91 nit fix).

## 3. Research note

- [ ] 3.1 `docs/research/clustered-amortised-benchmark.md` documenting methodology and cross-over result per encoding (MPSRung1, Rung3, Rung4).
- [ ] 3.2 Test the hypothesis from the proposal: measured cross-over within ~10× of the predicted ~2 ops? If much higher, surface the dominant cost (likely per-block `dataclasses.replace` round-trip — the caveat from `clustered-dictionary-recall-vs-flat.md`).
- [ ] 3.3 Emit `docs/research/data/clustered_amortised_benchmark.json` with the raw numbers (one record per encoding × op combination).
- [ ] 3.4 Cross-link the new note from `clustered-dictionary-recall-vs-flat.md`'s "follow-up status" subsection — replacing the "filing pending" placeholder.

## 4. Tests

- [ ] 4.1 `tests/benchmarks/test_clustered_amortised.py` covering: smoke test on the toy fixture (small `n_repeats`, all three ops); `cross_over_n_repeats` is non-None and ≥ 1 on a synthetic high-block-count fixture; `BenchmarkReport` round-trips through `to_dict()` / `from_dict()`.
- [ ] 4.2 `tests/test_examples.py::test_clustered_amortised_benchmark_smoke` covering the example script.

## 5. Validation

- [ ] 5.1 `openspec validate add-clustered-amortised-benchmark --strict`.
- [ ] 5.2 Full pytest pass.
- [ ] 5.3 Run the benchmark at full N=24,576 on the real SAE fixture; record results in `clustered-amortised-benchmark.md`.
- [ ] 5.4 Update `[[project-interpretability-bet]]` memory entry with the measured cross-over numbers — the "smaller + cheaper" sub-claim gets its first downstream throughput data.

## 6. Closing

- [ ] 6.1 Commit + PR.
