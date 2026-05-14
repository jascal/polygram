## 1. Code preparation

- [x] 1.1 Add `--assign-amp-knobs` CLI flag to `examples/rung_compression_coverage.py`. Threaded through to `EpochCompressor(assign_amp_knobs=...)`. Default False (back-compat with v2.1 results).
- [x] 1.2 Record the flag's value in the output JSON's `assign_amp_knobs` field for reproducibility.
- [x] 1.3 Print the flag's value in the per-run header so console captures include the cell identity.

## 2. Environment prep on the 2019 MBP

- [x] 2.1 Verify `pip install -e ".[behavioural]"` succeeds in the polygram `.venv`. (Per auto-memory, the user has a parallel personal-project `.venv` with torch 2.2.2 + transformers 4.49 + numpy<2 caps already working — this task is to ensure the polygram-dir `.venv` matches.) — `.venv` already had torch 2.2.2 + transformers 4.57.6 working; no install needed.
- [x] 2.2 Smoke-check: `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"` returns without error.
- [x] 2.3 Smoke-check: `python -c "from polygram import EpochCompressor; ec = EpochCompressor.__init__"` resolves without import error after the behavioural extras land.

## 3. Run the comparison (4 cells + 2 extended controls)

- [x] 3.1 Cell C1 (MPS baseline) → `axis1_mps.json`. 3540 zeroed; cumulative CE Δ = 0.291; `max_iterations` termination.
- [x] 3.2 Cell C2 (Rung4 default-knob control) → `axis1_rung4_amp_off.json`. 17 411 zeroed; cumulative CE Δ = 0.404; `max_iterations`.
- [x] 3.3 Cell C3 (Rung4 amp-on, load-bearing) → `axis1_rung4_amp_on.json`. **9 376 zeroed; cumulative CE Δ = 0.208 (lower than MPS)**; `max_iterations`.
- [x] 3.4 Cell C4 (Rung3 amp-on, generality) → `axis1_rung3_amp_on.json`. 6 465 zeroed; cumulative CE Δ = 0.235; `max_iterations`. Sits monotonically between MPS and Rung4 amp-on.
- [x] 3.5 Capture the console output for each cell in `runlog_<cell>.txt`. All four landed under `docs/research/data/runlog_C{1,2,3,4}_*.txt`.
- [x] 3.6 (added in flight) C3-extended (Rung4 amp-on, `--max-iterations 10`) → `axis1_rung4_amp_on_iter10.json`. 22 892 zeroed; cumulative CE Δ = 0.534; per-iter trajectory shows plateau (iters 0-5 ~3050) then collapse (iters 6-9), with anomalous iter-9 CE spike (0.168).
- [x] 3.7 (added in flight) C1-extended (MPS, `--max-iterations 10`) → `axis1_mps_iter10.json`. 10 847 zeroed; cumulative CE Δ = 0.432. Disambiguates the iter-9 spike as Rung4-amp-on-specific (MPS shows smooth monotonic decline, no comparable spike).

## 4. Compare + write up

- [x] 4.1 Compile the headline table comparing all cells. Done in `docs/research/rung4-viability-spike-v2.md` under "Axis 1 result (v2.2)".
- [x] 4.2 Apply the decision rule from `proposal.md`. **Verdict: PASS at 3-iter operating point** (Rung4 amp-on +165% features at −28% CE vs MPS, both metrics clear material thresholds). At 10-iter borderline PASS/PARTIAL (more features at moderately higher CE; features-per-CE-budget ratio stays 1.71× MPS).
- [x] 4.3 Write up the result as "## Axis 1 result (v2.2)" section. Mirrors v2.1 structure; includes 4-cell headline, verdict, per-iter trajectory observations, predictions-vs-actuals table, extended-iter analysis, and MPS 10-iter control / Pareto comparison subsections.
- [x] 4.4 Update `CHANGELOG.md` with the "Axis 1 measurement landed" entry. Also added a `### Fixed` entry for the `cumulative_cross_entropy_delta` example-script bug discovered + patched mid-run.

## 5. Closing

- [x] 5.1 Run `openspec validate run-axis1-compression-coverage-mbp --strict`.
- [x] 5.2 No new test code; the existing smoke test (`tests/test_examples.py::test_rung_compression_coverage_smoke`) covers the CLI surface.
- [ ] 5.3 Commit + PR with the artifacts and v2.2 write-up. Title: `docs(research): Axis 1 compression coverage — PASS`.

## 6. What this change explicitly defers

- [ ] 6.1 Axis 4 (sae-forge cross-repo). Separate run plan, separate hardware availability (sae-forge has more torch-heavy dependencies).
- [ ] 6.2 Multi-host comparison (Llama, Gemma, Qwen). 2019 MBP can't fit larger hosts; defer until rented-GPU access.
- [ ] 6.3 Default-flip decision. Even a strong PASS doesn't auto-flip `assign_amp_knobs=True` as default — that's a separate change with compatibility implications.
- [ ] 6.4 Pareto-curve studies over α / τ / quality-budget. Single-point only.
- [ ] 6.5 Prompt-set sensitivity studies. Canonical 8-prompt set only.
