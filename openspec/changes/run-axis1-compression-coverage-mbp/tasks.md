## 1. Code preparation

- [x] 1.1 Add `--assign-amp-knobs` CLI flag to `examples/rung_compression_coverage.py`. Threaded through to `EpochCompressor(assign_amp_knobs=...)`. Default False (back-compat with v2.1 results).
- [x] 1.2 Record the flag's value in the output JSON's `assign_amp_knobs` field for reproducibility.
- [x] 1.3 Print the flag's value in the per-run header so console captures include the cell identity.

## 2. Environment prep on the 2019 MBP

- [ ] 2.1 Verify `pip install -e ".[behavioural]"` succeeds in the polygram `.venv`. (Per auto-memory, the user has a parallel personal-project `.venv` with torch 2.2.2 + transformers 4.49 + numpy<2 caps already working — this task is to ensure the polygram-dir `.venv` matches.)
- [ ] 2.2 Smoke-check: `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"` returns without error.
- [ ] 2.3 Smoke-check: `python -c "from polygram import EpochCompressor; ec = EpochCompressor.__init__"` resolves without import error after the behavioural extras land.

## 3. Run the comparison (3 cells)

- [ ] 3.1 Cell C1 (MPS baseline):
  ```
  python examples/rung_compression_coverage.py \
    --encoding mps \
    --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \
    --output docs/research/data/axis1_mps.json
  ```
- [ ] 3.2 Cell C2 (Rung4 default-knob control):
  ```
  python examples/rung_compression_coverage.py \
    --encoding rung4 \
    --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \
    --output docs/research/data/axis1_rung4_amp_off.json
  ```
- [ ] 3.3 Cell C3 (Rung4 amp-on, load-bearing):
  ```
  python examples/rung_compression_coverage.py \
    --encoding rung4 --assign-amp-knobs \
    --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \
    --output docs/research/data/axis1_rung4_amp_on.json
  ```
- [ ] 3.4 (optional) Cell C4 (Rung3 amp-on):
  ```
  python examples/rung_compression_coverage.py \
    --encoding rung3 --assign-amp-knobs \
    --sae scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \
    --output docs/research/data/axis1_rung3_amp_on.json
  ```
- [ ] 3.5 Capture the console output for each cell in a `runlog_<cell>.txt` next to the JSON, so reviewers can verify no warnings/errors and inspect per-iteration trajectory.

## 4. Compare + write up

- [ ] 4.1 Compile the headline table comparing C1 / C2 / C3 (and C4 if run) on:
  - `n_features_zeroed_total`
  - `final_iteration.cumulative_cross_entropy_delta`
  - `n_iterations` actually executed
  - `final_iteration.convergence_state`
- [ ] 4.2 Apply the decision rule from `proposal.md`:
  - "Material difference" thresholds: ≥10% on features-zeroed, ≥20% on CE delta.
  - Bucket: PASS / PARTIAL / FAIL / INCONCLUSIVE.
- [ ] 4.3 Write up the result as a new "## Axis 1 result (v2.2)" section in `docs/research/rung4-viability-spike-v2.md`. Mirror the v2.1 (Axis 2) section's table-then-prose structure. Include:
  - The 3-cell (or 4-cell) headline table.
  - The verdict bucket + interpretation.
  - Any per-iteration trajectory observations (e.g., "Rung4 amp-on hit max_iterations terminal at iter 3 with N features zeroed, while MPS terminated at iter 2 via stable_clusters").
  - Honest predictions vs actuals (record the proposal-time predictions and whether they held).
- [ ] 4.4 Update `CHANGELOG.md` with a one-line "Axis 1 measurement landed" under the existing v2 work entry.

## 5. Closing

- [ ] 5.1 Run `openspec validate run-axis1-compression-coverage-mbp --strict`.
- [ ] 5.2 No new test code; the existing smoke test (`tests/test_examples.py::test_rung_compression_coverage_smoke`) covers the CLI surface.
- [ ] 5.3 Commit + PR with the artifacts and v2.2 write-up. Title: `docs(research): Axis 1 compression coverage — <verdict-bucket>`.

## 6. What this change explicitly defers

- [ ] 6.1 Axis 4 (sae-forge cross-repo). Separate run plan, separate hardware availability (sae-forge has more torch-heavy dependencies).
- [ ] 6.2 Multi-host comparison (Llama, Gemma, Qwen). 2019 MBP can't fit larger hosts; defer until rented-GPU access.
- [ ] 6.3 Default-flip decision. Even a strong PASS doesn't auto-flip `assign_amp_knobs=True` as default — that's a separate change with compatibility implications.
- [ ] 6.4 Pareto-curve studies over α / τ / quality-budget. Single-point only.
- [ ] 6.5 Prompt-set sensitivity studies. Canonical 8-prompt set only.
