## Why

[`docs/research/rung-viability-methodology.md`](../../../docs/research/rung-viability-methodology.md) proposed a 4-axis battery for rung-viability v2. Axis 2 (gram condition) ran locally and surfaced the load-bearing finding that real consumers never saw higher-rung capacity. PRs #63 + #64 fixed that — `from_sae_lens(assign_amp_knobs=True)` populates amp-branch knobs from decoder geometry, and `EpochCompressor` / `Compressor` thread the flag through to every internal `from_sae_lens` call.

**Axis 1 (compression coverage) is now actually meaningful and runnable**. Per the user's question about the 2019 MBP being sufficient: yes — GPT-2-small fits in 16GB RAM on Intel macOS, the canonical SAE checkpoint is already on disk, the analytic phase after the pre-pass is torch-free, and a single 4-cell comparison (MPS vs Rung4) × (amp-off vs amp-on) costs ~5 minutes per cell. This change formalizes the run plan, decision rule, and expected outputs.

The change is **most measurement, least new code** — the only code-shaped piece is the `--assign-amp-knobs` CLI flag on `examples/rung_compression_coverage.py` (a ~5 LOC addition; the upstream plumbing landed in #63/#64). The bulk is the run protocol, the falsifying invariants, and the deliverable artifacts. Lighter weight than the typical openspec; modelled on `tech-debt-backlog`'s proposal+tasks shape.

## What Changes

### Code change (small)

`examples/rung_compression_coverage.py` gains `--assign-amp-knobs` CLI flag, threaded through to the `EpochCompressor(assign_amp_knobs=...)` constructor. Default off; preserves the existing measurement at `assign_amp_knobs=False` for back-comparability with the v2 results note.

### Run protocol (4 cells)

Three runs on the canonical GPT-2-small SAE (`scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors`), all from the 2019 Intel MacBook Pro:

1. **`--encoding mps`** (default amp-off, no-op for MPSRung1). Baseline.
2. **`--encoding rung4 --assign-amp-knobs=False`** (omit the flag). Establishes that "Rung4 with default knobs = MPSRung1 in disguise" actually shows up in compression metrics, not just gram metrics.
3. **`--encoding rung4 --assign-amp-knobs`** (flag on). The load-bearing measurement — does the un-dormant Rung4 capacity actually translate to better compression coverage?

(A fourth cell — `--encoding rung3 --assign-amp-knobs` — is optional. Adds another data point but doesn't change the decision.)

### Falsifying decision rule

After the runs land, the v2.2 supplemental in `docs/research/rung4-viability-spike-v2.md` reports the table below. The decision rule maps to the PR-#60 v2 methodology's bucketing:

| Outcome | Interpretation |
|---|---|
| **Rung4-amp-on zeros materially more features than MPS at equal-or-better CE budget** | PASS for Axis 1. Capacity lift cashes out in production compression. Strong evidence to flip `assign_amp_knobs=True` as the default for higher-rung encodings. |
| **Rung4-amp-on ≈ MPS on features-zeroed, but cumulative CE delta lower at the same iteration count** | PARTIAL. Capacity lift produces "smoother" compression (less quality damage) but not more features compressed. Worth more investigation; opt-in stays. |
| **Rung4-amp-on ≈ MPS on both metrics** | FAIL for Axis 1. The capacity lift exists structurally (gram condition improves per Axis 2) but doesn't translate to compression metrics that this pipeline measures. Probably means downstream Axes (sae-forge faithfulness, behavioural validator gate TPR) are the right place to measure rung value. |
| **Rung4-amp-on zeros MORE features but at HIGHER CE delta** | INCONCLUSIVE. Quality/quantity trade-off; need a Pareto-curve study, not a single point. |

The verdict goes into the v2.2 note. None of these is a "kill the encoding" verdict — Rung4 stays opt-in regardless. The question is whether to flip the default and update the README's "when to use which rung" guidance.

### Expected artifacts

- `docs/research/data/axis1_mps_n_features_zeroed.json` — MPS baseline
- `docs/research/data/axis1_rung4_amp_off.json` — Rung4 default-knob (control)
- `docs/research/data/axis1_rung4_amp_on.json` — Rung4 amp-on (load-bearing)
- (optional) `docs/research/data/axis1_rung3_amp_on.json`
- `docs/research/rung4-viability-spike-v2.md` gains a "Axis 1 result (v2.2)" section with the table + verdict.

## Impact

### Affected specs

None modified directly. The `--assign-amp-knobs` CLI flag is a small addition to a non-spec-tracked example script.

### Affected code

- `examples/rung_compression_coverage.py` (+5 LOC for the flag, already in this branch)

### Affected docs

- `docs/research/rung4-viability-spike-v2.md` (new "Axis 1 result (v2.2)" section after the runs land)
- `CHANGELOG.md` (one-line "Axis 1 measurement landed" entry under the existing v2 work)

### Closes

The "Axis 1 / 4 pending a torch-enabled host" TODO from `docs/research/rung4-viability-spike-v2.md`, for the Axis 1 piece. Axis 4 (sae-forge cross-repo) remains separate work.

### What this change explicitly does NOT do

- **No Axis 4 (sae-forge faithfulness).** Cross-repo; needs a separate run plan.
- **No optimization sweep** over `α`, `τ`, panel size, or grid resolution. Single-point measurement at canonical kwargs.
- **No multi-host comparison** (Llama-3, Gemma-2). 2019 MBP can't fit larger hosts.
- **No automatic CI integration.** The runs are user-triggered on the MBP; CI just verifies the smoke-test skip path (already in place).
- **No commitment to flip `assign_amp_knobs=True` as default** post-run. That's a separate decision that this change's verdict feeds into.
