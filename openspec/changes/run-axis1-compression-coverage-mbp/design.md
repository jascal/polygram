## Context

The Axis 1 / Axis 4 measurements have been "pending a torch-enabled host" since `docs/research/rung4-viability-spike-v2.md` landed in PR #61. The blocking-est constraint was the structural-dormancy bug (PR #63 + #64 fixed). The next-blocking constraint is hardware: the polygram .venv on the personal Intel Mac doesn't currently have torch installed, but the auto-memory confirms a separate `.venv` with `torch 2.2.2 + transformers 4.49 + numpy<2 caps` exists for personal-project work on this machine. The 2019 MBP (16 GB RAM, no CUDA GPU) is hardware-sufficient for GPT-2-small forward passes.

## Goals / Non-Goals

**Goals**:
- Produce a 3-cell (optionally 4-cell) comparison table of compression coverage on the canonical SAE.
- Settle whether the un-dormanting work from PR #63/#64 translates to a measurable compression-pipeline win on GPT-2-small block 10.
- Update `docs/research/rung4-viability-spike-v2.md` with a v2.2 section reporting the verdict.

**Non-goals**:
- Multi-host comparison (Llama, Gemma, Qwen) — not feasible on the 2019 MBP within reasonable wall time.
- Pareto-curve studies over α/τ/quality-budget — single-point measurement only.
- Axis 4 (sae-forge end-to-end forge → faithfulness) — cross-repo; separate run plan.
- Automatic CI runs — user-triggered.

## Decisions

### Decision 1: 3 cells, 4 if Rung3 is cheap to add

The canonical 3-cell comparison is:

| Cell | encoding | assign_amp_knobs | Purpose |
|---|---|---|---|
| C1 | mps | (irrelevant — no-op for MPSRung1) | Baseline |
| C2 | rung4 | False | Confirms "Rung4 default = MPS in disguise" propagates from gram → compression metrics |
| C3 | rung4 | True | Load-bearing — does un-dormant Rung4 actually compress better? |
| C4 (optional) | rung3 | True | Rung3 amp-on data point for the rung-ladder comparison |

C1 vs C2 should show **near-identical** compression metrics (modulo the encoding's structural effect on panel size — Rung4 has cap 32 vs MPS's 8, which DOES affect `_select_panels`'s neighbour cap). If C1 and C2 are quite different, the panel-size effect dominates and the amp-knob path's effect (C2 vs C3) is what matters for "does un-dormanting help?".

C2 vs C3 is the load-bearing comparison.

### Decision 2: Use existing canonical kwargs

`examples/rung_compression_coverage.py` already ships with `--max-iterations 3 --coverage-target 0.5 --cosine-threshold 0.3 --n-prompts 8`. These match the polygram defaults on EpochCompressor for the fast iterative-loop preset. No tuning sweep — single-point at canonical kwargs across all cells. Any rung-N+1 win is then on equal-footing-kwargs grounds.

### Decision 3: Wall-time budget

GPT-2-small pre-pass on CPU ≈ 30-60 sec for 8 prompts × ~32 tokens. EpochCompressor's analytic phase ≈ 30-90 sec per iteration depending on panel count. With `max_iterations=3` and the canonical SAE (24,576 features), one cell is expected at **3-5 min wall time** on the 2019 MBP. Three cells: ~15 min total. Four cells: ~20 min total. Easily fits a single work session.

### Decision 4: What "more compressed" means

Primary metric: `n_features_zeroed_total`. Higher is better at equal-or-lower `cumulative_cross_entropy_delta`.

Secondary metric: `cumulative_cross_entropy_delta`. Lower is better at equal-or-greater `n_features_zeroed_total`.

Tertiary metric: `n_iterations`. Lower = faster convergence at equal coverage.

Reporting: a 4-row table per cell (final iteration's three counters + the convergence state), plus a per-iteration trajectory plot in the v2.2 section.

### Decision 5: What counts as a "material" difference

The PR-#63 falsifying invariant pinned `Frobenius(gram_off - gram_on) > 1e-3` on the toy fixture. For compression metrics on a real SAE, **material** is interpreted as:

- `n_features_zeroed_total`: ≥ 10% relative difference (e.g., 120 vs 110, or 1200 vs 1080).
- `cumulative_cross_entropy_delta`: ≥ 20% relative difference (CE deltas at this scale are inherently noisier than feature counts).

Smaller deltas land in the "PARTIAL / INCONCLUSIVE" rows of the proposal's decision-rule table.

### Decision 6: Determinism + reproducibility

The runs are deterministic given the same SAE checkpoint, prompts, kwargs, and torch version. The JSON artifacts carry every input field needed to reproduce, so a future re-run on a different host can verify. The amp-knob path is also deterministic (PCA-axis assignment is a pure forward pass).

## Risks / Trade-offs

- **Single-point measurement may miss the pareto frontier.** A run might find Rung4-amp-on at lower n_features_zeroed but lower CE delta, or vice versa. Documented in the proposal's decision-rule table as INCONCLUSIVE; the response is a separate Pareto-curve study, not a re-run of the same single-point comparison.
- **The canonical 8-feature §4.4 panel is small for a compression run.** EpochCompressor operates on the FULL SAE (24,576 features), not just the §4.4 panel — that's why this run is genuinely different from the gram-condition spike. But the Rung4 amp-knob path's effect depends on what features end up in panels during `_select_panels`, which is stochastic-in-spirit (priority-ordered, but the priority signal depends on activation statistics). A single run is one realization.
- **GPT-2-small SAE may not be the most informative fixture.** A Llama / Gemma SAE would exercise the rung ladder against modern SAE topologies. Out of scope for this MBP-bound run; flagged as future work.

## Open Questions

- **Should the verdict be "flip the default" or "leave opt-in" regardless of the outcome?** Default-flipping has compatibility cost (existing call sites that pass `encoding=Rung4()` see different behaviour). Even a strong Axis 1 PASS doesn't *require* flipping the default — it just supports doing so in a follow-up change. Defer the decision.
- **Should we measure with prompts ≠ the canonical 8?** The 8 prompts are diverse enough for a 30-token forward, but may not exercise the full SAE feature space. A 32-prompt run is 4× the wall time and might surface different feature-firing distributions. Defer; the 8-prompt single-point is the first-cut measurement.
