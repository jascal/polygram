## Context

PR #63 (`encoding-aware-knob-assignment`) un-dormanted Rung3/Rung4 amp-branch knobs. PR #64 plumbed that flag through `EpochCompressor` + `Compressor`. The 2026-05-15 GPT-2 bug report surfaced that **the same dormancy exists at the MPSRung1 level**: α and φ are MPS-substrate knobs that every encoding (MPSRung1, Rung3, Rung4) consumes, but `from_sae_lens` never assigns them — they default to 0 forever.

The fix mirrors PR #63's PCA-axis-extension pattern but targets phase knobs (α, φ) instead of amp-branch knobs. Two flags, two separate concerns, two separate code paths.

## Goals / Non-Goals

**Goals**:
- Make `from_sae_lens` populate α and φ from decoder geometry when opted in.
- Preserve byte-identical behaviour at the default `assign_phase_knobs=False`.
- Make the new path measurably effective: |G|² with the flag on must differ from |G|² with the flag off by a Frobenius distance well above FP noise; the sanity check predicts ~63% drop in mean off-diagonal on the toy fixture.
- Keep the two flags (`assign_phase_knobs`, `assign_amp_knobs`) orthogonal — users can enable either, both, or neither.

**Non-goals**:
- A unified single-flag interface ("activate-all-knobs"). The user explicitly asked for two flags.
- Default-flipping. Stays opt-in.
- Addressing the other 3 issues in the bug report (`plan_pareto`, EpochCompressor cross-panel, uniform-sphere docstring). Separate scopes.

## Decisions

### Decision 1: PCA-axis allocation — phase gets PC2/PC3, amp gets PC4-PC7

Throughout this doc, **PC_k** refers to the *k*-th principal component (1-indexed; PC1 is the top component, PC2 the second, etc.). In code, these correspond to `vt[k-1]` rows of the SVD output (0-indexed array access). Standard ML convention.

| Flag | Knob | Component | Code reference |
|---|---|---|---|
| (existing β strategy) | β | PC1 | `vt[0]` |
| `assign_phase_knobs` | α | PC2 | `vt[1]` |
| `assign_phase_knobs` | φ | PC3 | `vt[2]` |
| `assign_amp_knobs` | theta_amp | PC4 | `vt[3]` |
| `assign_amp_knobs` | psi_aux | PC5 | `vt[4]` |
| `assign_amp_knobs` | theta_amp_b | PC6 | `vt[5]` |
| `assign_amp_knobs` | psi_amp_b | PC7 | `vt[6]` |

Rationale:
- Phase knobs apply to **all** encodings (MPSRung1, Rung3, Rung4 share the MPS substrate), so they take precedence on the lowest available PCA components after β.
- Amp knobs apply only to Rung3/Rung4. They shift to PC4-PC7 when phase knobs occupy PC2-PC3.
- When `assign_phase_knobs=False` and `assign_amp_knobs=True`, amp knobs STILL go to PC4-PC7 (NOT PC2-PC5 as in PR #63). This is the load-bearing break — the PR-#63 results note's exact gram-condition numbers would not reproduce.

The break is acceptable because PR #63 is one PR old, the existing v2.1 results note's *qualitative* finding (un-dormanting works) survives, and the exact numbers were never load-bearing.

**Alternative considered**: Keep `assign_amp_knobs` at axes 2-5 when used alone, shift only when phase is also on. **Rejected** — makes the axis allocation depend on flag combinations, which is harder to reason about and harder to test. A clean "phase before amp" rule is simpler.

### Decision 2: Linear rescale into knob's natural range

Same as PR #63 (Decision 1 of that design.md): linear rescale `coord / abs_max → [-1, 1] → linearly into target range`. α has natural range `[0, 2π]`; φ has natural range `[0, 2π]`. Both rescale by the same pattern as `psi_aux`.

Sinusoidal rescale variant deferred (same Open Question as PR #63).

### Decision 3: Encoding-agnostic helper

`assign_phase_knobs_pca(projections, encoding)` does NOT branch on encoding type. Every shipped encoding (MPSRung1, Rung3, Rung4) has α and φ — so the helper just computes the values and returns them. The encoding type is only used (a) for the INFO-once log message confirming the flag's applicability, and (b) defensively rejecting `HEA_Rung2` which has a different knob structure.

This is simpler than PR #63's amp-knob helper (which had to branch on Rung3 vs Rung4 to determine which amp arrays to populate). Phase knobs are universal across encodings; no per-encoding branching needed.

### Decision 4: Test architecture — falsifying invariant + sanity-prediction floor

The cornerstone test is the falsifying invariant: gram must differ by `Frobenius > 1.0` and `mean_off_diag_on < 0.5 * mean_off_diag_off`. The 0.5× factor is calibrated against the actual sanity-check result (0.76 → 0.28, a 63% drop on the toy fixture). If the impl is correct, this test passes comfortably; if the impl is a no-op or marginal, this test fails loudly.

Additional tests mirror PR #63's pattern: byte-identity at default, MPSRung1 ≠ Rung3 ≠ Rung4 grams when phase-on, deterministic, degenerate-PCA fallback, `SAEImportConfig` propagation.

### Decision 5: Two-flag interaction is documented but not auto-enabled

Users opt into either, both, or neither. Default is `False` for both. The combinatoric matrix:

| `assign_phase_knobs` | `assign_amp_knobs` | Knobs populated (per applicable encoding) |
|---|---|---|
| False | False | β, γ only (current behavior) |
| True  | False | β, γ, **α, φ** |
| False | True  | β, γ, **theta_amp, psi_aux, theta_amp_b, psi_amp_b** (Rung3/Rung4 only) |
| True  | True  | β, γ, α, φ, theta_amp, psi_aux, theta_amp_b, psi_amp_b (FULL on Rung4) |

The last row is "FULL on Rung4" — every per-feature knob is populated from a different PCA axis of decoder geometry.

### Decision 6: Compression plumbing follows PR #64's pattern exactly

`EpochCompressor` and `Compressor` gain `assign_phase_knobs: bool = False` field. The fix in PR #64 surfaced 3 `from_sae_lens` call sites in the compression pipeline; this change updates all 3 to also pass `assign_phase_knobs=self.assign_phase_knobs`. The captured-kwargs test pattern from PR #64 catches any missed call site.

## Risks / Trade-offs

- **Backward-incompat for `assign_amp_knobs=True` reproducibility**. The PR-#63-era research artifacts (`docs/research/data/rung_gram_condition_{rung3,rung4}_amp_on.json`) capture exact gram-condition numbers that won't reproduce after the axis shift. **Mitigation**: the qualitative finding (amp-on materially changes the gram) still holds; the v2.1 results note gets a one-paragraph note about the axis shift; a regenerate pass updates the exact numbers if anyone cares.
- **Two-flag UX vs one-flag simplicity**. Two flags is the user's explicit choice (per design discussion). Single-flag `enable_pca_knob_assignment=True` would be simpler but conflates two different concerns (phase rotation vs amp branching). Two flags is the right factoring even if it's slightly more API.
- **Degenerate decoder geometry**. On a 2-feature SAE, PCA has only 1 non-zero axis; α and φ fall back to encoding defaults. Same fallback pattern as PR #63's amp helper.

## Open Questions

- **Should we ALSO populate γ for non-clustered profiles?** Currently `assign_gamma=True` (default) does per-cluster PCA for γ, which is correlated within cluster. A higher-PCA-axis γ would be more diverse. **Deferred** — orthogonal concern, separate change if it surfaces.
- **What about the sinusoidal rescale?** Same open question as PR #63 — defer.

## Migration Plan

- **Existing users** (no `assign_phase_knobs` kwarg): no change. Default `False` is byte-identical.
- **Users opted into `assign_amp_knobs=True`**: their exact gram-condition numbers change (axis shift). Qualitative behaviour same.
- **New users wanting MPSRung1 to use its full state space**: pass `assign_phase_knobs=True`.
- **New users wanting Rung4 with everything on**: pass both flags.

### Default-flip path (future, not in scope here)

The flags stay opt-in initially. A default-flip to `True` for higher-rung encodings would be considered only after:

1. **Axis 1 (compression coverage)** on a torch-enabled host shows materially better compression for Rung4-amp+phase vs Rung4-default (the run plan in `openspec/changes/run-axis1-compression-coverage-mbp/`).
2. **Axis 4 (sae-forge faithfulness)** confirms downstream forge-pipeline benefit on a real LM.
3. Two-flag UX has settled with no surprise behaviour reports from external users.

If those three conditions land, a follow-up change (e.g., `flip-phase-and-amp-defaults-for-higher-rungs`) proposes the default-flip with explicit migration guidance — existing `encoding=Rung4()` call sites would see different behaviour and need explicit `assign_*_knobs=False` to preserve pre-flip semantics. Out of scope here.
