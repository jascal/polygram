## Why

Polygram's `from_sae_lens` import path projects decoder geometry into
per-feature polygram knobs via a **hardcoded** axis-to-knob
permutation: `assign_amp_knobs_pca` and `assign_phase_knobs_pca`
unconditionally route PC1 → β, PC2 → α, PC3 → φ, then PC4..PC{3+2k}
into the amp-branch slots. The map is not learned from data; it's
baked into the helpers.

A small prototype in `examples/rung5_pareto_scans.py::run_learned_assignment`
(landed alongside the 0.7.0 rung5 Pareto scans) shows that greedy
axis-to-knob permutation search lifts decoder-Gram Spearman by **~3×**
on a synthetic 64-feature clustered SAE in 5 seconds of search:

| k | Spearman (baseline) | Spearman (greedy) | Δ | Cond ratio |
|---|---|---|---|---|
| 3 | +0.1042 | +0.3350 | +0.2309 | 3.8× better |
| 4 | +0.1161 | +0.3380 | +0.2219 | 2.1× better |

Both k values learn `α ← PC1` — the hardcoded baseline reserves PC1
for β-via-labels and starts α at PC2, wasting the cluster-bearing
direction. The greedy variant finds and exploits this. The fidelity
gain comes for free at runtime (the SAE is unchanged; only the
import-time projection is recalibrated) and the gram conditioning
also improves at the same time.

The "Rung5 buys conditioning, not fidelity" finding from
`docs/research/rung5-pareto-scans.md` scan 1 is therefore an artefact
of the hardcoded map, not of the encoding itself. With a learned
axis-to-knob assignment, Rung5 (and every prior encoding) recovers
substantial decoder-geometry headroom.

This change ships a production `LearnedKnobAssignment` strategy:
proper continuous optimisation on a small linear map `W` per knob
(not greedy permutation), pluggable objective, opt-in flag on
`from_sae_lens`, surfaced in `SelectionReport` for inspection.

## What Changes

- **New strategy class `polygram.geometry.LearnedKnobAssignment`**
  implementing the `KnobAssignment` protocol. Computes decoder PCA
  once, then optimises a small permutation-or-linear-combination map
  `W: R^{n_axes} → R^{n_knobs}` against a configurable objective.
  Two solver modes:
  - `solver="greedy"` — deterministic permutation search (the
    prototype). Pure numpy, ships in the base install.
  - `solver="scipy"` — continuous optimisation on `W` via scipy
    `minimize` (or `differential_evolution` for large k). Available
    behind the existing `polygram[opt]` extra.
- **Pluggable objective.** A `LearnedAxisObjective` protocol with
  three built-ins:
  - `"spearman"` (default) — Spearman rank correlation between the
    analytic gram's off-diagonal `|G|²` and the decoder cosine²
    matrix.
  - `"pearson"` — Pearson correlation, same off-diagonal entries.
    Cheaper than Spearman; correct when the relationship is roughly
    linear.
  - `"behavioural"` — bring-your-own ground-truth pair-similarity
    matrix (e.g. from behavioural co-activation). Optional; requires
    user-supplied data.
- **Opt-in `from_sae_lens` flag.** `learn_axis_assignment: bool |
  LearnedKnobAssignment | None = None`. When `True`, instantiates the
  default `LearnedKnobAssignment()` and uses it for the import. When
  a strategy instance is passed, uses that directly. When `False` or
  `None`, keeps the existing hardcoded `assign_amp_knobs_pca` +
  `assign_phase_knobs_pca` behaviour byte-for-byte.
- **`SelectionReport` extension.** New `learned_axis_assignment`
  field carrying the chosen knob → axis (or knob → axis-coefficient
  vector) map plus the achieved objective value and the baseline
  objective for comparison. `None` for runs that didn't use the
  learned strategy. Lets callers audit which axes ended up where.
- **CLI flag.** `polygram from-sae-lens --learn-axis-assignment`
  toggles the strategy on for command-line imports.
- **Documentation.** Research note linking back to the
  `rung5-pareto-scans.md` scan 4 motivation and the May 2026
  theoretical treatment's §9 (Algorithms) and §11 (Open problems)
  discussion of axis-assignment optimisation.
- **No SAE-side changes.** The SAE's `W_dec` is fixed; only the
  import-time projection into polygram-knob space is recalibrated.
  Out of scope: anything touching the SAE compress/regrow/train
  cycle (sae-forge owns that loop).

## Capabilities

### New Capabilities

- `learned-axis-assignment`: import-time calibration that optimises
  the PCA-axis-to-polygram-knob projection against a fidelity
  objective (decoder-Gram Spearman / Pearson / behavioural). Ships as
  a `KnobAssignment` strategy alongside the existing clustered and
  uniform-sphere strategies.

### Modified Capabilities

- `geometry-regimes`: `KnobAssignment` protocol gains a non-mandatory
  `axis_assignment` returned in `KnobAssignmentResult` so callers can
  audit which axis fed which knob. `LearnedKnobAssignment` populates
  it with the optimised map; clustered/uniform-sphere strategies
  populate it with the hardcoded baseline for parity.
- `sae`: `from_sae_lens` accepts `learn_axis_assignment` kwarg
  (default `None`, preserves byte-identical behaviour). When
  populated, the import path delegates to `LearnedKnobAssignment`
  instead of the hardcoded `assign_*_pca` helpers. `SelectionReport`
  gains `learned_axis_assignment` field.
- `tuning-config`: `SAEImportConfig` gains `learn_axis_assignment`
  field plumbing the kwarg through the dataclass-driven config path.

## Impact

- `polygram/geometry/learned_axis_assignment.py` — new module
  implementing the strategy class.
- `polygram/geometry/__init__.py` — re-export
  `LearnedKnobAssignment`, `LearnedAxisObjective`.
- `polygram/geometry/protocols.py` — extend
  `KnobAssignmentResult` with optional `axis_assignment` field
  (default `None`); add `LearnedAxisObjective` protocol.
- `polygram/sae_import.py` — accept `learn_axis_assignment` kwarg,
  delegate to the strategy when populated; populate
  `SelectionReport.learned_axis_assignment`.
- `polygram/cli.py` — add `--learn-axis-assignment` flag to the
  `from-sae-lens` subcommand.
- `polygram/config.py` — `SAEImportConfig.learn_axis_assignment`
  field.
- `tests/test_learned_axis_assignment.py` — new tests:
  - Greedy solver is deterministic seed-by-seed.
  - Scipy solver improves on greedy on the prototype's synth.
  - Hardcoded baseline byte-identical when `learn_axis_assignment`
    is `None`/`False`.
  - `SelectionReport.learned_axis_assignment` populated correctly.
- `tests/test_sae_import.py` — extend to cover the `learn_axis_assignment`
  kwarg path.
- `tests/test_cli.py` — `--learn-axis-assignment` smoke test.
- `examples/learned_axis_assignment_demo.py` — small demo
  reproducing scan 4's headline result with the production strategy.
- `docs/research/learned-axis-assignment.md` — design note linking
  back to `rung5-pareto-scans.md` scan 4 and the theoretical
  treatment's §9/§11.

**Depends on** the prototype's data shape in
`docs/research/data/rung5_pareto/learned_assignment.json` (already
shipped in PR #79). No other downstream blockers.

**No breaking changes.** Default `learn_axis_assignment=None`
preserves bit-exact existing behaviour. Callers opt in.

## Promote-to-default gate

Strict criteria for flipping `learn_axis_assignment` from opt-in to
default (raised in PR #80 review):

1. **Replication on ≥ 2 distinct real SAEs.** Both Gemma-Scope and
   GPT-2-small (or equivalent) must show a positive Δ on the
   training-objective Spearman *and* the held-out-pair validation
   Spearman, vs the hardcoded baseline, with seed=0 and seed=42.
2. **Downstream metric agreement.** At least one of the following
   must improve by a non-trivial margin at the same SAEs:
   - post-cancellation KL divergence on a behavioural-validation
     prompt set (lower is better),
   - decoder-Gram reconstruction fidelity (Frobenius distance to the
     analytic gram, lower is better).
3. **Calibration-cost budget.** The greedy solver completes in
   ≤ 5 s per import at N ≤ 1000 on commodity CPU; the scipy solver
   in ≤ 60 s at N ≤ 1000.

Until all three hold, the strategy stays opt-in. The Spearman win on
a synthetic clustered SAE is *necessary* but not *sufficient* for
the default flip — synthetic-only evidence is consistent with the
strategy fitting cluster-bearing noise.
