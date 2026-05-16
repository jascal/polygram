## Context

`polygram.geometry.amp_assignment.assign_amp_knobs_pca` and
`polygram.geometry.phase_assignment.assign_phase_knobs_pca` project
decoder PCA axes into polygram knob slots via a hardcoded
permutation. The map is hand-picked: PC1 → β-via-labels (cluster
ordinal), PC2 → α, PC3 → φ, PC4..PC{3+2k} → amp branch pairs.

Empirically (per
[rung5-pareto-scans.md scan 4](../../docs/research/rung5-pareto-scans.md)),
this permutation leaves substantial decoder-Gram fidelity on the
table on a synthetic clustered SAE: a 5-second greedy search over
permutations 3×s Spearman in both Rung5(k=3) and Rung5(k=4).

The fidelity is recoverable because the projection
`decoder PCA → knob slots` is a *small linear map* (a few axes × a
few knobs) and the objective (decoder-Gram fidelity) is closed-form
over polygram's analytic gram. Optimising the map is therefore cheap
calibration, not learning in the sae-forge sense.

This change ships a production strategy that replaces greedy
permutation with proper continuous optimisation, with the greedy
solver retained as a deterministic fallback.

Stakeholders: sae-forge (uses `from_sae_lens` in its pareto sweeps —
will see fidelity improvements without code changes); polygram CLI
users; future research notes that want to cite the achieved fidelity
on real SAEs.

## Goals / Non-Goals

**Goals:**

- Ship `LearnedAxisAssignment` as an opt-in `KnobAssignment`
  strategy. Default-off; existing callers see byte-identical
  behaviour.
- Two solvers: greedy permutation (deterministic, base install) and
  scipy-continuous (richer, `polygram[opt]` extra). Both produce a
  knob → axis (or knob → axis-coefficient-vector) map.
- Pluggable objective via `LearnedAxisObjective` protocol; three
  built-ins (Spearman, Pearson, behavioural).
- Surface the chosen map in `SelectionReport.learned_axis_assignment`
  so callers can audit / debug it.
- Reproduce the prototype's headline result (~3× Spearman lift) with
  the production strategy on the same synthetic SAE.

**Non-Goals:**

- Touching the SAE itself. `W_dec` is read-only input to this
  strategy. SAE-content optimisation lives in sae-forge.
- Replacing the hardcoded helpers (`assign_amp_knobs_pca`,
  `assign_phase_knobs_pca`). They stay as the cheap default when
  `learn_axis_assignment=None`. The learned strategy is additive.
- Online / per-sweep-step recalibration. The strategy runs once per
  import; it doesn't re-tune as the dictionary mutates.
- Learning the per-feature knob values themselves. The strategy
  learns the *projection*; the per-feature values still come from
  the data's PCA coords through that projection.
- HEA_Rung2 support. `HEA_Rung2` has a different per-feature θ
  tensor shape; learned assignment for it is out of scope for v1.
  The strategy SHALL log INFO-once and fall back to the hardcoded
  helper when given a `HEA_Rung2` encoding.

## Decisions

### Decision 1: Strategy class, not a flag on existing helpers

`LearnedAxisAssignment` lives as a peer of `ClusteredKnobAssignment`
and `UniformSphereKnobAssignment` — both already implementations of
the `KnobAssignment` protocol from `polygram.geometry.protocols`.
The learned variant follows the same shape: instance method
`assign(projections, feature_names, *, ...) -> KnobAssignmentResult`.

This keeps the geometry-regime architecture flat (every strategy is
a class; geometry profiles compose them) and avoids adding a
boolean-on-helper-function path that would balloon the
`assign_*_pca` signatures.

**Alternatives considered:**
- *Boolean flag on `assign_amp_knobs_pca` and
  `assign_phase_knobs_pca`*: doubles their parameter surface and
  conflates "compute knobs from PCA" with "decide which PCA axis
  feeds which knob." Rejected on separation-of-concerns.
- *Reuse `ClusteredKnobAssignment` with a learning mode*: the
  clustered strategy is opinionated about clusters first, knobs
  second. The learned strategy has the opposite ordering. Rejected
  on cohesion.

### Decision 2: Solvers — greedy + scipy

Two solvers ship side-by-side, chosen via `LearnedAxisAssignment(solver=...)`:

- **`solver="greedy"`** (default). Permutation search: for each
  knob slot in canonical order (α, φ, then amp pairs in qubit-index
  order), tries every unused PCA axis and locks in the axis whose
  addition gives the best objective value. Deterministic given the
  seed and the projection matrix. O(n_knobs × n_axes) objective
  evaluations, each O(N²) for the gram. No external dependencies.
- **`solver="scipy"`**. Continuous optimisation on a small linear
  map `W ∈ R^{n_knobs × n_axes}` (each row a sparse-ish vector of
  PCA-axis coefficients per knob). Initialised from the greedy
  result. Uses `scipy.optimize.minimize(method="Nelder-Mead")` for
  small problems and `differential_evolution` when `n_knobs ≥ 8`
  (Rung5 with k≥3). Requires the `polygram[opt]` extra.

The greedy solver is the headline default because:
1. Reproduces the prototype's published numbers exactly.
2. Determinism makes it trivial to test.
3. Doesn't pull scipy into the base install.

The scipy solver lifts performance further when the linear-map
expressiveness pays off (e.g., when no single PCA axis dominates a
knob's signal — common on noisy real SAEs).

**Alternative considered:** Riemannian gradient descent directly on
the polygram manifold (per §9.2 of the theoretical treatment).
Rejected for v1 — the manifold is `(S²)^n × S¹` and the map being
optimised is the *axis assignment*, not the per-feature knobs.
Riemannian descent would learn the knobs themselves; what we want
is calibration of how the data feeds them. Different objective.

### Decision 3: Pluggable objective via protocol

`LearnedAxisObjective` is a protocol:

```python
@runtime_checkable
class LearnedAxisObjective(Protocol):
    def __call__(
        self,
        analytic_gram: np.ndarray,  # (N, N) complex
        decoder_geom: np.ndarray,   # (N, N) real (e.g., cosine²)
        *,
        feature_names: list[str],
    ) -> float:
        ...
```

Three built-ins ship as small functions in
`polygram.geometry.objectives`:

- `spearman_objective(g, d, **_) -> float` — Spearman rank
  correlation on off-diagonal triangular entries.
- `pearson_objective(g, d, **_) -> float` — Pearson correlation,
  same entries.
- `behavioural_objective(reference_pair_sims)(g, d, **_) -> float`
  — factory returning a closure that ignores `d` and correlates
  against the supplied ground-truth matrix.

The default is `spearman_objective` (matches the prototype). Callers
pass a custom callable via
`LearnedAxisAssignment(objective=my_callable)`.

**Alternative considered:** Single hardcoded objective. Rejected
because the prototype's Spearman is one of several reasonable
choices, and behavioural-fidelity ground truth (from sae-forge's
co-activation matrices) is the natural next step.

### Decision 4: `KnobAssignmentResult.axis_assignment` is optional

Add `axis_assignment: dict[str, int | list[float]] | None = None`
to `KnobAssignmentResult`. Populated by `LearnedAxisAssignment` with
either a knob → PCA-axis-index map (greedy solver) or a
knob → list-of-axis-coefficients (scipy solver).

Existing `ClusteredKnobAssignment` / `UniformSphereKnobAssignment`
leave it `None`. Optional rather than mandatory so adding the field
doesn't churn every existing strategy implementation.

`SelectionReport.learned_axis_assignment` is populated from this
field when present, otherwise `None`. Callers can pretty-print it
for debugging.

### Decision 5: Opt-in via single kwarg, default-off

`from_sae_lens(..., learn_axis_assignment=None)` accepts:

- `None` / `False` — keep hardcoded behaviour (default).
- `True` — instantiate `LearnedAxisAssignment()` with defaults and
  use it.
- A `LearnedAxisAssignment` instance — use it directly.

Strict default-off preserves bit-exact existing behaviour for every
caller, including all 524-test regression coverage. The opt-in path
runs the new strategy.

**Alternative considered:** Make the learned strategy the default.
Rejected — would change every existing caller's dictionary contents
silently. The prototype shows it's a strict win on a synthetic SAE,
but real-SAE replication is a follow-up; promote to default later
if the win generalises.

### Decision 6: HEA_Rung2 fallback, not support

`LearnedAxisAssignment` checks `isinstance(encoding, HEA_Rung2)` in
`assign()` and falls back to the hardcoded helpers with an INFO-once
log. HEA's per-feature θ tensor has a different parameter shape; a
learned axis assignment for it would need a separate objective
plumbing (per-(rotation, depth, qubit) slot).

This mirrors what `assign_amp_knobs_pca` already does for HEA:
returns all-`None` with an INFO-once log. Consistent stance.

## Risks / Trade-offs

- **[Real-SAE generalisation untested]** — the 3× Spearman win is on
  a synthetic clustered SAE where the cluster-bearing direction
  doesn't match the hardcoded PC2 → α assignment. On a real
  Gemma-Scope SAE, the hardcoded map may already be well-aligned
  and the learned strategy might give a marginal Δ. → Mitigation:
  research note documents the synth limitation; promote to default
  only after real-SAE replication. v1 stays opt-in.

- **[Calibration cost at import time]** — greedy solver does
  O(n_knobs × n_axes) gram evaluations, each O(N²·n). At N=64,
  k=4, n_axes=32: 10×32×64²×7 = ~9M flops per evaluation × ~320
  evaluations = ~3 GF — sub-second. At N=1000+ on real SAEs the
  evaluation cost rises to ~1GB per gram × ~hundreds of
  evaluations: minutes, not seconds. → Mitigation: document the
  `Θ(n_knobs · n_axes · N²)` scaling in the strategy docstring; add
  a `max_axes` parameter (default 32) to cap n_axes; allow the
  scipy solver to early-terminate.

- **[Solver determinism]** — scipy solvers seeded but
  Nelder-Mead is famously sensitive to initial conditions and may
  converge to local optima. → Mitigation: scipy solver initialises
  from the greedy result (which is a strong starting point);
  multi-start from `[greedy, greedy+small-noise]` is offered via
  `LearnedAxisAssignment(scipy_restarts=N)`.

- **[Objective overfitting to the synth]** — Spearman on a 64-feature
  synth has ~2000 pairs; the strategy can in principle "tune to
  noise" rather than to structure. → Mitigation: hold-out half the
  pairs for a validation score returned in the result; warn when
  validation score is materially below training score.

- **[Behavioural objective requires ground truth]** — the
  third built-in needs a user-supplied pair-similarity matrix from
  outside polygram (e.g., from sae-forge's behavioural validator
  output). The polygram side doesn't enforce its existence; the
  factory pattern leaves it caller-supplied. → Mitigation: clear
  docstring + an example in
  `examples/learned_axis_assignment_demo.py` showing how to wire
  sae-forge's matrix in.

## Migration Plan

No data migration. Additive:

1. `LearnedAxisAssignment` + `LearnedAxisObjective` land in
   `polygram.geometry.learned_axis_assignment` behind the
   `learn_axis_assignment` opt-in. Nothing dispatches to them
   without explicit caller request.
2. `SelectionReport.learned_axis_assignment` defaults to `None`;
   existing report serialisations round-trip unchanged.
3. `SAEImportConfig.learn_axis_assignment` defaults to `None`;
   existing configs unchanged.

Rollback: remove the opt-in branch. No persisted state references
the strategy until callers explicitly enable it.

## Open Questions

- **Sample-complexity bound for axis recovery.** The theoretical
  treatment's Thm 7.5 gives 4n+1 generic overlap measurements for
  parameter recovery. The analogous question for *axis recovery*
  (i.e., given noisy overlaps, how many features are needed to
  identify the right knob→axis map?) is open. v1 doesn't need this;
  worth flagging in the theoretical treatment's §11 Open Problems
  alongside the existing identifiability bounds.

- **Real-SAE replication.** Run the strategy against a Gemma-Scope
  SAE loaded via `load_sae_safetensors` and measure the Δ Spearman.
  Out of scope here (no torch on the dev Mac); document as a
  follow-up. Lean: open a `research/learned-axis-real-sae` branch
  on the next machine with torch.

- **Behavioural-objective adapter for sae-forge output.** sae-forge
  emits behavioural-validator co-activation matrices in a specific
  shape; v1 leaves the format up to the caller. A small
  `behavioural_objective_from_saeforge_report(report)` factory
  would close that loop. Lean: ship the factory in a sae-forge
  PR once polygram 0.8.0 lands; polygram itself stays
  format-agnostic.

- **Default switch.** When does `learn_axis_assignment=True` become
  the default? Lean: after real-SAE replication confirms the win on
  ≥ 2 real SAEs and the calibration cost is < 5s per import for
  N ≤ 1000. Until then, opt-in.
