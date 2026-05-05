## Context

Polygram's `MPSRung1` encoding builds each feature's state as
`|ψ⟩ = R_α(qubit 0) ⊗ R_β(qubit 1) ⊗ R_γ(qubit 2)` with a single
phase knob φ entangling the three rotations. The 2-φ
phase-only optimizer that `Cancellation` runs against this shape
gives an analytic floor: per-pair squared overlap is bounded
to `[M − |V|, M + |V|]`, with M and V determined by the
(α, β, γ) assignments. The `polygram.cancellation.Cancellation`
result already exposes `structural_floor()` (which raises for
non-MPS encodings) — this MVP plugs into that field rather than
inventing a parallel surface.

`HEA_Rung2` adds a depth-1 hardware-efficient ansatz with a
single rotation knob, but that knob is still phase-shaped: its
eigenvalues are ±1, so squared overlap stays a pure sinusoid in
the knob value (per the floor memo). Practically, HEA_Rung2 is a
re-parameterization of MPSRung1's floor, not an escape from it.

Rung3's claim is that an *amplitude* knob θ_amp, applied to an
auxiliary qubit pair, modulates M itself rather than the
phase-relative position between M and V. If true, joint
optimization over (φ, θ_amp, ψ_aux) breaks below `M − |V|`.
The MVP measures whether that claim survives contact with
real data.

## Goals / Non-Goals

**Goals:**

- A `polygram.encoding.Rung3` class, parallel to MPSRung1 and
  HEA_Rung2, with analytic `compute_concept_gram(other)`.
- `Cancellation(encoding="rung3")` that jointly searches
  (φ, θ_amp, ψ_aux) and reports the post-optimization residual
  alongside the *MPS phase-only floor* of the same pair.
- Q-Orca emission of the 5-qubit Rung3 template from the
  existing `_qorca_emit` path.
- A worked example (`examples/rung3_viability_spike.py`) that
  runs the full 28-pair comparison and prints a decision banner
  per the four criteria.
- Zero changes to `BehaviouralValidator` or `Compressor`
  surfaces (both are encoding-agnostic).

**Non-Goals:**

- Shipping Rung3 as the production default. Conditional
  follow-up.
- Running Rung3 on non-GPT-2 models.
- Comparing Rung3 against depth-4 HEA. Separate probe.
- Auto-tuning (θ_amp, ψ_aux) initialization. 5×5 grid +
  scipy refine is the contract.
- Modifying `cancellation.structural_floor()` to support Rung3.
  The MVP keeps the floor concept anchored to MPSRung1 (the
  encoding whose floor we are trying to break).

## Decisions

### Decision 1 — Calibrated success criteria (the four lights)

The decision rule is the four-criterion table in `proposal.md`.
The numbers are calibrated against shipped data:

- **A's threshold (`r ≤ 0.3` strong)** comes from "broke 70%
  of the floor on average" — chosen because the floor is the
  thing the experiment is testing, and 70% is a clear
  magnitude-of-effect bar that distinguishes Rung3 from a
  well-tuned MPS optimizer.
- **B's threshold (`80%` strong, `66%` failure)** comes from
  §4.4 directly: the Polygram ≥ 0.7 gate currently has a 8/12
  ≈ 66.7% true-positive rate against Jaccard ≥ 0.30. Strong
  pass means Rung3 raised it to ≥ 80%; failure means Rung3
  regressed below the §4.4 baseline.
- **C's threshold (`+0.65` strong, `+0.50` failure)** comes
  from §4.4's +0.637 Spearman. Strong pass means Rung3 held
  ranker quality essentially flat or improved; failure means
  Rung3 destroyed the +0.637 ranker signal that was the
  loop-unblocking finding.
- **D's threshold (`90%` strong, `80%` failure)** comes from
  §4.4's 8/8 = 100% gate coverage of Jaccard ≥ 0.30 pairs.
  Strong is "essentially preserved"; failure is "hides one of
  five real redundancies."

The decision rule is applied in priority order (D first, then
A, then B + C). D-first reflects the safety property — Rung3
is unacceptable if it hides real redundancies, regardless of
how cleanly it cancels the ones it sees.

### Decision 2 — `Rung3` defaults to MPSRung1-equivalent behaviour

`Rung3.theta_amp` defaults to π/4; `psi_aux` defaults to 0.
At those defaults, the amplitude branch (qubits 3–4) reduces
to identity on the joint state and `Rung3.compute_concept_gram`
returns the same value as the corresponding MPSRung1 instance.

Rationale: a baseline Rung3 dictionary with default knobs
should behave identically to its MPSRung1 counterpart for the
purposes of cancellation_gap ranking, dictionary materialization,
and downstream validation. The optimizer, not the encoding
class, is what activates the new knobs.

This also gives the spike a clean A/B comparison: build *one*
Rung3 dictionary; run two cancellations on it (MPS-mode and
Rung3-mode); the geometry is identical, only the knob set
changes.

### Decision 3 — `Cancellation(encoding="rung3")` reports MPS floor as the residual baseline

A Rung3 cancellation result's `structural_floor` field SHALL
carry the *MPS phase-only floor* of the same pair. The Rung3
encoding does not have a closed-form floor (the amplitude branch
removes the analytic bound); reporting the MPS floor as the
baseline lets the spike compute `residual = post_rung3 /
floor_mps` directly without bookkeeping juggling.

Two consequences:

- The `Cancellation(encoding="rung3")` implementation runs an
  internal `MPSRung1.structural_floor()` call on the same
  (α, β, γ) tuple as a side computation; this is essentially
  free (closed-form arithmetic).
- `result.structural_floor` for a Rung3 result is *not* a floor
  the Rung3 optimizer was bounded by; it is the MPS-phase floor
  the Rung3 result was *trying to break*. Doc this clearly in
  the result's docstring.

Alternative considered: emit a separate `mps_floor_baseline`
field. Rejected because (a) downstream code already reads
`structural_floor`, and (b) the conceptual question the field
answers ("what was the bar this pair started at?") is
unchanged.

### Decision 4 — 5×5 outer grid; existing 2-φ inner; scipy final refine

The `Cancellation(encoding="rung3")` optimizer:

1. **Outer grid**: 5×5 over (θ_amp ∈ [0, π/2], ψ_aux ∈ [0, 2π)).
   Configurable via `Cancellation(grid_outer=(5, 5))`. The
   default 5×5 = 25 cells is calibrated against the existing
   single-knob phase-grid cost.
2. **Inner**: at each outer cell, run the existing MPSRung1
   2-φ optimizer to convergence. Cache per-cell best (φ, post).
3. **Scipy refine**: take the best outer cell, run
   `scipy.optimize.minimize` over (φ_a, φ_b, θ_amp, ψ_aux)
   from that cell's best as the initial point.

The 5×5 grid is chosen because:

- It mirrors the existing `Cancellation` phase-grid cadence
  (which is 8×8 = 64 phase cells; outer-amp is at the same
  order of magnitude).
- 25 outer cells × ~64 inner phase cells × 28 pairs × 2
  encodings ≈ 90,000 closed-form Gram evaluations. At ~1ms
  each that is ≈ 90 s wall-clock — well within the < 30 min
  spike budget.
- Coarser grids risk missing the optimum's basin; finer grids
  buy little for a viability test.

### Decision 5 — Q-Orca emission for 5-qubit Rung3

`polygram._qorca_emit` learns a `_emit_rung3_machine` path that
produces a `.q.orca.md` file with:

- `## context` declaring `qubits: list<qubit>` of length 5 per
  feature.
- `## state` declarations for the prepared state per feature
  (the existing emitter pattern, just with two extra qubits).
- `## transitions` referencing the new `apply_amp_branch` action
  (the analog of MPSRung1's `apply_R_alpha`).
- `## actions` listing the new action signatures.

The emitted machine MUST parse + verify clean against the
shipped `q-orca` verifier with no new verification rules. The
existing rules (unitarity, Schmidt rank > 1 for entangled
states) are sufficient for the 5-qubit shape; Rung3's auxiliary
qubit pair is locally unitary by construction, so unitarity
holds, and the encoded state is entangled iff the corresponding
MPSRung1 state was, so Schmidt rank checks still fire.

Out-of-scope: new q-orca verification rules. The MVP
deliberately stays inside the existing rule set.

### Decision 6 — Probe harness as a worked example, not a spec capability

`examples/rung3_viability_spike.py` is the harness. Same shape
as `examples/behavioural_validate.py`:

- CLI args (`--feature-ids`, `--sae-checkpoint`, `--output-dir`,
  `--n-prompts`).
- Skip path: SAE checkpoint or torch absent → exit 0 with
  hint message (matches §4.2 / §4.3 / §4.4 pattern).
- Prints the four-criterion banner + decision bucket.
- Writes per-pair JSON (`rung3_viability_spike.json`) for the
  research-note writeup.

The harness is *not* a spec capability because:

- It does not introduce a public Python surface (it is a
  script).
- The decision rule is documented in `proposal.md` and
  duplicated in the harness output, but the rule itself is a
  research artifact, not a contract for downstream code.

A smoke test in `tests/test_examples.py` mirrors the validator
smoke pattern — exit 0 on the SAE-or-torch-absent branch.

### Decision 7 — Research note shape

`docs/research/rung3-viability-spike.md` is added in the
*findings PR* (separately from this spec PR), once the harness
has been run. The findings PR's structure mirrors §4.2 / §4.3 /
§4.4:

- TL;DR table (MPS vs Rung3 numbers across the four
  criteria).
- 5-qubit circuit diagram.
- Per-pair before/after table.
- Decision (which bucket) + next steps.

The decision is **load-bearing**: it determines whether the
follow-up change is "make Rung3 the default" (strong pass),
"calibration probe" (partial), or "deprecate Rung3 / explore
hybrid" (fail).

### Decision 8 — One implementation PR + one findings PR cadence

Same as §4.4's PR pair (#24 + #25): the spec PR ships the
infrastructure (this change); a follow-up PR ships the
research note + harness output. The findings PR does not
modify the contract — it reports against it.

Justification: the spike's decision-driven nature means the
findings *are* the deliverable. Folding both into a single PR
would either (a) ship code without findings (bad cadence) or
(b) delay code review behind a long-running probe (bad
cadence).
