## Context

`Rung3` was shipped via `add-rung3-encoding-mvp` (2026-05-05) with an
amp-branch state `|amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩` on
qubits 3–4. This was a Bell-pattern parameterisation: at θ=π/4, ψ=0
the amp state is the maximally-entangled `(1/√2)(|00⟩ + |11⟩)`. The
parameterisation has 2 real DOF and was designed so that its default
knobs (θ=π/4, ψ=0) make the amp overlap factor identically 1, so
Rung3 grams reduce to MPSRung1 grams on the same (α, β, γ, φ) at
default.

The Rung3 viability spike (`docs/research/rung3-viability-spike.md`)
found that the joint optimizer degenerates to the trivial
amp-zeroing solution on every pair unless `min_amp_overlap > 0` is
enforced. The constrained re-run downgraded the bucket from
`strong_pass` to `partial_pass`. Rung3 stays opt-in.

The follow-up `rung3-rank-bound` finding revealed the *geometric*
reason for Rung3's limited extra leverage: the Bell-pattern
parameterisation only spans a 2-dim subspace of the 2-qubit amp
Hilbert space C⁴. The amp-branch's contribution to the total rank
cap is 2, not 4.

Rung4 fixes the parameterisation: two independent single-qubit amps,
each spanning its full C². The product spans all of C² ⊗ C² = C⁴.
Total cap: 8 · 4 = 32.

## Goals / Non-Goals

**Goals:**
- Ship `Rung4` as a new encoding parallel to `Rung3`, not a
  replacement.
- Analytic gram via the elementwise-product factorisation; reuse the
  existing single-qubit overlap math (the current
  `rung3_amp_overlap` IS a single-qubit overlap formula in disguise).
- Empirically verify the 32 cap via a Rung4-targeted run of the
  existing rank probe.
- Wire Rung4 through cancellation, Q-OrCA emit, and a viability
  spike — the full MVP loop, mirroring Rung3.
- Backwards compatibility: existing Rung3 dictionaries and tests
  unchanged.

**Non-Goals:**
- Generalising to a Schmidt-form 6-knob amp branch with entanglement
  between q3 and q4 (deferred — would be a hypothetical Rung5).
- Lifting q-orca-lang's safe-Rz matcher pin. Rung4 keeps the MPS-side
  pin and only changes the amp branch.
- Changing `MPSRung1` or `Rung3`. Both stay.
- A 4-qubit MPSRung1 extension — separate cross-repo change.
- Default-on Rung4 in `Dictionary.encoding` — opt-in only, same as
  Rung3.

## Decisions

**Decision 1 — `Rung4` is a new encoding class, not a Rung3 variant.**

Mutating `Rung3` in-place would invalidate the rung3-viability-spike's
conclusions and force migration on existing Rung3 users. A new class
is additive and reuses the existing dispatch pattern.

**Decision 2 — Feature gains `theta_amp_b` and `psi_amp_b` (additive fields).**

Per the user-facing pattern already in `Feature` (one float per knob),
the q4 single-qubit amp knobs become two new fields with default
`0.0`. Rung3 dictionaries that don't touch them are unaffected (the
Rung3 amp factorisation doesn't consume these fields). Rung4 reads
both pairs.

An alternative blob (`amp_knobs: tuple[float, ...] | None = None`)
would be cleaner long-term but is a larger refactor with no immediate
payoff; deferred.

Naming: existing `theta_amp` and `psi_aux` become **also** the q3
single-qubit amp knobs under Rung4's interpretation. The names don't
change. Backward compat is preserved at the field level.

**Decision 3 — Single-qubit overlap is extracted as a helper.**

The current `rung3_amp_overlap(θ_a, ψ_a, θ_b, ψ_b)` is mathematically
a single-qubit overlap `⟨u(θ_a, ψ_a) | u(θ_b, ψ_b)⟩` where
`|u(θ, ψ)⟩ = cos(θ)|0⟩ + e^(iψ) sin(θ)|1⟩` — modulo the half-angle
convention. Refactor it into `_single_qubit_overlap`, and have
`rung3_amp_overlap` and `rung4_amp_overlap` both call it. This keeps
the Rung3 math byte-identical (no behaviour change) and gives Rung4
its product-of-singles math for free.

**Decision 4 — Default knob values for Rung4 reduce to MPS gram.**

To match the Rung3 invariant ("default Rung3 = MPS gram on
(α,β,γ,φ)"), pick Rung4 defaults so the amp overlap factor is
identically 1 for the default-knob case. The simplest choice:
`theta_amp = theta_amp_b = 0`, `psi_aux = psi_amp_b = 0`. Then both
single-qubit amps are |0⟩, and ⟨0|0⟩·⟨0|0⟩ = 1.

This differs from Rung3 (which picks θ=π/4, ψ=0 for symmetry). Both
satisfy the invariant; Rung4's choice is simpler because the product
factorisation makes the |0⟩ state the natural identity.

**Decision 5 — Cancellation knob list for Rung4 is fixed at 6.**

`Cancellation(encoding="rung4")` requires the canonical 6-knob list
`[a.phi, b.phi, b.theta_amp, b.psi_aux, b.theta_amp_b, b.psi_amp_b]`.
Custom knob lists are not supported in v0 (mirrors Rung3's stance).
Feature A's amp stays anchored at defaults; only feature B's amp
knobs are optimised, plus both features' phi.

**Decision 6 — Joint optimiser: refactor `_run_rung3_joint` into shared `_run_amp_joint`.**

The Rung3 joint optimiser has three stages: outer grid over amp
knobs, inner 2-φ grid at every outer cell, scipy Nelder-Mead refine
over the full set. Generalising the outer-grid axes from 2D
(θ_amp, ψ_aux) to 4D (θ_amp, ψ_aux, θ_amp_b, ψ_amp_b) is a knob-list
parameterisation, not a structural rewrite. Refactor
`_run_rung3_joint` into `_run_amp_joint(amp_knob_count: int)` and
dispatch on encoding.

The 4D outer grid for Rung4 is `grid_outer ** 4` cells. At the
default `(5, 5)` setting from `Rung3CancellationConfig.grid_outer` that's
625 outer cells per pair (vs Rung3's 25). Worth checking wall-clock
on the viability spike fixture; may need to reduce default density
for Rung4.

**Decision 7 — Q-OrCA emit: simpler than Rung3.**

Rung3 emits a Bell-pattern amp branch with a CNOT between q3 and q4
(plus per-feature θ and ψ rotations). Rung4 emits two independent
single-qubit preparations on q3 and q4 — no CNOT between them. The
action signature carries 4 amp knobs per feature instead of 2; the
register stays at 5 qubits. q-orca's safe-Rz matcher pins q0–q2 / χ=2
and is agnostic to the q3–q4 amp shape, so no q-orca-lang work is
expected.

Verification: round-trip a 2-feature Rung4 dictionary through the
emit + parse path and confirm the analytic gram matches the
q-orca-evaluated gram to 1e-10.

**Decision 8 — Viability spike methodology mirrors Rung3.**

Run the same four-criterion (A/B/C/D) bucket analysis from the Rung3
spike (`docs/research/rung3-viability-spike.md`), against the same
GPT-2-small SAE pair fixture. Two runs: unconstrained and
constrained (`min_amp_overlap = 0.5`). Decision rule for
default-on: if the constrained re-run lands in `strong_pass` (unlike
Rung3, which landed in `partial_pass`), recommend making Rung4 the
default encoding. Otherwise it stays opt-in.

## Risks / Trade-offs

**Risk:** Rung4's joint optimiser may also degenerate to trivial
amp-zeroing solutions on the constrained re-run. The extra 2 amp
knobs give MORE degenerate paths, not fewer.

Mitigation: the viability spike's `min_amp_overlap` constraint
applies the same way to the product-amp overlap. If the constrained
re-run fails, Rung4 stays opt-in (same outcome as Rung3) and the
research note documents why. Not a blocker for shipping the
encoding itself — the rank-cap improvement (16 → 32) is independent
of the viability outcome.

**Risk:** Wall-clock cost of the 4D outer grid (625 cells default).

Mitigation: reduce `grid_outer` default for Rung4 (e.g., `(3, 3, 3, 3)`
= 81 cells), or split the outer grid into a coarse pass + scipy
refine without a full Cartesian product. Settle in design review of
the cancellation extension task.

**Risk:** `Feature` gaining two new fields could subtly affect
serialisation, hashing, or other invariants that assume the field
list.

Mitigation: add the fields as default-zero. Existing JSON/safetensors
round-trips that omit the fields will reconstruct them at zero.
Targeted regression tests for `Feature.__eq__`, `Feature.__hash__`,
and the SAE-import JSON round-trip.

## Sequencing

Depends on `per-encoding-feature-cap` (must merge first; this change
relies on `encoding.max_features` being the enforcement mechanism).

Within this change:

1. Encoding class + math (additive, low risk).
2. Rank verification (empirical confirmation of the 32 cap).
3. Dictionary.gram() dispatch.
4. Q-OrCA emit (round-trip test).
5. Cancellation extension (most code).
6. Viability spike (research output).

Each section's tests gate the next; the worked example only runs
once the full chain is green.

## Migration Notes

No migration. Users opt into Rung4 by constructing `Rung4()` and
passing it as `Dictionary.encoding`. Existing Rung3 and MPSRung1
dictionaries continue to work unchanged. The two new `Feature`
fields default to 0.0 — Rung3 gram math ignores them; Rung4 gram math
uses them.
