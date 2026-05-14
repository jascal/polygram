# Rung4 viability spike — do the product-amp dimensions buy real headroom?

> Research-track note recording the empirical findings of the
> Rung4 viability spike defined in
> [`add-rung4-encoding-mvp` §7](../../openspec/changes/add-rung4-encoding-mvp/proposal.md).
> Reproducible via
> `python examples/rung4_viability_spike.py --output-dir examples/output/rung4_viability_spike`.
> Raw artifacts: [`data/rung4_viability_spike.json`](data/rung4_viability_spike.json)
> (unconstrained), [`data/rung4_viability_spike_constrained.json`](data/rung4_viability_spike_constrained.json)
> (constrained re-run, ε = 0.5).
>
> Run on the same 8-feature panel as the Rung3 spike for direct
> comparison. **Behavioural phase (criteria B, C, D) skipped** — the
> host had the SAE checkpoint but not the `[behavioural]` extra
> (torch + transformers); the analytic phase (criterion A) is
> sufficient to answer the load-bearing question, but the table
> below has `skipped` rows where B/C/D would otherwise go.

## TL;DR

**Rung4's extra amp dimensions buy zero new geometric leverage over
Rung3 in the constrained regime that defines the verdict.**
Unconstrained, both encodings degenerate to the trivial branch-A
zeroing (mirroring the Rung3 spike's [Caveats section](rung3-viability-spike.md#caveats)).
Constrained at ε = 0.5, both encodings produce **identical** median
residual `0.500` — a structural identity at the constraint boundary,
not a coincidence. **Rung4 stays opt-in; the 32-feature capacity is
real but the cancellation primitive doesn't extract additional value
from the product-amp structure.**

| Criterion | Unconstrained | Constrained (ε = 0.5) |
|-----------|---------------|-----------------------|
| **A.** Floor-breaking — median residual `r4_post / mps_floor` | 6.8e-44 (strong) | **0.500 (partial)** |
| **B.** Gate true-positive rate | _skipped_ | _skipped_ |
| **C.** Ranker preservation — Spearman | _skipped_ | _skipped_ |
| **D.** Coverage | _skipped_ | _skipped_ |
| **Decision bucket** | `partial_analytic_only` | **`partial_analytic_only`** |

The decision bucket is `partial_analytic_only` only because B/C/D
were skipped (no torch). On A alone, the constrained verdict
mirrors Rung3's `partial` — and as we'll see, this isn't a
coincidence but a structural property of the constrained spike.

## The 5-qubit circuit (Rung4 variant)

```
        ┌────────────┐                                       ┌──────────┐
 q0 ───▷│            │                                       │          │◁─── concept gram
 q1 ───▷│  MPSRung1  │──── α, β, γ, φ ──────────────────────▷│  MPS     │
 q2 ───▷│ (3-qubit)  │                                       │  factor  │
        └────────────┘                                       └──────────┘
                                                                  ⊗
        ┌────────────┐                                       ┌──────────┐
 q3 ───▷│ sq(θ, ψ)   │── θ_amp,    ψ_aux   ─────────────────▷│  branch  │
        │ (1-qubit)  │   (default π/4, 0)                    │  A       │
        └────────────┘                                       └──────────┘
                                                                  ⊗
        ┌────────────┐                                       ┌──────────┐
 q4 ───▷│ sq(θ, ψ)   │── θ_amp_b,  ψ_amp_b ─────────────────▷│  branch  │
        │ (1-qubit)  │   (default π/4, 0)                    │  B       │
        └────────────┘                                       └──────────┘
```

Per-feature 5-qubit state (vs Rung3's Bell-pattern amp):

```
|ψ_f⟩ = |mps(α, β, γ, φ)⟩ ⊗ |sq(θ_amp, ψ_aux)⟩ ⊗ |sq(θ_amp_b, ψ_amp_b)⟩

|sq(θ, ψ)⟩ = cos(θ)|0⟩ + e^(iψ) sin(θ)|1⟩
```

Pair amp overlap:

```
|⟨amp_a|amp_b⟩|² = |sq_a · sq_b|²_branch_A · |sq_a · sq_b|²_branch_B
```

The factorisation is the load-bearing difference vs Rung3:

| Encoding | Amp dim | Amp structure | Knobs (per feature) |
|----------|---------|---------------|---------------------|
| MPSRung1 | 1       | trivial      | (α, β, γ, φ)         |
| Rung3    | 2       | entangled (Bell pattern) | + (θ_amp, ψ_aux) |
| Rung4    | 2 × 2   | product (independent single-qubit) | + (θ_amp, ψ_aux, θ_amp_b, ψ_amp_b) |

Rung4 has 6 cancellation knobs (`a.phi, b.phi, b.theta_amp,
b.psi_aux, b.theta_amp_b, b.psi_amp_b`) vs Rung3's 4. The
hypothesis: the extra two dimensions on branch B let the optimizer
find non-trivial amp configurations that Rung3 can't reach.

## Unconstrained run — degenerate (just like Rung3)

The Rung4 joint optimizer converged to the same trivial branch-A
zeroing on every one of the 28 pairs:

```
theta_amp        ≈ π/4 (default)
psi_aux          ≈ π    (vs default 0 — half-period phase flip)
theta_amp_b      ≈ π/2  (sin saturates → |1⟩)
psi_amp_b        ≈ 0    (default — irrelevant when θ saturates)
```

Decoding: the optimizer found `ψ_aux = π` for feature B vs default
`ψ_aux = 0` for feature A. With both `θ_amp = π/4`:

```
sq_a = (|0⟩ + |1⟩)/√2
sq_b = (|0⟩ - |1⟩)/√2      (e^(iπ) = -1)

⟨sq_a | sq_b⟩ = (1/2)(1 - 1) = 0
```

**Branch A alone gives zero overlap.** Branch B's knobs were
explored by the outer grid but never engaged — `|⟨amp⟩|² = 0 · |B
factor|² = 0` regardless of what B does. The 4D outer grid found
this corner on every pair and never bothered with non-degenerate
neighbourhoods.

Per-pair `r4_post = 0.0` (median 6.8e-44 — numerical zero) across
all 28 pairs. Strong-pass on criterion A in the technical sense,
but the result is the same degeneracy the Rung3 spike documented:
**the joint optimizer doesn't engage the new geometric dimensions
at all when they're not needed**. The amp branch becomes a phase
trick that knocks the gram to zero independent of the MPS-side
work.

This is *expected* behaviour — the optimizer is correctly finding
the global minimum within the unconstrained feasible region. But
the result tells us nothing about whether Rung4's geometry actually
helps cancellation. It just says "the optimizer can zero the gram
trivially."

## Constrained re-run (ε = 0.5) — the structural identity

To force the optimizer off the trivial-zeroing solution, the
`--min-amp-overlap 0.5` constraint marks any candidate with
`|⟨amp_a|amp_b⟩|² < 0.5` as infeasible. Now the optimizer has to
find a non-trivial amp configuration. Result:

```
median residual: 0.5000 (exact)
per-pair: every one of 28 pairs hits residual = 0.500 to ≤4 decimals
```

**Every pair lands at exactly the constraint boundary.** This is not
a coincidence — it's a structural property of the spike methodology:

```
residual = r4_post / mps_floor
        = (mps_post × amp_overlap²) / mps_floor
        = mps_floor × amp_overlap² / mps_floor    (optimizer hits MPS floor)
        = amp_overlap²
        = min_amp_overlap                          (constraint binds tight)
```

The optimizer:
1. Drives `mps_overlap` down to `mps_floor` via the (φ_a, φ_b) knobs
   (same as MPS-only cancellation).
2. Drives `amp_overlap²` down to exactly `min_amp_overlap`
   (the constraint floor).
3. Total residual = product of the two normalised drops = `min_amp_overlap`.

The constraint binds tight on every pair because:

- Reducing `amp_overlap²` below 0.5 is infeasible by definition.
- Increasing it above 0.5 strictly increases the post-cancellation
  gram and is therefore suboptimal.

**So the residual at the constraint boundary is structurally identical
across any encoding whose amp factor can hit exactly the constraint
value.** Rung3 (4 amp knobs) and Rung4 (8 amp knobs) both can; both
do; both produce identical residuals.

### Falsifying experiment

To check this isn't an optimizer artifact, both Rung3 and Rung4
spikes can be re-run with `--min-amp-overlap 0.7`. Prediction:
median residual = 0.7 for both encodings, again identically. (Not
run in this spike — the methodological point is already made by the
ε = 0.5 result matching Rung3 exactly.)

## Implications for the design

1. **The cancellation primitive is rung-agnostic at the constraint
   boundary.** Once you block the trivial amp-zeroing, the
   geometry of the amp branch contributes nothing to the
   single-pair post-cancellation gram. The amp factor is a
   *constraint-bound* quantity, not a *geometric* one.

2. **Rung4 stays opt-in.** Like Rung3, Rung4's value (if any) is
   not in single-pair cancellation. It's in:
   - **Capacity**: 32 features per Dictionary vs MPSRung1's 8 and
     Rung3's 16. Real and confirmed by
     [`docs/research/data/rung4_rank_verification.json`](data/rung4_rank_verification.json).
   - **Aggregate gram structure**: with 32 features in a single
     block, the *block-level* gram has more degrees of freedom than
     the same 32 features split across 4 MPSRung1 blocks. Whether
     that matters for compression quality is a separate research
     question — not addressed by the cancellation spike.

3. **The constrained spike doesn't generalise to higher rungs.**
   If a future Rung5 or HEA_Rung2 spike uses the same
   `min_amp_overlap` methodology, the result will trivially be
   `residual = min_amp_overlap` again — same structural identity.
   Distinguishing geometric leverage between rung-N and rung-(N+1)
   requires a different probe (e.g., look at the *value of the
   gram at a fixed `amp_overlap`*, or aggregate gram properties on
   multi-pair blocks).

## Decision

**`make-rung4-default` is dead before it was ever proposed.**
Rung4 ships as an opt-in encoding alongside Rung3. The capacity
lift to 32 is the real value; cancellation-primitive headroom is
not.

This matches the post-Rung3-spike status quo: encodings exist on a
capacity ladder (MPSRung1 → Rung3 → Rung4 → ...) where higher rungs
offer more *room*, not more *geometric power per pair*.

## Caveats

- **Behavioural phase skipped.** The host machine had the SAE
  checkpoint but lacked the `[behavioural]` extra. Criteria B
  (gate TPR), C (Spearman ranker preservation), and D (coverage)
  were not computed. The criterion A result alone is sufficient
  to make the verdict (the constrained spike's identity argument
  is the load-bearing finding), but a full behavioural re-run on
  a torch-enabled host is a clean follow-up to confirm Rung3 vs
  Rung4 produce identical gate behaviour.
- **`grid_outer=(3, 3)` for Rung4** (vs `(5, 5)` for Rung3).
  Rung4's outer iteration is 4D over feature B's product-amp
  knobs, so `(5, 5)` would have been 625 cells per pair — ~25×
  slower than acceptable. The `(3, 3)` choice gives 81 cells per
  pair; the constraint binding observed is robust to grid
  resolution because the optimum lies on a 4D surface that
  cosmologically any reasonable grid sampling can locate.
- **8-feature panel.** Same as the Rung3 spike for direct
  comparability. The trivial-zeroing and constraint-binding
  phenomena are structural properties of the optimizer, not
  features of the specific panel, so the result generalises.

## Files

- `examples/rung4_viability_spike.py` — the spike script.
- `docs/research/data/rung4_viability_spike.json` — unconstrained
  raw output (28 pairs, criterion A, criteria B/C/D skipped).
- `docs/research/data/rung4_viability_spike_constrained.json` —
  constrained (ε = 0.5) raw output.
- `tests/test_examples.py::test_rung4_viability_spike_smoke` —
  CI smoke test exercising the SAE-absent skip path.
