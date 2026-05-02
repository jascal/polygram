# Polygram `Cancellation` — phase-only search has a structural floor

**Status:** finding from v0 implementation (commit `16c4523`, 2026-05-02). Inverts a framing in the original `cancellation-primitive` proposal.

## The math

`Cancellation` varies only the φ knobs of the target pair. β/α/γ stay fixed. For any target pair the squared overlap factors as:

  **|⟨A|B⟩|² = M + V · cos(δ)**, where δ = φ_A − φ_B

- **M** — DC component, set by β/α/γ alignment between A and B
- **V** — phase swing; `Cancellation` modulates only this term

Range over all δ: `[M − |V|, M + |V|]`. There is a **structural floor at M − |V|** that pure-phase search cannot pierce. To genuinely zero out cross-overlap you'd need `|V| = M` — exact amplitude matching between the |0⟩- and |1⟩-branch contributions to the inner product. With shared α=γ=0 and per-cluster β (the `from_sae_lens` default), that equality is generically broken.

CS framing: phase is the sign bit. You can sign-flip to add destructively, but if magnitudes don't match the destructive sum leaves a residue. The residue lives in β/α/γ — outside the primitive's search space.

## Empirics on Animals-4

Toy dictionary, target `(dog_poodle, hawk_red)`: **M ≈ 0.68, V ≈ −0.089**. Because V is negative:

| direction | δ | cos(δ) | overlap | role |
|---|---|---|---|---|
| diagonal `φ_A = φ_B` | 0 | +1 | **0.59** | min — cancellation channel |
| antidiagonal `(0, π)` / `(π, 0)` | ±π | −1 | **0.77** | max |

The original proposal called the antidiagonal "where overlap collapses." That reading swapped min and max. The cancellation direction in this encoding is the **diagonal**.

## What was changed in the spec

`openspec/changes/cancellation-primitive/specs/experiment/spec.md`, scenario "grid backend reduces target overlap on Animals," tightened from:

  `after_overlap < before_overlap`  →  `after_overlap <= before_overlap + 1e-9`

The strict `<` was unachievable: starting from φ=0 already sits on the floor.

## What `Cancellation` actually is

A *constraint solver*: "find the matched-φ point — or the best feasible point under the tier-preservation constraint when matched-φ violates ordering." The materialized `.q.orca.md` artifact and `feasible_count` reporting still earn their keep, especially when starting from a non-zero φ configuration.

## Implication for future work

A "Cancellation v2" that genuinely drives overlap to zero needs at least one of:

1. **Search over β/α/γ as well as φ** — re-engineering the encoding, not just steering it.
2. **Multi-feature phase coordination** — coordinating phases across features within one cluster so the |0⟩/|1⟩ branch amplitudes can be balanced.
3. **A richer encoding** — more qubits / more parameters per feature.

All three are larger-than-v0 changes and should not be quietly grafted onto the current primitive.
