## Context

The 8-feature cap is a Hilbert-space dimensionality constraint that
varies across encodings. `MPSRung1` produces 3-qubit states in C⁸, so
N ≤ 8 features can be linearly independent. `Rung3` is a tensor
product C⁸ ⊗ C² (the amp branch's parameterisation restricts to a
2-dim subspace `span{|00⟩, |11⟩}` of C⁴), so N ≤ 16. `HEA_Rung2`
generalises to `n_qubits` qubits, so N ≤ 2^n_qubits.

The 8-cap was set when only MPSRung1 existed. `add-rung3-encoding-mvp`
(2026-05-05) shipped Rung3 without touching the loader cap because
the cap wasn't yet load-bearing for downstream Rung3 workflows — the
viability spike used ≤8-feature fixtures and the rank ceiling wasn't
empirically measured until the `rung3-rank-bound` research note.

The rank-bound finding (`docs/research/rung3-rank-bound.md`) now
documents the per-encoding caps as empirical, sharp limits. This
change lands them in the loader.

## Goals / Non-Goals

**Goals:**
- Per-encoding `max_features` attribute on `MPSRung1`, `Rung3`, `HEA_Rung2`.
- Loader, validator, compression all query the encoding for its cap.
- Honest error messages naming the encoding when the cap trips.
- Backwards-compatible: callers that don't change encoding see no
  behaviour difference.

**Non-Goals:**
- Adding new encodings (Rung4 is a separate change,
  `add-rung4-encoding-mvp`).
- Changing the math: this change only surfaces caps that already
  exist mathematically. No new analytic primitives.
- Touching Q-OrCA emit. The cap is import-time, not emit-time.

## Decisions

**Decision 1 — `max_features` as a class attribute, not a method.**

For `MPSRung1` and `Rung3` the cap is constant. For `HEA_Rung2` it
depends on `n_qubits` which is a frozen-dataclass field. A property
on `HEA_Rung2` (and class-level constants on the other two) gives
uniform `encoding.max_features` access without forcing all encodings
into a method shape. This matches existing patterns (e.g.,
`HEA_Rung2.theta_shape` is a property).

**Decision 2 — Keep `MAX_FEATURES_PER_DICTIONARY` exported for backcompat.**

Some downstream consumers (notably tests and external scripts under
`scratch/`, plus the `BehaviouralValidator.MAX_FEATURES_PER_DICTIONARY`
import at `validator.py:53`) reference the constant. Removing it
breaks them. Retain the constant as a top-level re-export holding the
MPSRung1 value (8), but stop using it for enforcement. New code uses
`encoding.max_features`.

**Decision 3 — `Rung3.max_features = 16` is locked by the rank-bound finding.**

The empirical probe at 1e-12 relative tolerance saturates at 16
across two seeds. The 15-order gap between σ[15] and σ[16] makes
this an algebraic, not numerical, bound. If a future Rung3 variant
generalises the amp branch (see `add-rung4-encoding-mvp`), it ships
as a new encoding rather than mutating Rung3's cap in place.

**Decision 4 — Audit compression/regrow comments, but expect no logic changes.**

Quick read of `compression/compressor.py:400` and
`compression/regrow.py:461` (the two sites where comments reference
the cap) suggests narrative-only mentions, not logic dependencies.
This change includes an audit task to confirm; if logic does depend
on the value, replace with the per-encoding query.

**Decision 5 — Error message names the encoding.**

The current error at `sae_import.py:611` reads "Pick a smaller …".
The new error reads "Encoding `<encoding>` supports at most
`<N>` features; got `<M>`. To go higher, use an encoding with a
larger Hilbert-space dimension (e.g., `Rung3` for 16, `HEA_Rung2`
with larger `n_qubits` for `2**n_qubits`)." Surfaces the cap's
origin and the path to higher capacity.

## Risks / Trade-offs

**Risk:** existing callers passing >8 features against `MPSRung1`
will continue to fail (correct). Callers passing 9–16 features
against `Rung3` will now succeed where they previously failed. If
any downstream code assumed the 8-cap as a load-bearing invariant
beyond the import gate, that breaks here.

Mitigation: the audit of `compression/{compressor,regrow}.py` covers
the two known references. A grep sweep for `MAX_FEATURES_PER_DICTIONARY`
across the whole repo before merging is part of the task list.

**Risk:** users may construct `Rung3` dictionaries with 9–16 features
and find that downstream consumers (cancellation, Q-OrCA emit) work
but produce results they don't expect at the higher feature count.
Cancellation's 2-φ analytic structural floor is defined for
`encoding=mps` only; on Rung3 the floor reduces to the MPS-phase-only
floor of the same (α, β, γ). Both behaviours are documented but
become more visible at N>8.

Mitigation: error message and design.md flag this; documentation in
the rank-bound research note already covers it.

## Migration Notes

No migration required. `MPSRung1` users see no change; the cap they
hit is the same 8. Users currently using `Rung3` with ≤8 features see
no change. Users wanting to push past 8 on `Rung3` can now do so up
to 16, after this change merges.
