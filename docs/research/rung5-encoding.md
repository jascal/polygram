# Rung5 encoding

**Date:** 2026-05-16
**Status:** shipped in polygram 0.7.0 (this change)
**Driver:** sae-forge pareto sweeps over feature counts

## Summary

`Rung5(n_amp_qubits=k)` generalises `Rung4`'s fixed-width product amp
branch to a configurable `k` independent single-qubit amps on qubits
`q3..q3+k-1`. Per-feature Hilbert dim is `8 · 2^k`; `Rung5(k=2)` is
numerically equivalent to `Rung4` (both reduce to two single-qubit
amps composed with the same MPSRung1 substrate). The encoding rejects
`k=0` — that case is `MPSRung1` and should be spelled directly.

| Encoding      | Amp qubits         | Per-feature dim |
|---------------|--------------------|-----------------|
| `MPSRung1`    | 0                  | 8               |
| `Rung3`       | 2 (Bell-pattern)   | 16              |
| `Rung4`       | 2 (product)        | 32              |
| `Rung5(k)`    | k (product)        | 8 · 2^k         |

The cap `RUNG5_MAX_N_AMP_QUBITS = 16` gives a hard upper bound of
`8 · 2^16 = 524288` features per dictionary. sae-forge can read the
constant directly to validate sweep ranges.

## Math

Each feature's amp state factorises as

```
|amp(θ_0, ψ_0, …, θ_{k-1}, ψ_{k-1})⟩
    = ⊗_{i=0}^{k-1} (cos(θ_i)|0⟩ + e^(iψ_i) sin(θ_i)|1⟩)_{q(3+i)}
```

No entangling gates between amp qubits — same product geometry as
Rung4 generalised to k factors. The pairwise amp overlap is the
k-fold product of single-qubit overlaps:

```
⟨amp_a | amp_b⟩ = ∏_{i=0}^{k-1} ⟨u(θ_a_i, ψ_a_i) | u(θ_b_i, ψ_b_i)⟩
```

Reuses the existing `_single_qubit_overlap` helper unchanged.

The default-knob property generalises trivially: every
`(θ_i, ψ_i) = (0, 0)` gives a single-qubit overlap factor of 1, so a
default-knob Rung5 dictionary's gram equals the MPSRung1-equivalent
gram on the same `(α, β, γ, φ)`. This is the same fixed-point
property Rung3 and Rung4 ship, scaled across k.

## Empirical rank verification

`examples/rung5_rank_verification.py` sweeps a k-ladder and computes
gram rank at `N ∈ {cap/4, cap/2, cap, 2·cap}` for each k. The committed
artifact [`docs/research/data/rung5_rank_verification.json`](data/rung5_rank_verification.json)
confirms saturation at `8 · 2^k` for k ∈ {2, 3, 4} across two seeds:

| k | cap (8·2^k) | rank @ N=cap | rank @ N=2·cap | seeds |
|---|-------------|--------------|----------------|-------|
| 2 | 32          | 32 ✓         | 32 ✓ (saturated) | {0, 42} |
| 3 | 64          | 64 ✓         | 64 ✓ (saturated) | {0, 42} |
| 4 | 128         | 128 ✓        | 128 ✓ (saturated) | {0, 42} |

The Rung4 viability-spike result (rank-32 cap) is recovered as the
`k=2` slice; the Rung5 cap-vs-k relationship holds tightly through
k=4, with no anomalies at the boundary.

## Design choices

### `k` is fixed at `Rung5` construction time

Per-feature variable `k` was explicitly out of scope. sae-forge selects
a single `k` per pareto point. Auto-growing `k` as features are added
is also out of scope — the dictionary commits to its `k` at
construction and stays there. A later dictionary can always be built
at a larger `k` if a sweep wants more headroom.

### `k=0` is rejected, not aliased to `MPSRung1`

A `Rung5(n_amp_qubits=0)` would be numerically identical to
`MPSRung1`. Forbidding it (with an error pointing callers at
`MPSRung1` directly) keeps the encoding ladder discriminator clean —
the *presence* of an amp branch is the load-bearing distinction
between `Rung5` and the MPS-only encoding. Two ways to spell the same
encoding invites drift.

### Cancellation goes scipy-only

The Rung4 outer-grid joint cancellation explodes as `(M·N)^k` at high
k. Rung5's joint optimiser instead runs scipy `differential_evolution`
over the full `(2 + 2k)`-dim bounded space — dimension-agnostic, no
per-k tuning. For sae-forge sweeps pushing k high, the recommended
flow is: pre-screen pairs with the cheap φ-only cancellation (φ knobs
only, no amp axes), and reserve the full joint solve for pairs that
the pre-screen flagged as ambiguous. The optimiser dimension grows
linearly in k (24-dim at k=10), so this stays tractable on commodity
hardware.

### Q-OrCA emission keeps the MPSRung1 substrate

The emitter writes the MPSRung1 staircase machine on q0..q2 unchanged
and appends a `## amp branch` table with one indexed `(θ_i, ψ_i)` pair
per amp qubit per feature. This matches the Rung3/Rung4 precedent —
the amp branch is informational from q-orca's perspective; the
analytic gram path in `Dictionary.gram()` applies the k-fold product
factor on top.

Forward-looking: if a future version wants q-orca to compile the amp
qubits into actual gates, the table format already carries the data
shape needed (`k` columns per qubit indexed unambiguously).

## Future direction: the general (M, k) family

`Rung5` is one slice of a broader two-axis encoding family
parameterised by `(M, k)` — `M` the MPS-core width (today fixed at
3 qubits, bond_dim=2) and `k` the product-amp register width.
`MPSRung1 = (3, 0)`, `Rung4 = (3, 2)`, `Rung5(k) = (3, k)`; per-
feature Hilbert dim is `2^M · 2^k` in the general case. Reframing
the encoding ladder as a `(M, k)` plane (rather than a 1D rung
sequence) makes the design space's open directions explicit: this
PR moves along the k-axis; future work can move along the M-axis
or jointly.

A unified `RungMPS(n_mps_qubits=M, n_amp_qubits=k)` encoding would
be the cleanest landing for that family, with `MPSRung1`, `Rung4`,
and `Rung5` recoverable as fixed slices. Out of scope for this
change. The `Rung5` name doesn't ossify the design space — if
`RungMPS` lands, `Rung5` can be re-expressed as
`RungMPS(n_mps_qubits=3, n_amp_qubits=k)` with no caller breakage,
since `Rung5`'s public API is its `n_amp_qubits` field and the
amp-overlap functions.

## Files

- `polygram/encoding.py` — `Rung5`, `Rung5State`,
  `rung5_amp_overlap`, `rung5_amp_overlap_squared`,
  `RUNG5_MAX_N_AMP_QUBITS`.
- `polygram/dictionary.py` — `Feature.amp_knobs`,
  `Feature.with_default_amp_knobs`, `Dictionary` validation +
  `gram()` dispatch + `with_knob` `amp_knobs[i].{theta,psi}` paths.
- `polygram/cancellation.py` — `"rung5"` SUPPORTED_ENCODINGS entry,
  `_run_rung5_joint` using scipy `differential_evolution`,
  k-independent `structural_floor` via `_mps_equivalent_floor`.
- `polygram/_qorca_emit.py` — Rung5 `## amp branch` table emission.
- `polygram/geometry/amp_assignment.py` — Rung5 PCA-axis branch for
  populating amp_knobs from decoder geometry.
- `examples/rung5_rank_verification.py` — empirical rank probe.
- `docs/research/data/rung5_rank_verification.json` — committed
  rank artifact across k ∈ {2, 3, 4} and seeds {0, 42}.

## See also

- `docs/research/rung3-rank-bound.md` — the dimensional analysis
  that motivates the product-amp design and the saturation bound
  formula.
- `docs/research/rung4-viability-spike.md` — viability decision for
  the `k=2` predecessor; Rung4 stays default-off, Rung5 inherits
  that opt-in default.
- `openspec/changes/add-rung5-encoding/` — full change proposal,
  design doc, and capability specs.
