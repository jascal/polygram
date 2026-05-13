# Rung3 rank bound — how many linearly-independent features does Rung3 actually hold?

> Research-track note resolving the empirical open question raised
> while scoping the per-encoding feature-cap change ("Stage 1" of the
> 8-feature-cap staircase). Reproducible via
> `python examples/rung3_rank_probe.py --json-out docs/research/data/rung3_rank_probe.json`.
> Raw artifact: [`data/rung3_rank_probe.json`](data/rung3_rank_probe.json).

## TL;DR

The naive headroom claim "Rung3 = 5 qubits = 2⁵ = 32 features" is
**wrong**. Rung3 saturates at **rank 16**, not 32, because the amp
branch's parameterization is restricted to a 2-dim subspace of the
2-qubit Hilbert space.

The cap is sharp and algebraic, not soft / numerical:

| Encoding | Predicted max rank | Empirical rank (relative tol 1e-12) | Spectrum gap at the cap |
|---|---|---|---|
| `MPSRung1` | 8 | **8** (saturates at N=8) | σ[7] = 1.4e-1, σ[8] = 1.0e-16 (14 orders) |
| `Rung3` | 16 (not 32) | **16** (saturates at N=16) | σ[15] = 7.5e-2, σ[16] = 1.6e-16 (15 orders) |

**Implication for the per-encoding-cap change:** the staircase still
works, but Rung3's contribution to the headroom story is 8 → 16
(2×), not 8 → 32 (4×). Path to a real 32: either Rung4 with a
generalized amp branch that reaches the full 2-qubit subspace, or
extend MPSRung1 to 4 qubits (no q-orca change required — q-orca's
`compute_concept_gram_mps` already handles arbitrary `n` qubits at
χ=2; see "Path-to-32" below).

## The math

Each Rung3 feature is the tensor product

|state(α, β, γ, φ, θ, ψ)⟩ = |mps(α, β, γ, φ)⟩ ⊗ |amp(θ, ψ)⟩

where:

- `|mps⟩` is the 3-qubit cross-coupled rung-1 staircase output —
  lives in `C^8` (full 3-qubit Hilbert space). Empirically (and
  expectedly) the parameterization spans all 8 dimensions; see the
  MPSRung1 row above.
- `|amp⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩` — lives in the
  **2-dimensional** subspace `span{|00⟩, |11⟩}` of the 2-qubit
  Hilbert space `C^4`. The encoding **cannot reach** `|01⟩` or
  `|10⟩` for any (θ, ψ).

So the total reachable subspace is

`dim = dim(H_mps) · dim(H_amp) = 8 · 2 = 16`

not the 8 · 4 = 32 that the qubit count alone suggests. The amp
qubits 3–4 are present in the circuit description but the amp-branch
state vector only uses 2 of their 4 basis directions.

## Empirical results

Diverse random parameter sampling, two seeds for sanity:

```
=== MPSRung1 — seed=0 ===
  N    rank@1e-12   σ_max       σ_min>0     σ_min
  4         4       1.57e+00    5.40e-01    5.40e-01
  8         8       2.50e+00    2.06e-02    2.06e-02   ← saturates
 16         8       3.84e+00    5.23e-01    2.67e-17   ← additional features → numerical noise
 32         8       7.23e+00    1.52e+00    7.16e-18

=== Rung3 — seed=0 ===
  N    rank@1e-12   σ_max       σ_min>0     σ_min
  4         4       1.58e+00    5.23e-01    5.23e-01
  8         8       2.60e+00    1.10e-01    1.10e-01
 12        12       2.95e+00    2.56e-02    2.56e-02
 16        16       3.12e+00    4.34e-04    4.34e-04   ← saturates
 20        16       3.48e+00    3.72e-02    6.06e-17   ← additional features → numerical noise
 32        16       4.59e+00    3.43e-01    2.58e-17

=== Rung3 — seed=42 (sanity) ===
  N    rank@1e-12
 16        16   ← matches seed=0 saturation
 32        16
```

Both seeds saturate at exactly 16. No seed-dependence; no soft cap.

### Spectrum shape at saturation

Rung3 singular values at N=32, normalized to σ_max:

```
σ[ 0] = 1.000000
σ[ 1] = 0.793717
σ[ 2] = 0.763084
...
σ[14] = 0.111204
σ[15] = 0.074641
σ[16] = 1.6e-16   ← below numerical floor
σ[17] = 1.4e-16
...
σ[31] = 5.6e-18
```

Two things to read off:

1. **The 16 nonzero values span a 13× ratio** (1.0 down to 0.075).
   No early degeneracy — features ARE making efficient use of the
   16-dim space, not collapsing into a smaller effective subspace.
2. **The gap between σ[15] and σ[16] is 15 orders of magnitude.**
   This is an algebraic bound, not a numerical near-degeneracy.
   Above N=16, additional features are linear combinations of the
   first 16 — they add no new information.

The MPSRung1 spectrum at N=16 shows the same pattern at the lower
cap: 8 healthy singular values, then a 16-order gap.

## Implications for the per-encoding-feature-cap change

### Cap values per encoding

| Encoding | `max_features` | Source |
|---|---|---|
| `MPSRung1` | 8 | `dim(C^8) = 8` (confirmed empirically) |
| `Rung3` | **16** | `dim(C^8 ⊗ C^2) = 16` (confirmed empirically) |
| `HEA_Rung2` | `2 ** n_qubits` | already parameterized; cap follows the existing knob |

### Path-to-32 is not "add more rungs of the same shape"

A naive Rung4 that repeats the Rung3 amp-branch pattern (another
2-qubit amp branch parameterized as `cos(θ)|00⟩ + e^(iψ)sin(θ)|11⟩`)
would give `8 · 2 · 2 = 32`, but only if the two amp branches are
**independent** — which they are by construction in a tensor product.
So that path works.

Cheaper alternative: a single richer amp branch that spans the full
2-qubit Hilbert space (e.g., parameterizing all four amplitudes of
`α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩` modulo a global phase) gives
`8 · 4 = 32` from the same 5 qubits Rung3 already uses. This is
geometrically equivalent to fixing the existing amp branch rather
than adding a new one.

A third path: extend `MPSRung1` to 4 qubits (full Hilbert dim 16),
no amp branch. Gives 16 directly, with a cleaner Q-OrCA emit shape.

The polygram `MPSRung1` docstring previously suggested this would
force a q-orca change, but a closer read of
`q_orca/compiler/concept_gram_mps.py` shows the pinning is on
`bond_dim` (χ=2), not on `n_qubits`: `infer_qubit_count` resolves the
qubit count from the machine's `qubits = [q0, q1, …]` context list
generically, and `_parse_staircase_effect` accepts arbitrary-length
CNOT staircases. So a 4-qubit MPSRung1 staircase at χ=2 needs **no
q-orca change** — only polygram-side updates to `_state.py` (raise
`N_QUBITS`) and `_qorca_emit.py` (emit a 4-qubit staircase). The
cross-repo coordination claim in the original "MPSRung1 to 4 qubits
would change q-orca's safe-Rz matcher pinning" framing is incorrect.
Cross-repo work IS needed for a true χ>2 extension (multi-CNOT KAK
per rung, transfer-matrix contraction in
`compute_concept_gram_mps`), but that's a different proposal.

Recommended order if Stage 1 wants to push past 16:

1. **Stage 1 ships with `Rung3.max_features = 16`.** Honest with
   the math, no new encoding work.
2. **Follow-up (Stage 1b):** generalize the Rung3 amp branch to
   span the full 2-qubit subspace → `Rung3v2.max_features = 32`.
   No qubit-count change; Q-OrCA emit shape is unchanged at the
   register level (still 5 qubits) but the action signature needs
   two extra knobs on the amp branch.
3. **Deferred:** 4-qubit `MPSRung1`. Polygram-side only (raise
   `N_QUBITS` in `_state.py`, extend `_qorca_emit.py` to the longer
   staircase); q-orca's `compute_concept_gram_mps` already supports
   arbitrary `n` at χ=2. A genuinely cross-repo change is only
   required for χ>2 MPS (rung-2+ in the literal MPS sense).

Real-SAE scale (N ≫ 32) is a Stage 3 / clustered-dictionary
problem regardless of how this resolves — see the broader
8-feature-cap-staircase discussion.

## Reproduction

```
$ python examples/rung3_rank_probe.py
$ python examples/rung3_rank_probe.py --json-out docs/research/data/rung3_rank_probe.json
```

Default sweep: `N ∈ {4, 8, 12, 16, 20, 24, 28, 32}` at seeds 0 and
42 for Rung3, seed 0 for MPSRung1. Override via `--sizes` and
`--seeds`. The JSON dump captures every singular value at every
(encoding, seed, N) for downstream plotting or follow-up analysis.

## Caveats

- The probe samples uniformly over the parameter ranges declared in
  each encoding's docstring. A degenerate sampling distribution
  (e.g., all features sharing one knob) would produce a lower rank,
  but the question is about the encoding's **achievable** rank, not
  any particular sampling.
- The probe relies on `Dictionary.gram()`'s elementwise-product
  factorization for Rung3 (`dictionary.py:294-306`). If a future
  Rung3 variant changes that factorization, this bound needs to be
  re-checked.
- The "32" claim was a math error on my part during the staircase
  brainstorm, not a contradiction with anything previously shipped.
  No existing code asserts `Rung3.max_features = 32`; this note
  resolves the question before the per-encoding-cap change locks
  in a number.
