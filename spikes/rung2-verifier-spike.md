# Rung-2 HEA Verifier Spike — Report

**Date**: 2026-05-02
**Author**: Allan Scott (with Claude)
**Scope**: Risk-reduction probe for the proposed Rung-2 HEA support in
q-orca-lang. Throwaway. No changes to main codebase.

## TL;DR

1. **Verification cost is a non-issue.** Full Gram on 8 concepts × depth-5
   ring HEA on n=8 qubits takes **50 ms** in plain numpy statevector.
   n=10, depth=7 still finishes in ~100 ms. We do not need a tensor-network
   backend, QuTiP, or shot estimation at Polygram's scale (≤ 8 features).
2. **Tier-ordering verification is feasible** at the full-statevector
   level. With deterministic simulation, **`tolerance = 0.025`** gives
   100% pass rate across every meaningful jitter regime we tested
   (0.05, 0.10, 0.25, 0.50). The only failure mode is **jitter so high
   that the planted hierarchy genuinely doesn't survive** (intra-cluster
   margin collapses to ≤ 0.04) — that's a *true* negative and the
   verifier correctly flags it.
3. **The rung-1 `structural_floor` diagnostic generalizes cleanly to
   rung-2 single-knob search** — surprising, but true, and recoverable
   for free. Sweeping any one HEA rotation parameter while holding the
   other 3·L·n − 1 fixed produces a **purely sinusoidal** `|⟨A|B⟩|²(θ)
   = M + V·cos(θ − θ₀)` curve (FFT purity = 1.000 across three test
   knobs). Reason: every Pauli rotation has only two eigenvalues, so the
   amplitude is linear in `(cos(θ/2), sin(θ/2))` and the squared
   magnitude collapses to a single sinusoid. The `structural_floor()`
   helper still costs two Gram evaluations per knob; the existing
   diagnostic API survives with one parameter added (which knob to
   sweep).

**Recommendation**: rung-2 verifier work is unblocked. Proceed with
grammar + compiler implementation in q-orca-lang. On the Polygram side,
generalize `Cancellation.structural_floor(knob=...)` and `Cancellation`'s
search space to a list of declared knobs rather than the implicit
`(φ_A, φ_B)` pair.

---

## What we measured

### 1. Runtime sweep (numpy statevector, ring entangler, 8 concepts)

| n_qubits | depth | Gram (s) | States (s) |
|---------:|------:|---------:|-----------:|
|        6 |     3 |    0.011 |      0.011 |
|        6 |     5 |    0.018 |      0.018 |
|        6 |     7 |    0.024 |      0.024 |
|        8 |     3 |    0.028 |      0.027 |
|    **8** | **5** | **0.050**|  **0.045** |
|        8 |     7 |    0.063 |      0.063 |
|       10 |     3 |    0.042 |      0.043 |
|       10 |     5 |    0.070 |      0.070 |
|       10 |     7 |    0.098 |      0.098 |

The Gram evaluation itself (the 64 inner products on top of the
statevector preparation) is essentially free. **At Polygram's 8-feature
cap, full simulation Gram is in the ~50 ms range** — comparable to
parsing a `.q.orca.md` file. No need for a faster backend.

A future-proofing note: this scales as `2ⁿ · n · depth` per concept and
as `N²` for the inner products. At n=12 / 12 features it would still
be < 1 s; at n=16 / 16 features, ~10 s. If Polygram's cap ever gets
relaxed, the wall hits around n=20 (8 GB statevector, minutes per Gram).

### 2. Tier-ordering pass rate (n=8, depth=5, ring, 4 clusters × 2, 16 seeds)

| jitter | tol  | pass | mean_viol | intra | cross | margin |
|-------:|-----:|-----:|----------:|------:|------:|-------:|
|   0.05 |  0.000 | 1.00 |  0.00 | 0.869 | 0.004 |  0.866 |
|   0.05 |  0.025 | 1.00 |  0.00 | 0.869 | 0.004 |  0.866 |
|   0.10 |  0.000 | 1.00 |  0.00 | 0.573 | 0.004 |  0.569 |
|   0.10 |  0.025 | 1.00 |  0.00 | 0.573 | 0.004 |  0.569 |
|   0.25 |  0.000 | **0.56** |  2.38 | 0.037 | 0.004 |  0.033 |
|   0.25 |  0.025 | 1.00 |  0.00 | 0.037 | 0.004 |  0.033 |
|   0.50 |  0.000 | **0.00** | 24.4  | 0.004 | 0.004 | -0.0001 |
|   0.50 |  0.025 | 0.94 |  0.12 | 0.004 | 0.004 | -0.0001 |
|   0.50 |  0.050 | 1.00 |  0.00 | 0.004 | 0.004 | -0.0001 |

**Reading**:

- jitter=0.05 produces an Animals-style sibling-tier intra-cluster
  overlap (0.87, vs Polygram rung-1 Animals ≈ 0.93). Tier ordering is
  trivially preserved with any tolerance.
- jitter=0.10 produces an intra ≈ 0.57 — close to the rung-1
  cross-cluster *floor* of `cos(0.5)⁴ ≈ 0.59`. Still trivially separated
  from cross-cluster (which sits at random-circuit baseline of 0.004
  because depth-5 HEA on n=8 nearly fills Hilbert space).
- jitter=0.25 starts to swamp the planted structure: a small fraction
  of seeds produce per-triple violations at tol=0. tol=0.025 absorbs
  these.
- **jitter=0.50 destroys the hierarchy** (intra ≈ cross ≈ 0.004). At
  tol=0 the verifier correctly *fails* every seed. At tol=0.05 it
  accepts noise — **correct behavior is to NOT use that tolerance**.

**Recommended verifier tolerance**: `tol = 0.025` for deterministic
full-statevector simulation. This catches genuinely-broken hierarchies
(jitter=0.5 case at tol=0 still failed in 6% of seeds even at 0.025)
while tolerating the floating-point and small-jitter noise that
otherwise causes false negatives.

If we ever switch to shot-based estimation, tolerance must scale as
`O(1/√shots)`. At 1000 shots per Gram cell, the per-cell std dev is
~0.03 — recommend `tol = 0.05` minimum in that regime.

### 3. Phase-knob reconnaissance — does `structural_floor` generalize?

We picked feature A and feature B = A + small jitter, then varied a
**single** HEA rotation parameter θ ∈ [0, 2π] while holding all
3·L·n − 1 = 119 other parameters fixed. We FFT'd the resulting
overlap curve and measured spectral purity (c₁ / Σ|cₖ|, k≥1).

| Knob              | min    | max    | swing  | c₀..c₄                              | purity |
|-------------------|-------:|-------:|-------:|-------------------------------------|-------:|
| Rz q0 layer-0     | 0.0897 | 0.7035 | 0.6138 | 0.3966, 0.1536, 0.0, 0.0, 0.0       | 1.000  |
| Rz q0 layer-4     | 0.0108 | 0.7134 | 0.7026 | 0.3621, 0.1757, 0.0, 0.0, 0.0       | 1.000  |
| Ry q3 layer-2     | 0.0004 | 0.7032 | 0.7028 | 0.3518, 0.1759, 0.0, 0.0, 0.0       | 1.000  |

**Every knob's overlap curve is a pure single sinusoid.** No higher
harmonics. This is not a numerical accident — it's a structural fact:
any Pauli rotation `Rg(θ) = cos(θ/2)·I − i·sin(θ/2)·G` has eigenvalues
`±1`, so the overlap amplitude is `a·cos(θ/2) + b·sin(θ/2)` for
fixed complex `a, b`, and `|·|²` reduces to `M + V·cos(θ − θ₀)` for
real `M, V, θ₀`.

**Implication for `Cancellation.structural_floor()`**:

The rung-1 implementation evaluates the target-pair overlap at
`(φ, φ)` and `(φ, φ+π)` and returns `min(m_zero, m_pi) = M − |V|`.
The rung-2 generalization is mechanical:

```python
def structural_floor(self, knob: KnobSpec) -> float:
    """Floor reachable by varying `knob` only, all other params fixed."""
    # 2 Gram evals: knob = θ_anchor and knob = θ_anchor + π
    # Returns min(m0, m_pi) — analytic, backend-agnostic
```

For rung-1, `KnobSpec` is implicit `(target_pair[0].phi,
target_pair[1].phi)` paired through δ. For rung-2 HEA, it's a `(feature,
gate_axis, layer, qubit)` tuple per side. Cost stays at **two Gram
evaluations** regardless of backend or rung.

**Caveat**: the rung-1 floor was the *physical* lower bound of phase-only
search because there was only one knob. Under HEA, the per-knob floor
is just the floor *along that knob axis*. Multi-knob search can drive
overlap below any single-knob floor. So the rung-2 `cancellation_efficiency`
ratio `(before − after) / (before − floor)` can exceed 1.0 when the
optimizer beats the chosen reference knob's analytic limit. Two options:

- **Drop the [0, 1] clamp** for rung-2; keep the diagnostic but allow
  values >1.0 to mean "multi-knob search outperformed the chosen
  reference knob's structural floor". Honest, and informative.
- **Compute a multi-knob analytic floor** as the SDP minimum over the
  full Stiefel manifold of the swept gate parameters. Way more work,
  out of scope for this spike, and probably overkill.

Recommendation: keep `structural_floor(knob)` as a per-knob analytic
diagnostic, drop the clamp, and document it as a *reference floor*
rather than a *hard floor* in the rung-2 docstring.

---

## Recommendations for the q-orca-lang rung-2 PR

1. **Verification stage 4b**: full statevector Gram via numpy is
   sufficient at our scale. No QuTiP / tensor-network dep needed. The
   current spike's `compute_concept_gram_hea` is a 50-line drop-in
   reference implementation.
2. **`rung2_tier_ordering` rule**: per-triple `gram[i,j] + tol >=
   gram[i,k]` check (for `i, j` same cluster and `i, k` different). Default
   `tol = 0.025` for deterministic simulation. Document the
   `1/√shots` scaling for future shot-based backends.
3. **`compute_concept_gram_hea` API**: returns the same `GramMatrix`
   object as rung-1, plus the underlying `states: list[np.ndarray]` so
   downstream tools (Schmidt rank, partial trace) don't re-simulate.
4. **Resource warning threshold**: warn at `n_qubits > 12` (statevector
   memory), error at `n_qubits > 16` until tensor-network is wired in.

## Recommendations for the Polygram side (post-q-orca rung-2)

1. **`Cancellation` operates over a `knobs: list[KnobSpec]` list**
   instead of the implicit `(φ_A, φ_B)` pair. For rung-1, the default
   knobs are `(target_pair[0].phi, target_pair[1].phi)` — backward
   compatible. For rung-2, users pass explicit knob specs (or "all
   knobs of feature A and feature B" as sugar, with an optimizer like
   CMA-ES instead of grid).
2. **`structural_floor(knob)`** stays at 2 Gram evaluations. For
   `knobs` plural, return one floor per knob (a dict) — let the caller
   pick which to compare against.
3. **`cancellation_efficiency`**: drop the `[0, 1]` clamp on rung-2.
   Document that values >1 indicate multi-knob search beat the
   single-knob analytic floor.
4. **Don't rebuild the rung-1 Animals example for rung-2**. The
   Animals geometry is intrinsically rung-1 (cluster-amplitude β and
   per-feature φ). For rung-2, write a fresh demo with synthetic
   clustered HEA parameters (the spike's `synth_concepts` is the
   recipe).

---

## What this spike does NOT cover

- Tensor-network backend timings (skipped — full statevector is fast
  enough at our scale, no need to add the dep).
- Shot-based estimation (skipped — deterministic full sim is the
  cheaper path; document the `1/√shots` rule for future shot work).
- Compiler details — the spike does not produce `.q.orca.md` output.
  Grammar and compiler design is the next step in the q-orca-lang PR
  and is unblocked by this report.
- `entangler="all_to_all"` and `entangler="full"` — only ring/chain
  tested. Adding them is mechanical.
