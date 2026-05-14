# Rung-viability methodology — design for higher-rung spikes

> Research-track design doc. Motivated by the structural identity
> finding in [`docs/research/rung4-viability-spike.md`](rung4-viability-spike.md):
> the existing single-pair-cancellation spike methodology can't
> discriminate higher-rung encodings, because all encodings that
> can saturate the `min_amp_overlap` constraint produce identical
> residuals at the boundary.
>
> This note proposes a replacement methodology grounded in the
> question the spike was actually trying to answer: **does the new
> rung buy real value over the previous one?**

## TL;DR

- The single-pair-cancellation spike (Rung3, Rung4) **answered a
  question we didn't need to ask** ("can the optimizer find the
  trivial amp-zeroing or pin the constraint?") and **failed to
  answer the one we did** ("does the extra capacity translate to
  downstream value?").
- New spike replaces the analytic single-pair probe with a
  **downstream-metric battery** across four axes. Each axis
  measures something a *real consumer* of the encoding cares
  about. The decision rule is honest: rung-(N+1) PASSes only if
  it beats rung-N on at least 2 of the 4 axes on real-SAE data.
- Required tooling already exists post-PRs #50/#51/#52/#55:
  `EpochCompressor(encoding=...)` is configurable, the Rung4
  cancellation primitive ships, `ClusteredDictionary` carries
  the encoding through `from_compression_panels`. So the new spike
  is a *measurement* exercise, not a *new code* exercise.

## What the old methodology measured (and why it failed)

The Rung3 / Rung4 viability spikes ran two probes per pair:

- **Unconstrained**: minimize gram_post over all knobs. Both rungs
  found the **trivial amp-zeroing**: `ψ_aux^B = π` flips branch-A
  to be orthogonal to branch-A^A, giving amp_overlap = 0,
  gram_post = 0. Strong-pass on criterion A, but the optimizer
  never engaged the higher-rung-specific dimensions.

- **Constrained** (`min_amp_overlap ≥ ε`): the optimizer pins
  `amp_overlap² = ε` exactly (lowering it is infeasible; raising
  it is suboptimal). Residual = `mps_post/mps_floor × amp_overlap²
  = 1 × ε = ε`. The result is **structurally identical across all
  encodings that can saturate the constraint**.

Both probes are blind to the actual difference between rungs.
The unconstrained probe rewards the trivial solution; the
constrained probe rewards hitting the boundary.

### What the rungs actually differ in

The structural difference between rungs **is the per-encoding
feature cap**:

| Encoding   | max_features | Amp branch structure                  |
|-----------|---------|---------------------------------------|
| MPSRung1   | 8       | none (3-qubit MPS only)               |
| Rung3      | 16      | 2-qubit Bell-pattern entangled        |
| Rung4      | 32      | 2-qubit × 2-qubit product             |

The cancellation primitive's job is single-pair gram minimization.
For that job, the MPS phase optimization does the work (`α, β, γ,
φ`). The amp branch is a multiplicative factor on the gram — once
the optimizer has enough freedom to span amp_overlap ∈ [0, 1],
adding *more* freedom doesn't reduce single-pair gram below what
the MPS branch already achieves.

**The rungs are fundamentally about capacity, not per-pair
cancellation power.** A new viability spike needs to ask
capacity-flavored questions.

## The new methodology: four-axis downstream battery

Replace single-pair-cancellation as the primary probe with **four
discriminating axes**, each measuring a *real consumer's*
benefit from the encoding-cap lift.

### Axis 1: Compression coverage at fixed quality budget

**Probe**: run `EpochCompressor(encoding=X)` on the same SAE
checkpoint with the same `quality_delta_multiplier` budget, for
X ∈ {MPSRung1, Rung-N, Rung-(N+1)}. Compare:

- `result.n_features_zeroed_total` — how many features actually
  got compressed
- `result.iterations[-1].cumulative_cross_entropy_delta` — the
  quality cost incurred
- `len(result.iterations)` — how many iterations were needed

**Why it discriminates**: PR #57 confirmed that the encoding-cap
lift cashes out in **block count** (fewer blocks at higher K) and
in **per-block panel size** (larger panels). EpochCompressor's
per-iteration cancellation runs on each block; larger blocks mean
more candidate pairs per cancellation pass, and the cancellation
primitive can find more redundant features per iteration.

**Decision threshold**:
- Rung-(N+1) beats rung-N if `n_features_zeroed_total` is higher
  at equal or better quality (`cumulative_cross_entropy_delta`
  within 10%) for the same fixture.

### Axis 2: Aggregate gram condition near capacity

**Probe**: build a single `Dictionary` (not Clustered) at exactly
the encoding's `max_features` using real-SAE features chosen for
high redundancy (top-K by cosine within a high-cosine cluster).
Compute the gram matrix and report:

- `λ_min(|gram|²)` — smallest eigenvalue of the squared-modulus
  gram. Lower = more nearly-singular = features less distinguishable.
- `||off-diagonal||_F / max_features` — average off-diagonal
  modulus. Higher = more "tangled" features.

**Why it discriminates**: the rank-verification work
([`rung4-rank-verification.json`](data/rung4_rank_verification.json))
confirmed Rung4 saturates at exactly 32 features (matching its
max_features). But *how cleanly* it saturates — i.e., the condition
number near the limit — is unknown. Higher rungs *should* have
better-conditioned gram matrices at K=max_features because the
state space is larger; whether the optimizer actually puts the
features in well-separated regions of that space is the empirical
question.

**Decision threshold**:
- Rung-(N+1) beats rung-N if `λ_min` is at least 2× larger at the
  encoding's max_features, on the same hand-picked high-redundancy
  panel.

### Axis 3: Multi-pair simultaneous orthogonalization

**Probe**: for a 4-feature subset, run the Rung-N cancellation
optimizer on the joint objective `Σᵢⱼ |gram[i,j]|²` (sum of
pairwise squared overlaps off-diagonal) rather than single-pair
gram. Report the achievable joint minimum.

**Why it discriminates**: single-pair cancellation is rung-
independent at the constraint boundary. But *simultaneous*
cancellation of multiple pairs requires more degrees of freedom —
each rung's extra knobs might let the optimizer satisfy more
constraints at once. Rung4's product-amp gives independent control
over two branch directions; Rung3's Bell-pattern entangles them.

**Decision threshold**:
- Rung-(N+1) beats rung-N if the joint minimum (under the same
  constraint) is at least 50% lower on a 4-feature redundancy
  cluster.

**Caveat**: this requires extending `Cancellation` to accept a
multi-pair objective. Not currently implemented; ~150 LOC of new
code per encoding. **Defer** if axes 1 + 2 are sufficient for the
verdict.

### Axis 4: Downstream sae-forge faithfulness

**Probe**: run `sae-forge`'s forge pipeline against the SAE
compressed with each rung. Measure `faithfulness_kl(host, forged)`
on a fixed eval set. Higher rung at the same compression ratio
should produce a forged transformer that better imitates the host.

**Why it discriminates**: this is the most honest downstream test.
sae-forge is the eventual consumer of polygram's compressed SAEs;
if higher rungs don't produce better-forgeable SAEs, the rung lift
has no end-to-end value.

**Caveat**: requires sae-forge tooling + the `[behavioural]` extra
(torch + transformers + host model). Heaviest axis.

**Decision threshold**:
- Rung-(N+1) beats rung-N if `faithfulness_kl` is at least 5%
  lower (better) on the same eval set and forge pipeline.

## Decision rule

Rung-(N+1) **PASSES** the viability test if it beats rung-N on:

- **≥ 2 of axes 1, 2, 3** when axis 4 is unavailable (no torch /
  no sae-forge environment), OR
- **≥ 2 of axes 1, 2, 4** when axis 3 is deferred (multi-pair
  optimizer not implemented), OR
- **≥ 3 of all 4 axes** when all are measurable.

Rung-(N+1) **FAILS** if it's worse than rung-N on the majority of
measurable axes.

Rung-(N+1) is **INCONCLUSIVE** otherwise — typically a mixed
result where the encoding-cap lift cashes out somewhere but not
where the next consumer needs it. Inconclusive results don't
prevent shipping the rung as opt-in (Rung3 and Rung4 are both
opt-in for exactly this reason), but they prevent the rung from
becoming default.

This is a stricter rule than the Rung3/Rung4 spike's "criterion A
+ B/C/D" structure because the underlying question is different:
"is this rung worth its complexity?" replaces "does the optimizer
break below the structural floor?".

## What this methodology DOESN'T test

- **Per-pair geometric power.** The Rung3/Rung4 spikes' criterion
  A is dropped. The constrained-residual structural identity
  shows it's uninformative. The unconstrained version rewards
  trivial-zeroing, which is also uninformative.
- **Q-OrCA round-trip behaviour.** Tested separately by the
  encoding's emit smoke test; not part of viability.
- **Computational cost.** A rung might pass on all 4 axes but be
  so slow that no consumer uses it. Cost analysis is a separate
  scope-the-encoding decision, not viability per se. (The
  `grid_outer=(3,3) vs (5,5)` finding from
  [`rung4-viability-spike.md`](rung4-viability-spike.md#caveats)
  is the prototype of this kind of analysis.)

## Implementation sketch

Most of what this methodology needs already exists. Per-axis lift:

- **Axis 1**: existing tooling. `EpochCompressor(encoding=X).run(...)`
  + standard reporting. ~50 LOC of glue per rung comparison.
- **Axis 2**: existing tooling. Build the dictionary; call
  `np.linalg.eigvalsh(np.abs(gram)**2)`. ~30 LOC.
- **Axis 3**: NEEDS new code. Extend `Cancellation` to accept a
  multi-pair objective, or write a one-off optimizer for the
  simultaneous case. ~150 LOC. Defer for v1 of the methodology.
- **Axis 4**: existing sae-forge tooling (`ForgePipeline`). Needs
  a torch-enabled host (no employer-M4 constraint applies on the
  Intel Mac per `[personal_vs_employer_separation]`). ~50 LOC of
  glue.

**v1 plan**: ship axes 1, 2, 4. Defer axis 3 until the v1 result
shows we need it (e.g., if axes 1 + 2 + 4 are split). Total
effort: ~150 LOC of measurement scripts + a results note.

## Predictions

Honest predictions before running the new spike (record so we
don't fool ourselves after the fact):

- **Rung4 vs Rung3 on Axis 1**: likely WIN. The 32-feature cap
  vs Rung3's 16 should let EpochCompressor find more
  redundancies per panel on real SAEs (per PR #57's "downstream
  consumers pay per-block cost" finding).
- **Rung4 vs Rung3 on Axis 2**: probably TIE or marginal win.
  Rung4 has more state space (32 vs 16 dim), so condition number
  near max_features could be better, but the difference may be
  smaller than the threshold. Don't have strong intuition.
- **Rung4 vs Rung3 on Axis 4**: likely TIE. faithfulness_kl
  reflects host-model imitation; the SAE's feature density is
  upstream of that. Hard to predict without running.

If Rung4 wins Axis 1 strongly, the viability story shifts from
"capacity but no cancellation power" (the current
[`rung4-viability-spike.md`](rung4-viability-spike.md) conclusion)
to "**capacity that cashes out in compression coverage**" — which
is a much stronger case for users opting in.

## What needs to land before running this

- [ ] Axis 1 measurement script: `examples/rung_compression_coverage.py`.
  Takes `--encoding mps|rung3|rung4`, runs full compression, reports
  the three coverage numbers. ~50 LOC.
- [ ] Axis 2 measurement script:
  `examples/rung_gram_condition.py`. Builds a max_features dictionary
  on the §4.4 panel (or a redundancy-rich subset), reports λ_min
  and off-diagonal mass. ~30 LOC.
- [ ] Axis 4 cross-repo run: `sae-forge`'s `ForgePipeline.run()`
  against a Rung-N compressed SAE produced by polygram. No new code
  needed; just a runbook. Document in
  `docs/research/cross-repo-forge-comparison.md`.
- [ ] Results note: `docs/research/rung4-viability-spike-v2.md`
  reporting all three axes on Rung3 vs Rung4. Replaces the v1
  "constrained spike" verdict.

## Relationship to existing spikes

`docs/research/rung3-viability-spike.md` and
`docs/research/rung4-viability-spike.md` are not retracted — they
remain as the record of the old methodology and what it found.
The structural-identity finding in the Rung4 note is the load-
bearing argument for this new methodology, and the methodology
explicitly references that finding as its motivation.

When axis-1/2/4 measurements land for an existing rung (Rung3 or
Rung4), they go into a v2 supplemental note rather than rewriting
the v1 spike doc.

## Open questions

- **Which SAE checkpoint is the canonical Axis-1 fixture?** The
  Rung3/Rung4 spikes used GPT-2-small block 10. For compression
  coverage, the fixture's redundancy *density* matters — a
  highly-redundant SAE will reward higher rungs more than a
  sparsely-redundant one. Pick: the same GPT-2-small block 10
  checkpoint for direct comparability with prior spikes.
- **Should Axis 1 measure at multiple quality budgets?** A single
  budget gives one data point; a curve (`coverage` vs
  `quality_delta`) gives a more nuanced picture. Defer:
  single-point first, curve if the single point is suggestive.
- **Should the methodology be parameterised over `--n-features`
  panel size?** Higher rungs might dominate at larger panels
  where their cap actually engages. The current 8-feature §4.4
  panel may not exercise the rung difference. Pick: run at panel
  sizes 8, 16, 32 to span the cap range; the rung that adapts
  best to its native cap wins.
