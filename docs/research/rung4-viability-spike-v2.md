# Rung4 viability — v2 results (axes 2 partial, 1 / 4 pending)

> Research-track note. v2 supplemental to
> [`docs/research/rung4-viability-spike.md`](rung4-viability-spike.md)
> under the new methodology proposed in
> [`docs/research/rung-viability-methodology.md`](rung-viability-methodology.md).
>
> **Status**: partial. Axis 2 (gram condition) ran locally and
> produced an unexpected finding that invalidates the axis's
> discriminating power as designed. Axis 1 (compression coverage)
> and Axis 4 (sae-forge faithfulness) require a torch-enabled
> host and are deferred.

## TL;DR

- **Axis 2 doesn't discriminate encodings as designed.** At default
  knobs (the `from_sae_lens` initialization that real consumers
  use), Rung3 and Rung4 reduce to MPSRung1-equivalent gram on the
  same `(α, β, γ, φ)` per the encodings' own design — confirmed
  here at K=8 with bit-identical metrics across encodings.
- The K=max_features measurements differed across encodings only
  because they were implicitly measuring at *different K's*.
  Comparing rungs requires holding K fixed and varying the
  encoding, OR varying knob-assignment to exploit the larger
  state space — `from_sae_lens` doesn't do the latter.
- **All three encodings produce rank-deficient gram (λ_min ≈ 0)
  on the top-redundancy K=cap subset of the real GPT-2-small SAE.**
  This isn't a new failing of higher rungs — it's an existing
  property of how `from_sae_lens` assigns knobs from decoder
  geometry on highly-redundant features.
- **Methodology revision needed** before Axis 2 can yield a
  rung-discriminating verdict. Two paths: knob optimization
  (expensive) or encoding-specific knob assignment in
  `from_sae_lens` (separate change).

| Axis | Status | Verdict |
|---|---|---|
| 1 — Compression coverage | Script ready; needs torch + SAE | Pending |
| 2 — Gram condition | Ran on real SAE | **Methodology invalid as designed** |
| 3 — Multi-pair simultaneous | Deferred to v3 | n/a |
| 4 — sae-forge faithfulness | Cross-repo; needs torch | Pending |

## Axis 2 in detail

### What we measured

For each encoding `E ∈ {MPSRung1, Rung3, Rung4}`:

1. Selected the top-K most-redundant features in the real SAE
   (`scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors`,
   24576 features × 768 d_model), where `K = E.max_features` (8 / 16 / 32).
   Selection: greedy density expansion from the highest-cosine pair.
2. Built a `Dictionary` on those K features with encoding `E`.
3. Computed `|gram|²` and reported λ_min, λ_max, condition number,
   off-diagonal Frobenius mass.

### Results

#### At each encoding's K=max_features

| Encoding | K | mean off-diag | Frobenius off-diag / k | λ_min | λ_max | Cond # |
|---|---|---|---|---|---|---|
| MPSRung1 | 8  | 0.7725 | 2.115 | ~0 (-4e-17) | 6.45  | inf |
| Rung3    | 16 | 0.7835 | 3.137 | ~0 (-5e-16) | 12.78 | inf |
| Rung4    | 32 | 0.8236 | 4.718 | ~0 (-4e-15) | 26.81 | inf |

All three are rank-deficient. The Frobenius mass scales with K
because there are more off-diagonal entries; the *mean* off-diag
modestly increases with K (more features in the cluster = some
are less similar to each other = mean is dragged up by the
less-similar tail).

This **looks** like a meaningful comparison, but it's not — the
encodings are at different K's.

#### At fixed K=8 (the real comparison)

| Encoding | K | mean off-diag | Frobenius off-diag / k | λ_min | λ_max | Cond # |
|---|---|---|---|---|---|---|
| MPSRung1 | 8 | 0.7725 | 2.115 | -4.1e-17 | 6.447 | inf |
| Rung3    | 8 | 0.7725 | 2.115 | -4.1e-17 | 6.447 | inf |
| Rung4    | 8 | 0.7725 | 2.115 | -4.1e-17 | 6.447 | inf |

**Bit-identical.** This is not a measurement artifact — it's the
intended behavior of the encodings. From `polygram.encoding.Rung4`:

> Default-knob Rung4 reduces to MPSRung1-equivalent gram on the same
> (α, β, γ, φ).

At default amp-branch knobs (`theta_amp=π/4`, `psi_aux=0`,
`theta_amp_b=π/4`, `psi_amp_b=0`), the amp factors are 1, and the
gram becomes the MPS-only gram. `from_sae_lens` doesn't currently
assign non-default amp-branch knobs from decoder geometry — so
real-SAE-derived dictionaries with Rung3/Rung4 *always* land at
amp = 1, gram-identical to MPSRung1.

### What this means for the methodology

The hypothesis ("higher rungs have larger state spaces → better
gram conditioning at K=max_features") is **untestable** with the
current loader. The encoding's larger state space exists in theory
but is never populated for real consumers.

Two ways to fix Axis 2 so it actually discriminates:

1. **Optimize the amp knobs** to find the *best* gram each
   encoding can achieve at fixed K and a fixed decoder-geometry
   base. Compute `argmin over (θ_amp_*, ψ_aux_*, θ_amp_b_*,
   ψ_amp_b_*) ||off-diag(|gram|²)||_F`. This tests whether the
   encoding's extra parameter space *can* help reduce off-diagonal
   mass. Expensive (per-feature 2-4 extra knobs × K features × an
   optimizer pass).
2. **Add encoding-specific knob assignment to `from_sae_lens`**.
   E.g., use a second PCA axis to assign `θ_amp` like the current
   loader uses the first axis for `β`. This is its own change
   and would be a real product improvement, not just a
   measurement fix.

Neither is in scope for this v2 results note. The v2 status of
Axis 2 is: **methodology invalid; revision proposed in Open
Questions of the methodology doc**.

## Axes 1 + 4 — pending torch-enabled host

The measurement scripts ship:

- `examples/rung_compression_coverage.py` — runs `EpochCompressor`
  on the configured encoding against a real SAE, captures the
  per-iteration coverage trajectory + final cross-entropy delta.
  Graceful skip when torch / SAE is missing.
- (Axis 4 — sae-forge `ForgePipeline` against a polygram-compressed
  SAE — needs cross-repo orchestration; not implemented yet.)

The local Intel Mac doesn't have torch in the polygram .venv (per
[`personal_vs_employer_separation`](../../../.claude/projects/-Users-allans-code/memory/personal_vs_employer_separation.md);
the employer M4 with torch installed is off-limits for this work).
A torch-enabled run on rented GPU or a personal NVIDIA box would
land the Axis 1 + Axis 4 numbers.

## Decision

**Inconclusive.** With only Axis 2 measurable and Axis 2's
methodology invalid, we don't have evidence to revise the v1
"Rung4 stays opt-in" verdict.

The structural-identity argument from the v1 spike still holds —
the cancellation primitive is rung-blind at the constraint
boundary. The new question (does the capacity lift cash out
downstream?) remains open pending Axis 1 / Axis 4 measurements.

## What would change the verdict

- **Axis 1 strong win for Rung4**: rung-default candidate. If a
  Rung4 EpochCompressor run zeros materially more features at
  the same quality budget than the MPSRung1 baseline on the same
  SAE, that's the load-bearing evidence the rung lift is
  consumer-visible.
- **Axis 1 wash**: confirms the v1 opt-in verdict. The capacity
  lift exists structurally but isn't worth the optimizer cost.
- **Axis 4 win for Rung4**: end-to-end downstream confirmation;
  even stronger evidence for default.

## What we learned from running this exercise

Two findings worth recording independent of the verdict:

1. **`from_sae_lens` doesn't exploit non-MPS rungs at default
   knobs.** Higher rungs ship with rank capacity but no knob-
   assignment story for the extra dimensions. A natural follow-up
   is an issue: "Add encoding-specific knob assignment to
   `from_sae_lens`" — without it, real consumers never see
   Rung3/Rung4 in non-default-equivalent state, which means a lot
   of polygram's encoding work is currently dormant in the loader
   path.
2. **The v1 viability methodology's structural identity at the
   constraint boundary generalises**: any methodology that asks
   "does this encoding produce a different gram at default knobs?"
   gets a NO for free, because the encodings are designed to
   reduce to MPSRung1 at defaults. Discriminating axes need to
   either probe non-default knobs (via optimization or
   structured assignment) or measure downstream consumer behavior
   (Axes 1, 4).

## Files

- `examples/rung_compression_coverage.py` — Axis 1 script (script
  ready; awaits torch-enabled run)
- `examples/rung_gram_condition.py` — Axis 2 script
- `docs/research/data/rung_gram_condition_{mps,rung3,rung4}.json`
  — Axis 2 raw outputs (K=max_features each, **amp-knob defaults**)
- `docs/research/data/rung_gram_condition_{rung3,rung4}_k8.json`
  — Axis 2 same-K control (proves bit-identity)
- `docs/research/data/rung_gram_condition_{rung3,rung4}_amp_on.json`
  — Axis 2 raw outputs **with PCA-axis amp-knob assignment** (post-
  `encoding-aware-knob-assignment` change). See Resolved section.
- `tests/test_examples.py::test_rung_compression_coverage_smoke`
- `tests/test_examples.py::test_rung_gram_condition_smoke`
- `tests/test_amp_knob_assignment.py` — the falsifying-invariant
  tests that pin the amp-knob assignment as actually doing work.

## Resolved (encoding-aware-knob-assignment)

The "Axis 2 methodology is invalid as designed" finding is partially
addressed by [`encoding-aware-knob-assignment`](../../openspec/changes/encoding-aware-knob-assignment/proposal.md)
(P0 from the post-#61 strategic review). That change adds
`assign_amp_knobs: bool = False` to `from_sae_lens`. When set to
`True`, the loader populates higher-rung amp-branch knobs from
decoder PCA — un-dormanting the encodings' larger state spaces.

Re-running Axis 2 at K=max_features with `--assign-amp-knobs` on
the same real GPT-2-small SAE fixture:

| Metric | MPSRung1 K=8 (unchanged) | Rung3 K=16 amp-off | Rung3 K=16 **amp-on** | Rung4 K=32 amp-off | Rung4 K=32 **amp-on** |
|---|---|---|---|---|---|
| mean off-diag | 0.7725 | 0.7835 | **0.4787** | 0.8236 | **0.3206** |
| Frobenius off-diag / k | 2.115 | 3.137 | **2.397** | 4.718 | **2.345** |
| λ_min | ~0 | ~0 | ~0 | -3.7e-15 | **+2.9e-16** |
| λ_max | 6.45 | 12.78 | 9.37 | 26.81 | **12.30** |
| Condition # | inf | inf | inf | inf | **4.2e+16** |

**Headline finding**: Rung4 amp-on drops mean off-diagonal by 61%
(0.82 → 0.32) and flips λ_min from negative-FP-noise (rank-deficient)
to positive-near-zero (numerically rank-full). The encoding's
state space is no longer collapsing to MPSRung1; the gram is
materially different from the K=8 baseline.

MPS amp-on is bit-identical to MPS amp-off (no amp branch — the flag
is a no-op for MPSRung1).

**Decision update**: with the un-dormant path available, Axis 1
(compression coverage) and Axis 4 (sae-forge faithfulness) become
meaningful experiments for the first time — they'd actually be
testing higher-rung capacity rather than MPS-equivalent aliases.
Verdict on Rung4 viability is still **inconclusive pending
Axis 1 / 4 on a torch-enabled host**, but the load-bearing
methodological blocker is resolved.
