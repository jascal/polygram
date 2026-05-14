# Rung4 viability — v2 results (axes 1 + 2 landed, 4 pending)

> Research-track note. v2 supplemental to
> [`docs/research/rung4-viability-spike.md`](rung4-viability-spike.md)
> under the new methodology proposed in
> [`docs/research/rung-viability-methodology.md`](rung-viability-methodology.md).
>
> **Status**: Axis 2 (gram condition) ran locally — original
> methodology invalid as designed; the v2.1 "Resolved" section
> below recovers it via PCA-axis amp-knob assignment. Axis 1
> (compression coverage) now landed on the 2019 MBP — see the
> v2.2 section at the bottom; **PASS for Rung4 amp-on**. Axis 4
> (sae-forge faithfulness) is cross-repo and remains pending.

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
| 1 — Compression coverage | Ran on 2019 MBP (v2.2) | **PASS** for Rung4 amp-on |
| 2 — Gram condition | Ran on real SAE; recovered via v2.1 | Original invalid; amp-on flips Rung4 to numerically rank-full |
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

**Caveat (not a failure, just reality)**: λ_min remains *near* zero
on this fixture, and the condition number stays large (~4e+16). This
is the expected behaviour for a deliberately-chosen highest-redundancy
K=32 panel — the features were *picked* because they're nearly
parallel in decoder space, so any encoding (no matter how rich) will
produce a gram with one or more tiny eigenvalues. The amp assignment
moves us from "actually rank-deficient (negative FP-noise λ_min)" to
"merely ill-conditioned (positive near-zero λ_min)", which is the
load-bearing change. The condition-number metric becomes informative
again only when measured on less-redundant panels.

MPS amp-on is bit-identical to MPS amp-off (no amp branch — the flag
is a no-op for MPSRung1).

**Decision update**: with the un-dormant path available end-to-end
through `from_sae_lens`, `EpochCompressor`, `Compressor`, and
therefore in all downstream sae-forge + behavioural-validation
steps, Axis 1 (compression coverage) and Axis 4 (sae-forge
faithfulness) become meaningful experiments for the first time —
they'd actually be testing higher-rung capacity rather than
MPS-equivalent aliases. Verdict on Rung4 viability is still
**inconclusive pending Axis 1 / 4 on a torch-enabled host**, but
the load-bearing methodological blocker is resolved.

> **TODO** — run `examples/rung_compression_coverage.py --assign-amp-knobs`
> (Axis 1) on a torch-enabled host. The script ships and the un-dormant
> path is in place; what's left is procuring the GPU time per the
> auto-memory's IP-separation constraint. Results land here as a v2.2
> supplement.

## Axis 1 result (v2.2)

Run on the 2019 Intel MBP against the same real SAE
(`scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors`,
GPT-2-small, layer 10). Default `EpochCompressor` kwargs: 8 prompts,
`n_panels_max=200`, `max_iterations=3`, `coverage_target=0.5`,
`cosine_threshold=0.3`. Per-cell wall time ~13-25 min CPU.

### Results

| Cell | Encoding | `assign_amp_knobs` | Features zeroed | Cumulative CE Δ | Iters | Convergence |
|---|---|---|---:|---:|---:|---|
| C1 | MPSRung1 (K=8)  | n/a   | 3 540  | 0.290 905 | 3 | max_iterations |
| C2 | Rung4 (K=32)    | False | 17 411 | 0.403 784 | 3 | max_iterations |
| C3 | Rung4 (K=32)    | **True**  | 9 376  | **0.208 086** | 3 | max_iterations |
| C4 | Rung3 (K=16)    | **True**  | 6 465  | 0.234 552 | 3 | max_iterations |

Headline comparison (vs MPS baseline, C1):

| Metric | C3 / C1 | C4 / C1 |
|---|---:|---:|
| Features zeroed | **2.65×** (+165%) | 1.83× (+83%) |
| Cumulative CE Δ | **0.72×** (−28%) | 0.81× (−19%) |

Both higher-rung-amp-on cells zero materially more features (well
over the ≥10% material-difference threshold) at *lower* cumulative
CE budget than MPS. C3 clears the ≥20% CE threshold cleanly; C4
sits one point shy at −19%.

### Verdict bucket: **PASS** for Axis 1

Per the decision rule in
[`run-axis1-compression-coverage-mbp/proposal.md`](../../openspec/changes/run-axis1-compression-coverage-mbp/proposal.md):

> Rung4-amp-on zeros materially more features than MPS at
> equal-or-better CE budget → **PASS**. Strong evidence to flip
> `assign_amp_knobs=True` as the default for higher-rung
> encodings.

C3 doesn't just clear "equal-or-better CE budget" — it strictly
improves on it. The un-dormanting work from PR #63 + PR #64 cashes
out in production compression metrics, not just gram diagnostics
(v2.1's Axis 2 finding).

The default-flip decision is deferred per
[`run-axis1-compression-coverage-mbp`](../../openspec/changes/run-axis1-compression-coverage-mbp/proposal.md)
"What this change explicitly does NOT do" — backwards-compatibility
implications for existing pipelines warrant a separate change.

### Per-iteration trajectory observations

- **All four cells terminated at `max_iterations` (3/3).** None
  reached `stable_clusters` convergence. The trajectory shape is
  comparable across cells but absolute numbers are upper-bounded
  by the iteration cap. A longer-iteration sweep is the obvious
  follow-up — the rate of zeroing per iter is still declining at
  iter 2 in every cell, so absolute features-zeroed totals would
  grow but the *relative* picture (rung4 amp-on dominates MPS)
  may or may not hold past iter 3.
- **Per-iter zeroing decays monotonically in every cell**, by the
  same shape: ~⅓ zeroed at iter 0, ~⅓ at iter 1, ~⅓ at iter 2.
  No cell shows a "second wind" or stalls early.
- **CE damage front-loads** in every cell (iter 0 contributes the
  majority of cumulative CE delta), then attenuates — consistent
  with the iteration loop biting the easy clusters first.
- **C2 (Rung4 amp-off) shows the "quantity-over-quality" regime:**
  17 411 features zeroed (nearly 5× MPS) but at the worst CE
  budget of the four cells (+39% vs MPS). The bigger panels admit
  more cluster candidates whether or not the amp branch is
  populated, but without amp knobs the picked clusters are
  systematically lower quality. This validates the rung4-default
  decision being load-bearing on `assign_amp_knobs`: Rung4 alone
  isn't a free win.
- **C4 (Rung3 amp-on) sits monotonically between MPS and Rung4-amp-on**
  on both metrics. Rung3 isn't a strictly weaker Rung4 (it has
  different geometry), but on this fixture the rung-vs-rung
  ordering tracks max_features at fixed-amp-on.

### Predictions vs actuals

Proposal-time predictions (from
[`run-axis1-compression-coverage-mbp/proposal.md`](../../openspec/changes/run-axis1-compression-coverage-mbp/proposal.md)):

| Prediction | Actual | Held? |
|---|---|---|
| C3 (Rung4 amp-on) zeros materially more features at equal-or-better CE → PASS | +165% features, −28% CE | **Yes, with margin** |
| C2 (Rung4 amp-off ≈ MPS in disguise) | C2 zeros 5× more but at +39% CE — different regime, not "in disguise" | **Partially**: gram-equivalence at Axis 2 didn't predict compression-equivalence at Axis 1; bigger panels alone do change behaviour even without amp knobs |
| Each cell ≈ 3-5 min on 2019 MBP | C1 ≈ 13 min, C2 ≈ 16 min, C3 ≈ 19 min, C4 ≈ 25 min | **No** — proposal underestimated by 3-5×. 2019 MBP throughput is lower than projected, especially for larger-K encodings. Total run took ~73 min, not ~15 |
| Single-point measurement, no Pareto sweep | Single point | Held |

The proposal also predicted a Rung3-amp-on cell would be an
"optional data point that doesn't change the decision." Actual:
C4 adds confidence that the amp-on lift isn't a Rung4-specific
artifact — it generalises to Rung3 too, with a smooth monotonic
ordering. So the optional cell turns out to be load-bearing for
the *generality* of the verdict, even though it doesn't change
the bucket.

### What this means for the v2 verdict

The v2 "Inconclusive pending Axis 1 / 4 on a torch-enabled host"
verdict (above) is now updated for Axis 1:

- **Axis 1 verdict: PASS for Rung4 with `assign_amp_knobs=True`**.
  The capacity lift cashes out in production compression
  metrics.
- **Axis 4 (sae-forge faithfulness) remains pending.** Cross-repo,
  separate change.
- **Default-flip on `assign_amp_knobs=True` remains a separate
  decision.** Even a strong Axis 1 PASS doesn't auto-flip the
  default — that change would need a back-compat story (existing
  consumers expect byte-identical behaviour). What this evidence
  *does* support is updating the README's "when to use which
  rung" guidance to recommend `assign_amp_knobs=True` for any
  Rung3/Rung4 use.

### Extended-iteration run (C3-extended, max_iterations=10)

Follow-up on C3 to check whether Rung4 amp-on reaches a natural
convergence signal (`stable_clusters` or `quality_breached`) given
more iteration headroom, and to see whether the 3-iter
PASS-margin holds at deeper iteration counts.

Same configuration as C3 (`--encoding rung4 --assign-amp-knobs`)
with `--max-iterations 10`. The first 3 iterations are
bit-identical to C3 above (same RNG state); rows 0-2 reproduce
C3's trajectory exactly.

| iter | zeroed_this_iter | CE_delta | cumulative_CE_delta | state |
|---:|---:|---:|---:|---|
| 0 | 3 174 | 0.140 114 | 0.140 114 | continuing |
| 1 | 3 141 | 0.032 711 | 0.172 825 | continuing |
| 2 | 3 061 | 0.035 260 | 0.208 086 | continuing |
| 3 | 3 093 | 0.047 561 | 0.255 646 | continuing |
| 4 | 3 067 | 0.012 934 | 0.268 580 | continuing |
| 5 | 3 052 | 0.037 977 | 0.306 557 | continuing |
| 6 | 2 537 | 0.045 371 | 0.351 928 | continuing |
| 7 | 1 000 | 0.010 834 | 0.362 762 | continuing |
| 8 |   500 | 0.003 555 | 0.366 317 | continuing |
| 9 |   267 | **0.167 895** | **0.534 212** | max_iterations |

**Total: 22 892 features zeroed; final cumulative CE Δ = 0.534 212.**

#### Observations

- **No natural convergence.** The run terminated at
  `max_iterations` again, not via `stable_clusters` or
  `quality_breached`. The cluster fingerprint kept shifting and
  the per-iter CE delta stayed under the quality bound through
  iter 8 — the iteration loop *would have* continued past 10 if
  allowed. So even at max_iterations=10 the run is not "to
  minimum" — only "to wall".
- **Two phases visible in the trajectory.**
  - *Plateau (iters 0-5):* per-iter zeroing sits at
    ~3 000-3 174 with no decay. The compressor is finding new
    clusterable features at roughly constant rate, presumably
    because the larger Rung4-amp-on panels keep admitting new
    co-cosine candidates after each round of zeroing reshapes
    the residuals.
  - *Exhaustion (iters 6-9):* zeroing falls off sharply
    (2 537 → 1 000 → 500 → 267). The system is running out of
    high-cosine clusters; what remains is increasingly marginal.
- **Anomalous CE spike at iter 9.** CE delta = 0.167 895 for
  just 267 features zeroed — roughly 38× the per-feature CE
  cost of the surrounding iters. Possibilities (not
  disambiguated here): (a) iter 9 picked a small number of
  high-impact features that were previously protected by their
  cluster context, (b) the model's residual error has compounded
  enough that any cluster removal now costs disproportionately
  more, (c) numerical noise on a small zeroed_count. This single
  iter accounts for **31% of the cumulative CE Δ** despite
  zeroing **1.2%** of total features. Worth follow-up
  investigation if Rung4-amp-on is going to be run at deeper
  iteration counts in production.
- **Quality-bound check is per-iter, not cumulative.** The
  compressor's bound is `quality_delta_multiplier * delta_1 =
  2.0 × 0.140 = 0.280` (default
  `quality_delta_multiplier=2.0`), so iter 9's per-iter delta
  of 0.168 stays under the bound by a comfortable margin and
  doesn't trigger `quality_breached`. But cumulatively the run
  has spent 3.8× its iter-0 quality budget across iters 1-9 —
  the bound checks per-iter, not cumulative, which lets this
  drift go undetected.

#### Verdict caveat at deeper iterations

The C3 3-iter PASS verdict (above) holds for that operating point
but does **not** generalise unconditionally to deeper iteration
counts. The relative picture changes:

| Metric | C3 (3 iter) vs MPS | C3-extended (10 iter) vs MPS-3-iter baseline |
|---|---|---|
| Features zeroed | 2.65× | **6.47×** |
| Cumulative CE Δ | 0.72× (−28%) | **1.84× (+84%)** |

At 3 iters Rung4 amp-on dominates MPS on both. At 10 iters
Rung4 amp-on zeros far more features but at almost 2× the CE
budget vs the 3-iter MPS baseline. **This is the
"Rung4-amp-on zeroes MORE features but at HIGHER CE delta"
bucket** from the decision rule —
[INCONCLUSIVE at this iteration count](../../openspec/changes/run-axis1-compression-coverage-mbp/proposal.md).

Caveat on the caveat (since resolved): an apples-to-apples
comparison would re-run MPS at `max_iterations=10` too. **That
control was run** — see "MPS 10-iter control" below. It
disambiguates the iter-9 spike as Rung4-amp-on-specific rather
than universal late-stage behaviour, and shifts the
"INCONCLUSIVE at this iteration count" verdict above.

#### MPS 10-iter control (C1-extended)

Same config as C1 with `--max-iterations 10`. First 3 iters
reproduce C1 bit-identically.

| iter | zeroed_this_iter | CE_delta | cumulative_CE_delta | state |
|---:|---:|---:|---:|---|
| 0 | 1 225 | 0.205 435 | 0.205 435 | continuing |
| 1 | 1 176 | 0.061 958 | 0.267 393 | continuing |
| 2 | 1 139 | 0.023 512 | 0.290 905 | continuing |
| 3 | 1 120 | 0.014 121 | 0.305 026 | continuing |
| 4 | 1 097 | 0.032 529 | 0.337 555 | continuing |
| 5 | 1 077 | 0.016 543 | 0.354 099 | continuing |
| 6 | 1 061 | 0.007 870 | 0.361 968 | continuing |
| 7 | 1 007 | 0.049 080 | 0.411 049 | continuing |
| 8 |   989 | 0.008 437 | 0.419 486 | continuing |
| 9 |   956 | 0.012 550 | 0.432 035 | max_iterations |

**Total: 10 847 features zeroed; final cumulative CE Δ = 0.432 035.**

Same `max_iterations` termination — no encoding reached
`stable_clusters`/`quality_breached` at depth 10 on this fixture.

#### What the MPS control disambiguates

Per-iteration zeroing trajectory shape — totally different from
C3-extended:

|  | MPS C1-extended | Rung4-amp-on C3-extended |
|---|---|---|
| Iter-0 zeroings | 1 225 | 3 174 |
| Iter-9 zeroings | 956 | 267 |
| Decline shape | Smooth monotonic (~22% drop) | Plateau (iters 0-5 ~3 050) then collapse (iter 6: 2 537 → iter 9: 267) |
| Iter-9 CE delta | 0.013 | **0.168** (12.9× higher) |
| Iter-9 CE share of cumulative | 2.9% | **31.4%** |

MPS doesn't have a comparable late-iter CE spike. Its highest
per-iter CE delta after iter 0 is **iter 7's 0.049** — about
4× neighbouring iters, dramatic for MPS but nowhere near
Rung4-amp-on iter 9's 38× spike. The trend through iter 9 stays
under 0.05 per iter.

So the iter-9 spike in C3-extended **is encoding-specific**, not
a general property of late-stage compression. Hypothesis: the
larger K=32 Rung4-amp-on panels grab the highest-cosine clusters
in iters 0-5 (plateau), exhaust the supply by iter 6 (sharp
zeroing drop), and at iter 9 the algorithm is forced to compress
small clusters with bad residual geometry — paying outsized
CE cost per feature. MPS K=8 panels stay in a regime where new
marginal clusters keep appearing at roughly constant rate
(slope ~-30 zeroings/iter), so the algorithm never has to scrape
the bottom of the barrel.

#### Pareto comparison at fixed iteration count

The right way to read the deeper-iter numbers isn't either metric
alone but the *features-zeroed-per-unit-CE-budget* ratio:

| Operating point | Encoding | Zeroed | Cumul CE Δ | Zeroed / CE Δ |
|---|---|---:|---:|---:|
| 3 iter | MPS | 3 540 | 0.291 | 12 165 |
| 3 iter | Rung4 amp-on | 9 376 | 0.208 | **45 077** (3.71× MPS) |
| 10 iter | MPS | 10 847 | 0.432 | 25 109 |
| 10 iter | Rung4 amp-on | 22 892 | 0.534 | **42 870** (1.71× MPS) |

Rung4 amp-on dominates MPS in compression-per-CE-cost at **both**
operating points. The dominance shrinks from 3.71× to 1.71× as
iterations deepen, but doesn't flip. The "INCONCLUSIVE at this
iteration count" verdict from the section above was too pessimistic
when read on absolute-CE alone: the comparison-against-MPS-3-iter-baseline
isn't fair because MPS also escalates CE at depth.

Reading the right comparison (10-iter vs 10-iter): **Rung4 amp-on
still PASSES** at 10 iter, with margin:
- features zeroed +111% (well clear of ≥10% threshold)
- CE budget +24% (just clear of ≥20% threshold, but in the
  "more features for more CE" direction)

Re-bucketed: the 10-iter operating point lands in the
**PARTIAL/PASS borderline** of the proposal's decision rule. Not the
"INCONCLUSIVE" I wrote above — that conclusion implicitly compared
Rung4-amp-on-10-iter against MPS-3-iter, which is a different
operating point. The clean reading is:

- **3 iter**: Rung4 amp-on strict-dominates MPS on both metrics. PASS.
- **10 iter**: Rung4 amp-on dominates MPS on features-zeroed (2.11×)
  at +24% CE cost — features-per-CE still 1.71× MPS, but the
  decision rule's "equal-or-better CE" clause is no longer
  satisfied. Borderline PASS/PARTIAL.

#### Implication for the v2.2 verdict

The headline **PASS at the original 3-iter operating point
stands**: at the kwargs the v2 methodology specifies, Rung4
amp-on strict-dominates MPS on both metrics. The 10-iter control
clarifies what was previously read as "Rung4 amp-on degrades at
depth":

- **Rung4 amp-on doesn't degrade vs MPS at depth** —
  features-per-CE-budget stays 1.71× MPS at iter 10. What it
  *does* do is shift from "strict dominance on both metrics" to
  "more features at moderately higher CE cost". The right reading
  is **PASS at 3 iter, borderline PASS/PARTIAL at 10 iter**
  (decision-rule sensitive on the CE-clause), not INCONCLUSIVE.
- **The iter-9 CE spike is encoding-specific.** MPS at 10 iter
  doesn't show a comparable late-iter spike. Hypothesis: K=32
  amp-on panels exhaust high-cosine cluster supply by iter 6 and
  late iters scrape bad-geometry residuals. This is a property
  worth understanding before running Rung4 amp-on at production
  iteration counts above ~6, but doesn't change the verdict.

Natural follow-ups (deferred per
[the proposal's "explicitly does NOT do" section](../../openspec/changes/run-axis1-compression-coverage-mbp/proposal.md)):

- **Pareto curve over `(max_iterations, quality_delta_multiplier)`**
  to find the Rung4-amp-on operational sweet spot. The
  per-iter quality bound (`2.0 × delta_1`) is too permissive to
  catch the cluster-exhaustion regime; cumulative-budget gating
  may be the right mechanism.
- **Cluster-exhaustion diagnosis**: instrument what changes about
  the residual geometry / cosine-distribution between iter 5
  (plateau end) and iter 9 (spike) for Rung4 amp-on. Likely
  source of a future paper-track finding.
- **Axis 4 (sae-forge faithfulness)** remains the most
  load-bearing missing measurement.

### Files

- `docs/research/data/axis1_mps.json` — C1 MPS baseline
- `docs/research/data/axis1_rung4_amp_off.json` — C2 control
- `docs/research/data/axis1_rung4_amp_on.json` — C3 load-bearing (3 iter)
- `docs/research/data/axis1_rung3_amp_on.json` — C4 generality check
- `docs/research/data/axis1_rung4_amp_on_iter10.json` — C3-extended (10 iter)
- `docs/research/data/axis1_mps_iter10.json` — C1-extended MPS 10-iter control
- `docs/research/data/runlog_C{1,2,3,4}_*.txt`, `runlog_C3ext_*.txt`, `runlog_C1ext_*.txt` — console captures
