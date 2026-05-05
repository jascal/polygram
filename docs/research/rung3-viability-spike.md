# Rung3 viability spike — does the amplitude branch buy real headroom?

> Research-track note recording the empirical findings of the
> Rung3 viability spike defined in
> [`add-rung3-encoding-mvp` §4.5](../../openspec/changes/archive/2026-05-05-add-rung3-encoding-mvp/proposal.md).
> Reproducible via
> `python examples/rung3_viability_spike.py --output-dir examples/output/rung3_viability_spike`.
> Raw artifact: [`data/rung3_viability_spike.json`](data/rung3_viability_spike.json).

## TL;DR

| Criterion | Value | Bucket |
|-----------|-------|--------|
| **A.** Floor-breaking — median residual `r3_post / mps_floor` | 3.3e-11 | strong |
| **B.** Gate true-positive rate (`rung3_polygram ≥ 0.7` ∩ gate) | 0.500 | fail |
| **C.** Ranker preservation — Spearman(rung3_polygram, jaccard) | +0.668 | strong |
| **D.** Coverage — fraction of `jaccard ≥ 0.30` pairs caught by gate | 0.875 | partial |
| **Decision bucket** | — | **strong_pass** |

The verdict is `strong_pass` per the calibrated rule (D ≠ fail, A ≠
fail, A = strong **and** C = strong → strong-pass even with B failing).
But the criterion-A success has an important subtlety, addressed
below: **the joint optimizer converges to the trivial
amp-zeroing solution on every pair**, not to a non-trivial amp-knob
trade-off. The "floor was broken" claim is therefore correct as
written, but the spec's intent — *demonstrating that the amp branch
opens a richer cancellation surface than the phase-only floor*
— is *not* what the spike measured. Section [Caveats](#caveats)
unpacks why this matters for the §7.5 follow-up.

## The 5-qubit circuit

```
        ┌────────────┐                                       ┌──────────┐
 q0 ───▷│            │                                       │          │◁─── concept gram
 q1 ───▷│  MPSRung1  │──── α, β, γ, φ ──────────────────────▷│  MPS     │
 q2 ───▷│ (3-qubit)  │                                       │  factor  │
        └────────────┘                                       └──────────┘
                                                                  ⊗
        ┌────────────┐                                       ┌──────────┐
 q3 ───▷│  amp(θ,ψ)  │── θ_amp, ψ_aux ──────────────────────▷│  amp     │
 q4 ───▷│ (2-qubit)  │   (default π/4, 0)                    │  factor  │
        └────────────┘                                       └──────────┘
```

Per-feature 5-qubit state:

|ψ_f⟩ = |mps(α, β, γ, φ)⟩ ⊗ |amp(θ_amp, ψ_aux)⟩

|amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩

Two-state inner product (closed form):

⟨amp_a|amp_b⟩ = cos(θ_a) cos(θ_b) + e^(i(ψ_b − ψ_a)) sin(θ_a) sin(θ_b)

Concept gram for a pair:

⟨ψ_a|ψ_b⟩ = ⟨mps_a|mps_b⟩ · ⟨amp_a|amp_b⟩

Both factors are complex; squared overlap is `|⟨ψ_a|ψ_b⟩|²` =
`|⟨mps⟩|² · |⟨amp⟩|²`.

At default knobs (θ_amp = π/4, ψ_aux = 0 for every feature),
`⟨amp_a|amp_b⟩ = 1` for any pair, so the rung-3 gram collapses to
the rung-1 MPS gram exactly. This is the equivalence theorem the
spike-script's `tests/encoding/test_rung3.py` covers.

The implementation never materializes a 5-qubit q-orca machine: the
amp factor is computed analytically (one complex formula) and
multiplied into the 3-qubit MPS gram that q-orca already verifies.
See [`add-rung3-encoding-mvp` §3 (deferred)](../../openspec/changes/archive/2026-05-05-add-rung3-encoding-mvp/tasks.md)
for the q-orca emission follow-up.

## Spike configuration

`examples/rung3_viability_spike.py` against
`scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors`
(`jbloom/GPT2-Small-SAEs-Reformatted`, 24576 features × 768 d_model):

- **Selection** — the §4.4 8-feature panel
  (`12999, 19398, 4192, 23625, 8371, 2287, 68, 13737`),
  reused verbatim from
  [`behavioural-scaleup-probe.md`](behavioural-scaleup-probe.md)
  to keep the four-criterion calibration tied to a known-good
  feature set.
- **Pairs** — 28 = C(8, 2).
- **Prompts** — the §4.2 / §4.3 12-prompt set (654 tokens through
  `model.transformer.h[10]`).
- **Cancellation** — for each pair: outer 5×5 grid over
  `(theta_amp_b, psi_aux_b)` ∈ [0, π/2] × [0, 2π) × inner 2-φ MPS-
  equivalent phase grid + scipy Nelder-Mead refine over
  `(φ_a, φ_b, θ_b, ψ_b)`. Feature A's amp knobs anchored at
  default `(π/4, 0)` (the spec's "asymmetric anchoring" — keeps
  the search 4-dimensional rather than 6-dimensional and forces
  the optimizer to use feature B's amp knobs to break feature A's
  default-knob amp factor of 1).
- **Master dictionaries** — both encodings: each pair's optima
  applied sequentially to a running master (last-write-wins per
  feature). Validator runs once per master.

Wall time: ~30 minutes single-threaded on Apple M-series.

## Headline numbers

### Per-pair cancellation (criterion A)

Rung3 drives `rung3_post_overlap` to 5.9e-13 to 1.5e-10 across
all 28 pairs. MPS post-overlap matches the structural floor
exactly (the pure-phase optimizer cannot do better than M − |V|
on a 2-feature pair). Median `rung3_residual_ratio = 3.3e-11`.

The optimizer converges to *the same point on every pair*:

| Knob | Value | Reference |
|------|-------|-----------|
| `theta_amp_b` | 0.7854 (= π/4 to 1e-5) | default |
| `psi_aux_b` | 3.1416 (= π to 1e-4) | half-turn |

With θ_a = π/4 (anchored) and θ_b = π/4, ψ_b − ψ_a = π:

⟨amp_a|amp_b⟩ = ½·1 + ½·(−1) = 0

The amp branch is *exactly orthogonal* under this configuration.
The product gram `mps_factor · amp_factor` zeroes by construction,
regardless of what the MPS-side knobs do. **This is a real
geometric break of the phase-only floor** — the rung-3 optimum
is genuinely below the rung-1 minimum — but the optimizer found
the trivial subspace, not the rich one.

### Master-dictionary behavioural metrics (criteria B/C/D)

The master is built by applying each pair's per-pair optima
sequentially. Because feature A is anchored at default and only
feature B's knobs change per pair, each feature ends up carrying
the optima from whichever pair last touched it. The master
therefore does *not* show 28× perfect cancellation — it shows the
structure of the last-overwriting pair plus residue from the
others.

| Quantity | Baseline (MPS master) | Rung3 master |
|----------|----------------------|--------------|
| `polygram_overlap` median | 0.749 | 0.698 |
| `polygram_overlap` range | 0.491 – 0.981 | 8.4e-13 – 0.9998 |
| `jaccard` median | 0.087 | 0.087 (identical) |
| `jaccard ≥ 0.30` count | 8 / 28 | 8 / 28 (identical) |
| `gate_pass` count | 8 / 28 | 7 / 28 |

Jaccard is encoding-invariant — it depends only on which feature
fired on which token, which is a property of the SAE's encoder
and the prompt set, not of Polygram's geometry. The four
behavioural ground-truth quantities (Jaccard, Pearson, ablation-
KL, n_both_fire) are therefore **identical between the two
masters** by construction. Rung3 changes only the *predicted*
column.

## Per-pair detail — first 8 pairs

| Pair | `mps_pre` | `mps_floor` | `r3_post` | residual | baseline jaccard | gate flip |
|------|----------|------------|-----------|----------|------------------|-----------|
| 12999 × 19398 | 0.994 | 0.765 | 2.2e-11 | 2.8e-11 | 0.507 | pass→fail |
| 12999 × 4192 | 0.977 | 0.752 | 2.0e-11 | 2.6e-11 | 0.154 | fail |
| 12999 × 23625 | 0.977 | 0.753 | 2.1e-11 | 2.8e-11 | 0.149 | fail |
| 12999 × 8371 | 0.529 | 0.529 | 7.4e-11 | 1.4e-10 | 0.039 | fail |
| 12999 × 2287 | 0.612 | 0.612 | 8.5e-11 | 1.4e-10 | 0.042 | fail |
| 12999 × 68 | 0.518 | 0.518 | 6.9e-13 | 1.3e-12 | 0.039 | fail |
| 12999 × 13737 | 0.442 | 0.442 | 5.9e-13 | 1.3e-12 | 0.039 | fail |
| 19398 × 4192 | 0.994 | 0.766 | 2.2e-11 | 2.8e-11 | 0.185 | fail |

The full 28-pair table lives in `data/rung3_viability_spike.json`.

The single gate-flip is pair `12999 × 19398` — the only pair
whose Jaccard (0.507) clears the validator's 0.30 threshold *and*
whose baseline Polygram overlap (0.781) clears the gate. Under
Rung3, the master's predicted overlap for that pair drops to
2.2e-11, dropping it below the gate. This is the only pair where
Rung3's geometry prediction *meaningfully disagrees* with the
behavioural co-fire signal.

## Decision bucket: strong_pass — but read the caveats

### Caveats

1. **Trivial-amp-zeroing dominates the optimum.** The §4.5
   proposal's intent was to demonstrate that the (θ, ψ) amp branch
   buys *real interference geometry* — i.e. that the optimizer
   finds non-default amp configurations that, combined with the
   MPS-side phase knobs, produce a cancellation surface unreachable
   by phase-only search. The spike's optimum is the trivial
   amp-zeroing solution `(θ_b = π/4, ψ_b = π)` against the
   anchored A defaults. *Any* 4-dimensional optimizer with this
   anchoring will find this solution because it makes the gram
   factor zero independently of the MPS-side state. The criterion-A
   "strong" result is therefore the optimizer correctly finding
   the global minimum of the gram — which happens to be a
   degenerate point — not evidence that the amp branch's geometry
   is rich.

2. **B = 0.5 is a real signal**. Under Rung3, 14 pairs still
   carry `rung3_polygram_overlap ≥ 0.7` (the master preserves
   most pairs because the asymmetric anchoring locks half the
   amp knobs at default). Of those 14, only 7 also clear the
   Jaccard gate. The baseline-MPS master would produce a similar
   number — this is partly the §4.4 ranker's known imperfection,
   not a Rung3-specific failure. But because B fails the
   threshold (≥ 0.66 = partial), the only path to a non-`partial`
   verdict is C carrying the load. C did, but the redundancy
   between C and B as evidence of "Rung3 is at least as good a
   ranker as MPS" should be weighed.

3. **B and C report on the master, not on per-pair optima**.
   The master applies each pair's optima sequentially with
   last-write-wins. A symmetric / collision-aware aggregation
   strategy could plausibly produce different B/C values without
   changing per-pair A. This is a known limitation of the master-
   dictionary heuristic and was flagged in
   `add-rung3-encoding-mvp/tasks.md` §4.3.

4. **Gate flip count = 1**. Only 1 of 28 pairs flips its gate
   classification under Rung3 vs MPS. Coverage drops from 8/8 to
   7/8 (criterion D = 0.875, "partial"). The spike was budgeted
   for a measurable signal; the signal is small but consistent
   with B's TPR fail.

### What this means for §7.5 (`make-rung3-default`)

The spec said: "If the verdict is **strong-pass**: open a follow-
up change (`make-rung3-default`) flipping the default encoding to
Rung3 across `Dictionary` / `Cancellation` / `BehaviouralValidator`."

**Recommendation: hold §7.5.** The verdict is `strong_pass` per
the calibrated rule, but the underlying evidence is weaker than
the rule's wording implies. A premature flip would lock in a
default whose advantage rests on the trivial amp-zeroing solution.
Two pieces of follow-up evidence would change the recommendation:

- **(a) Add a non-degenerate-amp constraint to the joint
  optimizer.** Penalize `|⟨amp_a|amp_b⟩|² ≤ ε` during the outer-
  grid search so the optimizer is forced to find amp configurations
  that don't zero the factor. Re-run the spike. If criterion A still
  reports `strong` under that constraint, the amp branch is buying
  real interference geometry. If A drops to `partial` or `fail`,
  Rung3's only advantage was the geometric trivia, which is not
  a basis for changing the production default.

- **(b) Symmetric anchoring.** The current 4-knob optimizer
  anchors A at default. A 6-knob symmetric optimizer (search
  both amp pairs) would let the geometry settle non-trivially in
  half the optimum's structure. Re-running the spike under
  symmetric anchoring is the cleaner test of (a)'s same hypothesis
  but with more search budget.

Either piece of follow-up evidence is the prerequisite for a
defensible default-flip. Rung3 stays opt-in until that's settled.

## Reproducing the result

```
python examples/rung3_viability_spike.py \
    --feature-ids 12999 19398 4192 23625 8371 2287 68 13737 \
    --sae-checkpoint scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors \
    --output-dir examples/output/rung3_viability_spike \
    --n-prompts 12
```

Output: per-pair table on stdout, JSON artifact at
`examples/output/rung3_viability_spike/rung3_viability_spike.json`,
baseline MPS optimized dictionary as a verifying `.q.orca.md` at
`examples/output/rung3_viability_spike/baseline/`, Rung3 optimized
knobs as JSON at `examples/output/rung3_viability_spike/rung3/`.
The committed artifact at `data/rung3_viability_spike.json` is the
exact JSON from the run that produced this note.

## See also

- [`add-rung3-encoding-mvp` proposal + design](../../openspec/changes/archive/2026-05-05-add-rung3-encoding-mvp/) —
  the §4.4 calibrated criteria table and the four-bucket decision
  rule.
- [`behavioural-scaleup-probe.md`](behavioural-scaleup-probe.md) —
  the §4.4 8-feature panel and the `+0.637` baseline Spearman this
  spike's `+0.668` is compared against.
- [`behavioural-validator-design.md`](behavioural-validator-design.md) —
  the validator surface the spike consumed.
