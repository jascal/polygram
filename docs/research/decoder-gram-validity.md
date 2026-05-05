# Decoder-Gram validity — does Polygram's predicted Gram track the real SAE?

> Research-track note recording the empirical findings of the
> decoder-Gram validity spike defined in
> [`tech-debt-backlog` §4.1](../../openspec/changes/tech-debt-backlog/tasks.md).
> Reproducible via `python examples/decoder_gram_validity.py`.

## Context

PR #16 ([`cross-encoding-stability.md`](cross-encoding-stability.md))
closed the *internal* consistency question: `MPSRung1` and
`HEA_Rung2(depth=2)` agree on which pairs cross the kept-edge gates
across three fixtures including a real GPT-2 SAE. Its closing caveat
named the *external* question explicitly:

> compares two encodings to each other, not either encoding to actual
> SAE behaviour on text.

Until this is answered, every downstream prediction —
`BatchExperiment.cancellation_efficiency`, the
`build_separation_graph` "must-separate" flagging, the entire
disentanglement-loop sketch deferred in
[`spec-disentanglement-loop.md`](spec-disentanglement-loop.md) —
could be tracking a signal that lives only inside the encoded
representation, not the SAE.

The open question:

> For a feature pair `(i, j)` Polygram represents via `(β, α, γ, φ)`
> after `from_sae_lens`, does Polygram's predicted squared overlap
> `|⟨ψ_i|ψ_j⟩|²` correlate with the actual SAE decoder squared cosine
> `(W_dec[:,i] · W_dec[:,j])² / (‖W_dec[:,i]‖² · ‖W_dec[:,j]‖²)`?

If the answer is "no correlation," the entire toolchain is reading
encoding-internal interference that has no counterpart in the SAE,
and downstream work needs to start from
`from_sae_lens` itself.

## Method

For each fixture we (a) computed the real squared-cosine Gram of the
decoder columns directly from the projection vectors, (b) built a
`Dictionary` via `from_sae_lens(...)` (default knobs, φ=0,
`assign_gamma=True`), (c) computed `|D.gram()|²` under both
`MPSRung1()` and `HEA_Rung2(depth=2)` from the same extraction, and
(d) compared the off-diagonal entries via Pearson correlation,
Spearman rank correlation, max absolute drift, and a 2x2 contingency
table at threshold `FLOOR_BLOCK = 0.5` (the same constant
`build_sharing_graph` uses to drop unreachable pairs).

Two fixtures:

1. **Toy SAE.** Features `[0, 1, 4, 5]` from
   `tests/fixtures/toy_sae.json` — small 8-dim projections,
   hand-designed for cluster orthogonality.
2. **Real GPT-2 SAE.** Features `[7836, 13953, 15796, 11978]` from
   `jbloom/GPT2-Small-SAEs-Reformatted`'s `blocks.0.hook_resid_pre`
   layer (24576-feature SAE, 768-dim residual stream) — the same
   projection-similar selection PR #15/#16 used. These were *picked
   because their decoder vectors are similar*, which puts the real
   Gram in a high-baseline regime.

The Animals fixture is excluded; it is hand-crafted and has no SAE
decoder to compare against.

## Findings

### Finding 1 — Spearman ranking is fixture-dependent

| Fixture | Pearson(real, MPS) | Pearson(real, HEA) | Spearman(real, MPS) | Spearman(real, HEA) |
|---|---:|---:|---:|---:|
| Toy SAE  | +0.901 | +0.985 | +0.543 | +0.657 |
| Real SAE | +0.892 | +0.740 | **+0.943** | **+0.943** |

On the **Real SAE**, both encodings rank pair overlaps highly
consistently with the real decoder geometry — Spearman 0.943, well
into the "encoding tracks real geometry" outcome bucket defined in
the spike spec.

On the **Toy SAE**, the same encodings land in the middle bucket
(Spearman 0.5–0.7). The discrepancy is informative — see Finding 2.

### Finding 2 — Polygram over-predicts cross-cluster overlap when real cross-cluster pairs are near-orthogonal

The Toy SAE's real cross-cluster squared overlaps are essentially
zero — the fixture was hand-designed for clean cluster separation:

```
Toy SAE off-diagonal squared overlaps:
  pair                            G_real  G_mps   G_hea
  dog_poodle ↔ dog_beagle         0.951   0.939   0.939   (within)
  dog_poodle ↔ hawk_red           0.028   0.576   0.770   (cross)
  dog_poodle ↔ hawk_cooper        0.025   0.432   0.732   (cross)
  dog_beagle ↔ hawk_red           0.019   0.713   0.732   (cross)
  dog_beagle ↔ hawk_cooper        0.013   0.609   0.770   (cross)
  hawk_red   ↔ hawk_cooper        0.969   0.961   0.961   (within)
```

Within-cluster: real ≈ MPS ≈ HEA, all near 0.95. Match.

Cross-cluster: real ≈ 0.02, but MPS predicts 0.43–0.71 and HEA
predicts 0.73–0.77. Polygram's encoding *cannot* represent
near-orthogonal cross-cluster directions when the feature
parameters live in `(β, α, γ, φ)` with `β ∈ [-0.5, 0.5]`. The
1-qubit cluster-tier rotation `Ry(βπ)` maps β=±0.5 to states whose
inner product is `cos((β_a − β_b)π) = cos(π) = −1`, giving squared
overlap **1**. The bracket of available cross-cluster overlaps is
bounded below by what `(α, γ)` can subtract from that — and
`(α, γ)` enter via much smaller-magnitude rotations, so the
practical floor on cross-cluster squared overlap is ≈ 0.4 for MPS
and ≈ 0.73 for HEA(depth=2). Real decoder geometry can be
arbitrarily orthogonal; the encoding cannot.

### Finding 3 — Polygram tracks ranking well when the SAE subset has high cross-cluster baseline

The Real SAE selection criterion (the `projection-similar 4
features` heuristic that surfaces `--assign-gamma`) deliberately
picks features whose decoder vectors are similar across clusters.
The result is a high-baseline real Gram:

```
Real GPT-2 SAE off-diagonal squared overlaps:
  pair                            G_real  G_mps   G_hea
  feat_7836  ↔ feat_13953         0.957   0.635   0.767
  feat_7836  ↔ feat_15796         0.904   0.464   0.745
  feat_7836  ↔ feat_11978         0.992   0.987   0.987   (within: clusters dogs/dogs)
  feat_13953 ↔ feat_15796         0.960   0.939   0.939   (within: hawks/hawks)
  feat_13953 ↔ feat_11978         0.953   0.695   0.745
  feat_15796 ↔ feat_11978         0.913   0.547   0.767
```

Real overlaps are tightly clustered in [0.90, 0.99]. MPS spreads
the same six pairs across [0.46, 0.99]; HEA collapses them to [0.74,
0.99]. The *ranking* is preserved (Spearman 0.94) — the lowest real
overlap is also the lowest predicted overlap, etc. The *magnitudes*
diverge by up to 0.44 squared-overlap units.

### Finding 4 — Classification accuracy at threshold 0.5 mirrors the magnitude problem

| Fixture | Encoding | TP | TN | miss | false alarm | accuracy |
|---|---|---:|---:|---:|---:|---:|
| Toy SAE  | MPS | 2 | 1 | 0 | 3 | 0.50 |
| Toy SAE  | HEA | 2 | 0 | 0 | 4 | 0.33 |
| Real SAE | MPS | 5 | 0 | 1 | 0 | 0.83 |
| Real SAE | HEA | 6 | 0 | 0 | 0 | 1.00 |

False alarms dominate Toy SAE: Polygram flags cross-cluster pairs
as "high overlap" when they are actually near-orthogonal in the
real decoder. On Real SAE there are no false alarms because every
pair really is high-overlap; classification accuracy is largely an
artefact of the fixture's degenerate label distribution.

## Interpretation

**Polygram is a ranker, not a magnitude predictor.** Specifically:

- The *order* in which Polygram ranks pair overlaps tracks the order
  in which real decoder geometry ranks the same pairs (Spearman 0.94
  on the Real SAE; lower on Toy SAE because the cross-cluster
  near-orthogonality lives outside Polygram's representable range).
- The *magnitude* Polygram assigns each pair systematically diverges
  from real decoder magnitude by up to 0.44 squared-overlap units.
  Quantitative claims like "this pair has 35% irreducible overlap"
  are claims about encoding-internal structure, not the SAE.
- Polygram's encoding has a structural floor on cross-cluster
  overlap (≈ 0.4 MPS, ≈ 0.73 HEA(depth=2)) that real decoders are
  not bound by. When real cross-cluster overlap drops below this
  floor — which happens when an SAE has clean cluster separation —
  Polygram over-predicts.

The first blocker in
[`spec-disentanglement-loop.md`](spec-disentanglement-loop.md)
("gradient signal exists") gets **partial evidence**: there *is*
ranking signal, so a disentanglement primitive that uses Polygram
predictions to rank candidate pairs would be operating on real
information. But a primitive that uses Polygram magnitudes as a
quantitative loss surface would be optimizing encoding-internal
artefacts.

## Practical implications

1. **Use Polygram triage for pair *ranking*, not for absolute
   "headroom" claims.** The
   `BatchExperiment` and `build_*_graph` outputs are correctly
   ordered relative to real geometry; the per-pair magnitudes
   embedded in `cancellation_gap`, `structural_floor`, and the
   `cancellation_efficiency` denominator should be read as
   encoding-internal numbers, not real-SAE numbers.
2. **The `from_sae_lens` projection-similar feature selection
   matters more than was previously documented.** It puts the
   selected subset in a regime where Polygram's ranking signal is
   strong (Spearman 0.94). Random-feature subsets are likely to
   land in the Toy-SAE regime where Polygram over-predicts
   cross-cluster overlap.
3. **Don't shop Polygram numbers as quantitative compression
   predictions.** The user-supplied "Spec-DisEntanglement Loop
   v0.1" sketch's `compression_score = (steering × sparsity) /
   recon` would be combining Polygram-predicted magnitudes with
   real-model metrics; that combination would mix two scales whose
   relationship is now empirically known to be noisy. If a
   real-model loop is pursued, it should treat Polygram as
   "candidate ranker" only, with all magnitudes derived from the
   actual model.

## Caveats and what this doesn't tell us

- **Two fixtures, both 4-feature.** Six pairs each. Spearman
  estimates with N=6 swing easily; the Toy-vs-Real gap is real but
  the precise numbers should be treated as point estimates with
  wide uncertainty.
- **HEA(depth=2)'s structural floor on cross-cluster overlap is
  likely a depth-2 artefact**, mirroring the Finding 2 of the
  cross-encoding spike. Higher-depth HEA should pull the floor
  down. Worth running at depth=4/8 alongside that follow-up.
- **The "real" Gram here is the *decoder cosine Gram*, not the
  *behavioural* Gram from forward passes.** Two SAE features can
  have orthogonal decoder columns but still co-fire on the same
  inputs (and vice versa). The behavioural-Gram comparison would
  need a forward-pass infrastructure Polygram doesn't have. The
  decoder Gram is the cheaper, more falsifiable proxy: if the
  encoding can't even predict the decoder geometry, we don't need
  to ask the more elaborate question.
- **No `assign_gamma=False` baseline.** With `assign_gamma=True`,
  γ is chosen via per-cluster PCA — meaning cross-cluster γ
  contributions to the Gram are coupled to projection geometry.
  With `assign_gamma=False` (γ=0 for all features) Polygram's
  predictions would presumably degrade further on the Real SAE.
  A useful follow-up to confirm.

## Reproducibility

```bash
python examples/decoder_gram_validity.py
```

The Real SAE fixture is auto-skipped if
`./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors`
isn't present. To download it (~144 MB):

```bash
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download(repo_id='jbloom/GPT2-Small-SAEs-Reformatted', \
    filename='blocks.0.hook_resid_pre/sae_weights.safetensors', \
    local_dir='./scratch/real-sae')"
```

The full run reproduces in under a second on a laptop.

## Status

- **Findings 1 and 3** (Spearman tracks well on real SAE; magnitudes
  drift) are robust on the Real SAE fixture and consistent with the
  algebra of the encoding's cross-cluster overlap floor.
- **Finding 2** (encoding can't represent near-orthogonal
  cross-cluster pairs) is structural; the Toy-SAE numbers
  illustrate it but the conclusion follows from the rotation-only
  parameterization independent of fixture.
- **Finding 4** (classification accuracy) is fixture-degenerate
  and shouldn't be over-interpreted at N=6.
- **The decoder-Gram validity question itself** is partially
  answered: Polygram has a real ranking signal but no real
  magnitude signal. Downstream work that depends on ranking is
  unblocked; work that depends on magnitudes (the
  disentanglement-loop's loss surface; the user-supplied Gemma
  loop's compression score) needs a different signal source for the
  magnitude inputs.
- **Follow-ups worth running**: (a) Larger feature subsets (N=8,
  the `from_sae_lens` cap) for tighter Spearman estimates;
  (b) `assign_gamma=False` baseline; (c) HEA depth=4/8 to confirm
  the depth-2 cross-cluster overlap floor isn't intrinsic.
