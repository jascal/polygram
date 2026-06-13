# Readout-aligned dictionary geometry — build the Gram on the basis the model reads through

Add a `readout-aligned` geometric profile that builds Polygram's dictionary geometry (clustering + knob
assignment) on feature directions **projected onto the model's readout subspace**, instead of on the **raw
decoder vectors** it uses today. This is the highest-leverage test of Polygram's load-bearing claim — and it is
now backed by hard cross-model evidence, not a hunch.

## Why

Polygram's load-bearing claim is **behavioural**: the predicted dictionary geometry must track real co-firing /
ablation-KL — measured as `Spearman(Polygram-overlap, co-firing Jaccard) = 0.637`
(`docs/research/behavioural-scaleup-probe.md`). But that geometry is built on **raw decoder vectors**:
`polygram/geometry/clustered.py` feeds the raw per-feature projections straight into k-means
(`clustered.py:165`), the cluster centroids (`:176`), the residual-variance fidelity (`:177`), and the
per-cluster-PCA γ assignment (`:182`). The raw decoder/activation spectrum is exactly the **deceptively
compressible** basis the research program warns against — the *behaviourally* relevant object is the
**readout-aligned decision geometry** (the directions the model's argmax actually reads through), not raw
decoder norm.

That used to be a hypothesis; it is now a **measured, model-general** fact. A powered cross-model run
(fieldrun `tau_star_powered.py`, wikitext 20k tokens, 3 seeds) shows a trained rank-`r` projection onto the
**readout-aligned** subspace beats the frozen SVD lens on open-class next-token recovery at every rung:
**GPT-2 +52pp, Pythia-70m +31pp, Pythia-160m +40pp** (open-class R@32, all 6 model×rank cells win). The
readout-aligned directions, not the energy-weighted decoder spectrum, govern behaviour — across architectures.

**So if Polygram fits its geometry in the raw-decoder basis, its 0.637 Spearman is partly measuring the
basis, not the model.** Fixing the basis is the single highest-leverage change to the geometry-tracks-behaviour
claim — and it is a fair, falsifiable test of **Reckoning #3** (is the quantum Gram load-bearing, or a
removable scaffold?): if the *right* basis raises the Spearman, the Gram earns its keep; if it doesn't move,
the predictive power isn't basis-limited and the prune becomes more likely.

## What

### 1. A `readout-aligned` geometric profile — `polygram/geometry/readout_aligned.py`

Mirror `clustered.py`'s `ClusteredKnobAssignment` / `clustered()` factory with a `ReadoutAlignedKnobAssignment`
/ `readout_aligned()` that, **before** clustering + knob assignment, projects the decoder vectors onto the
**readout subspace**:

```
readout subspace  R = top-r right singular directions of  (gain ⊙ U)     # U: (vocab, d_model) unembed
proj_readout      = projections @ Rᵀ                                      # (n_features, r)
# then k-means / centroids / residual-variance / γ-PCA run on proj_readout, not the raw projections
```

This is the same readout-aligned construction R2 validated (the unembed the model reads through, optionally
weighted by the final-norm `gain`). The strategy's `.assign(projections, ...)` does the projection internally,
so the `from_sae_lens` call site (`sae_import.py:813`) is otherwise unchanged. Register `readout_aligned()` in
`polygram/geometry/__init__.py` alongside `clustered()` / `uniform_sphere()`.

### 2. Thread the readout geometry in — `from_sae_lens(u_matrix=, gain=)`

`from_sae_lens` already takes `profile=`. Add net-new optional `u_matrix: np.ndarray | None` (the host unembed,
`(vocab, d_model)`) and `gain: np.ndarray | float | None` (final-norm gain, default ones). When
`profile="readout-aligned"`, `u_matrix` is **required** — raise a clear `ValueError` if missing. Polygram does
**not** load a host model itself (keeps the core numpy-only / low-dep); the caller supplies `U`/`gain` (the
behavioural extra's `runtime._load_host_model` already exposes a host for the probe).

### 3. Before/after behavioural comparison

Extend `examples/behavioural_gram_scaleup.py` to run **both** profiles (`clustered` vs
`readout-aligned`, same features/SAE/host) and emit a `polygram_overlap_readout_aligned` column into
`docs/research/data/scaleup_pairs.csv`; record the head-to-head Spearman table in
`docs/research/behavioural-scaleup-probe.md` (no new page).

## Scope / what this is NOT

- **Geometry / knob-assignment only.** The profile changes how β/γ are assigned; the encoding's structural form
  and the **Q-Orca emission are transparent** to it (`_qorca_emit.py` / `Dictionary.gram()` consume the final
  knob values, not the projection vectors). No change to the quantum machine structure.
- **Caller supplies `U`/`gain`.** No host-model load inside the Polygram core (no new hard dep on transformers).
- **v1 host: GPT-2-small** (the existing scaleup-probe substrate). Cross-arch (Pythia, via the now-merged
  forge adapter) is a follow-up once the GPT-2 result lands.
- **`clustered` stays the default.** `readout-aligned` is opt-in; existing dictionaries/tests are byte-identical
  unless the profile is selected.

## Falsifiable acceptance gate (descriptive, both outcomes first-class)

Re-run the behavioural scaleup probe on GPT-2-small head-to-head (raw `clustered` vs `readout-aligned`), same
panels:

- **WIN:** readout-aligned raises `Spearman(Polygram-overlap, co-firing Jaccard)` to **≥ 0.70** (from 0.637) →
  readout-alignment is the right basis; the Gram's predictive power was basis-limited (and the quantum lens
  earns its keep on the right basis).
- **NO IMPROVEMENT:** Spearman does not move → the Gram's predictive power is **not** basis-limited; an
  equally-valuable result that makes **Reckoning #3**'s prune (quantum lens = removable scaffold) more likely.

Either way the verdict is the head-to-head table, descriptive; no necessity claims.

### Gate RESULT (2026-06-13) — NO IMPROVEMENT; the readout basis is *wrong for co-firing*

Implemented + run head-to-head on GPT-2-small `blocks.10` (`examples/behavioural_gram_scaleup.py
--profile {clustered,readout-aligned} --readout-rank R`):

| profile / rank | `Spearman(Polygram-overlap, co-firing Jaccard)` |
|---|---|
| `clustered` (raw decoder, baseline) | **0.640** |
| `readout-aligned`, rank 64 (default) | 0.267 |
| `readout-aligned`, rank 256 | 0.067 |
| `readout-aligned`, rank 768 (full `d_model`) | **0.640** (exactly the baseline) |

**Readout-alignment does NOT raise the Spearman — it strictly *hurts*, recovering the baseline only at full
rank.** The full-rank result is the key tell: a full-rank readout projection is just an orthonormal rotation
(plus gain weighting), and k-means is rotation-invariant — so it reproduces `clustered` exactly (0.640). Every
*truncated* rank discards residual dimensions and degrades the co-firing signal.

**The mechanism — and the real lesson — is a basis/metric mismatch:** Polygram's behavioural metric is
**co-firing** (which features *activate together* — an **encoder-side** phenomenon living in the full-residual
feature geometry), whereas R2's readout subspace is **decode-side** (the directions the model's *argmax* reads
through). Readout-alignment is the right basis for the *decode* tax (R2's +52/+31/+40pp) but the **wrong** basis
for *co-firing*. So this is the **NO-IMPROVEMENT** branch — and it **vindicates** Polygram's raw-decoder
geometry for the co-firing claim: the 0.640 is *not* a basis artifact, and Reckoning #3's basis-limited concern
does not apply here.

**What ships:** the `readout-aligned` profile is implemented, tested, and correct — it is a legitimate tool for
*decode-relevant* geometry questions; it is simply **not** the right basis for the co-firing Gram, which this
gate establishes cleanly. (A future probe with a *decode-side* behavioural metric — e.g. logit-attribution
overlap — is where readout-alignment would be expected to help.)

## Related

- `docs/research/behavioural-scaleup-probe.md` — the 0.637 baseline this targets.
- Manifesto Reckoning #3 (quantum lens load-bearing?) — this is a fair test of it.
- fieldrun R2 / `tau_star_powered.py` — the powered, model-general evidence that readout-aligned directions
  govern behaviour; sae-forge `add-capability-ceiling-diagnostic` — the consumer (readout-aligned atom
  *selection* is the action when its `selection_gap` is large).
