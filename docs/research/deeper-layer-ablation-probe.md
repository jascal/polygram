# Deeper-layer ablation-KL probe — does single-feature ablation become informative below `blocks.0`?

> Research-track note recording the empirical findings of the
> deeper-layer ablation-KL probe defined in
> [`tech-debt-backlog` §4.3](../../openspec/changes/tech-debt-backlog/tasks.md).
> Reproducible via `python examples/behavioural_gram_probe.py --layer {0,5,10}`.

## Context

PR #20 ([`behavioural-gram-probe.md`](behavioural-gram-probe.md))
landed the §4.2 one-pair behavioural-Gram probe and surfaced an
unexpected blocker for any future loop that wants to use ablation-KL
as a behavioural-impact signal: at `blocks.0.hook_resid_pre` on
GPT-2 small, individual SAE feature ablations produce ~5e-5 nats of
KL on the next-token distribution — too small to discriminate one
feature from another. The §4.2 substitutability metric trivially
"passed" for both pairs because the underlying KL was indistinguishable
from float32 noise. PR #20 named the natural follow-up:

> Pick a deeper SAE layer for ablation-impact metrics.
> `blocks.0.hook_resid_pre` is the input to the model. Individual
> SAE features at this point have ~5e-5 nats of KL impact, which is
> noise.

This probe rerun the §4.2 harness at `blocks.5.hook_resid_pre` and
`blocks.10.hook_resid_pre` to answer one question: **does ablation-KL
grow with depth, stay flat, or peak somewhere in the middle?**

## Method

Same script as §4.2, with a new `--layer {0, 5, 10}` flag. The flag:

- Resolves the SAE checkpoint path
  `./scratch/real-sae/blocks.{layer}.hook_resid_pre/sae_weights.safetensors`
  (same `jbloom/GPT2-Small-SAEs-Reformatted` HuggingFace repo for all
  three layers).
- Moves both forward-pre-hooks from `model.transformer.h[0]` to
  `model.transformer.h[layer]`. The capture hook reads the residual
  stream entering block `layer`; the ablation hook subtracts
  `f_A · W_dec[A, :]` at that same position.

Same prompts (12 paragraphs, 654 GPT-2 tokens), same feature ids
(`[7836, 13953, 15796, 11978]`), same pair set (`(7836, 11978)` +
`(7836, 15796)`).

**Important caveat about feature identity.** The feature ids 7836,
11978, 15796 were chosen for the layer-0 SAE (PR #16's projection-
similarity selection). The blocks.5 and blocks.10 SAEs are
*independently trained* SAEs — they share the 24576-feature dictionary
size but their feature indices have no semantic correspondence to
layer-0's. At deeper layers, "feat_7836" is whatever feature happens
to occupy index 7836 in that layer's SAE. The probe uses these
indices as **arbitrary anchor features** for measuring per-feature
ablation-KL magnitude. The §4.3 question is "does ablation-KL become
informative for arbitrary single SAE features at depth?" — not "do
these specific semantic features become informative." The pair-level
co-occurrence and substitutability metrics at layers 5 and 10 are
**not** comparing the same semantic feature pair as layer 0; treat
them as a sanity check on the harness, not as cross-layer behavioural
claims.

## Findings

### Finding 1 — Ablation-KL jumps 4 orders of magnitude from layer 0 to layer 5

Per-feature ablation-KL averaged on tokens where that feature fires:

| Layer | KL(ablate-7836) on 7836-fire tokens | KL(ablate-11978) | KL(ablate-15796) |
|------:|---:|---:|---:|
| 0     | 5.18e-05 | 3.86e-05 | 7.20e-05 |
| 5     | 1.04 | 1.93 | 1.30 |
| 10    | 1.12 | 1.03 | 0.56 |

The 0 → 5 transition is roughly four orders of magnitude. The
5 → 10 transition is essentially flat. Layer 0 is the anomaly;
layers 5 and 10 carry comparable signal magnitudes within a factor
of ~3.

This is the load-bearing finding: **ablation-KL is well-defined at
mid- and late-layer residual-stream SAEs** for GPT-2 small. Loops
that want to use single-feature ablation impact as a behavioural
signal can do so as long as they hook at `blocks.5` or deeper.

### Finding 2 — Deeper-layer features at these indices fire much more sparsely

Number of tokens (out of 654) where each feature fires:

| Layer | n_fires(7836) | n_fires(11978) | n_fires(15796) |
|------:|---:|---:|---:|
| 0     | 118 | 97 | 78 |
| 5     | 13 | 12 | 12 |
| 10    | 22 | 14 | 12 |

At layer 0 these specific features fire on 12-18% of tokens. At layer
5 and 10 they fire on 2-3% of tokens — roughly an order of magnitude
sparser. Two things are likely going on:

1. **Layer-local SAE feature distribution shift.** Different SAEs
   trained on different layer activations have different sparsity
   profiles. The layer-5 / layer-10 SAEs may simply have lower
   per-feature firing rates than the layer-0 SAE. The 24576-feature
   capacity is the same; the sparsity is set by the L1 penalty during
   SAE training and may differ between the three checkpoints.
2. **Arbitrary-index narrowness.** Picking feature ids that are
   meaningful at layer 0 means picking arbitrary features at layers
   5 and 10. With ~24k features, an arbitrary index is statistically
   biased toward the rare end of the distribution (typical SAE
   features fire on a small fraction of tokens; only a handful are
   common-firing).

The substantive consequence: per-token KL averages at layers 5 and 10
are based on 12-22 tokens, not 78-118. Statistical confidence in
those averages is correspondingly lower, but the magnitudes are large
enough (1+ nats vs 5e-5 nats) that even a high-variance estimate
clears the layer-0 noise floor by orders of magnitude.

### Finding 3 — Pearson +1.000 at depth signals near-degenerate firing patterns

| Layer | Pair | Pearson(act_i, act_j) | Co-fire / fire-i | Co-fire / fire-j |
|---:|:---|---:|---:|---:|
| 0  | within (7836↔11978) | +0.245 | 49/118 = 0.42 | 49/97 = 0.51 |
| 0  | cross  (7836↔15796) | +0.190 | 31/118 = 0.26 | 31/78 = 0.40 |
| 5  | within | +1.000 | 12/13 = 0.92 | 12/12 = 1.00 |
| 5  | cross  | +1.000 | 12/13 = 0.92 | 12/12 = 1.00 |
| 10 | within | +0.9999 | 12/22 = 0.55 | 12/14 = 0.86 |
| 10 | cross  | +0.9999 | 12/22 = 0.55 | 12/12 = 1.00 |

At layers 5 and 10, the Pearson correlation across firing tokens
collapses to +1.000. Combined with the very high co-fire rates (most
fires are co-fires) this means the three arbitrary anchor features at
each layer fire on overlapping token sets with linearly-related
activation magnitudes. The simplest explanation is that these
particular features at layers 5 and 10 are responding to a small
shared substrate (a specific punctuation, BOS-adjacent positions, or a
narrow attention pattern) that produces near-identical activations
when triggered.

This is **not a property of deep-layer SAE features in general** —
it's a property of *these specific arbitrary indices* at depth.
Picking different feature ids at layers 5 and 10 (selected by, say,
projection similarity within each layer's own SAE) would likely give
much more varied behaviour. The probe carries this caveat because the
spec deliberately reuses layer-0-chosen indices to keep the harness
simple; the load-bearing finding remains the per-feature absolute KL
magnitudes (Finding 1), not the pair-level correlations.

### Finding 4 — Substitutability ratios become meaningful at depth

Recall §4.2 found KL ratios of 1.18 and 0.72 at layer 0, technically
in the "substitutable" `[0.5, 2.0]` band but uninformative because
both numerator and denominator were ~5e-5. At depth:

| Layer | Pair | KL_i (both-fire) | KL_j (both-fire) | Ratio i/j | In [0.5, 2.0]? |
|---:|:---|---:|---:|---:|:---:|
| 0  | within | 3.64e-05 | 3.09e-05 | 1.18 | yes (uninformative) |
| 0  | cross  | 4.86e-05 | 6.78e-05 | 0.72 | yes (uninformative) |
| 5  | within | 1.12 | 1.93 | 0.58 | yes |
| 5  | cross  | 1.12 | 1.30 | 0.87 | yes |
| 10 | within | 2.04 | 1.20 | 1.70 | yes |
| 10 | cross  | 2.04 | 0.56 | 3.64 | **no** |

At layer 5 both pairs land cleanly inside the band; at layer 10 the
within-cluster pair stays inside (1.70) but the cross-cluster pair
escapes (3.64). The substitutability metric now has discriminative
power — the ratio escapes the band when ablating one feature has
clearly more impact than ablating the other.

(Caveat from Finding 3 still applies: the layer-5 / layer-10 features
at these indices are correlated near-degenerately, so the ratios
shouldn't be read as evidence about Polygram's predictions on those
specific pairs — they're evidence that the **metric** has
discriminative range at these layers.)

## Interpretation

Mapping to the §4.3 spec's three outcome buckets:

> **KL grows monotonically with depth** (e.g., 5e-5 → 1e-3 → 1e-2
> nats): ablation-KL becomes informative deeper in the stack.

This is the bucket we land in, with a wrinkle: the growth is not
gradual. KL jumps four orders of magnitude from layer 0 to layer 5
and then plateaus. Layer 0 is the outlier; layers 5 and 10 are
roughly equivalent for ablation-impact purposes.

The structural interpretation: the residual stream at `blocks.0` is
just embedding + position_embedding. Eleven downstream transformer
blocks attend, MLP, and re-mix the signal so thoroughly that any
single 768-dim direction's contribution gets averaged out. By
`blocks.5` we're past the heavy attention-mixing zone — direction-
specific signals at that position survive to the unembedding with
much more leverage. By `blocks.10` we're close enough to the output
that further depth doesn't add much (only two more blocks of mixing
remain).

The picture, combined with PR #18 and PR #20:

1. **Polygram is a directional ranker for both decoder geometry and
   behavioural similarity** (PR #18, PR #20).
2. **Polygram's magnitudes are not real-behaviour magnitudes** (PR #18,
   PR #20).
3. **Ablation-KL is informative at depth (`blocks.5+` for GPT-2
   small) but not at the input layer.** Future compression-loop work
   that wants to use ablation impact as a behavioural signal should
   target a mid- or late-layer SAE.

## Practical implications

1. **Pick `blocks.5` or deeper for any ablation-based behavioural
   metric.** The layer-0 dead zone is structural — eleven downstream
   blocks compensate for any one input-layer feature's removal — and
   does not improve with feature selection or longer prompts. Move
   the hook, or use a different metric entirely.
2. **`blocks.5` ≈ `blocks.10` for ablation magnitude on GPT-2 small.**
   Either layer carries comparable signal. `blocks.10` is closer to
   the output and may be marginally more sensitive to features that
   directly write to the unembedding direction; `blocks.5` keeps more
   downstream computation available for measuring secondary effects.
   In the absence of a specific reason to prefer one, `blocks.10` is
   slightly safer (less downstream compensation) and `blocks.5` is
   slightly more compute-efficient (fewer blocks per ablation
   forward pass).
3. **Pair selection must be layer-local.** PR #16's
   projection-similarity feature subset was chosen on the layer-0 SAE.
   Repeating that selection on each layer's own SAE (different decoder
   geometry, different feature set) is a precondition for any
   behavioural comparison across layers — which is why this probe
   only reports per-feature KL magnitudes at depth and explicitly
   does *not* claim cross-layer behavioural redundancy.
4. **The PR #20 caveat against the peer-agent `compression_score` at
   layer 0 stands; at layer 5 / 10 it would no longer hit the
   zero-signal floor.** If a loop wants to revisit ablation-KL as
   part of its objective, the path is open — at depth.

## Caveats and what this doesn't tell us

- **N = 3 anchor features per layer.** The KL magnitudes are
  per-feature averages over a small token set. They are large enough
  to be informative against the layer-0 floor (4 OOM gap) but the
  point estimates have wide error bars. A scaled run (~30 features
  selected via per-layer projection similarity) would tighten this.
- **GPT-2 small only.** Other architectures (Gemma, Llama, Mistral)
  may have different residual-stream mixing rates. The qualitative
  result — input-layer ablation is uninformative, mid-/late-layer is
  informative — should generalize, but the specific layer thresholds
  will not. For Gemma-2-2b's 26 blocks, the equivalent of `blocks.5`
  on GPT-2's 12 blocks is roughly `blocks.10`.
- **Same-index features ≠ same semantic features across layers.**
  The pair-level metrics at layers 5 and 10 are not comparing the
  same Polygram-predicted pair as at layer 0. Findings 3 and 4 carry
  this caveat in-text. The Polygram → behavioural correspondence
  question is the §4.4 (scale-up) target, where pairs would be
  selected per-layer.
- **Ablation-KL = next-token only.** Multi-token text generation,
  attention pattern shifts, or downstream-layer activation deltas
  could show different patterns. A logit-lens measurement at the
  ablated layer (skipping further blocks) would test how much of the
  KL signal is "this feature directly writes the unembedding" vs
  "this feature triggers downstream behaviour."
- **Pearson +1.000 at depth is suspicious.** It strongly suggests
  the arbitrary anchor features at layers 5 and 10 happen to fire
  on a small near-identical token subset. The next-step probe should
  pick layer-local features deliberately, both to avoid this
  degeneracy and to make the pair-level metrics interpretable.

## Reproducibility

```bash
# Layer 0 (the §4.2 baseline; ~5e-5 nats per ablation)
python examples/behavioural_gram_probe.py --layer 0

# Layer 5 (~1 nat per ablation)
python examples/behavioural_gram_probe.py --layer 5

# Layer 10 (~1 nat per ablation)
python examples/behavioural_gram_probe.py --layer 10
```

Each run auto-skips with a clear message if the corresponding SAE
checkpoint isn't on disk. Download command (per layer):

```bash
hf download jbloom/GPT2-Small-SAEs-Reformatted \
    --include="blocks.{0,5,10}.hook_resid_pre/sae_weights.safetensors" \
    --local-dir ./scratch/real-sae
```

Each layer takes ~2-3 minutes on CPU (12 prompts, two ablation passes
per pair, two pairs). Raw outputs from the run that produced the
tables above are checked in at
[`docs/research/data/deeper_layer_probe_layer{0,5,10}.log`](data/).

## Status

- **Finding 1 (KL grows from 5e-5 → 1+ nats between layers 0 and 5
  and plateaus thereafter)** is the load-bearing result. It is
  robust against the small-N caveat because the gap is four orders
  of magnitude.
- **Finding 2 (deeper-layer arbitrary-index features are sparser)**
  is partly an SAE training artefact and partly a feature-selection
  artefact. Doesn't change the load-bearing result but shapes the
  interpretation of pair-level metrics.
- **Findings 3 and 4 (Pearson +1.000 at depth, substitutability
  becomes discriminative)** are sanity checks on the harness, not
  cross-layer behavioural claims about specific Polygram pairs.
- **Outcome bucket:** monotonic with plateau. Layer 0 is the dead
  zone; layers 5 and 10 are roughly equivalent.
- **Follow-ups worth running**:
  (a) Repeat with layer-local feature selection — apply the
      projection-similarity selection within each layer's own SAE
      separately, then compare per-pair Polygram predictions to
      per-pair behavioural stats. Resolves the same-index-different-
      feature caveat and lets pair-level results carry weight.
  (b) §4.4 scale-up: 30+ pairs at varied Polygram overlaps, picked
      per-layer, at `blocks.5` or `blocks.10`. Lets us measure the
      Polygram → behavioural-Jaccard correlation slope at a layer
      where ablation-KL is also a usable signal.
  (c) Logit-lens variant of the ablation hook: measure the KL after
      bypassing further blocks (i.e., apply final layernorm + unembed
      directly to the ablated residual stream at the hook layer).
      Tells us how much of the ablation-KL is direct unembed write
      vs downstream propagation.
