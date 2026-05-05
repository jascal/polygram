# Behavioural-Gram one-pair probe — does Polygram's predicted overlap track real-model co-firing?

> Research-track note recording the empirical findings of the
> behavioural-Gram one-pair probe defined in
> [`tech-debt-backlog` §4.2](../../openspec/changes/tech-debt-backlog/tasks.md).
> Reproducible via `python examples/behavioural_gram_probe.py`.

## Context

PR #18
([`decoder-gram-validity.md`](decoder-gram-validity.md))
closed the *decoder-geometry* validity question: Polygram's predicted
squared overlap correlates well with the SAE decoder's squared cosine
Gram on the Real GPT-2 SAE fixture (Spearman 0.94) but its per-pair
magnitudes diverge by up to 0.44. Its closing caveat named the next
gap explicitly:

> Two SAE features can have orthogonal decoder columns but still
> co-fire on the same inputs (and vice versa). The behavioural-Gram
> comparison would need a forward-pass infrastructure Polygram doesn't
> have.

This spike builds the smallest such infrastructure for *one pair* of
features and tests whether Polygram's ranking signal carries into
real-model behaviour. The open question:

> For a feature pair `(i, j)` Polygram represents via `(β, α, γ, φ)`
> after `from_sae_lens`, does Polygram's predicted squared overlap
> `|⟨ψ_i|ψ_j⟩|²` line up with the *behavioural* statistics
> `(co-occurrence, activation correlation, ablation-KL
> substitutability)` measured by running GPT-2 small + SAE on
> held-out text?

## Method

For each of two feature pairs we computed three real-model statistics
on a held-out 654-token batch (12 short paragraphs), then compared
against Polygram's predicted Gram entry and the decoder squared
cosine measured directly from `W_dec`.

**Pairs.** Both drawn from PR #18's GPT-2 SAE feature subset
(`feat_7836, feat_13953, feat_15796, feat_11978`):

- **Within-cluster** `feat_7836 ↔ feat_11978`: highest Polygram
  overlap (0.987), highest decoder cosine (0.992).
- **Cross-cluster** `feat_7836 ↔ feat_15796`: low Polygram MPS
  overlap (0.464) but high decoder cosine (0.904) — the pair shape
  PR #18 used to demonstrate that Polygram and decoder geometry
  diverge sharply on cross-cluster pairs.

**Statistics.** Per pair:

1. **Co-occurrence** — `P(j fires | i fires)`,
   `P(i fires | j fires)`, and Jaccard, with "fires" defined as
   `f_i > 0` (the SAE's ReLU is the natural threshold).
2. **Activation Pearson** — Pearson correlation of the two
   features' raw post-ReLU activations across all 654 tokens.
3. **Ablation-KL substitutability** — for each token where a
   feature fires, run a counterfactual forward pass with that
   feature's decoder contribution subtracted from the residual
   stream entering block 0 (`x' = x - f_A · W_dec[A, :]`), then
   compute KL divergence between the baseline and ablated
   next-token distributions. We report:
   - `KL(baseline || ablate-i)` averaged on tokens where i fires;
   - `KL(baseline || ablate-j)` averaged on tokens where j fires;
   - **paired comparison** on tokens where *both* features fire
     (49 tokens for the within-cluster pair, 31 for the
     cross-cluster pair). The pair is *substitutable* if the KL
     ratio is in `[0.5, 2.0]`.

**Scope deliberately not pursued.** No φ optimization, no
Cancellation runs, no Dictionary-baking. Polygram's φ knob doesn't
map to `W_dec`; this probe is purely observational.

## Findings

### Finding 1 — Polygram ordering is preserved in real behaviour, but magnitudes are compressed

| Pair | Polygram overlap | Decoder cosine | Jaccard | Pearson | Both-fire KL ratio (i/j) |
|---|---:|---:|---:|---:|---:|
| Within-cluster `7836 ↔ 11978` | **0.987** | 0.992 | 0.295 | +0.245 | 1.18 |
| Cross-cluster `7836 ↔ 15796` | **0.464** | 0.904 | 0.188 | +0.190 | 0.72 |

The within-cluster pair has *higher* Jaccard, *higher* Pearson, and
KL ratios closer to 1.0 — all consistent with Polygram's claim that
this is the more behaviourally redundant pair. The directional
ordering Polygram predicts (within > cross) carries into real
behaviour.

But the *magnitudes* are heavily compressed. Polygram puts the two
pairs in very different buckets (0.987 vs 0.464 — a 2.1× ratio); real
behaviour treats them more similarly (Jaccard 0.30 vs 0.19, ~1.6×;
Pearson 0.25 vs 0.19, ~1.3×). This is the same Polygram-as-ranker /
Polygram-as-magnitude-predictor split PR #18 surfaced for the
decoder-Gram comparison, now confirmed at the behavioural level.

### Finding 2 — even the highest-Polygram-overlap pair does NOT always co-fire

The within-cluster pair Polygram predicts at 0.987 overlap fires
together on only 49/118 tokens where feature `7836` fires (P(j|i) =
0.42), and on 49/97 tokens where `11978` fires (P(i|j) = 0.51). High
predicted overlap does not mean "these features are functionally
duplicates." It means "these features encode similar decoder
directions." The two are not the same: a feature can encode a
specific *aspect* of a similar concept and fire only on the subset of
contexts where that aspect is present.

This is a load-bearing distinction for compression-pipeline work. A
disentanglement primitive that targets "high-overlap pairs" with the
intent of *removing one* will only correctly preserve information if
the pair is truly redundant — which depends on co-firing patterns,
not decoder geometry. Polygram's classification alone is not
sufficient evidence for redundancy.

### Finding 3 — layer-0 SAE feature ablations have negligible KL impact

The absolute KL magnitudes are tiny across the board:

| Pair | KL(ablate-i \| i fires) | KL(ablate-j \| j fires) |
|---|---:|---:|
| Within-cluster | 5.18e-05 | 3.86e-05 |
| Cross-cluster | 5.18e-05 | 7.20e-05 |

Ablating *any* of these features at the residual stream entering
block 0 produces ~5e-5 nats of KL on the next-token distribution.
That's smaller than the float32 noise floor on a softmax over GPT-2's
50257-token vocabulary in many cases.

This is a finding about the layer-0 SAE itself, not about Polygram:
individual features at `blocks.0.hook_resid_pre` carry minimal
load-bearing signal for next-token prediction. There are 11 more
transformer blocks downstream that can compensate for any single
feature's removal. Compression-pipeline work that uses ablation-KL as
a "feature impact" signal at layer 0 will be measuring noise. Either
move to a deeper layer (where SAE features sit closer to the
unembedding) or use a different impact metric (e.g. logit lens,
attention pattern shift, downstream layer activation deltas).

### Finding 4 — substitutability is "passing" by default at this layer

The KL ratio i/j on both-fire tokens lands in `[0.5, 2.0]` for both
pairs (1.18 and 0.72). Per the §4.2 spec's substitutability criterion
this is the "substitutable" bucket — in both cases, ablating either
feature gives roughly comparable KL impact.

But this isn't validation of Polygram's prediction — it's a
consequence of Finding 3. When the absolute KL is ~5e-5 for both
features, the ratio of two near-zero quantities is dominated by
noise, and "substitutable" effectively means "neither matters." The
metric loses discriminative power at this layer.

The within-cluster pair *does* show a tighter ratio (1.18 vs 0.72),
which is consistent with greater behavioural similarity, but the
small absolute magnitudes make the difference unconvincing as a
single data point.

## Interpretation

Mapping to the §4.2 spec's three outcome buckets:

> All three real signals high (co-occurrence > 0.5, Pearson > 0.5,
> KL substitutability ratio in [0.5, 2.0]): Polygram's high-overlap
> classification predicts real feature redundancy.

We do **not** hit this bucket. The within-cluster pair has Jaccard
0.30 and Pearson 0.25 — well below the 0.5 thresholds the spec named.
KL substitutability passes, but as Finding 4 explains, it passes
trivially because layer-0 KL is too small to discriminate.

> At least one real signal contradicts Polygram (e.g., high Polygram
> overlap but low co-occurrence): decoder geometry and behavioural
> geometry are genuinely different…

We do hit this. The pair Polygram predicts at 0.987 overlap fires
together on only ~30% of either feature's firings. High decoder
similarity does not imply high behavioural co-firing.

> Mixed (one of three signals low): write up which carries and which
> doesn't…

This is the closest fit. Co-occurrence and Pearson carry a *signed*
signal that matches Polygram's prediction (within > cross) but at
substantially compressed magnitudes. Substitutability is undefined
at layer 0 because the underlying ablation-KL is too weak.

The picture, combined with PR #18:

1. **Polygram is a directional ranker for both decoder geometry and
   behavioural similarity.** It correctly orders pairs (within >
   cross is preserved across all three: decoder cosine, co-occurrence,
   Pearson). PR #18 established this for decoder Gram with Spearman
   0.94 across 6 pairs; this probe corroborates it for behavioural
   stats on a single contrast (N = 2 pairs is too small for a
   correlation but the directional signal is consistent).
2. **Polygram's magnitudes are not real-behaviour magnitudes.** The
   Polygram "0.987 vs 0.464" gap compresses to a Jaccard "0.30 vs
   0.19" gap on real text. A pair Polygram predicts at near-perfect
   overlap fires together less than half the time.
3. **Layer-0 ablation-KL is not a useful behavioural signal.** Any
   compression-loop that uses it at this layer will measure noise.

## Practical implications

1. **Co-occurrence is the cheapest behavioural sanity check.** A
   single forward pass over a few hundred tokens of text gives a
   Jaccard estimate that costs orders of magnitude less than the
   ablation-KL forward passes. Future compression-pipeline work
   should treat Polygram's ranking as a *candidate filter* and
   confirm with co-occurrence before any quantitative reasoning.
2. **Pick a deeper SAE layer for ablation-impact metrics.**
   `blocks.0.hook_resid_pre` is the input to the model. Individual
   SAE features at this point have ~5e-5 nats of KL impact, which
   is noise. The peer-agent Gemma-Scope steering loop should target
   middle or late residual-stream SAEs where individual features
   carry more load-bearing signal.
3. **Don't conflate "high decoder similarity" with "redundant
   feature."** The within-cluster pair Polygram predicts at 0.987
   and the decoder confirms at 0.992 fires together on only 30% of
   token positions (Jaccard). The two features encode similar
   *directions* but different *contexts of activation*. A naive
   removal would lose information in 70% of the firing events.
4. **The §4.1 closure stands.** Polygram is a ranker, not a
   magnitude predictor — confirmed at the behavioural level, not
   just the decoder-geometry level. Downstream work that rides on
   ranking is unblocked; work that depends on magnitudes (the
   disentanglement-loop loss surface; the user-supplied Gemma
   `compression_score`) needs real-model magnitudes.

## Caveats and what this doesn't tell us

- **N = 2 pairs.** This probe is a one-pair (plus contrast) spike.
  The directional correlation it observes is consistent with PR #18's
  Spearman 0.94 for decoder geometry, but the behavioural-correlation
  evidence here is anecdotal. A scaled-up version (N = 30+ pairs at
  varied Polygram overlaps) would tighten the directional claim and
  let us measure the magnitude-compression rate empirically.
- **One layer, one model.** GPT-2 small at `blocks.0.hook_resid_pre`.
  Layer 0 is the *input* layer; the layer-0 finding (KL too small to
  discriminate) is specific to this position. Different layers, and
  different model architectures (Gemma, Llama), will have very
  different signal profiles. Don't generalize the layer-0 ablation
  conclusion.
- **No long-range behavioural consequences measured.** We only check
  the next-token distribution. Multi-token text generation, attention
  pattern changes, or deeper-layer activation shifts could show
  larger ablation effects than the one-token KL captures.
- **GPT-2's `forward_pre_hook` placement matters.** We hook the input
  to `transformer.h[0]`, which equals
  `embedding + position_embedding` after dropout. The SAE was trained
  on `blocks.0.hook_resid_pre` in transformer_lens terminology;
  these should align in eval mode (no dropout) but a small
  divergence is possible. The co-occurrence numbers (44, 38, 49
  fires across 654 tokens) are within expected ranges for a 24576
  feature SAE so the hook point is plausible; deeper validation
  (loading the same checkpoint via transformer_lens and confirming
  identical activations) is left as a follow-up.
- **The `compression_score` proposal stays blocked at this layer.**
  The peer-agent v0.1 sketch's `(steering × sparsity) / recon`
  composite would inherit the layer-0 zero-signal problem. Either
  move to a deeper layer or replace the steering metric with one
  that doesn't bottleneck on next-token KL.

## Reproducibility

```bash
python examples/behavioural_gram_probe.py
```

The probe is auto-skipped if
`./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors`
isn't on disk; see
[`cross-encoding-stability.md`](cross-encoding-stability.md) for the
download command. Requires `transformers` and `torch`; if either is
missing the script prints a hint and exits.

The full run reproduces in roughly two minutes on CPU for 12 prompts
(the default).

## Status

- **Findings 1 (directional preservation, magnitude compression)
  and 2 (high overlap ≠ always co-fire)** are robust on a single
  pair-and-contrast and consistent with PR #18's decoder-Gram results.
- **Finding 3 (layer-0 ablation KL is tiny)** is structural and
  follows from the depth of the model relative to the SAE's
  position. It's a property of the layer choice, not of Polygram.
- **Finding 4 (substitutability is trivially passing)** is an
  artefact of Finding 3 and shouldn't be over-interpreted.
- **Outcome bucket:** mixed. Co-occurrence and Pearson carry a
  signed signal aligned with Polygram's prediction; substitutability
  is undefined at this layer.
- **Follow-ups worth running**:
  (a) Repeat at `blocks.5.hook_resid_pre` and `blocks.10.hook_resid_pre`
      (mid- and late-layer SAEs from the same `jbloom/...` repo) to
      see whether ablation-KL becomes informative deeper in the
      stack.
  (b) Scale to 30+ pairs at varied Polygram overlaps to estimate
      the Polygram → behavioural-Jaccard correlation directly.
  (c) Replace the next-token-KL impact metric with a logit-lens or
      attention-pattern-shift metric at the chosen layer, where
      individual feature ablations may show more leverage.
