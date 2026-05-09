# SAE geometry regimes

**Status**: motivation document for the
[`add-sae-geometry-regimes`](../../openspec/changes/add-sae-geometry-regimes/)
OpenSpec change. Captures the five-SAE smoke probe that established
why polygram needed named projection-space profiles and why
modality is not the right selector.

## TL;DR

A five-SAE panel (Whisper × 2, Qwen-Scope, Llama-Scope L0R + L12R)
showed that polygram's pre-v0.2 defaults — calibrated on small
dense LM SAEs at the GPT-2-small scale — collapse on every other
modern SAE we tested. The single empirical predictor is **width ×
d_model**: any SAE with `d_model ≥ ~1K` and `n_features ≥ ~16K`
sits on a near-uniform sphere where Pearson `tier_preservation`
becomes selection-driven noise and the rung-1 binary β-spread
saturates `encoding_suitability_score` at ~1e-5.

## The five-SAE panel

| SAE | training | layer | n_features × d_model | decoder norm (mean ± std) | cosine std | tier_preservation (random subset) |
|---|---|---|---|---|---|---|
| Whisper-tiny enc.b2 (audio) | TopK | mid (b2/4) | 6,144 × 384 | 1.020 ± 0.071 | 0.056 | -0.408 |
| Whisper-large-v1 enc.b16 (audio) | TopK | mid (b16/24) | 20,480 × 1,280 | 0.996 ± 0.043 | 0.028 | -0.000 |
| Qwen-Scope L14 W32K (text 1.7B) | TopK | mid (L14/28) | 32,768 × 2,048 | 1.000 ± 0.000 | 0.035 | +0.297 |
| Llama-Scope L0R 8x (text 8B) | unknown | first (L0/32) | 32,768 × 4,096 | 1.998 ± 0.260 | 0.020 | -0.089 |
| Llama-Scope L12R 8x (text 8B) | JumpReLU | mid (L12/32) | 32,768 × 4,096 | 1.001 ± 0.002 | 0.016 | +0.210 |

All five share:

- **Quasi-uniform sphere** projection geometry: mean off-diagonal
  cosine ≈ 0, cosine std ∈ `[0.016, 0.056]`, real clusters
  appearing only at k≈256.
- **Pearson `tier_preservation` is selection-driven noise**: across
  random / cosine-clustered / anti-clustered subsets it sometimes
  flips sign (Whisper-tiny: -0.41 / +0.27 / -0.40), sometimes
  collapses to zero (Whisper-large random: -0.00), sometimes
  reverses ordering (Llama L0R anti-clustered +0.26 > cos-clustered
  +0.21). Not a fidelity signal on this regime.
- **`encoding_suitability_score` saturates** at 1e-5 to 1e-7
  regardless of `n_clusters` or layer choice.
- **Cancellation efficiency hits 0.999** on the top-|V| pair across
  all five — a signature of the rung-1 k=2 binary β-spread hitting
  its structural floor immediately, not a sign of faithful encoding.

## What the panel eliminates as confounds

The five SAEs span four candidate regime indicators that earlier
framings treated as load-bearing:

- **Modality**: audio (Whisper × 2) and text (Qwen, Llama × 2) both
  land in the same regime. *Modality is not the selector.*
- **Training recipe**: TopK (Whisper × 2, Qwen, Llama L0R) and
  JumpReLU (Llama L12R) both land in the same regime. *Sparsity
  mechanism is not the selector.*
- **Decoder normalization**: strict unit-norm at floating
  precision (Qwen at 0.000 std, Llama L12R at 0.002 std), drifty
  unit-norm (Whisper × 2 at 0.04–0.07 std), and **non-unit-norm**
  (Llama L0R at mean 1.998, range [0.41, 3.14]) all land in the
  same regime. *Decoder normalization is not the selector.*
- **Layer position**: first-layer (Llama L0R) and mid-stack (the
  other four) both land in uniform-sphere. Layer L0 has a heavier
  cosine tail (max 0.826 in a 1000-sample sweep vs 0.075 at L12R)
  — pre-mixing residual-stream features include some near-
  duplicates — but the bulk distribution is still uniform-sphere
  by std and q95. *Layer is not the selector.*

## What predicts the regime

**Width × d_model.** Once an SAE crosses approximately
`(d_model ≥ ~1K) × (n_features ≥ ~16K)`, the decoder rows
distribute near-uniformly on the unit sphere regardless of
training recipe, modality, layer, or whether the trainer
explicitly normalised the decoder. The five-SAE evidence is
unanimous on this axis.

The narrow corner where polygram's pre-v0.2 defaults are calibrated
— small dense LM SAEs at GPT-2-small scale (d_model ≤ 768, ≤24K
features) — is the **exception**, not the rule, in the modern SAE
landscape. Calling `clustered` (the v0.1.0-equivalent default) on
a Qwen-Scope or Llama-Scope SAE silently degrades.

## What the v0.2 profiles do

Two named profiles ship:

- **`clustered`** (default): the v0.1.0-equivalent path. K=2 k-means,
  β = ±0.5 antipodal spread, Pearson `tier_preservation`. Calibrated
  scope: GPT-2-small.
- **`uniform-sphere`**: K≥16 k-means on unit-normalised projections;
  β derived from top-1 PCA-axis coordinate (continuous geometric
  position rather than cluster ordinal); rank-recall@k as fidelity
  (rank-based, bounded `[0, 1]`, doesn't sign-flip on uniform-sphere
  data). Calibrated scope: the five-SAE panel above.

Profile selection is consumer-driven — sae-forge and similar tools
have meta-knowledge of each SAE's pedigree at orchestration time and
pass `profile="uniform-sphere"` for any SAE in the broader regime.

## Open boundary question

We have no SAE in the panel between GPT-2-small (d=768, ≤24K
features) and Qwen3-1.7B (d=2048, 32K features). The exact regime
boundary is therefore unknown — does GPT-2-medium (d=1024) still
sit in `clustered`? Does a 16K-feature Pythia-160M SAE? Out of
scope for v0.2; consumers who hit ambiguity should fall back to
`clustered` (fail loud) rather than guess.

## Reproducing

The raw probe artifacts live under:

- `scratch/whisper_sae/` — Whisper-tiny enc.b2 (cherrvak/topkautoencoder_baseline)
- `scratch/whisper_large_sae/` — Whisper-large-v1 enc.b16 (cherrvak/large_v1_block_16_audioset_topk_16)
- `scratch/qwen_scope/` — Qwen-Scope L14 W32K (Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50)
- `scratch/llama-scope/Llama3_1-8B-Base-L0R-8x/` — Llama-Scope L0R 8x
- `scratch/llama_scope_l12/` — Llama-Scope L12R 8x (suchitg/llama_scope_lxr_8x)

The `scratch/` directory is git-ignored; download via `hf download`
and run the same probe (3-subset selection × tier_preservation,
1000-sample cosine sweep, top-|V| Cancellation) to reproduce the
table.
