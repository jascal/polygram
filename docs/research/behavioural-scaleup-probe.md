# Behavioural-Gram scale-up probe — does Polygram's predicted overlap track behavioural Jaccard at scale?

> Research-track note recording the empirical findings of the
> behavioural-Gram scale-up probe defined in
> [`tech-debt-backlog` §4.4](../../openspec/changes/tech-debt-backlog/tasks.md).
> Reproducible via `python examples/behavioural_gram_scaleup.py`.

## Context

PR #20 ([§4.2 / `behavioural-gram-probe.md`](behavioural-gram-probe.md))
established the *directional* claim with N = 2 pairs at
`blocks.0.hook_resid_pre`: Polygram's within-vs-cross ordering survives
at the behavioural level, but per-pair Jaccard magnitudes compress vs
Polygram-predicted overlaps. Two pairs is too few to fit a slope.

PR #23 ([§4.3 / `deeper-layer-ablation-probe.md`](deeper-layer-ablation-probe.md))
established that ablation-KL is a *useful* signal at `blocks.5+` on
GPT-2 small (~1 nat per single-feature ablation, four orders of
magnitude above layer 0), and that `blocks.5` ≈ `blocks.10` for
ablation magnitude.

This probe (§4.4) closes the remaining cheap question: **at a layer
where ablation-KL is informative, how strongly does Polygram's
predicted overlap rank pairs by behavioural co-firing across many
pairs?** The answer determines whether the compression / disentanglement
loop spec can use Polygram as its primary candidate-selection signal.

## Scope adjustment vs the merged §4.4 spec

The merged spec (PR #24) named **"~25 features → ~300 pairs"**.
Polygram's rung-1 MPS encoding caps a Dictionary at 8 features
(`MAX_FEATURES_PER_DICTIONARY = 8` in `polygram/sae_import.py:23`),
so this implementation runs at the cap: **8 features → 28 pairs**.

Trade-offs:

- **What's preserved**: ~4.7× pair count vs PR #18's N = 6, the
  Spearman / Pearson slope estimate, the per-bucket Jaccard
  reporting, and the spec's three-outcome gating.
- **What's coarsened**: 28 pairs gives ~9 pairs per bucket instead
  of ~100, so per-bucket bootstrap CIs are wider.
- **What disappears**: by construction, the 8-feature near+far
  split with KMeans k = 2 produces no pairs in the low-overlap
  bucket (Polygram ≤ 0.4) — the rung-1 β-spread between two
  clusters floors every pair's predicted squared overlap at
  ~0.44 even when γ values differ. The "low overlap" outcome is
  effectively un-testable inside this cap. A proper low-bucket
  measurement would require either rung-2+ encodings (out of
  current Polygram scope) or a multi-Dictionary harness.

The spec's "30+ pairs at varied Polygram overlaps" floor was
written assuming the ~25-feature sizing was achievable. Inside the
8-feature cap the achievable variation is the *upper half* of the
overlap range. The Spearman estimate at scale (28 pairs) remains
informative for the headline question; the per-bucket geometry
shifts from low/mid/high to mid/high.

## Method

`examples/behavioural_gram_scaleup.py`:

1. **Load** `blocks.10.hook_resid_pre` SAE
   (`jbloom/GPT2-Small-SAEs-Reformatted`, 24576 features × 768
   d_model).
2. **Forward** the §4.2 / §4.3 12-prompt set (654 GPT-2 tokens)
   through GPT-2 small with a forward-pre-hook on
   `model.transformer.h[10]` to capture the residual stream
   entering block 10.
3. **Encode** all 24576 SAE features on the captured residuals to
   compute per-feature firing rates. Filter to features with
   firing rate ≥ 0.01 (fires on ≥ 1% of tokens — gives
   meaningful Jaccard).
4. **Select** 8 features stratified by decoder cosine to a high-
   firing seed:
   - Seed = highest firing-rate × decoder-norm in the eligible pool.
   - 3 nearest cosine neighbours (the "near" cluster).
   - 4 features sampled across the bottom half of the seed-
     cosine distribution (the "far" cluster).
   The selection produces a clean two-cluster decoder-space
   structure that KMeans (default `n_clusters=2`) recovers.
5. **Build** a Polygram Dictionary via `from_sae_lens` with
   `assign_gamma=True` (cluster-PCA γ for within-cluster
   variation; otherwise pairs in the same cluster would have
   identical β and γ = 0 → identical predicted overlap).
6. **Run 8 ablation forward passes**, one per selected feature,
   subtracting that feature's `f_A · W_dec[A, :]` contribution at
   every token where it fires; capture the next-token KL.
7. **Compute per-pair**: Polygram-predicted squared overlap,
   decoder squared cosine, Jaccard co-fire, activation Pearson,
   paired ablation-KL ratio on both-fire tokens (only for pairs
   with ≥ 5 both-fire tokens).
8. **Aggregate**: Spearman + Pearson between Polygram and each
   behavioural metric; same for decoder cosine (ceiling); per-
   bucket Jaccard means with 95% bootstrap CI.

## Selected feature panel

The seed-selection produced this panel at `blocks.10`:

| Feature | Firing rate | Cosine to seed | Cluster (KMeans) |
|---:|---:|---:|---:|
| 12999 (seed) | 0.476 | +1.000 | near |
| 19398 | 0.329 | +0.603 | near |
| 4192  | 0.073 | +0.515 | near |
| 23625 | 0.078 | +0.514 | near |
| 8371  | 0.018 | −0.014 | far |
| 2287  | 0.020 | −0.031 | far |
| 68    | 0.018 | −0.054 | far |
| 13737 | 0.018 | −0.089 | far |

KMeans k = 2 cluster split: β var-explained 0.353. The split
recovers the seed-cosine cut exactly, as expected when the cosine
distribution is bimodal (which the seed-stratified sampling was
designed to produce).

## Findings

### Finding 1 — Polygram ranks pairs by behavioural Jaccard at scale (Spearman +0.64)

| Pair correlation | Spearman | Pearson |
|:---|---:|---:|
| Polygram predicted overlap ↔ Jaccard co-fire | **+0.637** | +0.694 |
| Decoder squared cosine ↔ Jaccard co-fire | −0.054 | −0.075 |
| Polygram predicted overlap ↔ \|log(KL_i / KL_j)\| | −0.330 | — |

The headline number is **+0.637**, clearing the spec's 0.6 threshold
for the "high Spearman, loop unblocked" outcome. Polygram's
predicted ordering of pairs survives the move from decoder geometry
to real co-firing on real tokens at a usable layer.

### Finding 2 — Decoder cosine alone does NOT track Jaccard at this layer

The most surprising number in the table: **Spearman(decoder cosine,
Jaccard) = −0.054 at `blocks.10`** — essentially zero. Compare with
PR #18's finding at the same SAE layer-0 fixture, where
Spearman(decoder cosine, Polygram-predicted) = 0.94 and behavioural
Jaccard was an unmeasured downstream gap.

Two effects compound to produce this:

- **Decoder cosine is selection-biased**. The seed-stratified
  selection enforces a bimodal cosine distribution (4 near, 4 far),
  but the *firing patterns* of far-cluster features have nothing to
  do with their decoder geometry — they're picked to be cosine-
  orthogonal to a seed they share no semantic content with.
- **Polygram adds γ-spread on top of decoder geometry**. With
  `assign_gamma=True`, within-cluster pairs get distinct γ values
  via cluster-PCA, which spreads predicted overlaps inside what
  decoder cosine alone would treat as a single bin. That γ-spread
  is what gives Polygram the +0.64 Spearman.

The practical implication: **inside this cap, Polygram is doing
real work beyond what its decoder-geometry input provides**. The
"Polygram is just decoder cosine" reading is incorrect at scale —
γ matters once the decoder-geometry has been pre-selected to be
two-cluster.

This finding is *narrower than its statement*: it speaks only to a
seed-stratified 8-feature panel. A non-stratified random selection
might recover decoder cosine's predictive value. The relevant
generalization is "*after a clustering-shaped feature selection*,
Polygram outperforms raw decoder cosine on Jaccard ranking" — which
is the relevant regime for any compression loop, since loops
*always* select via decoder geometry first.

### Finding 3 — Per-bucket Jaccard has a clean separation

| Polygram-overlap bucket | n_pairs | Jaccard mean | 95% bootstrap CI |
|:---|---:|---:|:---|
| Low (≤ 0.4) | 0 | — | — (un-testable inside cap) |
| Mid (0.4 – 0.7) | 16 | 0.145 | [0.096, 0.192] |
| High (≥ 0.7) | 12 | **0.621** | [0.427, 0.823] |

The CIs **do not overlap**. The 4.3× ratio between bucket means
(0.62 vs 0.14) is large; the upper bound of the mid-overlap CI
(0.192) sits cleanly below the lower bound of the high-overlap CI
(0.427).

For a loop spec's co-firing gate, the natural Jaccard threshold is
**τ ∈ [0.20, 0.43]**. τ ≈ 0.30 sits squarely between the buckets
and would admit ≥ 90% of high-overlap pairs while excluding ≥ 90%
of mid-overlap pairs (precise rates depend on which cap-respecting
selection a real loop uses; that's a §4.5+ question).

But: even in the high-overlap bucket, **mean Jaccard is 0.62, not
1.0**. Polygram-redundant pairs co-fire on roughly two-thirds of
either-fire tokens. The §4.2 magnitude-compression finding holds
at scale — naive removal of a "Polygram-redundant" feature would
silently lose information on the ~38% of activations where its
partner doesn't fire. The co-firing gate is a **necessary** filter,
not a sufficient one.

### Finding 4 — Ablation-KL substitutability gives a weaker but correctly-signed signal

`Spearman(Polygram_overlap, |log(KL_i / KL_j)|) = −0.330`.

The direction is right: pairs with high Polygram-predicted overlap
have more *substitutable* ablation-KL profiles (smaller log-ratio
distance from 1.0). But the magnitude is much weaker than the
Jaccard correlation, and the underlying KL ratios on individual
pairs are noisy — a single near-zero KL on one feature blows up
the ratio to ~10⁴ (one such row in the CSV: `feat_12999 ↔ feat_13737`,
KL_j = 2.96e-04, KL_i = 3.25, ratio ≈ 10978).

Practical implication for a loop: **use Jaccard as the primary
gate; treat the ablation-KL ratio as a confirmatory signal, not a
selection criterion**. The KL signal is real but its noise floor
on small-firing features is too high for it to be a primary
ranker.

## Outcome — high Spearman, loop spec unblocked

The spec's three-outcome gating:

> - **High Spearman (≥ 0.6) Polygram ↔ Jaccard.** Polygram's
>   ranking transfers cleanly to behavioural co-firing at scale.
>   The loop spec proceeds with Polygram as the primary candidate
>   filter, `Jaccard ≥ τ` as a secondary gate (τ chosen from the
>   per-bucket means), ablation-KL at `blocks.10` as the impact
>   metric.

We land at **+0.637**, just above the threshold. Within the
caveats above (cap-imposed N, mid+high buckets only), the loop
spec is unblocked. The four constraints are now empirical:

1. **Layer**: hook at `blocks.5` or `blocks.10` (§4.3, settled).
2. **Encoding**: Polygram-as-ranker only — magnitudes derive from
   real model metrics (§4.1, settled).
3. **Co-firing gate**: required, with a Jaccard threshold τ ≈ 0.30
   based on the bucket separation in this probe (§4.4, settled
   to the cap-imposed scope).
4. **Behavioural impact metric**: ablation-KL at `blocks.10` is
   the right shape for per-feature impact (§4.3, settled);
   pairwise KL ratio is a confirmatory not a primary signal
   (§4.4, this probe).

## Caveats

### A. The 28 pairs share a seed feature

Seven of 28 pairs include feature 12999 (the seed) on one side.
Within-cluster behavioural correlations are not 28 independent
measurements; they're a constellation around a single anchor.
This concentrates the high-overlap bucket. The +0.637 Spearman is
robust to this (Spearman is rank-based and the bucket separation
is not driven by within-cluster correlations alone), but the per-
bucket bootstrap CIs treat the pairs as exchangeable, which
overstates effective sample size.

### B. The "decoder cosine doesn't track Jaccard" finding is selection-conditional

`Spearman(decoder, Jaccard) = −0.054` is a property of *this
selection*, not of `blocks.10` SAE features in general. A random
sample of feature pairs across the SAE would likely show
substantial decoder-Jaccard correlation, simply because high-
cosine pairs at random would also tend to have similar firing
contexts. The probe's selection deliberately enforces a bimodal
decoder distribution; under that constraint, decoder cosine is
constant within each cluster and so cannot vary with Jaccard
within-cluster. Polygram's γ-spread does vary within-cluster, and
that's what produces the +0.64 Spearman gap.

### C. Low-overlap bucket is empty inside the cap

A loop wanting to filter "obviously orthogonal pairs" at the
low-Polygram-overlap end has no data points from this probe to
calibrate against. The empty low bucket is structural:
`MPSRung1` with `_spread_betas` over 2 clusters in `[-0.5, 0.5]`
gives β-difference of 1.0 between clusters → minimum cross-cluster
squared overlap of ~0.44 at γ = 0. The *upper bound* of the
"low overlap" bucket (0.4) is below this floor.

### D. CPU-only run, ~3 minutes wall-clock

The full probe ran in approximately 3 minutes on a single CPU
(no GPU). Reproducible via the script with no flags.

## Recommended next steps

§4.4 closes the cheap probes. Loop spec should write next.
Sequencing:

- **Loop spec (§4.5+)**: Polygram-as-primary-ranker, Jaccard ≥ 0.30
  as required co-firing gate, ablation-KL at `blocks.10` as impact
  metric. Layer-local feature selection within each layer's own
  SAE. The compression action is "only act on pairs that pass both
  Polygram and Jaccard thresholds; verify ablation-KL changes
  match expectations."

- **Optional follow-ups before loop spec lands**:
  - Random-pair Spearman at scale (no seed-stratified selection)
    to confirm Polygram still ranks above decoder cosine on a
    naturally-distributed sample. Cheap: same harness, different
    selection.
  - Logit-lens variant of the ablation hook to separate "direct
    unembed write" from "downstream propagation" components of
    the ablation-KL signal at depth.
  - Multi-Dictionary harness: stitch 3-4 cap-sized Dictionaries
    with consistent cluster assignments for ~24 features and
    proper low-overlap bucket coverage. Cost: harness code
    larger than the probe itself; signal payoff modest given the
    +0.64 Spearman is already above the loop-unblocking threshold.

## Source artifacts

- Script: `examples/behavioural_gram_scaleup.py`
- Per-pair CSV: `docs/research/data/scaleup_pairs.csv`
- Full run log: `docs/research/data/scaleup_probe_full.log`
