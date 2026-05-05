# Cross-encoding stability — does rung-1 triage agree with HEA?

> Research-track note recording the empirical findings of the
> cross-encoding stability spike defined in
> [`tech-debt-backlog` §3.1](../../openspec/changes/tech-debt-backlog/tasks.md).
> Reproducible via `python examples/cross_encoding_stability.py`.

## Context

Both [`add-sharing-graph-triage`](../../openspec/changes/archive/2026-05-04-add-sharing-graph-triage)
and [`add-batch-experiment`](../../openspec/changes/archive/2026-05-04-add-batch-experiment)
ride on the rung-1 closed-form `(M, V, structural_floor, cancellation_gap)`
decomposition that the
[`analysis-triage-layer`](../../openspec/changes/archive/) shipped. That
decomposition is exact under `MPSRung1`. Under `HEA_Rung2` it has no
analytic backing — `Cancellation.structural_floor` raises
`NotImplementedError` outside the canonical 2-φ rung-1 shape (see
[`polygram/cancellation.py`](../../polygram/cancellation.py:268)).

The open question:

> Does a feature pair classified as "good sharing candidate" or
> "must separate" under `MPSRung1` stay in that bucket under
> `HEA_Rung2(depth=2)` on the same `(β, α, γ, φ)` configuration?

If MPS and HEA disagree on real data, the closed-form rung-1 predictions
are telling us about the *encoding* rather than the SAE's intrinsic
geometry — which would be a load-bearing finding before any
compression-pipeline work.

## Method

For each of three fixtures, we built two `Dictionary` instances with
the same features (same names, clusters, β/α/γ/φ values) but different
encodings: `MPSRung1()` and `HEA_Rung2(depth=2)`. The HEA instance's
features carry `theta=None`, so `_default_hea_theta(...)` synthesizes a
θ tensor from the four scalar knobs — the documented spike layout in
[`polygram/dictionary.py`](../../polygram/dictionary.py).

Three fixtures of increasing realism:

1. **Animals.** 4 hand-crafted features in 2 clusters of 2, identical
   to the bundled rung-1 example — controlled, known geometry.
2. **Toy SAE.** Features `[0, 1, 4, 5]` from
   `tests/fixtures/toy_sae.json` (4 features, 8-dim projections,
   `assign_gamma=True`).
3. **Real GPT-2 SAE.** Features `[7836, 13953, 15796, 11978]` from
   `jbloom/GPT2-Small-SAEs-Reformatted`'s `blocks.0.hook_resid_pre`
   layer (24576-feature SAE, 768-dim residual stream,
   `assign_gamma=True`). Same projection-similarity-selected set that
   surfaced the [`--assign-gamma` finding](../../README.md#loading-from-safetensors).

For each fixture we ran `triage_dictionary(...)` against both
encodings, compared per-pair `(current_overlap, structural_floor, V)`,
and built sharing + separation graphs at threshold = 0.0 and at the
default thresholds (0.5 and 0.2 respectively) to compare kept-edge
sets.

## Findings

### Finding 1 — within-cluster pair overlaps are encoding-invariant

For every within-cluster pair across all three fixtures, MPS and HEA
report the same `current_overlap` to within numerical noise (`Δ ≤ 1e-4`):

| Fixture | Within-cluster pair | MPS `current` | HEA `current` | Δ |
|---|---|---:|---:|---:|
| Animals | dog_poodle ↔ dog_beagle | 0.9997 | 0.9999 | +0.0001 |
| Animals | bird_hawk ↔ bird_sparrow | 0.9997 | 0.9999 | +0.0001 |
| Toy SAE | dog_poodle ↔ dog_beagle | 0.9388 | 0.9388 | 0.0000 |
| Toy SAE | hawk_red ↔ hawk_cooper | 0.9610 | 0.9610 | 0.0000 |
| Real SAE | feat_7836 ↔ feat_11978 | 0.9870 | 0.9870 | 0.0000 |
| Real SAE | feat_13953 ↔ feat_15796 | 0.9388 | 0.9388 | 0.0000 |

This is striking. The within-cluster Gram entry is determined by the
sibling pair's `(β, α, γ, φ)` configuration in a way that survives
swapping rung-1 MPS for depth-2 HEA. (The structural floor differs
slightly more — up to ±0.034 on Animals — because that's a function
of the off-diagonal V too.)

### Finding 2 — cross-cluster pair overlaps differ systematically

For every cross-cluster pair across all three fixtures, HEA reports a
*higher* `current_overlap` than MPS:

| Fixture | Cross-cluster pair | MPS `current` | HEA `current` | Δ |
|---|---|---:|---:|---:|
| Animals | dog_poodle ↔ bird_hawk | 0.6202 | 0.7686 | +0.1484 |
| Animals | dog_beagle ↔ bird_sparrow | 0.6212 | 0.7691 | +0.1479 |
| Toy SAE | dog_poodle ↔ hawk_cooper | 0.4319 | 0.7320 | +0.3001 |
| Toy SAE | dog_beagle ↔ hawk_cooper | 0.6094 | 0.7696 | +0.1602 |
| Real SAE | feat_7836 ↔ feat_15796 | 0.4641 | 0.7449 | +0.2808 |
| Real SAE | feat_15796 ↔ feat_11978 | 0.5473 | 0.7666 | +0.2193 |

HEA is "smoothing" cross-cluster overlaps upward into a narrow band
around 0.75, while MPS retains wider variation tied to per-feature γ.

A second pattern is even stronger: under HEA, cross-cluster pairs
with the same `|Δβ|` converge to nearly identical values regardless of
the γ differences between the features. On the Real SAE:

```
HEA cross-cluster overlaps: {0.7449, 0.7449, 0.7666, 0.7666}
MPS cross-cluster overlaps: {0.4641, 0.5473, 0.6351, 0.6949}
```

HEA produces only two distinct values across four pairs; MPS produces
four distinct values. The HEA depth-2 ansatz is bounded by β-spread —
γ contributes less leverage than under rung-1 MPS. This is a property
of `depth=2`, not of HEA in general.

### Finding 3 — kept-edge classifications agree

At every threshold and on every fixture, the sharing-graph and
separation-graph kept-edge sets *agree* between MPS and HEA:

| Fixture | Kind | Threshold | MPS edges | HEA edges | Common |
|---|---|---:|---:|---:|---:|
| Animals | sharing | 0.5 (default) | 0 | 0 | 0 |
| Animals | sharing | 0.0 | 0 | 0 | 0 |
| Animals | separation | 0.2 (default) | 6 | 6 | 6 |
| Animals | separation | 0.0 | 6 | 6 | 6 |
| Toy SAE | sharing | 0.5 | 0 | 0 | 0 |
| Toy SAE | separation | 0.2 | 6 | 6 | 6 |
| Real SAE | sharing | 0.5 | 0 | 0 | 0 |
| Real SAE | separation | 0.2 | 6 | 6 | 6 |

Note: the sharing-graph edge sets are empty across the board at the
default thresholds because every within-cluster pair has
`current_overlap > 0.93` and every cross-cluster pair has
`structural_floor > FLOOR_BLOCK = 0.5` — both gates zero the weight.
This is a property of the fixtures, not the encodings.

### Finding 4 — suitability scores agree numerically

Encoding suitability is dominated by the `(1 − max_pairwise_overlap)`
factor, which equals `(1 − max_within_cluster_overlap)` because every
within-cluster pair in our fixtures sits at ≥ 0.94 overlap. Since
within-cluster overlaps are encoding-invariant (Finding 1), so is the
suitability score:

| Fixture | MPS score | HEA score |
|---|---:|---:|
| Animals | 0.0000 | 0.0000 |
| Toy SAE | 0.0028 | 0.0028 |
| Real SAE | 0.0010 | 0.0010 |

## Interpretation

**The algorithm's discrete classifications are encoding-stable; the
continuous magnitudes are encoding-dependent.** Specifically:

- The output of `build_sharing_graph` and `build_separation_graph` —
  *which pairs cross which threshold* — agrees between MPS and HEA on
  every fixture we tested.
- The output of `triage_dictionary` — *the per-pair `(M, V,
  structural_floor)` triple* — does NOT agree between MPS and HEA on
  cross-cluster pairs. Magnitudes differ by up to 0.30.
- Within-cluster pair magnitudes happen to agree to within numerical
  noise; this looks like a property of how `(β, α, γ, φ)` enter both
  encodings via similar Ry-axis machinery in the inner rung. Worth
  understanding analytically as a follow-up but not critical for
  practice.

## Practical implications

1. **Closed-form rung-1 triage is a sound cheap pre-filter for the
   "which pairs do I study" question.** The kept-edge classifications
   it produces agree with depth-2 HEA on the fixtures we tested.
   `BatchExperiment(top_k=K)` can ride on top of either encoding and
   produce the same pair selection.
2. **Don't carry per-pair magnitudes across encodings.** If your
   downstream experiment uses HEA, regenerate `M`, `V`, and
   `structural_floor` with HEA before drawing quantitative conclusions
   like "this pair has 23% phase headroom". A 0.30 magnitude drift on
   cross-cluster `current_overlap` is enough to break those numbers.
3. **HEA depth=2 has bounded expressivity on cross-cluster pairs.**
   When two cross-cluster pairs share `|Δβ|`, depth-2 HEA tends to
   collapse them to nearly the same Gram entry regardless of γ
   variation. This isn't necessarily wrong — it might genuinely
   reflect what depth-2 ansätze can resolve — but it means HEA at
   depth=2 is *less informative* than rung-1 MPS for ordering
   cross-cluster pairs by compression difficulty. Higher-depth HEA
   should recover the per-feature γ leverage; testing that is a
   follow-up.

## Caveats and what this doesn't tell us

- **Three fixtures, all 4-feature.** The classifications could drift on
  larger feature sets where the threshold gates land more
  marginally. The kept-edge agreement we observed is strongest evidence
  for fixtures where both encodings push every pair clearly above or
  below the gate.
- **Default thresholds aren't validated.** Sharing threshold 0.5 and
  separation threshold 0.2 produced the all-or-nothing edge sets above
  on every fixture. A workload where the gates land in the
  "ambiguous" middle would be a stronger test.
- **No ground-truth comparison.** This experiment compares two
  encodings to each other, not either encoding to actual SAE
  behaviour on text. The question "does Polygram's compressibility
  prediction track ablation impact on a real transformer" is the
  next research-track step and is what
  [`docs/research/spec-disentanglement-loop.md`](spec-disentanglement-loop.md)
  reaches for.

## Reproducibility

```bash
python examples/cross_encoding_stability.py
```

If `./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors`
isn't present, the third fixture is automatically skipped. To download
it (~144 MB):

```bash
hf download jbloom/GPT2-Small-SAEs-Reformatted \
    --include="blocks.0.hook_resid_pre/sae_weights.safetensors" \
    --local-dir ./scratch/real-sae
```

The full run reproduces in under a second on a laptop.

## Status

- **Finding 1, 3, 4** are robust across our three fixtures and align
  with what the algebra suggests.
- **Finding 2** is the most interesting one and warrants the
  follow-up: confirm the depth-vs-γ-leverage relationship by re-running
  at `HEA_Rung2(depth=4)` and `(depth=8)`, see whether cross-cluster
  magnitudes recover MPS-like variation.
- **The encoding-stability question itself** — at the
  kept-edge-classification level — appears settled in the favourable
  direction for the v0 toolchain: closed-form rung-1 triage and HEA
  depth-2 agree on *which* pairs to study, even when they disagree on
  the magnitudes attached to each.
