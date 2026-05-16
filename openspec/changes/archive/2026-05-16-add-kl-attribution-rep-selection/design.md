## Context

`Compressor._pick_representative` (compression/compressor.py) chooses one cluster member to keep when applying a union-find compression plan. Two modes ship today:

- `n_fires` — sums `pair.n_fires_i` / `n_fires_j` across all pairs touching the cluster, picks the feature with the highest total. Geometry-blind.
- `scale_aware` — combines decoder-norm magnitude with the n_fires score (`_score_scale_aware` in compressor.py). Geometry-aware but still indirect: a feature with a large decoder norm and high firing count is *assumed* to be the cluster's behavioural anchor.

Both proxies are **encoding-geometry approximations** of the question we actually care about: *which feature, if retained alone, preserves the most of the cluster's contribution to downstream model behaviour?*

`BehaviouralValidator` already measures this directly. For every candidate pair `(i, j)`, it records `kl_ablate_i` and `kl_ablate_j` — the KL divergence between the unablated model's output distribution and the model with feature `i` (resp. `j`) zeroed on the held-out token stream. Higher `kl_ablate_f` means feature `f` is more load-bearing for downstream behaviour. The fields are already on `CandidatePair` (`polygram/behavioural/report.py:62`); the validator computes them; the report serialises them; the Compressor reads `validation_report.pairs` to build its cluster plan.

**The recon-aware rep_selection signal is already in scope.** No activation pass-through, no new interface, no new dependency. Just a different reducer over data Compressor already has.

## Goals / Non-Goals

**Goals:**
- Add `rep_selection="kl_attribution"` that picks the cluster rep with maximum `kl_ablate` score, breaking ties by `n_fires` and feature_id.
- Preserve byte-identity for all existing callers (default stays `scale_aware`; explicit `n_fires` and `scale_aware` opt-ins unchanged).
- Surface a clear `ValueError` when the report's behavioural fields are all-NaN (e.g. came through `DecoderGeometryConfirmer`).
- Gracefully fall back to `scale_aware` per-feature when individual `kl_ablate` values are NaN within an otherwise-behavioural cluster.
- Add scenarios pinning the tiebreak rule and the NaN handling.

**Non-Goals:**
- Adding an activation pass-through to Compressor.
- Changing the default `rep_selection`.
- New strategies for `compression.strategy` (zero / merge).
- New fields on `ValidationReport` or `CandidatePair`.
- Auto-selection of rep_selection based on report contents.
- Joint cluster-optimisation (greedy minimisation of total cluster recon loss).

## Decisions

### Decision 1 — Use `kl_ablate_*` as the recon-aware signal, not a synthetic loss

`CandidatePair.kl_ablate_i` and `kl_ablate_j` are already the right signal. They measure, for a feature `f`, "how much does ablating only `f` shift the model's output distribution." Higher value means `f` is more behaviourally load-bearing, so retaining `f` as the cluster's rep preserves more behaviour.

**Alternative considered**: synthesise a per-feature recon score by reconstructing per-token activations from a single feature direction. Rejected — requires Compressor to ingest activations (real interface widening), and would re-do work BehaviouralValidator already did.

**Alternative considered**: use `pearson_activation` as a behavioural signal. Rejected — pearson is *pairwise correlation*, not a per-feature importance score. Useful for clustering, not for ranking cluster members.

### Decision 2 — Aggregate per-feature `kl_ablate` as the mean across pairs

A feature `f` may appear as `pair.i` in some pairs and `pair.j` in others. Each occurrence carries one `kl_ablate_f` measurement. In principle these measurements should all be the same value (since `kl_ablate_f` is single-feature, not pair-local), but in practice they may vary slightly due to measurement noise.

The aggregation rule: `kl_ablate(f) = mean of {pair.kl_ablate_i if pair.i == f else pair.kl_ablate_j for pair in pairs containing f}`. Mean smooths over per-pair measurement noise.

**Alternative considered**: take any single value (assume measurements are exactly equal). Rejected — measurement noise is observable; mean is more robust.

**Alternative considered**: take the max. Rejected — biases toward outlier measurements.

### Decision 3 — Tiebreak first on `n_fires_total`, then on feature_id ascending

Ties in `kl_ablate` are unusual but possible (e.g. two features with very similar firing patterns). The deterministic tiebreak matches the existing `n_fires` rep_selection (which already tiebreaks on feature_id):

1. Higher `n_fires_total` wins — more-fired features have more reliable KL measurements.
2. Lower feature_id wins (cross-platform-deterministic).

**Alternative considered**: tiebreak on `scale_aware` score. Rejected — adds a code path that pulls in W_dec; the n_fires tiebreak is cheaper and the cases where it matters are vanishingly rare in practice.

### Decision 4 — Per-feature NaN falls back to `scale_aware`; all-cluster NaN raises

`kl_ablate` can be NaN for two reasons:
- The feature fires too rarely for the validator's KL measurement to be statistically meaningful (validator's `min_firing_rate` gate produces a NaN signal for borderline cases).
- The confirmation strategy doesn't compute behavioural fields (`DecoderGeometryConfirmer` and `ClusterConfirmer` from `add-confirmation-strategies` populate only the geometric fields; behavioural columns are NaN — see `polygram/confirmation/decoder_geometry.py`).

The two cases warrant different handling:

- **Per-feature NaN within an otherwise-behavioural cluster**: silently fall back to `scale_aware` rep scoring **for that one feature only**. The other cluster members can still use their KL scores; the NaN feature gets a finite scale_aware score and competes on the same axis. Documented in the rep_selection docstring.
- **All-NaN cluster** (every member's `kl_ablate` is NaN): raise `ValueError` with the message `"Compressor: rep_selection='kl_attribution' requires behavioural confirmation; this cluster's kl_ablate values are all NaN (likely came through DecoderGeometryConfirmer or ClusterConfirmer). Use rep_selection='scale_aware' or 'n_fires' for geometry-only reports."` Surfaces the caller's mis-configuration loudly rather than silently degrading.

**Alternative considered**: silently fall back to `scale_aware` for entire all-NaN clusters too. Rejected — masks a real configuration error; the caller asked for behavioural rep selection and the report doesn't have the data.

**Alternative considered**: refuse at `Compressor.__post_init__` time if no pair in the report has non-NaN `kl_ablate`. Rejected — partial-validator runs (e.g. a mix of behavioural and geometric pairs) are conceivable; the per-cluster check is more granular.

### Decision 5 — `kl_attribution` is opt-in; default stays `scale_aware`

The default `CompressionConfig(rep_selection="scale_aware")` is unchanged. Callers explicitly opt in with `CompressionConfig(rep_selection="kl_attribution")`.

**Why opt-in**: byte-identity. The existing test suite, sae-forge byte-equivalence tests, and downstream callers all assume `scale_aware` behaviour. Flipping the default would be a breaking change disguised as a quality improvement.

**Why not deprecate `scale_aware`**: cheap geometric proxy is useful for geometry-only reports (which can't use `kl_attribution`); `scale_aware` and `n_fires` keep paying their rent.

**When `kl_attribution` becomes the default candidate**: after empirical evidence that it Pareto-dominates `scale_aware` across (a) frontier shape post-forge, (b) Compressor wall-time, (c) edge cases (low-firing features, partial-behavioural reports). That decision is a separate openspec change.

### Decision 6 — Document `kl_attribution` semantics in a new capability spec

Rep selection isn't pareto-specific (it applies to `plan()`, `plan_with_target()`, and `plan_pareto()`), so the algorithm spec lives in a new `recon-aware-rep-selection` capability rather than as a delta to `pareto-compression`. The `tuning-config` delta covers only the `rep_selection` field's supported-value list extension; the algorithm itself gets its own spec file with the scenarios pinning behaviour.

**Alternative considered**: fold the algorithm into the `tuning-config` spec under "extended `rep_selection` scenarios". Rejected — bloats `tuning-config` (which is about *configurations*, not *algorithms*), and obscures the algorithm's discoverability for future readers.

## Risks / Trade-offs

- **`kl_attribution` is only as good as `BehaviouralValidator`'s KL measurements.** If the validator runs on a small prompt corpus, the per-feature `kl_ablate` values are noisy; rep selection becomes less reliable. The `--validation-prompts` corpus quality is a load-bearing dependency. Sae-forge's `add-auto-materialise-sweep` (PR #34) makes this knob explicit, partially addressing the issue.

- **NaN fallback to `scale_aware` per feature is silent.** A user who runs `kl_attribution` on a mostly-behavioural report with a handful of low-firing features will get a mixed-criterion cluster representative. The docstring documents this; logging at WARNING level when fallback fires is a future polish but not part of this change.

- **No empirical case for `kl_attribution`'s superiority is bundled.** The proposal is motivated *conceptually* (a behavioural question deserves a behavioural answer, and the data is already in scope) rather than empirically. The K=4-vs-K=8 KL inversion observed in sae-forge PR #33's N=32 Rung4 smoke is **not** load-bearing evidence for this change — that smoke operated in the `degenerate` quality regime (sae-forge PR #35's tagging), where rep choice can't compensate for rank-1-to-4 bases against a 768-dim residual. Pareto-dominance over `scale_aware` on real forge runs is an open research question that this change unblocks but does not answer. The natural test bed is an Axis-4 sweep using sae-forge's `--auto-materialise --rep-selection kl_attribution` (PR #34) filtered to `quality_tier in {"good", "saturated"}` rows (PR #35) — that's where the comparison is meaningful and that's where the change pays off (or doesn't).

- **`kl_attribution` does not joint-optimise across the cluster.** It picks the single best per-feature score; the cluster's *aggregate* recon loss under the chosen rep is not directly minimised. The deferred follow-up (greedy cluster minimisation) is the structural improvement when this matters in practice.
