## Context

`from_sae_lens` (`polygram/sae_import.py`) today bakes one
geometric assumption into every call: features cluster into a
small number of antipodal groups, β = ±0.5 spans the cluster
axis, γ is a small in-cluster PCA perturbation, and Pearson
correlation between projection cosines and Polygram Gram is the
fidelity stat. That calibration is locked in for *small dense
LM SAEs* — specifically GPT-2-small — by the §4.4 calibration
and the resolved cross-encoding stability spike (per project
memory).

A three-SAE smoke probe established the calibration scope is
narrower than originally framed:

| SAE | n_features × d_model | decoder norm | cosine std | tier_pres random |
|---|---|---|---|---|
| Whisper-tiny enc.b2 (audio) | 6,144 × 384 | 1.020 ± 0.071 | 0.056 | -0.408 |
| Whisper-large-v1 enc.b16 (audio) | 20,480 × 1,280 | 0.996 ± 0.043 | 0.028 | -0.000 |
| Qwen-Scope L14 W32K (text) | 32,768 × 2,048 | 1.000 ± 0.000 | 0.035 | +0.297 |

(See `scratch/whisper_sae/`, `scratch/whisper_large_sae/`,
`scratch/qwen_scope/` for the raw artifacts and the conversation
history for the methodology.)

All three SAEs sit on a quasi-uniform sphere: mean off-diagonal
cosine ≈ 0, decoder rows unit-norm (Qwen-Scope to floating-
point precision; Whisper to ~3% std), real clusters appearing
only at k≈256. On the audio SAEs the Pearson
`tier_preservation` flips sign with selection strategy; on
Qwen-Scope it stays positive but small (+0.30 random vs +0.37
clustered) — slightly more recoverable structure than audio,
but the same regime. `encoding_suitability_score` saturates at
~1e-5 to 1e-4 across all three, regardless of `n_clusters` or
layer choice. Cancellation efficiency hits 0.999 on the top-|V|
pair across all three — a signature of polygram's k=2 binary
β-spread hitting its structural floor immediately rather than
evidence of a faithful encoding.

The crucial reframing: **the calibration mismatch is not text-
vs-audio.** Qwen-Scope is a text SAE, and it lands in the same
regime as the audio SAEs, not the same regime as polygram's
GPT-2-small calibration baseline. The empirical split runs
along **scale + decoder normalization + training regime**:

- Today's defaults match small dense LM SAEs (GPT-2-small d=768,
  ≤24K features, dense ReLU / JumpReLU).
- A second regime — TopK or strict-unit-norm decoders, large
  width (≥16K features), large d_model (≥1K) — covers
  audio SAEs and large LM SAEs alike.

Downstream consumer sae-forge has meta-knowledge of each SAE's
pedigree at orchestration time. Polygram doesn't need to
*infer* the regime from projection geometry — it needs to
*expose* named regimes that the consumer selects explicitly.

### Relationship to the stashed `add-qwen-scope-loader` work

The earlier `spec/add-qwen-scope-loader` branch (stashed,
unmerged) stalled on a separate concern: the **TopK fidelity
ceiling**. Polygram caps Dictionaries at 8 features (rung-1 =
3-qubit register); Qwen-Scope's TopK k is 50 or 100. A
Dictionary built from Qwen-Scope features will always hold
fewer features than k, so the TopK gate is structurally vacuous
at the Dictionary level. The proposed `TopKMPSRung1` marker
was provenance-only; the stash design called this out
honestly.

The Phase-1.5 Qwen-Scope probe surfaced an *additional*
problem that loader would have hit: the `tier_preservation`-
collapses-into-noise / suitability-score-saturates issue this
change addresses. Two separate problems would have stacked.
This change resolves the second; the TopK fidelity ceiling
remains out of scope and is a candidate for a future
`add-topk-fidelity` change after the encoding cap is lifted.

## Goals / Non-Goals

**Goals:**

- Make polygram's SAE-handling layer parameterised by a named
  geometric regime, not by a single hardcoded calibration.
- Preserve v0.1.0 behaviour byte-for-byte when no profile is
  passed, so existing text-SAE callers (including the
  `examples/`, `tests/`, and `polygram analyze` CLI flows) need
  no change.
- Expose a third-party-friendly registry so sae-forge — and
  future image/video-SAE consumers — can register custom
  profiles without forking.
- Land an audio-calibrated profile (`uniform-sphere`) that
  sae-forge can use today on Whisper-style SAEs.

**Non-Goals:**

- Automatic regime detection from projection geometry. Phase-1
  data shows simple stats (mean cosine, k-means tightness) do
  flag uniform-sphere reliably, but auto-selection couples
  polygram more tightly to specific calibrations and steps on
  the consumer's meta-knowledge. Defer.
- Image / video profiles. We have no Phase-1 data for them; the
  registry makes adding them later cheap.
- Raising the rung-1 8-feature cap. Audio data shows real
  clusters at k≈256, so the cap will eventually need to lift,
  but that's an encoding-side change touching Q-Orca register
  width — separate decision, separate change.
- Profile-aware `encoding_suitability_score`. Today's score is
  rung-1-specific; rewriting it across profiles is a follow-up
  once we have a second profile in production for a quarter.
- Behavioural-validator generalisation (HostModel /
  EffectMetric). That was Phase-2 in the original audio
  investigation; it's a separate, larger change targeting the
  text-only `BehaviouralValidator` / `Regrower` / `EpochCompressor`
  pipeline.

## Decisions

### Profile is a bundle, not three separate kwargs

We could add three independent kwargs to `from_sae_lens`:
`knob_assignment=`, `fidelity_metric=`, `default_n_clusters=`.
Rejected because:

- The three knobs co-vary by design (e.g. `pca_axis` β makes no
  sense paired with Pearson `tier_preservation`). Bundling
  prevents nonsense combinations.
- Consumers want to say "this is an audio SAE", not "use
  pca_axis β with rank-recall@k fidelity at k=16". The bundle
  is the right abstraction at the consumer surface.
- A registry of named bundles is much easier to document and
  to extend in third-party code than three independent
  registries.

The cost is a small amount of plumbing inside the bundle that
mirrors what would otherwise be three constructor args.

### Strategy resolution order: kwargs > config > profile > strategy internal defaults

Today's `Cancellation` / `Compressor` config-vs-kwarg precedence
is "per-field kwargs > config > dataclass defaults"
(`polygram/config.py`). The profile slots in *between* config and
strategy internal defaults — explicit per-field kwargs and
explicit `SAEImportConfig` values both win over profile
defaults. This keeps the existing override surface intact.

Rejected: putting profile defaults *above* `SAEImportConfig`
fields. That would silently change behaviour for callers who
set `n_clusters=2` on a config and then later pass
`profile="uniform-sphere"` — the explicit `2` should win.

### Profile names follow geometry, not modality

Earlier drafts named the profiles `text-clustered` and
`uniform-sphere`. The Qwen-Scope probe falsified the modality-
flavoured naming: Qwen-Scope is a text SAE that lands in the
uniform-sphere regime, not in the GPT-2-small regime. Calling
the default `text-clustered` would invite future callers to
pass it on Qwen-Scope / Gemma-Scope / Llama-Scope by analogy
with "text", silently degrading.

Renamed to `clustered` (the default, calibrated on small dense
LM SAEs — GPT-2-small specifically) and `uniform-sphere` (the
broader regime: audio + Qwen-Scope, plausibly other large LM
SAEs). The names describe the *projection-space property the
profile assumes*, not the source modality.

Documented contract:

- `clustered` is appropriate when the SAE has recoverable
  small-k cluster structure visible in cosine geometry. The
  empirical scope is GPT-2-small at ≤24K features.
- `uniform-sphere` is appropriate when features sit on a near-
  uniform sphere with cosine std ≤ ~0.06 and `tier_preservation`
  is selection-driven noise. The empirical scope so far is
  audio TopK SAEs + Qwen-Scope.
- For SAEs outside both characterised regimes, register a
  custom profile or fall back to `clustered` (fail loud rather
  than silently mis-calibrate).

### `clustered` is the default, named, and observable

Three options for "what does omitting `profile=` mean":

1. Resolve to `clustered` at call time and record
   `report.profile = "clustered"`. **Chosen.**
2. Leave `report.profile = None` when not passed.
3. Resolve to a synthetic `"default"` alias.

Option 1 wins because consumers downstream (sae-forge,
analysis tools) can branch unambiguously on the profile name
without special-casing `None`. The cost is one concept's worth
of "implicit default ≡ named default" mapping that has to be
documented; the benefit is symmetry between the GPT-2-small
and Qwen-Scope / audio call sites.

### β = `pca_axis` strategy for `uniform-sphere`

For the uniform-sphere profile, β assignment options were:

- **`pca_axis`** (chosen): project each feature onto the top-1
  PCA component of the centered selected-subset's projection
  vectors, rescale to `(-0.5, 0.5)`. β becomes a continuous
  geometric coordinate. Cluster identity is still tracked
  (k-means at k=16 by default) for tier reporting, but β no
  longer encodes cluster ordinal.
- `orthogonal`: every feature gets a unique β quantised over
  `(-0.5, 0.5)`. Equivalent to today's k=N k-means path on
  Phase-1 audio data; doesn't add information.
- `cosine_to_seed`: pick a seed feature, β = cosine to seed.
  Coupling β to one feature is brittle.

`pca_axis` aligns β with the dominant geometric variation in the
selected subset. On the Phase-1 cosine-clustered audio subset
(8 features, mean within-cluster cosine 0.38) this gives β a
real spread (>0.6 of the (-0.5, 0.5) range) instead of the
saturating ±0.5 ordinal that today's k=2 path produces.

### `rank_recall_at_k` for `uniform-sphere` fidelity

Pearson on n=28 off-diagonal pairs (8-feature setup) is
high-variance and direction-flipping on uniform-sphere data
(empirically demonstrated in Phase-1). Three alternatives:

- **`rank_recall_at_k`** (chosen): top-k Polygram-Gram pairs ∩
  top-k cosine pairs / k. Rank-based, bounded `[0, 1]`,
  direction unambiguous (higher is better). At k=4 on the audio
  fixture this gives a stable signal where Pearson does not.
- Spearman correlation: still rank-based, but as variance-
  prone as Pearson on n=28.
- KL-divergence between sorted-overlap distributions: harder to
  interpret, no clear "good" threshold.

`k = max(3, len(features) // 2)` keeps the metric meaningful
across the 4-to-8-feature operating range.

### Centralise resolution in `from_sae_lens`, not `SAEImportConfig`

`SAEImportConfig.profile` defaults to `None` (not to a resolved
`GeometricProfile`). Resolution happens once, at
`from_sae_lens` entry, against the live registry. This means a
serialised config from polygram v0.1.x (no `profile` field at
all) deserialises cleanly under v0.2 and resolves to
clustered. It also means downstream packages that register
custom profiles after import don't have to time their
registration against config construction.

### Strategy lives outside `polygram.sae_import`

New package `polygram/geometry/`:

```
polygram/geometry/
  __init__.py        # public API: GeometricProfile, register_profile,
                     # get_profile, available_profiles, clustered,
                     # uniform_sphere
  profile.py         # GeometricProfile dataclass
  protocols.py       # KnobAssignment, GeometricFidelity protocols
  registry.py        # register/get/available; built-ins registered
                     # in __init__.py to avoid circular imports
  clustered.py  # the v0.1.0-equivalent strategy + fidelity
  uniform_sphere.py  # the audio-calibrated strategy + fidelity
```

`polygram/sae_import.py` keeps its public surface (`from_sae_lens`,
`SAEFeatureRecord`, `SelectionReport`, `load_sae_safetensors`,
`load_toy_sae`) and gains a small dispatch shim that hands off
to the active profile's `KnobAssignment` and
`GeometricFidelity`. The cluster_assignments / from_labels
precedence paths stay in `sae_import.py` (they're upstream of
strategy dispatch).

## Risks / Trade-offs

[Risk] **Default-equivalence regression**: extracting today's
hardcoded path into a `clustered` strategy could subtly
diverge (e.g. a different k-means seed flow, different float
rounding in β spread). **Mitigation**: ship a frozen golden
fixture (`tests/fixtures/golden_clustered.json`)
generated from the v0.1.0 baseline, and assert byte-equality of
`Dictionary.features` and `SelectionReport` fields in a
regression test. The spec scenario already names this fixture.

[Risk] **Third-party profiles can break analysis-layer
assumptions**. `polygram.analysis.encoding_suitability_score`
hardcodes the rung-1 closed-form `(M, V)` decomposition;
profiles that move features into a different parameterisation
(e.g. a future image profile that wants α ≠ 0) would silently
get nonsensical suitability scores. **Mitigation**: this change
does NOT alter `analysis.suitability_score`. Profiles in this
change keep the rung-1 (α=0, β/γ/φ-only) parameterisation. A
follow-up change can make the suitability score profile-aware
once a non-rung-1-compatible profile is on the table.

[Risk] **`tier_preservation` field stays around as a partial
metric**. Keeping it for backwards-compat means callers who
generic-dispatch on it might think the field is universally
populated. **Mitigation**: spec is explicit that
`tier_preservation` is the v0.1.0 Pearson and is `None` outside
the clustered profile (and any opt-in third-party reuses).
The new `geometric_fidelity` is the field consumers should
read going forward; the README needs an updated note pointing
to `geometric_fidelity` as the canonical fidelity stat.

[Risk] **Documentation drift**. README, `polygram/__init__.py`
re-exports, the `Choosing an encoding` README section, and
`docs/research/` cross-references all point at today's single
calibration. **Mitigation**: the tasks file explicitly enumerates
each doc surface; a single PR lands all of them.

[Trade-off] **No automatic profile detection.** Consumers who
forget to pass `profile=` get clustered output regardless
of whether their SAE is text. On a Whisper SAE this manifests
as a low `tier_preservation` and a saturating
`encoding_suitability_score` — diagnosable, not silent failure,
but not as nice as auto-detection would be. We accept this in
v0.2 because the consumer (sae-forge) already has the
meta-knowledge to pass the right profile, and auto-detection
is a separate research question (what cosine-distribution
threshold separates the regimes? does it generalise?).

## Migration Plan

1. **No external migration required for v0.1 callers.** Omitting
   `profile=` reproduces v0.1.0 behaviour byte-for-byte.
2. **Internal call sites** (`examples/`, `tests/`, CLI flows,
   `polygram analyze`) are updated in the same PR to either pass
   `profile="clustered"` explicitly (where it documents
   intent) or leave it implicit (where it's unimportant). No
   logic changes.
3. **Downstream consumer (sae-forge)** lands a parallel change
   on its side: at SAE-record construction time, check the
   modality tag and pass
   `profile=("uniform-sphere" if modality in {"audio"} else "clustered")`
   into `from_sae_lens`. Out of scope for this polygram change;
   tracked as a sae-forge-side ticket.
4. **CHANGELOG**: bump to 0.2.0 since the API surface grows
   (additive, not breaking). Note `geometric_fidelity` as the
   new canonical fidelity field; `tier_preservation` is retained
   but profile-scoped.

Rollback: revert is safe — the public surface only adds new
optional kwargs and new dataclass fields. Downstream code that
read `tier_preservation` continues to work on clustered
output.

## Open Questions

- **Should `available_profiles()` distinguish built-in vs
  third-party-registered?** Probably not in the v0.2 surface;
  add only if a sae-forge debugging need surfaces it.
- **Does `uniform_sphere` need a different default
  `gamma_range`?** Phase-1 used `(-0.25, 0.25)` (today's
  default) and γ via per-cluster PCA worked fine inside the
  k=16 buckets. Leaving it unchanged for v0.2; revisit if
  audio-SAE follow-ups indicate γ-range tuning matters.
- **Should the registry persist third-party registrations
  across `importlib.reload`?** Current design says no — re-
  importing `polygram.geometry` resets the registry to the
  built-ins. sae-forge will need to call `register_profile`
  in its package init. Acceptable for v0.2.
