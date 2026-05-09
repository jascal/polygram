## Why

A five-SAE smoke probe — Whisper-tiny `encoder.blocks.2` and
Whisper-large-v1 `encoder.blocks.16` (audio, TopK), Qwen-Scope
Qwen3-1.7B layer 14 (text 1.7B, TopK), Llama-Scope Llama3.1-8B
L0R 8x and L12R 8x (text 8B; L0R unknown training, L12R
JumpReLU) — revealed that polygram's current `from_sae_lens`
defaults are calibrated for one specific corner of SAE-space
and collapse outside it.

Findings (full numbers in
`docs/research/sae-geometry-regimes.md`):

- All five SAEs sit on a quasi-uniform sphere: mean off-
  diagonal cosine ≈ 0, std 0.016–0.056. Decoder normalization
  varies wildly (Llama L0R has mean norm 2.0 with range
  [0.41, 3.14]; Qwen-Scope is at floating-point precision
  unit-norm; Whisper sits in between) yet all five give the
  same uniform-sphere cosine signature.
- Pearson `tier_preservation` is selection-driven noise across
  the board — Whisper-tiny flips -0.41 / +0.27 / -0.40 across
  random / clustered / anti-clustered subsets; Llama-Scope
  L0R goes -0.09 / +0.21 / +0.26 (anti-clustered higher than
  cos-clustered, geometrically backwards). Not a fidelity
  signal on this regime.
- `encoding_suitability_score` saturates at 1e-5 to 1e-7 on
  all five, regardless of `n_clusters` or layer choice.
- Cancellation efficiency hits 0.999 on the top-|V| pair
  across all five — a signature of polygram's `k=2` binary
  β-spread hitting its structural floor immediately rather
  than evidence of a faithful encoding.

The five-SAE panel **eliminates four candidate regime
indicators as confounds**: modality (audio + text both land
here), training recipe (TopK + JumpReLU both land here),
decoder normalization (strict + drifty + non-unit all land
here), and layer position (first + mid both land here, with
first-layer just having a heavier cosine tail). The single
predictor that survives is **width × d_model**: any SAE with
`d_model ≥ ~1K` and `n_features ≥ ~16K` lands in the uniform-
sphere regime.

Today's defaults match small dense LM SAEs at the GPT-2-small
scale (d=768, ≤24K features) — locked in by the §4.4
calibration and the resolved cross-encoding stability spike
(per project memory). That narrow corner is the *exception*,
not the rule, in the modern SAE landscape. They mismatch
**every other SAE we've measured**, including text SAEs at
1.7B and 8B scale.

Polygram has *one* implicit geometric profile when there are at
least two empirical regimes. Downstream consumers (sae-forge
today, future image/video-SAE consumers later) have meta-
knowledge of the SAE's pedigree at orchestration time and would
benefit from selecting the appropriate profile explicitly.

## What Changes

- Introduce a `GeometricProfile` concept: a named bundle of a
  `KnobAssignment` strategy + a `GeometricFidelity` metric +
  recommended `from_sae_lens` defaults (`n_clusters`,
  `gamma_range`).
- Ship two named profiles, **named after projection-space
  geometry rather than modality or architecture**:
  `"clustered"` (the new default alias for today's behaviour —
  k=2 k-means, β = ±0.5 antipodal spread, Pearson
  `tier_preservation`; calibrated on small dense LM SAEs,
  GPT-2-small specifically — d_model ≤ 768, ≤24K features)
  and `"uniform-sphere"` (k≥16 k-means, β derived from
  PCA-axis coordinates rather than cluster ordinal, rank-
  recall@k as fidelity; calibrated on five SAEs spanning audio
  + text, TopK + JumpReLU, and the full normalization range —
  the predictor is `d_model ≥ ~1K` and `n_features ≥ ~16K`).
- Add `from_sae_lens(..., profile: str | GeometricProfile | None =
  None)`. `None` and `"clustered"` both resolve to today's
  behaviour exactly (no observable change for existing callers).
- Extend `SelectionReport` with a `profile: str` field and a
  `geometric_fidelity: float | None` field that records the
  profile's metric output. Retain `tier_preservation` as a
  field; it stays populated for `clustered` and is `None`
  for profiles that don't define a Pearson-style metric.
- Expose a `polygram.geometry` module with the registry,
  `KnobAssignment` protocol, and `GeometricFidelity` protocol so
  third-party consumers (sae-forge) can register custom profiles
  without forking.
- Document the consumer contract: callers that know the SAE's
  pedigree (small text-LM SAE / large LM SAE / TopK audio SAE /
  future image-video SAE collections) SHOULD pass `profile=...`;
  callers that don't get the `clustered` default. Document
  explicitly that the appropriate profile for **large LM SAEs
  like Qwen-Scope is `uniform-sphere`, not `clustered`** —
  modality alone is not a reliable selector.

## Capabilities

### New Capabilities
- `geometry-regimes`: the `GeometricProfile` concept, the named
  profile registry, the `KnobAssignment` strategy protocol, and
  the `GeometricFidelity` metric protocol. Ships two built-in
  profiles (`clustered`, `uniform-sphere`) and the
  third-party registration API.

### Modified Capabilities
- `sae`: `from_sae_lens` accepts an optional `profile` argument
  selecting a `GeometricProfile`; `SelectionReport` gains
  `profile` and `geometric_fidelity` fields; the existing
  `tier_preservation` field is retained (populated by the
  clustered profile, `None` for profiles that don't compute
  a Pearson fidelity).

## Impact

- Affected modules: `polygram/sae_import.py` (knob-assignment
  becomes a strategy plug-in), `polygram/config.py`
  (`SAEImportConfig` gains `profile`), new `polygram/geometry/`
  package (registry + protocols + bundled profiles).
- Affected specs: `sae` (delta), new `geometry-regimes` spec.
- No breaking changes: `from_sae_lens` keyword surface is purely
  additive, all existing kwargs (`assign_gamma`, `gamma_range`,
  `n_clusters`, `cluster_assignments`, `config`) keep their
  semantics. When `profile` is omitted, behaviour is byte-for-byte
  identical to v0.1.0.
- Downstream: sae-forge can pass `profile="uniform-sphere"` for
  audio SAEs *and* for large LM SAEs (Qwen-Scope, plausibly
  Gemma-Scope and Llama-Scope at width). Small dense LM SAEs
  (GPT-2-small style) keep the `clustered` default. The
  modality tag is *not* the selector; pedigree (size class +
  training regime + decoder normalization) is.
- Relationship to the stashed `add-qwen-scope-loader` work:
  that change stalled on a separate concern (TopK fidelity
  ceiling — Polygram's 8-feature cap < Qwen-Scope's k=50,
  so the gate is structurally vacuous at the Dictionary level).
  The Phase-1.5 Qwen-Scope probe surfaced an *additional*
  problem the loader would have hit: even if `from_qwen_scope`
  had landed, the resulting Dictionary would have collapsed
  into noise on the `tier_preservation` / suitability-score
  axes the same way audio SAEs do. This change addresses that
  second problem directly; the TopK fidelity ceiling remains
  out of scope here.
- Out of scope (separate follow-ups): automatic profile detection
  from projection geometry; image/video profiles (no data yet);
  raising the rung-1 8-feature cap; resolving the TopK fidelity
  ceiling; modifying `encoding_suitability_score` to be
  profile-aware.
