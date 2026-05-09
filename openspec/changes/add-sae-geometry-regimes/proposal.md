## Why

Phase-1 audio-SAE probe (Whisper-tiny encoder.blocks.2 and
Whisper-large-v1 encoder.blocks.16, both TopK SAEs) showed that
audio SAE features sit in a quasi-uniform sphere — mean off-diag
cosine ≈ 0, std ≈ 0.03–0.06, real clusters appearing only at
k≈256. Polygram's current `from_sae_lens` defaults bake in the
*opposite* assumption (k=2 binary clustering, β spread over
`(-0.5, 0.5)`, Pearson `tier_preservation` as fidelity stat),
which collapses on uniform-sphere geometries: `tier_preservation`
flipped sign across feature-selection strategies (-0.41 random,
+0.27 clustered, -0.40 anti-clustered) and
`encoding_suitability_score` saturated around 1e-5 regardless of
n_clusters or layer.

These defaults are correct for text SAEs — the GPT-2-small
calibration and the resolved cross-encoding stability spike (per
project memory) lock that in as the production baseline. The
mismatch is that Polygram has *one* implicit geometric profile
when there's now evidence of at least two distinct regimes, and
downstream consumers (sae-forge today) have meta-knowledge of
the SAE's modality and would benefit from selecting the
appropriate profile explicitly.

## What Changes

- Introduce a `GeometricProfile` concept: a named bundle of a
  `KnobAssignment` strategy + a `GeometricFidelity` metric +
  recommended `from_sae_lens` defaults (`n_clusters`,
  `gamma_range`).
- Ship two named profiles: `"text-clustered"` (the new default
  alias for today's behaviour — k=2 k-means, β = ±0.5 antipodal
  spread, Pearson `tier_preservation`) and `"uniform-sphere"`
  (k≥16 k-means, β derived from PCA-axis coordinates rather than
  cluster ordinal, rank-recall@k as fidelity).
- Add `from_sae_lens(..., profile: str | GeometricProfile | None =
  None)`. `None` and `"text-clustered"` both resolve to today's
  behaviour exactly (no observable change for existing callers).
- Extend `SelectionReport` with a `profile: str` field and a
  `geometric_fidelity: float | None` field that records the
  profile's metric output. Retain `tier_preservation` as a
  field; it stays populated for `text-clustered` and is `None`
  for profiles that don't define a Pearson-style metric.
- Expose a `polygram.geometry` module with the registry,
  `KnobAssignment` protocol, and `GeometricFidelity` protocol so
  third-party consumers (sae-forge) can register custom profiles
  without forking.
- Document the consumer contract: callers that know the SAE's
  modality (text/audio/image/video) SHOULD pass `profile=...`;
  callers that don't get the text-clustered default and the
  current behaviour.

## Capabilities

### New Capabilities
- `geometry-regimes`: the `GeometricProfile` concept, the named
  profile registry, the `KnobAssignment` strategy protocol, and
  the `GeometricFidelity` metric protocol. Ships two built-in
  profiles (`text-clustered`, `uniform-sphere`) and the
  third-party registration API.

### Modified Capabilities
- `sae`: `from_sae_lens` accepts an optional `profile` argument
  selecting a `GeometricProfile`; `SelectionReport` gains
  `profile` and `geometric_fidelity` fields; the existing
  `tier_preservation` field is retained (populated by the
  text-clustered profile, `None` for profiles that don't compute
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
  audio SAEs. No change required for its existing text-SAE flows.
- Out of scope (separate follow-ups): automatic profile detection
  from projection geometry; image/video profiles (no data yet);
  raising the rung-1 8-feature cap; modifying
  `encoding_suitability_score` to be profile-aware.
