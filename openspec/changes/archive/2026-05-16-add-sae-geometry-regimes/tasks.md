## 1. Scaffold the geometry package

- [x] 1.1 Create `polygram/geometry/` package with `__init__.py`, `profile.py`, `protocols.py`, `registry.py`, `clustered.py`, `uniform_sphere.py`
- [x] 1.2 Define `GeometricProfile` frozen dataclass in `profile.py` with fields `name`, `knob_assignment`, `geometric_fidelity`, `default_n_clusters`, `default_gamma_range`
- [x] 1.3 Define `KnobAssignment` and `GeometricFidelity` `Protocol`s in `protocols.py` plus a `KnobAssignmentResult` dataclass for the strategy return shape
- [x] 1.4 Implement `register_profile`, `get_profile`, `available_profiles` in `registry.py` with a module-level `_REGISTRY: dict[str, GeometricProfile]`; `register_profile` raises `ValueError` on duplicate names
- [x] 1.5 Wire `polygram/geometry/__init__.py` to import the two built-in profiles and call `register_profile` for each at import time

## 2. Extract the v0.1.0 path into the clustered profile

- [x] 2.1 Move today's `_kmeans`, `_spread_betas`, `_centroids`, `_variance_explained`, `_gamma_via_cluster_pca` helpers from `polygram/sae_import.py` into `polygram/geometry/clustered.py` (or import them from there) — preserve their signatures and float behaviour
- [x] 2.2 Implement `ClusteredKnobAssignment.assign(...)` that wraps those helpers and returns a `KnobAssignmentResult` with `cluster_method="kmeans"`
- [x] 2.3 Move `_tier_preservation` (today in `sae_import.py`) into `clustered.py` as `TierPreservationFidelity.compute(...)` returning `float | None`
- [x] 2.4 Implement `clustered() -> GeometricProfile` factory with `default_n_clusters=2`, `default_gamma_range=(-0.25, 0.25)`
- [x] 2.5 Generate a frozen golden fixture `tests/fixtures/golden_clustered.json` from the v0.1.0 baseline on the toy SAE: capture full `Dictionary.features` (name, cluster, beta, gamma, phi), `report.cluster_assignments`, `report.beta_variance_explained`, `report.tier_preservation`
- [x] 2.6 Add regression test `tests/test_clustered_golden.py` asserting byte-equality of v0.2 output against the golden fixture for a deterministic fixture call. The test docstring SHALL state that the fixture is regenerated **only on intentional default changes** to the `clustered` profile (renamed kwargs, retuned β-spread, etc.) and never as a side effect of refactoring; if a future change needs to ship multiple golden sets, version the filename (`golden_clustered_v2.json`) rather than overwriting.

## 3. Implement the uniform-sphere profile

- [x] 3.1 Implement `UniformSphereKnobAssignment.assign(...)` in `uniform_sphere.py`: k-means on unit-normalised projections with `n_init>=4`, β via top-1 PCA component of the centered selected subset rescaled into `(-0.5, 0.5)`, γ via per-cluster PCA when `assign_gamma=True`; `cluster_method="pca_axis"`; `beta_variance_explained` defined as the fraction of selected-subset variance captured by the top-1 PCA component (not the cluster centroids)
- [x] 3.2 Implement `RankRecallAtKFidelity.compute(...)` with `k = max(3, len(features) // 2)`; returns `None` when fewer than `k+1` off-diagonal pairs exist
- [x] 3.3 Implement `uniform_sphere() -> GeometricProfile` factory with `default_n_clusters=16`, `default_gamma_range=(-0.25, 0.25)`
- [x] 3.4 Add unit tests for `UniformSphereKnobAssignment` (β span ≥ 60% of `(-0.5, 0.5)` on the audio fixture) and `RankRecallAtKFidelity` (returns float in `[0, 1]` on a synthetic clustered case)

## 4. Wire profile dispatch into from_sae_lens

- [x] 4.1 Add `profile: str | GeometricProfile | None = None` kwarg to `polygram.from_sae_lens` (`polygram/sae_import.py`)
- [x] 4.2 Add `profile: str | None = None` field to `polygram.config.SAEImportConfig`; preserve `to_dict` / `from_dict` round-trip behaviour (`profile` serialises as a string, never a `GeometricProfile` instance)
- [x] 4.3 Implement profile resolution at `from_sae_lens` entry: kwarg > config > registry default (`clustered`); resolve string names via `get_profile`
- [x] 4.4 Refactor the k-means path to dispatch to `profile.knob_assignment.assign(...)`; keep `cluster_assignments` and `from_labels` precedence paths upstream of strategy dispatch (unchanged)
- [x] 4.5 Always invoke `profile.geometric_fidelity.compute(...)` after building the Dictionary, regardless of which cluster-assignment path was taken
- [x] 4.6 Add `profile: str` and `geometric_fidelity: float | None` fields to `SelectionReport`; populate `tier_preservation` only when the active profile's fidelity is the v0.1.0 Pearson (i.e. `clustered`)
- [x] 4.7 Verify `SelectionReport` JSON serialisation round-trips the new fields

## 5. Tests for the new dispatch surface

- [x] 5.1 Add `tests/test_geometry_registry.py`: built-ins registered at import, duplicate registration raises `ValueError`, `get_profile("nonexistent")` raises `KeyError` with available-names message
- [x] 5.2 Add `tests/test_from_sae_lens_profile_dispatch.py`: omitting `profile=` is bit-equal to passing `profile="clustered"`; per-field `n_clusters` overrides profile default; `cluster_assignments` bypasses strategy but profile fidelity is still computed
- [x] 5.3 Add `tests/test_uniform_sphere_profile.py`: full from_sae_lens call with `profile="uniform-sphere"` on the audio-style fixture; assert β span, `report.profile == "uniform-sphere"`, `report.tier_preservation is None`, `report.geometric_fidelity` in `[0, 1]`
- [x] 5.4 Add an audio-style fixture under `tests/fixtures/`: a small synthetic SAE whose decoder rows are unit-normalised quasi-orthogonal vectors with one tight 8-feature cluster (mean within-cluster cosine ≈ 0.4) — captures the Phase-1 audio signature without shipping a 200 MB checkpoint
- [x] 5.5 Re-run the full polygram test suite and confirm zero regressions on text-SAE flows

## 6. Public API and re-exports

- [x] 6.1 Re-export `GeometricProfile`, `register_profile`, `get_profile`, `available_profiles`, `clustered`, `uniform_sphere` from `polygram.__init__`
- [x] 6.2 Update `polygram/__init__.__all__` to include the new names
- [x] 6.3 Confirm `polygram.config.SAEImportConfig` re-export still works after the new field is added

## 7. Documentation

- [x] 7.1 Update `README.md` "SAE import" section: document `profile=` kwarg, the two built-in profiles, when to use which, and the consumer-meta-knowledge framing. Explicitly state the resolution order in both the README and the `from_sae_lens` public docstring: **per-field kwarg > `SAEImportConfig.profile` > registry default (`clustered`)** — surface this in caller-facing prose, not just in the spec scenarios.
- [x] 7.2 Update `README.md` "Choosing an encoding" section: cross-link profiles to encodings (clustered + MPSRung1 is the validated combo; uniform-sphere is provisional pending behavioural validation)
- [x] 7.3 Add `docs/research/sae-geometry-regimes.md` capturing the Phase-1 audio findings (cosine distributions on Whisper-tiny block-2 and Whisper-large-v1 block-16, n_clusters sweep, why Pearson fidelity collapses on uniform-sphere data) — references the conversation that motivated this change
- [x] 7.4 Update `CHANGELOG.md` for v0.2.0: new `profile` kwarg (additive, default unchanged), new `geometric_fidelity` field on `SelectionReport`, retained-but-scoped `tier_preservation`, new `polygram.geometry` module
- [x] 7.5 Update `polygram/__init__.py` module docstring's surface-area summary

## 8. Adjacent bug fixes surfaced by the probe panel

- [x] 8.1 Fix `load_sae_safetensors(feature_ids=...)` bfloat16 slice path: `_load_subset` in `polygram/sae_import.py` calls `np.asarray(slc[:, fid_int], dtype=float)` on a raw bf16 tensor and crashes with `TypeError: data type 'bfloat16' not understood`. The eager full-decoder path works because it goes via torch's float-conversion. Llama-Scope L0R surfaced this; modern LLM SAEs (Llama-Scope, Gemma-Scope, plausibly Qwen-Scope at scale) ship bf16 by default, so this is load-bearing for the implementation pass — not optional cleanup.
- [x] 8.1.1 Extract a module-private `_safe_to_float32(tensor) -> np.ndarray` helper in `polygram/sae_import.py` that dispatches to torch's `.to(torch.float32).numpy()` for any non-numpy-native dtype (bf16, fp16, fp8 if it ever ships) and falls through to `np.asarray(..., dtype=np.float32)` otherwise. Use it from both the eager and sliced paths so they stay in lockstep and future loaders don't re-discover the bf16 quirk.
- [x] 8.2 Add a regression test `tests/test_load_sae_safetensors_bfloat16.py` that constructs a tiny bf16 fixture and calls `load_sae_safetensors(path, feature_ids=[0, 1])`; assert it returns float32 records without raising

## 9. Validation pass

- [x] 9.1 Run `openspec validate add-sae-geometry-regimes` and resolve any structural issues
- [x] 9.2 Re-run the five-SAE smoke probe panel (Whisper × 2, Qwen-Scope, Llama-Scope L0R + L12R) with the new `profile="uniform-sphere"` path; capture before/after numbers in `docs/research/sae-geometry-regimes.md` and confirm `geometric_fidelity` reads as a stable float across selection strategies (the Pearson sign-flip we observed pre-change should not recur under rank-recall@k)
- [x] 9.3 Confirm `polygram analyze` CLI on the bundled toy SAE produces identical output before and after this change (clustered default path)
