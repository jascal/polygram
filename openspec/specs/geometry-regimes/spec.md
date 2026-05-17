# geometry-regimes Specification

## Purpose
TBD - created by archiving change add-sae-geometry-regimes. Update Purpose after archive.
## Requirements
### Requirement: GeometricProfile is a named bundle of strategy + metric + defaults

`polygram.geometry.GeometricProfile` SHALL be a frozen dataclass
that bundles, for a single SAE geometric regime:

- `name: str` — kebab-case identifier (e.g. `"clustered"`,
  `"uniform-sphere"`)
- `knob_assignment: KnobAssignment` — a strategy object
  responsible for mapping selected projection vectors to
  per-feature `(beta, gamma)` knobs and a cluster partition
- `geometric_fidelity: GeometricFidelity` — a metric strategy
  that, given the projection matrix and the built `Dictionary`,
  returns `float | None`
- `default_n_clusters: int | None` — k-means default when
  `from_sae_lens` falls back to its k-means path; `None` means
  "the strategy decides"
- `default_gamma_range: tuple[float, float]` — γ-range default
  passed to γ assignment when `assign_gamma=True`

`GeometricProfile` SHALL be hashable, immutable, and safe to
pass through configuration round-trips (e.g.
`SAEImportConfig.to_dict()`).

#### Scenario: profile is hashable and immutable

- **WHEN** code attempts to mutate `profile.name` after
  construction, or stores two equal-by-content profiles in a set
- **THEN** the mutation raises `FrozenInstanceError`, and the
  set has exactly one element

### Requirement: KnobAssignment protocol governs (β, γ, cluster) assignment

`polygram.geometry.KnobAssignment` SHALL be a protocol whose
implementations expose `assign(projections: np.ndarray,
feature_names: list[str], *, n_clusters: int | None,
gamma_range: tuple[float, float], assign_gamma: bool, seed: int)
-> KnobAssignmentResult` returning:

- `cluster_per_feature: list[str]` — same length as
  `feature_names`
- `betas: list[float]` — same length as `feature_names`
- `gammas: list[float]` — same length as `feature_names`
- `cluster_method: str` — recorded in `SelectionReport`
  (e.g. `"kmeans"`, `"pca_axis"`, `"orthogonal"`)
- `beta_variance_explained: float` — defined per strategy;
  always in `[0.0, 1.0]`

Strategies SHALL NOT consume `cluster_assignments` or
`from_labels` paths — those are upstream of strategy dispatch in
`from_sae_lens` and bypass the strategy entirely.

#### Scenario: protocol contract is length-consistent

- **WHEN** any built-in `KnobAssignment` is invoked with an
  N-length projections array
- **THEN** `cluster_per_feature`, `betas`, `gammas` are each
  length N, and `0.0 <= beta_variance_explained <= 1.0`

### Requirement: GeometricFidelity protocol returns a profile-specific scalar

`polygram.geometry.GeometricFidelity` SHALL be a protocol whose
implementations expose `compute(projections: np.ndarray,
dictionary: Dictionary) -> float | None`. The returned value is
the profile's headline fidelity metric (the moral analogue of
today's `tier_preservation`). Returning `None` SHALL signal "not
defined for this geometry / sample size."

The contract SHALL NOT prescribe a fixed direction (higher vs
lower better) — direction is documented per strategy.

#### Scenario: fidelity returns float or None, never raises

- **WHEN** any built-in `GeometricFidelity` is called on a
  single-feature dictionary (no off-diagonal pairs available)
- **THEN** it returns `None` without raising

### Requirement: clustered profile reproduces v0.1.0 defaults exactly

`polygram.geometry.clustered()` SHALL return a
`GeometricProfile(name="clustered", ...)` whose behaviour
is byte-for-byte identical to polygram v0.1.0's
`from_sae_lens` k-means path:

- `default_n_clusters = 2`
- `default_gamma_range = (-0.25, 0.25)`
- `KnobAssignment` runs k-means on raw projections (not unit
  vectors), spreads β evenly across cluster ordinals within
  `(-0.5, 0.5)`, and derives γ via per-cluster PCA on centered
  projections (the v0.1.0 `_gamma_via_cluster_pca` path)
- `GeometricFidelity` is the v0.1.0 `tier_preservation` Pearson
  correlation between off-diagonal `|G|²` of the projection-
  space cosine-overlap matrix and the analytic Polygram Gram
- `cluster_method` returned by the strategy is `"kmeans"`

This profile SHALL be the registry default, and SHALL be the
profile selected when `from_sae_lens` is called without a
`profile` argument.

#### Scenario: clustered output matches v0.1.0 byte-for-byte

- **WHEN** `from_sae_lens(records, ids,
  profile="clustered")` is called on the bundled toy SAE
  fixture (4-feature subset)
- **THEN** the returned `Dictionary.features`,
  `report.cluster_assignments`,
  `report.beta_variance_explained`, and
  `report.tier_preservation` are identical to the v0.1.0
  baseline (recorded in a frozen golden fixture under
  `tests/fixtures/golden_clustered.json`)

#### Scenario: omitting profile resolves to clustered

- **WHEN** `from_sae_lens(records, ids)` is called without a
  `profile` argument on the bundled toy SAE fixture
- **THEN** the returned dictionary, report, and fidelity values
  are bit-equal to those returned by the same call with
  `profile="clustered"`

### Requirement: uniform-sphere profile targets quasi-uniform feature geometries

`polygram.geometry.uniform_sphere()` SHALL return a
`GeometricProfile(name="uniform-sphere", ...)` whose strategy is
calibrated for quasi-orthogonal feature dictionaries (mean off-
diagonal cosine ≈ 0, no recoverable 2-cluster structure):

- `default_n_clusters = 16`
- `default_gamma_range = (-0.25, 0.25)` (unchanged — γ via
  per-cluster PCA still works inside small tight clusters when
  the chosen k surfaces them)
- `KnobAssignment` runs k-means on **unit-normalised**
  projection vectors with `n_init >= 4`, then assigns β by
  projecting each feature onto the top-1 PCA component of the
  *full selected subset's centered projections* and rescaling
  into `(-0.5, 0.5)` — `cluster_method = "pca_axis"`. β is no
  longer cluster-ordinal; clusters carry tier identity but β
  carries continuous geometric position.
- `GeometricFidelity` is `rank_recall_at_k` with `k =
  max(3, len(features) // 2)`: the fraction of top-`k`
  off-diagonal pairs by Polygram-Gram `|G|²` that are also in
  the top-`k` off-diagonal pairs by projection-space cosine
  `|<p_i, p_j>|`. Range `[0.0, 1.0]`, higher is better.

This profile SHALL NOT be the default. Consumers MUST opt in by
passing `profile="uniform-sphere"` (or constructing it
explicitly).

#### Scenario: uniform-sphere produces non-degenerate β spread on audio SAE subset

- **WHEN** `from_sae_lens(records, ids, profile="uniform-sphere")`
  is called on the audio-SAE fixture with a tight cluster of 8
  features (mean within-cluster cosine ~0.4)
- **THEN** the returned `Dictionary.features` have β values
  spanning at least 60% of `(-0.5, 0.5)` (i.e.
  `max(beta) - min(beta) >= 0.6`), and
  `report.geometric_fidelity` is in `[0.0, 1.0]`

### Requirement: profile registry supports built-in lookup and third-party registration

`polygram.geometry` SHALL expose:

- `register_profile(profile: GeometricProfile) -> None` —
  adds a profile to the registry, keyed by `profile.name`;
  raises `ValueError` if `name` is already registered (no
  silent overrides)
- `get_profile(name: str) -> GeometricProfile` — returns the
  registered profile or raises `KeyError` with a message that
  lists currently-registered names
- `available_profiles() -> list[str]` — returns the sorted
  list of registered profile names

The two built-in profiles (`"clustered"`,
`"uniform-sphere"`) SHALL be registered at import time of
`polygram.geometry`. Third-party packages can register
additional profiles (e.g. `"image-clip"`, `"video-vjepa"`) by
calling `register_profile` after import, without forking
polygram.

#### Scenario: built-ins are registered at import

- **WHEN** `import polygram.geometry` runs
- **THEN** `polygram.geometry.available_profiles()` returns at
  minimum `["clustered", "uniform-sphere"]` (sorted)

#### Scenario: duplicate registration is rejected

- **WHEN** `register_profile(GeometricProfile(name="clustered", ...))`
  is called after import
- **THEN** `ValueError` is raised, the existing
  `clustered` profile is unchanged, and the message names
  the conflicting key

### Requirement: `KnobAssignmentResult` carries an optional `axis_assignment` field

`polygram.geometry.protocols.KnobAssignmentResult` SHALL include
`axis_assignment: dict[str, int | list[float]] | None = None`. The
field surfaces *which PCA axis (or which linear combination of
axes)* fed each polygram knob slot during the import.

`ClusteredKnobAssignment` and `UniformSphereKnobAssignment` SHALL
leave the field at the default `None`. `LearnedKnobAssignment`
SHALL populate it per the
[`learned-axis-assignment` capability spec](../learned-axis-assignment/spec.md).

The field is optional rather than mandatory so adding it does not
churn every existing strategy implementation.

#### Scenario: existing strategies leave the field None

- **WHEN** `ClusteredKnobAssignment().assign(...)` or
  `UniformSphereKnobAssignment().assign(...)` returns
- **THEN** `result.axis_assignment is None`

#### Scenario: learned strategy populates the field

- **WHEN** `LearnedKnobAssignment().assign(...)` returns
- **THEN** `result.axis_assignment` is a non-empty dict whose keys
  are knob names

### Requirement: `LearnedAxisObjective` protocol formalises the objective surface

`polygram.geometry.LearnedAxisObjective` SHALL be a
`runtime_checkable` Protocol with the call signature
`__call__(analytic_gram: np.ndarray, decoder_geom: np.ndarray, *,
feature_names: list[str] | None = None) -> float`. The
`feature_names` kwarg is optional (default `None`) to reduce
boilerplate for simple objectives that ignore cluster context. The
return value is a scalar that the learned-assignment solver
maximises.

Three built-in objectives SHALL be available in
`polygram.geometry.objectives`:

- `spearman_objective` — Spearman rank correlation on off-diagonal
  upper-triangle of `|analytic_gram|²` and `decoder_geom`.
- `pearson_objective` — Pearson correlation on the same entries.
- `behavioural_objective(reference_pair_sims)` — factory returning
  a closure that scores against a user-supplied square matrix.

User-defined objectives are accepted by the strategy as long as
they satisfy the protocol.

#### Scenario: built-in objectives satisfy the protocol

- **WHEN** `isinstance(spearman_objective, LearnedAxisObjective)`,
  `isinstance(pearson_objective, LearnedAxisObjective)`, and
  `isinstance(behavioural_objective(some_matrix), LearnedAxisObjective)`
  are evaluated
- **THEN** all three return `True`

#### Scenario: user callable accepted as objective

- **WHEN** a user defines `def my_obj(g, d, *, feature_names=None):
  return float(np.real(g).mean())` and passes it via
  `LearnedKnobAssignment(objective=my_obj)`
- **THEN** the strategy accepts it without complaint and uses it
  during the search

#### Scenario: simple objective omitting feature_names accepted

- **WHEN** a user defines `def simple_obj(g, d): return
  float(-np.linalg.norm(np.abs(g) - d))` and passes it via
  `LearnedKnobAssignment(objective=simple_obj)`
- **THEN** the strategy accepts it (the protocol's `feature_names`
  default is `None`) and uses it during the search

