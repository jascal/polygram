## ADDED Requirements

### Requirement: from_sae_lens accepts an optional GeometricProfile

`polygram.from_sae_lens` SHALL accept a keyword argument
`profile: str | GeometricProfile | None = None`.

- When `profile` is a `str`, it SHALL be resolved via
  `polygram.geometry.get_profile(profile)`.
- When `profile` is a `GeometricProfile`, it SHALL be used
  directly.
- When `profile is None`, the function SHALL resolve to
  `polygram.geometry.text_clustered()` — the v0.1.0-equivalent
  default. This resolution SHALL be observable behaviour (i.e.
  `report.profile == "text-clustered"`).

The selected profile SHALL govern the k-means path's defaults
(`n_clusters`, `gamma_range`) and the strategy used to compute
`(β, γ, cluster_method)` and `geometric_fidelity`. Per-field
kwargs (`n_clusters`, `gamma_range`, `assign_gamma`,
`cluster_assignments`, `config`) SHALL retain precedence over
profile defaults: kwargs > config > profile defaults > strategy
internal defaults.

The `cluster_assignments` and `from_labels` precedence paths
(documented in the existing "from_sae_lens selection-first
import" requirement) SHALL run before profile dispatch and
bypass the profile's `KnobAssignment` strategy entirely. The
profile's `GeometricFidelity` SHALL still be computed regardless
of which cluster-assignment path was taken.

When `config: SAEImportConfig` is supplied and contains a
`profile` field, that profile SHALL be used unless the
per-field `profile` kwarg also supplies one. `SAEImportConfig`
SHALL gain an optional `profile: str | None = None` field;
`None` resolves to the registry default (text-clustered) at
`from_sae_lens` time, not at `SAEImportConfig` construction
time, so the resolution path is centralised.

#### Scenario: omitting profile is byte-equal to passing text-clustered

- **WHEN** `from_sae_lens(records, [0, 1, 4, 5])` is called on
  the bundled toy SAE fixture without a `profile` argument
- **THEN** the returned `Dictionary` and `SelectionReport` are
  bit-equal to those returned by the same call with
  `profile="text-clustered"`, and `report.profile` is the string
  `"text-clustered"`

#### Scenario: per-field n_clusters overrides profile default

- **WHEN** `from_sae_lens(records, ids,
  profile="uniform-sphere", n_clusters=4)` is called
- **THEN** k-means runs with k=4 (the per-field kwarg), not 16
  (the uniform-sphere default), and `report.profile` is
  `"uniform-sphere"`

#### Scenario: cluster_assignments bypasses profile strategy

- **WHEN** `from_sae_lens(records, ids,
  profile="uniform-sphere", cluster_assignments={...})` is
  called with explicit cluster assignments
- **THEN** the returned `Dictionary` has the user-supplied
  cluster partition, `report.cluster_method == "user"`, but
  `report.profile == "uniform-sphere"` and
  `report.geometric_fidelity` is the uniform-sphere fidelity
  computed against that partition

### Requirement: SelectionReport records the active profile and its fidelity

`SelectionReport` SHALL gain two fields:

- `profile: str` — the `name` of the `GeometricProfile` used
  for this build; never `None` (defaults to `"text-clustered"`
  when `from_sae_lens` was called without a profile)
- `geometric_fidelity: float | None` — the output of the
  active profile's `GeometricFidelity` metric for this build;
  `None` when the profile's metric returns `None` (e.g. for
  a single-feature dictionary)

The existing `tier_preservation: float | None` field SHALL be
retained and SHALL be populated whenever the active profile's
`GeometricFidelity` is the v0.1.0 Pearson correlation (i.e.
exactly the `text-clustered` profile, plus any third-party
profiles that opt to reuse that metric). For profiles that use
a different metric, `tier_preservation` SHALL be `None` and the
profile's chosen scalar appears in `geometric_fidelity` instead.

#### Scenario: text-clustered populates both fields with equal values

- **WHEN** `from_sae_lens(records, ids,
  profile="text-clustered")` is called
- **THEN** `report.profile == "text-clustered"`, and
  `report.tier_preservation == report.geometric_fidelity`
  (or both are `None`)

#### Scenario: uniform-sphere populates geometric_fidelity but not tier_preservation

- **WHEN** `from_sae_lens(records, ids,
  profile="uniform-sphere")` is called on a multi-feature
  fixture
- **THEN** `report.profile == "uniform-sphere"`,
  `report.tier_preservation is None`, and
  `report.geometric_fidelity` is a float in `[0.0, 1.0]`
  (rank-recall is bounded)

#### Scenario: serialised report round-trips the profile field

- **WHEN** a `SelectionReport` is serialised to JSON via the
  existing `.to_json()` / `to_dict()` path and re-loaded
- **THEN** the deserialised report has the same `profile`
  string, `geometric_fidelity` value, and
  `tier_preservation` value as the original
