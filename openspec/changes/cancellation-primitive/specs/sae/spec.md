## ADDED Requirements

### Requirement: SelectionReport reports per-feature reconstruction error

`SelectionReport` SHALL expose `reconstruction_error: dict[str,
float]` — for each selected feature, the Euclidean distance from
its projection vector to the centroid of its assigned cluster (in
projection space). This complements the aggregate
`beta_variance_explained` by surfacing per-feature outliers.

#### Scenario: identical projections in a cluster have zero error

- **WHEN** every selected feature in a single cluster has an
  identical projection vector
- **THEN** every entry of `report.reconstruction_error` for those
  features is 0.0 (within 1e-12)

### Requirement: SelectionReport reports tier preservation

`SelectionReport` SHALL expose `tier_preservation: float | None`
— Pearson correlation between off-diagonal `|G|²` entries of the
projection-space cosine-overlap matrix (`G_proj[i,j] = <p_i, p_j>
/ (|p_i| |p_j|)`) and the analytic Polygram Gram of the built
`Dictionary` at φ=0 for all features. None when `len(feature_ids)
<= 1` (no off-diagonals to correlate).

#### Scenario: identical projections collapse correlation to undefined or 1.0

- **WHEN** two features in the same cluster have identical
  projections (so projection-space and Polygram-Gram both report
  `|G|² == 1` between them)
- **THEN** `tier_preservation` is in `[-1.0, 1.0]` (or NaN if
  collinearity makes the correlation undefined; the value SHALL
  not raise an exception)

### Requirement: from_sae_lens optional gamma auto-assignment

`from_sae_lens` SHALL accept `assign_gamma: bool = False` and
`gamma_range: tuple[float, float] = (-0.25, 0.25)` keyword
arguments. When `assign_gamma=True`, each feature's γ SHALL be
its projection vector's coefficient on the first principal
component of its assigned cluster's centered projection
vectors, rescaled into `gamma_range`. `report.gamma_method`
records `"zero"` (default) or `"projection_pca"` accordingly.
α and φ defaults are unchanged (both 0).

#### Scenario: assign_gamma writes nonzero per-feature gamma

- **WHEN** `from_sae_lens(records, [0, 1, 4, 5],
  assign_gamma=True)` is called on the bundled toy fixture
- **THEN** `report.gamma_method == "projection_pca"`, the
  returned Dictionary has at least one feature with `gamma != 0`,
  and all `feature.gamma` values lie within `gamma_range`

#### Scenario: default keeps gammas at zero

- **WHEN** `from_sae_lens(records, [0, 1, 4, 5])` is called
  without `assign_gamma`
- **THEN** every returned feature has `gamma == 0` and
  `report.gamma_method == "zero"`
