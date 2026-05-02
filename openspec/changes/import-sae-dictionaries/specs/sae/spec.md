## ADDED Requirements

### Requirement: SAE feature record schema

Polygram SHALL expose `SAEFeatureRecord` — a frozen dataclass holding
the data the importer needs from an SAE feature. Required fields:
`feature_id: int`, `name: str`, `projection: np.ndarray` (1D, real,
finite). Optional fields: `label: str | None = None`,
`activation_mean: float | None = None`, `activation_std: float | None
= None`. The `projection` is the decoder column (or other direction
vector) for that feature in residual-stream space.

#### Scenario: record validates projection is 1D and finite

- **WHEN** an `SAEFeatureRecord` is constructed with a 2D projection
  array or one containing NaN
- **THEN** construction raises `ValueError` naming the offending field

### Requirement: Toy SAE fixture loader

`polygram.sae_import.load_toy_sae(path)` SHALL parse a JSON file with
schema `{"features": [<record>, ...]}` where each record carries the
`SAEFeatureRecord` fields, and return a `dict[int, SAEFeatureRecord]`
keyed by `feature_id`. Projection arrays SHALL be coerced to
`np.ndarray` of dtype `float64`.

#### Scenario: load round-trips bundled fixture

- **WHEN** `load_toy_sae("tests/fixtures/toy_sae.json")` is called
- **THEN** the result is a dict of length 16, each value is an
  `SAEFeatureRecord`, and every projection array has shape `(8,)`

### Requirement: from_sae_lens selection-first import

`polygram.sae_import.from_sae_lens(records, feature_ids, ...)` SHALL
build a `Dictionary` from an explicit subset of SAE features and
return a `(Dictionary, SelectionReport)` pair. Cluster assignment
follows this precedence:

1. If `cluster_assignments: dict[int, str]` is provided, use it
   verbatim (cluster method recorded as `"user"`).
2. Otherwise, if every selected record's `label` matches
   `"<cluster>/<feature>"`, parse the prefix as the cluster
   (`"from_labels"`).
3. Otherwise, run k-means on the projection vectors with
   `n_clusters` (default 2), using cluster index as the cluster
   name (`"cluster_0"`, `"cluster_1"`, ...; recorded as
   `"kmeans"`).

β values SHALL be spread evenly across cluster means within
`beta_range` (default `(-0.5, 0.5)`). α and γ default to 0 unless a
future overload supplies per-feature values. φ defaults to 0.

The function SHALL reject `len(feature_ids) > 8` with a `ValueError`
naming the limit and the selected count, before any clustering.

#### Scenario: explicit cluster assignments honored

- **WHEN** `from_sae_lens(records, [1, 2, 5, 6],
  cluster_assignments={1: "A", 2: "A", 5: "B", 6: "B"})` is called
- **THEN** the returned `Dictionary` has hierarchy
  `{"A": [name_of(1), name_of(2)], "B": [name_of(5), name_of(6)]}`
  and `report.cluster_method == "user"`

#### Scenario: from-labels cluster path

- **WHEN** every selected record's `label` matches `"<cluster>/..."`
  and no `cluster_assignments` is provided
- **THEN** `report.cluster_method == "from_labels"` and the
  Dictionary's hierarchy keys are exactly the unique label prefixes

#### Scenario: k-means default fallback

- **WHEN** records have no parseable labels and no
  `cluster_assignments` are provided
- **THEN** `report.cluster_method == "kmeans"` and all selected
  features land in one of `n_clusters` (default 2) generated cluster
  buckets named `cluster_0..cluster_{n-1}`

#### Scenario: capacity limit enforced

- **WHEN** `from_sae_lens(records, [1, 2, 3, 4, 5, 6, 7, 8, 9])` is
  called (9 features)
- **THEN** `ValueError` is raised before any clustering work,
  naming the limit (8) and the selected count (9)

### Requirement: SelectionReport surfaces fidelity stats

`SelectionReport` SHALL be a frozen dataclass with fields:

- `n_input_features: int` — `len(records)`
- `n_selected: int` — `len(feature_ids)`
- `cluster_assignments: dict[str, str]` — feature name → cluster name
- `cluster_method: str` — one of `"user"`, `"from_labels"`, `"kmeans"`
- `beta_variance_explained: float` — in `[0.0, 1.0]`. Computed as
  `1 - SS_residual / SS_total`, where `SS_total` is the sum of
  squared distances of selected projection vectors from their
  collective centroid, and `SS_residual` is the sum of squared
  distances from each vector to *its assigned cluster's centroid*.
  This measures how much projection-space variance the cluster
  partition captures.
- `warnings: list[str]` — non-empty iff a clustering pathology was
  detected (e.g. an empty cluster bucket, single-cluster fallback)

#### Scenario: identical-projection cluster collapses report variance to 1.0

- **WHEN** every selected feature in a single cluster has an
  identical projection vector
- **THEN** `report.beta_variance_explained == 1.0` (within 1e-9)

#### Scenario: empty-cluster pathology emits a warning

- **WHEN** k-means with `n_clusters=4` is run on 4 selected features
  whose projections are nearly collinear so that one bucket ends up
  empty
- **THEN** `report.warnings` contains a non-empty string mentioning
  the empty cluster
