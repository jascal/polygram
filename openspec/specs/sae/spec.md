# sae Specification

## Purpose

The SAE → Polygram bridge. `polygram.from_sae_lens` consumes
SAE Lens-style feature records (id, projection vector, cluster
hint), runs cluster assignment and feature-knob derivation (β, α,
optional γ via per-cluster PCA), and returns a built `Dictionary`
plus a `SelectionReport` with reconstruction error per feature and
a tier-preservation correlation between projection-space cosine
overlaps and the analytic Polygram Gram.
## Requirements
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

### Requirement: load_sae_safetensors reads decoder columns from a single .safetensors file

`polygram.load_sae_safetensors(path: str | Path, *, names: dict[int, str] | None = None) -> dict[int, SAEFeatureRecord]` SHALL read a single `.safetensors` file from disk and return the dict shape that `polygram.from_sae_lens` already consumes.

The function SHALL:

1. Lazily import `safetensors.numpy` so package import does not fail when the optional `[sae]` extra is not installed; missing imports SHALL raise a `ImportError` whose message points at `pip install polygram[sae]`.
2. Auto-detect the decoder weight tensor key by trying, in order, `W_dec`, `decoder.weight`, `dec`. The first match wins; if none are present, the function SHALL raise `ValueError` whose message lists every key in the file.
3. Treat the matched tensor as 2D. Non-2D tensors SHALL raise `ValueError` naming the offending tensor key and shape.
4. Map decoder rows to features (one row → one `SAEFeatureRecord`). When the matched key is `decoder.weight` and the tensor is non-square, the loader SHALL transpose first (PyTorch `nn.Linear` weight convention is `out × in`, where `out = d_model` and `in = d_sae`); square matrices SHALL NOT be transposed.
5. Default each feature's name to `f"feat_{i}"`. The `names` parameter SHALL override per-feature names; absent keys keep the default. `names` keys outside `[0, n_features)` SHALL raise `ValueError` naming the offending key.
6. Set every returned record's `label`, `activation_mean`, and `activation_std` to `None`. The loader SHALL NOT infer or attach these.
7. Coerce projection vectors to `numpy.ndarray` with `dtype=float64` (matches the existing `SAEFeatureRecord` projection coercion in `from_sae_lens`).

#### Scenario: W_dec key takes precedence and rows are features

- **GIVEN** a `.safetensors` file containing tensors `W_dec` (shape `(n=4, d=8)`) and `dec` (shape `(2, 8)`)
- **WHEN** `load_sae_safetensors(path)` is called
- **THEN** the returned dict has exactly 4 entries keyed by `0..3`
- **AND** `records[i].projection` is the numpy array of `W_dec[i, :]` for every `i`
- **AND** `records[i].name == f"feat_{i}"`
- **AND** `records[i].label is None`

#### Scenario: decoder.weight is transposed when non-square

- **GIVEN** a `.safetensors` file whose only matching key is `decoder.weight` with shape `(d=8, n=4)` (PyTorch out × in convention with `out = d_model = 8`, `in = d_sae = 4`)
- **WHEN** `load_sae_safetensors(path)` is called
- **THEN** the returned dict has exactly 4 entries
- **AND** `records[i].projection.shape == (8,)`
- **AND** `records[i].projection` equals `decoder.weight[:, i]` for every `i`

#### Scenario: dec key is the terse fallback

- **GIVEN** a `.safetensors` file whose only matching key is `dec` with shape `(n=3, d=4)`
- **WHEN** `load_sae_safetensors(path)` is called
- **THEN** the returned dict has 3 entries with `records[i].projection` equal to `dec[i, :]`

#### Scenario: missing decoder key surfaces every key in the file

- **GIVEN** a `.safetensors` file whose tensors are `enc`, `b_enc`, `b_dec` (none of `W_dec`, `decoder.weight`, `dec`)
- **WHEN** `load_sae_safetensors(path)` is called
- **THEN** a `ValueError` is raised
- **AND** the message lists `W_dec`, `decoder.weight`, `dec` (the auto-detect precedence)
- **AND** the message lists `enc`, `b_enc`, `b_dec` (the keys actually present)

#### Scenario: non-2D decoder tensor rejected

- **GIVEN** a `.safetensors` file whose `W_dec` tensor has shape `(n, d, k)` (3D)
- **WHEN** `load_sae_safetensors(path)` is called
- **THEN** a `ValueError` is raised naming the key `W_dec` and the shape `(n, d, k)`

#### Scenario: names override applies per feature

- **GIVEN** a `.safetensors` file with `W_dec` of shape `(4, 8)`
- **WHEN** `load_sae_safetensors(path, names={0: "dog_poodle", 2: "bird_hawk"})` is called
- **THEN** `records[0].name == "dog_poodle"`
- **AND** `records[1].name == "feat_1"` (default)
- **AND** `records[2].name == "bird_hawk"`
- **AND** `records[3].name == "feat_3"`

#### Scenario: names key out of range rejected

- **GIVEN** a `.safetensors` file with `W_dec` of shape `(4, 8)`
- **WHEN** `load_sae_safetensors(path, names={5: "ghost"})` is called
- **THEN** a `ValueError` is raised naming the offending key `5` and the valid range `[0, 4)`

#### Scenario: missing safetensors install raises a clear hint

- **GIVEN** a Python environment without the `safetensors` package installed
- **WHEN** `load_sae_safetensors(path)` is called for any path
- **THEN** an `ImportError` is raised
- **AND** the message names `pip install polygram[sae]` as the install hint

### Requirement: load_sae_safetensors returns the from_sae_lens-consumable shape

The dict returned by `load_sae_safetensors` SHALL be directly consumable by `polygram.from_sae_lens(records, feature_ids, ...)` with no further coercion. Specifically:

- The dict's values SHALL be instances of `polygram.SAEFeatureRecord`.
- Each record's `feature_id` SHALL equal its dict key.
- Each record's `projection` SHALL be a 1-D numpy array of dtype `float64` and length matching the decoder's column count (after any orientation correction).

#### Scenario: round-trip through from_sae_lens

- **GIVEN** a `.safetensors` file with `W_dec` of shape `(8, 16)` and four features `[0, 1, 4, 5]` selected by id
- **WHEN** `records = load_sae_safetensors(path)` and `dictionary, _ = from_sae_lens(records, [0, 1, 4, 5])` are called
- **THEN** `from_sae_lens` returns a `Dictionary` with 4 features whose names match the records'
- **AND** the call raises no errors

### Requirement: load_sae_safetensors supports lazy row slicing via feature_ids

`load_sae_safetensors` SHALL accept a `feature_ids: list[int] | None = None` keyword argument. When `None` (the default), the loader behaves as the eager path documented above and reads the full decoder tensor into memory.

When `feature_ids` is set, the loader SHALL:

1. Open the file via `safetensors.safe_open(path, framework="numpy")` instead of `safetensors.numpy.load_file`. The full decoder tensor SHALL NOT be loaded into memory.
2. Auto-detect the decoder key and apply the same orientation rule as the eager path (decoder.weight non-square → operate on columns rather than rows).
3. Slice each requested feature_id individually via `safe_open(...).get_slice(matched)[fid, :]` (or `[:, fid]` post-orientation), reading at most `d_model × dtype_size` bytes per requested feature.
4. Return a `dict[int, SAEFeatureRecord]` keyed by exactly the requested `feature_ids`. Iteration order SHALL match the input list.
5. Reject out-of-range entries in `feature_ids` with `ValueError` naming the offending id and the valid range — using the same `[0, n_features)` rule as `names` validation.

The lazy path SHALL be observably equivalent to the eager path: for any `path` and any `ids`, `load_sae_safetensors(path, feature_ids=ids)` SHALL produce records whose `projection` arrays equal the corresponding entries from `load_sae_safetensors(path)` element-wise.

#### Scenario: lazy load reads only the requested rows

- **GIVEN** a `.safetensors` file with `W_dec` of shape `(8, 16)`
- **WHEN** `load_sae_safetensors(path, feature_ids=[0, 4])` is called
- **THEN** the returned dict has exactly two entries keyed by `0` and `4`
- **AND** the iteration order of the returned dict yields `0` then `4` (matching the input list order)
- **AND** the projection arrays match the eager-path output for the same ids

#### Scenario: lazy load preserves orientation correction

- **GIVEN** a `.safetensors` file whose only matching key is `decoder.weight` with shape `(8, 4)` (PyTorch out × in convention)
- **WHEN** `load_sae_safetensors(path, feature_ids=[0, 1, 2, 3])` is called
- **THEN** the returned dict has 4 entries
- **AND** each `records[i].projection` equals `decoder.weight[:, i]` (column slicing post-orientation)
- **AND** each `records[i].projection.shape == (8,)`

#### Scenario: out-of-range feature_id rejected in lazy mode

- **GIVEN** a `.safetensors` file with `W_dec` of shape `(4, 8)`
- **WHEN** `load_sae_safetensors(path, feature_ids=[0, 9])` is called
- **THEN** a `ValueError` is raised naming the offending id `9` and the valid range `[0, 4)`

