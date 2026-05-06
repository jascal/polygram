## ADDED Requirements

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
