# tuning-config Specification (delta)

## MODIFIED Requirements

### Requirement: polygram.config exposes tuning dataclasses

The `polygram.config` module SHALL export the same six frozen
dataclasses as before. `CompressionConfig` SHALL grow two
additional fields supporting target-feature-count compression and
Pareto-path planning:

- `CompressionConfig` —
  `strategy: str = "merge"`,
  `rep_selection: str = "scale_aware"`,
  `merge_mode: str = "freq_weighted"`,
  `confirmer: str | None = None`,
  `target_n_features_kept: int | None = None`,
  `score_field: str = "polygram_overlap"`.

The remaining five dataclasses (`EpochCompressionConfig`,
`CancellationConfig`, `ValidationConfig`, `RegrowConfig`,
`SAEImportConfig`) are unchanged in field set and defaults.

`CompressionConfig.target_n_features_kept` semantic: matches the
existing `CompressionReport.n_features_kept`, which counts cluster
*representatives*, not the SAE's total surviving feature count. The
field's docstring SHALL state this explicitly (see Decision 1 in
`design.md`).

`__post_init__` validation SHALL additionally enforce:

- `target_n_features_kept` is either `None` or an integer `>= 1`.
- `score_field` is one of `"polygram_overlap"`, `"jaccard"`,
  `"decoder_overlap"` (the three CandidatePair fields that are
  bounded `[0, 1]` similarity-like quantities — see Decision 3 in
  `design.md`).

Existing range/value validation on `strategy`, `rep_selection`,
`merge_mode` is unchanged.

#### Scenario: CompressionConfig accepts target_n_features_kept

- **WHEN** `CompressionConfig(target_n_features_kept=200)` is
  constructed
- **THEN** the resulting config exposes `target_n_features_kept == 200`
  and `score_field == "polygram_overlap"` (the default)

#### Scenario: CompressionConfig rejects non-positive target

- **WHEN** `CompressionConfig(target_n_features_kept=0)` is
  constructed
- **THEN** `__post_init__` raises `ValueError` naming
  `target_n_features_kept` and the valid range (`>= 1`)

#### Scenario: CompressionConfig rejects unknown score_field

- **WHEN** `CompressionConfig(score_field="bogus")` is constructed
- **THEN** `__post_init__` raises `ValueError` naming `score_field`
  and listing the three supported values

#### Scenario: CompressionConfig rejects KL score_fields

- **WHEN** `CompressionConfig(score_field="kl_log_ratio_abs")` is
  constructed (a real `CandidatePair` field, deliberately excluded
  from valid `score_field` values)
- **THEN** `__post_init__` raises `ValueError` listing the three
  supported values

#### Scenario: CompressionConfig is byte-identical when new fields are unset

- **WHEN** `CompressionConfig()` is constructed without the new
  fields
- **THEN** `target_n_features_kept is None`,
  `score_field == "polygram_overlap"`, and a `Compressor`
  constructed from this config exhibits the historical `plan()` →
  `apply()` byte output on the existing toy fixture (verified via
  `CompressionReport.to_json()` reference comparison)

### Requirement: Configs round-trip through dict

Each config dataclass SHALL continue to expose `.to_dict()` and
`.from_dict(cls, data: Mapping[str, Any]) -> Self` via the existing
`_ConfigMixin` field-iteration machinery
(`polygram/config.py:55`). `CompressionConfig.to_dict()` SHALL
include the two new fields automatically (via the mixin);
`CompressionConfig.from_dict()` SHALL accept dicts missing those
keys and fall back to defaults (`None` and `"polygram_overlap"`
respectively).

#### Scenario: CompressionConfig.from_dict tolerates missing new fields

- **WHEN** `CompressionConfig.from_dict({"strategy": "merge", "rep_selection": "scale_aware", "merge_mode": "freq_weighted"})`
  is called (no new fields)
- **THEN** the result has `target_n_features_kept is None` and
  `score_field == "polygram_overlap"`

#### Scenario: CompressionConfig round-trips with new fields populated

- **WHEN** `CompressionConfig(target_n_features_kept=500, score_field="jaccard").to_dict()`
  is fed back through `CompressionConfig.from_dict`
- **THEN** the resulting instance equals the original
