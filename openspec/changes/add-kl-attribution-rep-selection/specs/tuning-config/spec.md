# tuning-config Specification (delta)

## MODIFIED Requirements

### Requirement: polygram.config exposes tuning dataclasses

The `polygram.config` module SHALL continue to export the same six frozen dataclasses. `CompressionConfig.rep_selection` SHALL accept one additional supported value (`"kl_attribution"`) on top of the existing two (`"n_fires"`, `"scale_aware"`):

- `CompressionConfig` —
  `strategy: str = "merge"`,
  `rep_selection: str = "scale_aware"`,
  `merge_mode: str = "freq_weighted"`,
  `confirmer: str | None = None`,
  `target_n_features_kept: int | None = None`,
  `score_field: str = "polygram_overlap"`.

The remaining five dataclasses (`EpochCompressionConfig`, `CancellationConfig`, `ValidationConfig`, `RegrowConfig`, `SAEImportConfig`) are unchanged.

`__post_init__` validation SHALL accept `rep_selection` from the extended set `("n_fires", "scale_aware", "kl_attribution")`. The default remains `"scale_aware"`; existing call paths byte-identical when `rep_selection` is unset or set to one of the two pre-existing values.

#### Scenario: CompressionConfig accepts kl_attribution

- **WHEN** `CompressionConfig(rep_selection="kl_attribution")` is constructed
- **THEN** the resulting config exposes `rep_selection == "kl_attribution"` and the other fields take their existing defaults

#### Scenario: CompressionConfig rejects unknown rep_selection

- **WHEN** `CompressionConfig(rep_selection="bogus")` is constructed
- **THEN** `__post_init__` raises `ValueError` whose message lists all three supported values (`n_fires`, `scale_aware`, `kl_attribution`)

#### Scenario: CompressionConfig is byte-identical when rep_selection is unset

- **WHEN** `CompressionConfig()` is constructed with no kwargs
- **THEN** `rep_selection == "scale_aware"` and a `Compressor` constructed from this config exhibits the historical `plan()` → `apply()` byte output on the existing toy fixture (the kl_attribution code path is not entered)

#### Scenario: CompressionConfig round-trips kl_attribution through dict

- **WHEN** `CompressionConfig(rep_selection="kl_attribution").to_dict()` is fed back through `CompressionConfig.from_dict`
- **THEN** the resulting instance equals the original
