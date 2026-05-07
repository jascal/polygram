## MODIFIED Requirements

### Requirement: from_sae_lens optional gamma auto-assignment

`from_sae_lens` SHALL accept `assign_gamma: bool = True` and
`gamma_range: tuple[float, float] = (-0.25, 0.25)` keyword
arguments. When `assign_gamma=True` (the default), each
feature's γ SHALL be its projection vector's coefficient on the
first principal component of its assigned cluster's centered
projection vectors, rescaled into `gamma_range`. When
`assign_gamma=False`, every feature's γ SHALL be 0.0.
`report.gamma_method` records `"projection_pca"` (default) or
`"zero"` accordingly. α and φ defaults are unchanged (both 0).

The default flips from `False` to `True` because γ=0 collapses
every in-cluster feature onto the same encoded state on real
SAEs; the README has long warned this is "almost always wrong"
on production inputs. Callers that genuinely want the legacy
γ=0 behaviour SHALL pass `assign_gamma=False` explicitly.

`from_sae_lens` SHALL also accept a keyword argument
`config: SAEImportConfig | None = None`. When `config` is
supplied, its `assign_gamma`, `gamma_range`, and `n_clusters`
fields SHALL provide values for the corresponding parameters.
Per-field kwargs override `config`; `config` overrides
`SAEImportConfig` defaults.

#### Scenario: assign_gamma writes nonzero per-feature gamma

- **WHEN** `from_sae_lens(records, [0, 1, 4, 5])` is called on
  the bundled toy fixture (using the new default)
- **THEN** `report.gamma_method == "projection_pca"`, the
  returned Dictionary has at least one feature with `gamma != 0`,
  and all `feature.gamma` values lie within `gamma_range`

#### Scenario: explicit assign_gamma=False keeps gammas at zero

- **WHEN** `from_sae_lens(records, [0, 1, 4, 5],
  assign_gamma=False)` is called
- **THEN** every returned feature has `gamma == 0` and
  `report.gamma_method == "zero"`

#### Scenario: config supplies assign_gamma when kwarg omitted

- **GIVEN** `cfg = SAEImportConfig(assign_gamma=False)`
- **WHEN** `from_sae_lens(records, [0, 1, 4, 5], config=cfg)` is
  called
- **THEN** every returned feature has `gamma == 0` and
  `report.gamma_method == "zero"`

#### Scenario: kwarg overrides config

- **GIVEN** `cfg = SAEImportConfig(assign_gamma=False)`
- **WHEN** `from_sae_lens(records, [0, 1, 4, 5], config=cfg,
  assign_gamma=True)` is called
- **THEN** `report.gamma_method == "projection_pca"` and at
  least one feature has `gamma != 0`
