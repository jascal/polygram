# tuning-config Specification

## Purpose

Polygram's tuning-config layer: frozen dataclasses in
`polygram.config` that capture the tunable knobs of the public
constructors (`Compressor`, `EpochCompressor`, `Cancellation`,
`BehaviouralValidator`, `Regrower.from_compression_report`,
`from_sae_lens`). Configs are shareable, dict-serialisable, and
override the constructors' own defaults while still yielding to
explicit per-field keyword arguments — giving callers one place
to declare a tuning profile without breaking legacy call sites.
## Requirements
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

### Requirement: Public constructors accept config= kwarg with override precedence

Public constructors SHALL accept an optional keyword argument `config` whose type is the matching dataclass from `polygram.config`. The constructors covered are `Compressor`, `EpochCompressor`, `Cancellation`, `BehaviouralValidator`, `Regrower.from_compression_report`, and `from_sae_lens`. When `config` is supplied, fields from `config` SHALL be used as the source of truth except where a per-field keyword argument is also explicitly supplied (i.e. is non-None for optional fields, or otherwise present in the call). Per-field kwargs SHALL win over `config`; `config` SHALL win over the constructor's own dataclass defaults.

When `config` is `None` (the default), behaviour SHALL be
identical to today — per-field kwargs apply with their existing
defaults.

#### Scenario: per-field kwarg overrides config

- **GIVEN** `cfg = CompressionConfig(strategy="merge",
  rep_selection="scale_aware")`
- **WHEN** `Compressor(validation_report=..., sae_checkpoint=...,
  config=cfg, strategy="zero")` is constructed
- **THEN** `compressor.strategy == "zero"` and
  `compressor.rep_selection == "scale_aware"` (config wins for
  the unsupplied field, kwarg wins for the supplied one)

#### Scenario: config-only call uses config values

- **GIVEN** `cfg = EpochCompressionConfig(coverage_target=0.9,
  max_iterations=3)`
- **WHEN** `EpochCompressor(config=cfg)` is constructed
- **THEN** the compressor's `coverage_target == 0.9` and
  `max_iterations == 3`, and other fields take the
  `EpochCompressionConfig` defaults

#### Scenario: no-config legacy call still works

- **WHEN** `EpochCompressor(coverage_target=0.7)` is constructed
  with no `config` argument
- **THEN** the compressor has `coverage_target == 0.7` and the
  remaining fields take the new dataclass defaults
  (`n_visits_per_feature=1`, `max_iterations=1`, …)

### Requirement: EpochCompressor named presets

`EpochCompressor` SHALL provide two classmethod presets:

- `EpochCompressor.fast()` — returns an instance whose tuning
  matches the dataclass defaults (`coverage_target=0.5`,
  `n_visits_per_feature=1`, `max_iterations=1`,
  `cosine_threshold=0.30`).
- `EpochCompressor.thorough()` — returns an instance with the
  pre-change "exhaustive offline run" defaults
  (`coverage_target=0.95`, `n_visits_per_feature=3`,
  `max_iterations=5`, `cosine_threshold=0.30`).

Both presets SHALL accept the same `**overrides` keyword surface
so a caller can write `EpochCompressor.fast(max_iterations=2)`.

#### Scenario: fast() matches defaults

- **WHEN** `a = EpochCompressor.fast()` and `b =
  EpochCompressor()` are constructed
- **THEN** `a.coverage_target == b.coverage_target`,
  `a.n_visits_per_feature == b.n_visits_per_feature`, and
  `a.max_iterations == b.max_iterations`

#### Scenario: thorough() restores prior defaults

- **WHEN** `EpochCompressor.thorough()` is constructed
- **THEN** the resulting instance has `coverage_target == 0.95`,
  `n_visits_per_feature == 3`, and `max_iterations == 5`

#### Scenario: preset accepts overrides

- **WHEN** `EpochCompressor.fast(coverage_target=0.6)` is
  constructed
- **THEN** the instance has `coverage_target == 0.6` and the
  other tuning fields match `EpochCompressor.fast()` values

### Requirement: Regrower.from_compression_report requires model_name and layer

`Regrower.from_compression_report` SHALL declare `model_name` and
`layer` as required keyword arguments (no default value). Calls
omitting either SHALL surface as `TypeError` at construction
time. The previous defaults (`model_name="gpt2"`, `layer=10`)
SHALL be removed.

#### Scenario: omitting model_name raises

- **WHEN** `Regrower.from_compression_report(report=...,
  sae_checkpoint=..., strategy="residual_kmeans", layer=10)` is
  called without `model_name`
- **THEN** Python raises `TypeError` naming the missing
  `model_name` keyword

#### Scenario: omitting layer raises

- **WHEN** `Regrower.from_compression_report(report=...,
  sae_checkpoint=..., strategy="residual_kmeans",
  model_name="pythia-160m")` is called without `layer`
- **THEN** Python raises `TypeError` naming the missing
  `layer` keyword

### Requirement: RegrowConfig accepts an optional top_k cap

`RegrowConfig` SHALL declare an optional `top_k: int | None = None` field. When `None` (the default), the regrower regrows every zeroed slot in the supplied `CompressionReport` — preserving the pre-change byte-identical behavior. When set to a non-negative integer, the regrower regrows only the first `top_k` slots in `RegrowPlan.populations` order, and the remaining slots stay zero in the output checkpoint.

`RegrowConfig.__post_init__` SHALL reject negative values with a clear `ValueError` naming the field and the offending value. Zero is a valid value (equivalent to skipping the regrower entirely).

The field SHALL round-trip cleanly through `RegrowConfig.to_dict()` / `RegrowConfig.from_dict()` so downstream consumers (e.g. sae-forge's FSM context plumbing) can serialize it without special handling.

#### Scenario: default None preserves byte-identical behavior

- **GIVEN** a `RegrowConfig(model_name="pythia-160m", layer=10, strategy="residual_kmeans", seed=0)` (no `top_k` set)
- **WHEN** `Regrower.from_compression_report(...).run(out)` is called against a fixture compression report with `n_features_zeroed = 12`
- **THEN** the output checkpoint's SHA matches the pre-change baseline (captured before this change landed)
- **AND** all 12 zeroed slots are regrown in the output

#### Scenario: top_k caps the regrown slot count

- **GIVEN** the same config but with `top_k=3`
- **WHEN** `run` is called
- **THEN** exactly 3 slots are regrown (the first three in plan order)
- **AND** the remaining 9 slots remain zero in the output checkpoint
- **AND** `RegrowReport.populations` has length 3

#### Scenario: top_k above plan size is a no-op cap

- **GIVEN** the same config but with `top_k=999` (well above `n_features_zeroed`)
- **WHEN** `run` is called
- **THEN** all 12 slots are regrown — equivalent to `top_k=None`
- **AND** no error is raised

#### Scenario: top_k zero is a valid no-regrow

- **GIVEN** the same config but with `top_k=0`
- **WHEN** `run` is called
- **THEN** the output checkpoint equals the input `CompressionReport.output_checkpoint` (no rows changed)
- **AND** `RegrowReport.populations` is empty

#### Scenario: negative top_k is rejected at construction

- **WHEN** `RegrowConfig(model_name="pythia-160m", layer=10, top_k=-1)` is constructed
- **THEN** `ValueError` is raised
- **AND** the message contains `"top_k"` and `"-1"`

### Requirement: Regrower.from_compression_report accepts top_k as a per-field kwarg

`Regrower.from_compression_report(...)` SHALL accept an optional `top_k: int | None = None` keyword argument. The precedence rule SHALL match the existing per-field-kwarg-vs-config behavior: when both the kwarg and `config.top_k` are set, the kwarg wins. When neither is set, the resulting regrower's `top_k` is `None`.

#### Scenario: per-field kwarg wins over config

- **GIVEN** `cfg = RegrowConfig(model_name="pythia-160m", layer=10, top_k=5)`
- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., config=cfg, top_k=2)` is called
- **THEN** the resulting regrower has `top_k == 2` (the kwarg, not 5)

#### Scenario: kwarg-only path

- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., model_name="pythia-160m", layer=10, top_k=3)` is called (no `config` kwarg)
- **THEN** the resulting regrower has `top_k == 3`

#### Scenario: config-only path

- **GIVEN** `cfg = RegrowConfig(model_name="pythia-160m", layer=10, top_k=4)`
- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., config=cfg)` is called (no `top_k` kwarg)
- **THEN** the resulting regrower has `top_k == 4`

#### Scenario: neither set defaults to None

- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., model_name="pythia-160m", layer=10)` is called (no `config`, no `top_k`)
- **THEN** the resulting regrower has `top_k == None`
- **AND** the regrower regrows every zeroed slot — pre-change behavior

