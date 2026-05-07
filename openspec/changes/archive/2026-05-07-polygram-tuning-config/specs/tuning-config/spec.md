## ADDED Requirements

### Requirement: polygram.config exposes tuning dataclasses

The `polygram.config` module SHALL export five frozen dataclasses
representing the configurable knobs of polygram's public
constructors:

- `CompressionConfig` — `strategy: str = "merge"`,
  `rep_selection: str = "scale_aware"`,
  `merge_mode: str = "freq_weighted"`, `confirmer: str | None = None`.
- `EpochCompressionConfig` — `coverage_target: float = 0.5`,
  `cosine_threshold: float = 0.30`,
  `n_visits_per_feature: int = 1`,
  `max_iterations: int = 1`,
  `quality_delta_multiplier: float = 2.0`,
  `validation: ValidationConfig | None = None`.
- `CancellationConfig` — `tolerance: float = 0.05`,
  `preserve_tiers: bool = True`,
  `optimize: dict[str, Any] = {"method": "grid", "max_steps": 50}`,
  `grid_outer: tuple[int, int] = (5, 5)`,
  `min_amp_overlap: float = 0.0`.
- `ValidationConfig` — `polygram_overlap_threshold: float = 0.7`,
  `jaccard_threshold: float = 0.30`,
  `min_firing_rate: float = 0.01`,
  `min_both_fire: int = 5`,
  `allow_layer_zero: bool = False`.
- `RegrowConfig` — `strategy: str = "residual_kmeans"`,
  `prompts: tuple[str, ...] | None = None`,
  `seed: int = 0`, `n_init: int = 4`,
  `model_name: str` (required, no default),
  `layer: int` (required, no default),
  `device: str | None = None`.
- `SAEImportConfig` — `assign_gamma: bool = True`,
  `gamma_range: tuple[float, float] = (-0.25, 0.25)`,
  `n_clusters: int = 2`.

Each dataclass SHALL be frozen (`frozen=True`) so a single config
instance can be shared across calls without mutation hazards. Each
dataclass SHALL run range/value validation in `__post_init__`
mirroring the validation that previously lived on the consuming
constructor.

#### Scenario: every config is importable from polygram.config

- **WHEN** `from polygram.config import CompressionConfig,
  EpochCompressionConfig, CancellationConfig, ValidationConfig,
  RegrowConfig, SAEImportConfig` is executed
- **THEN** all six names resolve and each is a `dataclass` whose
  `__dataclass_params__.frozen` is `True`

#### Scenario: invalid range raises in __post_init__

- **WHEN** `EpochCompressionConfig(coverage_target=1.5)` is
  constructed
- **THEN** `__post_init__` raises `ValueError` naming
  `coverage_target` and the valid range `(0, 1]`

#### Scenario: RegrowConfig requires model_name and layer

- **WHEN** `RegrowConfig(model_name="gpt2-medium")` is
  constructed without `layer`
- **THEN** Python raises `TypeError` for the missing required
  keyword argument; the same applies if `model_name` is omitted

### Requirement: Configs round-trip through dict

Each config dataclass SHALL expose `.to_dict()` and a classmethod
`.from_dict(cls, data: Mapping[str, Any]) -> Self`.

`.to_dict()` SHALL produce a JSON-serializable mapping (tuples
serialised as lists, nested configs serialised recursively).

`.from_dict()` SHALL:

- Accept either tuples or lists for tuple-typed fields and coerce
  to the declared type.
- Recurse into nested-config fields (e.g.
  `EpochCompressionConfig.validation`) when those keys are present.
- Emit a `UserWarning` (via `warnings.warn`) and ignore the value
  when an unknown key is supplied; this preserves forward
  compatibility when a downstream caller's stored config predates
  a knob being added.

#### Scenario: dict round-trip preserves field values

- **GIVEN** `cfg = EpochCompressionConfig(coverage_target=0.7,
  validation=ValidationConfig(polygram_overlap_threshold=0.8))`
- **WHEN** `EpochCompressionConfig.from_dict(cfg.to_dict())`
  is evaluated
- **THEN** the result equals `cfg` (including the nested
  `validation`)

#### Scenario: unknown key warns and is ignored

- **WHEN** `CompressionConfig.from_dict({"strategy": "merge",
  "futurefield": 42})` is called
- **THEN** a `UserWarning` is emitted naming `futurefield`, and
  the returned instance has `strategy == "merge"` and all other
  fields at their defaults

#### Scenario: list deserialises to tuple-typed field

- **WHEN** `CancellationConfig.from_dict({"grid_outer": [3, 4]})`
  is called
- **THEN** the returned instance has
  `grid_outer == (3, 4)` (a `tuple`, not a `list`)

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
