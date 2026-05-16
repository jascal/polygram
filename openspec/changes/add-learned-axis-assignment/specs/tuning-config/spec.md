## ADDED Requirements

### Requirement: `SAEImportConfig.learn_axis_assignment` plumbs the kwarg

`polygram.config.SAEImportConfig` SHALL gain a
`learn_axis_assignment: bool | None = None` field. When the config
is consumed by `from_sae_lens` (via the `config=` keyword), the
field's value SHALL flow through to `from_sae_lens(...,
learn_axis_assignment=...)`.

The config field accepts only `None`, `False`, or `True` — explicit
`LearnedKnobAssignment` instances are not config-serialisable and
must be passed directly to `from_sae_lens(...)`. Callers who need
custom strategies bypass the config path.

The field defaults to `None`, preserving bit-exact existing
behaviour for every existing config-driven import.

#### Scenario: config field defaults to None

- **WHEN** `SAEImportConfig()` is instantiated without
  `learn_axis_assignment`
- **THEN** `config.learn_axis_assignment is None`

#### Scenario: config field flows through to from_sae_lens

- **WHEN** `from_sae_lens(records, ids, config=
  SAEImportConfig(learn_axis_assignment=True))` is called
- **THEN** the import uses the default
  `LearnedKnobAssignment` strategy

#### Scenario: config field round-trips through dict

- **WHEN** `SAEImportConfig(learn_axis_assignment=True).to_dict()`
  is called and the resulting dict is round-tripped via
  `SAEImportConfig.from_dict(d)`
- **THEN** the reconstructed config has `learn_axis_assignment is
  True`
