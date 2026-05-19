## MODIFIED Requirements

### Requirement: SAEImportConfig knob-assignment defaults follow sm-sae measured recommendations

`polygram.config.SAEImportConfig` SHALL default `assign_amp_knobs` and `assign_phase_knobs` to `True`, matching the [sm-sae](https://jascal.github.io/sm-sae/) benchmark's measured recommendation for ground-truth-rich SAE fixtures.

Previously both defaulted to `False`. Callers that depend on the legacy
`False` behaviour SHALL pass the kwargs explicitly to `from_sae_lens`
or `SAEImportConfig`.

The class docstring SHALL cite sm-sae's recommendation alongside the
existing `assign_gamma=True` rationale.

#### Scenario: SAEImportConfig() exposes True defaults for amp/phase knobs

- **WHEN** `SAEImportConfig()` is constructed with no arguments
- **THEN** `cfg.assign_amp_knobs is True` and
  `cfg.assign_phase_knobs is True`

#### Scenario: from_sae_lens populates amp and phase knobs by default

- **WHEN** `from_sae_lens(records, ids, encoding=Rung3())` is called
  on a Rung3-capable fixture with neither `assign_amp_knobs` nor
  `assign_phase_knobs` supplied
- **THEN** the returned `Dictionary`'s feature knobs include populated
  amp-branch entries (the encoding's `theta_amp` / `psi_aux` knobs)
  and populated MPS-substrate `alpha` / `phi` knobs

#### Scenario: explicit False preserves legacy behaviour

- **WHEN** `from_sae_lens(records, ids, assign_amp_knobs=False,
  assign_phase_knobs=False)` is called
- **THEN** the returned `Dictionary`'s features have unpopulated
  amp-branch knobs and zero-valued `alpha` / `phi`, matching the
  pre-change behaviour
