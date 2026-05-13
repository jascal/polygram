## MODIFIED Requirements

### Requirement: SAE loader enforces the per-encoding feature cap

`polygram.sae_import.from_sae_lens` and `polygram.sae_import.load_sae_safetensors` SHALL enforce the target encoding's `max_features` value when validating the requested feature subset.

The previous behaviour (using a hardcoded module constant
`MAX_FEATURES_PER_DICTIONARY = 8`) is replaced. The constant is
retained as a top-level export for back-compat at the MPSRung1 value
but no loader code path consults it after this change.

#### Scenario: MPSRung1 8-feature load unchanged

- **WHEN** `from_sae_lens(..., encoding=MPSRung1())` is called with 8
  feature ids
- **THEN** the result is identical to the pre-change implementation
  (same dictionary shape, same SelectionReport)

#### Scenario: Rung3 12-feature load now succeeds

- **WHEN** `from_sae_lens(..., encoding=Rung3())` is called with 12
  feature ids
- **THEN** no `ValueError` is raised and a 12-feature Dictionary is
  returned

#### Scenario: Rung3 17-feature load raises with the encoding-named error

- **WHEN** `from_sae_lens(..., encoding=Rung3())` is called with 17
  feature ids
- **THEN** a `ValueError` is raised whose message contains both the
  string `Rung3` and the string `16`
