## MODIFIED Requirements

### Requirement: Compressor accepts rep_selection and merge_mode parameters
`Compressor` SHALL accept two new optional parameters at construction time:

- `rep_selection: str = "n_fires"` — controls how cluster representatives are chosen. Supported values: `"n_fires"`, `"scale_aware"`.
- `merge_mode: str = "freq_weighted"` — controls norm averaging for `strategy="merge"`. Supported values: `"freq_weighted"`, `"simple_mean"`.

Both parameters SHALL be validated in `__post_init__` and raise `ValueError` on unrecognised values.

`strategy` SHALL now accept `"merge"` in addition to `"zero"`.

All existing defaults (`strategy="zero"`, `rep_selection="n_fires"`) SHALL be preserved so existing call sites require no changes.

#### Scenario: default construction is backward-compatible
- **WHEN** `Compressor` is constructed with only `validation_report`, `sae_checkpoint`, and `strategy="zero"`
- **THEN** the result is identical to pre-change behaviour

#### Scenario: merge strategy accepted
- **WHEN** `Compressor` is constructed with `strategy="merge"`
- **THEN** no `ValueError` is raised and `apply()` uses the merge strategy
