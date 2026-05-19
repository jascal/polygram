## MODIFIED Requirements

### Requirement: SAE loader enforces the per-encoding feature cap

`polygram.sae_import.from_sae_lens` and `polygram.sae_import.load_sae_safetensors` SHALL enforce the target encoding's `max_features` value, with a three-state `clustered` kwarg controlling the behaviour when the requested subset exceeds that cap:

- `clustered=False` — strict. Raises `ValueError` whose message names the encoding and the cap.
- `clustered=True` — always build a `ClusteredDictionary`, regardless of subset size.
- `clustered=None` — **auto** (the new default). When `len(feature_ids) <= cap` a flat `Dictionary` is returned; when `len(feature_ids) > cap` the loader auto-promotes to `ClusteredDictionary` and appends a warning to `SelectionReport.warnings` of the form `"auto-promoted to clustered: N=<N> exceeds <Encoding>.max_features=<cap>"`.

The previous behaviour (the default raised when N exceeded the cap)
is replaced by the auto branch above. Callers can recover the strict
error by passing `clustered=False` explicitly.

#### Scenario: MPSRung1 8-feature load unchanged

- **WHEN** `from_sae_lens(..., encoding=MPSRung1())` is called with 8
  feature ids
- **THEN** a flat `Dictionary` is returned and
  `report.warnings` contains no auto-promote entry

#### Scenario: oversized subset auto-promotes by default

- **WHEN** `from_sae_lens(records, list(range(16)))` is called against
  the default `MPSRung1` encoding (cap = 8) with no `clustered`
  kwarg
- **THEN** the returned object is a `ClusteredDictionary`, and
  `report.warnings` contains an entry whose text begins with
  `"auto-promoted to clustered"`

#### Scenario: explicit clustered=False keeps the strict error

- **WHEN** `from_sae_lens(records, list(range(9)), clustered=False)`
  is called against `MPSRung1` (cap = 8)
- **THEN** a `ValueError` is raised whose message contains both
  `"MPSRung1"` and `"8"`

#### Scenario: explicit clustered=True succeeds regardless of cap

- **WHEN** `from_sae_lens(records, list(range(16)), clustered=True)`
  is called against `MPSRung1`
- **THEN** the returned object is a `ClusteredDictionary` and
  `report.warnings` contains no auto-promote entry
