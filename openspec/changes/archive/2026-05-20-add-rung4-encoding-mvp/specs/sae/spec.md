## MODIFIED Requirements

### Requirement: Feature carries q4 amp knobs in addition to q3 amp knobs

`polygram.dictionary.Feature` SHALL gain two new fields:

- `theta_amp_b: float = 0.0` — single-qubit amp θ knob for qubit 4
  under Rung4. Default 0.0 makes the Rung4 amp factor equal 1 with
  default knobs.
- `psi_amp_b: float = 0.0` — single-qubit amp ψ knob for qubit 4
  under Rung4.

The existing `theta_amp` and `psi_aux` fields SHALL be reinterpreted
as the q3 single-qubit amp knobs under Rung4's amp factorisation.
The field names do NOT change; only the geometric interpretation
under Rung4 differs from Rung3.

Default values for all four amp fields SHALL produce identity-like
overlaps:

- Under Rung3: `theta_amp = π/4`, `psi_aux = 0` makes amp factor = 1.
  Defaults remain `theta_amp = π/4`, `psi_aux = 0` (unchanged).
- Under Rung4: `theta_amp = theta_amp_b = 0`, `psi_aux = psi_amp_b = 0`
  makes amp factor = 1. New fields default to 0; existing fields
  remain at their pre-change defaults (π/4 and 0).

#### Scenario: existing Rung3 feature reconstruction is unaffected

- **WHEN** a `Feature` is constructed with the pre-change signature
  (no `theta_amp_b` or `psi_amp_b` arguments)
- **THEN** the result has `theta_amp_b == 0.0` and `psi_amp_b == 0.0`,
  and Rung3 gram math is byte-identical to the pre-change result

#### Scenario: Feature equality and hashing include the new fields

- **WHEN** two `Feature` instances differ only in `theta_amp_b`
- **THEN** `f1 == f2` is False and `hash(f1) != hash(f2)`

#### Scenario: SAE-import JSON round-trip handles pre-change fixtures

- **WHEN** a Dictionary saved before this change (no `theta_amp_b`
  / `psi_amp_b` columns) is loaded
- **THEN** the loaded Features reconstruct with the new fields at
  their 0.0 defaults

### Requirement: from_sae_lens accepts up to 32 features for Rung4

`polygram.sae_import.from_sae_lens` SHALL accept up to 32 feature ids
when the target encoding is `Rung4`. Above 32 the loader SHALL raise
the standard per-encoding-cap error, naming `Rung4` and `32`.

This requirement composes with `per-encoding-feature-cap`'s mechanism
— the loader queries `encoding.max_features` rather than the module
constant.

#### Scenario: 24-feature Rung4 load succeeds

- **WHEN** `from_sae_lens(... , encoding=Rung4())` is called with 24
  feature ids
- **THEN** no `ValueError` is raised and a 24-feature Dictionary is
  returned

#### Scenario: 33-feature Rung4 load fails

- **WHEN** `from_sae_lens(... , encoding=Rung4())` is called with 33
  feature ids
- **THEN** a `ValueError` is raised whose message contains the string
  `Rung4` and the string `32`
