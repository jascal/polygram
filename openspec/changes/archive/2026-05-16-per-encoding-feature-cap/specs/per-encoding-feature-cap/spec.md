## ADDED Requirements

### Requirement: Every encoding declares its max-features cap

Every encoding class in `polygram.encoding` SHALL expose a `max_features` attribute or property returning the maximum number of linearly-independent features the encoding's state space can hold. The value SHALL equal the dimension of the encoding's reachable Hilbert subspace.

This applies to `MPSRung1`, `Rung3`, `HEA_Rung2`, and any future encodings.

The values for the currently-shipped encodings SHALL be:

- `MPSRung1.max_features == 8` (`dim C⁸ = 8`).
- `Rung3.max_features == 16` (`dim C⁸ ⊗ C² = 16`; the amp branch
  parameterisation restricts to a 2-dim subspace, see
  `docs/research/rung3-rank-bound.md`).
- `HEA_Rung2.max_features == 2 ** self.n_qubits`.

#### Scenario: MPSRung1 cap unchanged

- **WHEN** `MPSRung1().max_features` is accessed
- **THEN** the value 8 is returned

#### Scenario: Rung3 cap matches the rank-bound finding

- **WHEN** `Rung3().max_features` is accessed
- **THEN** the value 16 is returned

#### Scenario: HEA cap scales with n_qubits

- **WHEN** `HEA_Rung2(depth=1, n_qubits=4).max_features` is accessed
- **THEN** the value 16 is returned

- **WHEN** `HEA_Rung2(depth=2, n_qubits=10).max_features` is accessed
- **THEN** the value 1024 is returned

### Requirement: Encoding cap is consulted at every enforcement site

Every site that previously enforced `MAX_FEATURES_PER_DICTIONARY` SHALL instead query the relevant encoding's `max_features` attribute. The sites in scope:

- `polygram.sae_import.from_sae_lens` and `load_sae_safetensors` —
  loader-time cap.
- `polygram.behavioural.validator.BehaviouralValidator.feature_ids` —
  validator-time cap.
- Any other site catalogued during the §4 audit.

The error message raised on a cap violation SHALL name the encoding
and its declared cap, and suggest the next encoding tier if the user
wants more capacity. The exact phrasing is implementation-detail but
the requirement that the encoding name and value appear in the
message is testable.

#### Scenario: error message names the encoding and the cap

- **WHEN** `from_sae_lens(...)` is called with 17 features against a
  `Rung3` encoding
- **THEN** the raised `ValueError` message contains the string
  `Rung3` and the string `16`

#### Scenario: cap consultation is per-call, not cached

- **WHEN** two `from_sae_lens` calls in the same process target
  different encodings (`MPSRung1`, then `Rung3`) with feature counts
  9 and 12 respectively
- **THEN** the first call raises (MPSRung1 cap 8) and the second
  succeeds (Rung3 cap 16)

### Requirement: MAX_FEATURES_PER_DICTIONARY is retained as a back-compat re-export

`polygram.sae_import.MAX_FEATURES_PER_DICTIONARY` SHALL continue to
exist as a module-level constant with value 8 (the MPSRung1 cap).
The constant SHALL NOT be used for enforcement after this change.

This requirement exists to avoid breaking external scripts and
existing `from polygram.sae_import import MAX_FEATURES_PER_DICTIONARY`
imports while the actual cap moves into per-encoding methods.

#### Scenario: back-compat import still works

- **WHEN** `from polygram.sae_import import MAX_FEATURES_PER_DICTIONARY`
  is executed
- **THEN** the import succeeds and the constant equals 8
