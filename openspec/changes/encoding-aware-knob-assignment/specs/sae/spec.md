# encoding-aware-knob-assignment Specification

## Purpose

The `encoding-aware-knob-assignment` capability extends `from_sae_lens` so the loader can populate higher-rung encodings' amp-branch knobs from decoder geometry, rather than leaving them at their MPS-equivalent defaults.

With the default `assign_amp_knobs=False`, behaviour is byte-identical to the pre-change loader. With `assign_amp_knobs=True`, the loader propagates per-feature `theta_amp`, `psi_aux`, and (for Rung4) `theta_amp_b`, `psi_amp_b` values into the resulting `Dictionary`'s `Feature` objects — producing a gram that *uses* the encoding's full state space rather than aliasing MPSRung1.

## ADDED Requirements

### Requirement: `from_sae_lens` accepts `assign_amp_knobs`

`polygram.sae_import.from_sae_lens` SHALL accept a new keyword argument `assign_amp_knobs: bool = False`.

When `assign_amp_knobs=False` (the default), the loader produces `Feature` objects with the encoding's default amp-branch knob values (`theta_amp=π/4, psi_aux=0, theta_amp_b=π/4, psi_amp_b=0`). The resulting `Dictionary.gram()` is bit-identical to the pre-change loader on the same inputs.

When `assign_amp_knobs=True` and the encoding has amp-branch knobs (`Rung3` or `Rung4`), the loader SHALL populate the relevant amp-knob fields on each `Feature` from the chosen strategy's assignment. The default strategy (PCA-axis extension, see Decision 1 in design.md) maps higher PCA axes of the projection vectors into the amp-knob ranges.

When `assign_amp_knobs=True` and the encoding has no amp-branch knobs (`MPSRung1`) or has a non-amp knob structure (`HEA_Rung2`), the flag SHALL be a no-op. The loader SHOULD emit a `logging.debug` message naming the encoding so callers can verify the flag's effect.

#### Scenario: default `assign_amp_knobs=False` preserves byte-identity

- **WHEN** `from_sae_lens(records, encoding=Rung4())` is called without `assign_amp_knobs`
- **THEN** the resulting `Dictionary.gram()` is bit-identical to `from_sae_lens(records, encoding=Rung4(), assign_amp_knobs=False)`
- **AND** bit-identical to the pre-change loader (verified by existing differential regression tests)

#### Scenario: `assign_amp_knobs=True` changes Rung4 gram

- **WHEN** `from_sae_lens(records, encoding=Rung4(), assign_amp_knobs=True)` is called on a toy SAE fixture with ≥5 features
- **THEN** the resulting `Dictionary.gram()`'s squared modulus differs from the `assign_amp_knobs=False` path by Frobenius distance > 1e-3
- **AND** the difference is not concentrated on the diagonal (off-diagonal Frobenius distance also > 1e-3)

#### Scenario: `assign_amp_knobs=True` changes Rung3 gram

- **WHEN** `from_sae_lens(records, encoding=Rung3(), assign_amp_knobs=True)` is called on a toy SAE fixture with ≥3 features
- **THEN** the resulting `Dictionary.gram()`'s squared modulus differs from the `assign_amp_knobs=False` path by Frobenius distance > 1e-3

#### Scenario: `assign_amp_knobs=True` is a no-op for MPSRung1

- **WHEN** `from_sae_lens(records, encoding=MPSRung1(), assign_amp_knobs=True)` is called
- **THEN** the resulting `Dictionary.gram()` is bit-identical to the same call with `assign_amp_knobs=False`

#### Scenario: degenerate PCA falls back to defaults

- **WHEN** `from_sae_lens(records, encoding=Rung4(), assign_amp_knobs=True)` is called on a 2-feature SAE (insufficient non-zero PCA axes for full Rung4 assignment)
- **THEN** the call does not raise
- **AND** amp-branch knobs corresponding to unavailable PCA axes fall back to the encoding's default value (π/4 for theta_amp, 0 for psi_aux, etc.)

### Requirement: `SAEImportConfig` propagates `assign_amp_knobs`

`polygram.config.SAEImportConfig` SHALL gain an `assign_amp_knobs: bool = False` field. `from_sae_lens` SHALL honor this field's value when the kwarg is not passed explicitly, mirroring the existing `assign_gamma` precedence pattern.

#### Scenario: SAEImportConfig field propagates to the loader

- **WHEN** `from_sae_lens(records, encoding=Rung4(), config=SAEImportConfig(assign_amp_knobs=True))` is called
- **THEN** the loader behaves as if `assign_amp_knobs=True` were passed directly

## MODIFIED Requirements

### Requirement: `KnobAssignment` strategy supports amp-knob assignment

`polygram.geometry.protocols.KnobAssignment.assign` SHALL accept two new keyword arguments: `assign_amp_knobs: bool = False` and `encoding: object = None`. The defaults preserve back-compat for any strategy that doesn't yet accept the new kwargs (via `**kwargs` or an explicit update).

Strategies that implement amp-knob assignment SHALL populate the result's `theta_amps`, `psi_auxes`, `theta_amp_bs`, `psi_amp_bs` fields with per-feature values derived from the projection geometry. The per-knob arrays MAY be `None` for unused knob channels (e.g., `theta_amp_bs` and `psi_amp_bs` remain `None` when `encoding=Rung3()` since Rung3 has no branch-B).

Strategies that do not implement amp-knob assignment SHALL leave the result's amp-knob fields as `None`. The loader treats `None` fields as "use encoding defaults".

#### Scenario: ClusteredKnobAssignment honours `assign_amp_knobs=True`

- **WHEN** `ClusteredKnobAssignment().assign(projections, feature_names, ..., assign_amp_knobs=True, encoding=Rung4())` is called
- **THEN** the returned `KnobAssignmentResult` has non-`None` values for all four amp-knob arrays
- **AND** each array has length equal to `len(feature_names)`
- **AND** the values are bounded by the per-knob ranges (`theta_amps ∈ [0, π/2]`, `psi_auxes ∈ [0, 2π]`, etc.)

#### Scenario: UniformSphereKnobAssignment honours `assign_amp_knobs=True`

- **WHEN** `UniformSphereKnobAssignment().assign(projections, feature_names, ..., assign_amp_knobs=True, encoding=Rung4())` is called
- **THEN** the returned `KnobAssignmentResult` has non-`None` values for all four amp-knob arrays with the same shape + range invariants as ClusteredKnobAssignment

### Requirement: `KnobAssignmentResult` carries optional amp-knob arrays

`polygram.geometry.protocols.KnobAssignmentResult` SHALL declare four new optional fields:

- `theta_amps: list[float] | None = None`
- `psi_auxes: list[float] | None = None`
- `theta_amp_bs: list[float] | None = None`
- `psi_amp_bs: list[float] | None = None`

When non-`None`, each list SHALL have length equal to `len(cluster_per_feature)`. `from_sae_lens` consumes these fields when constructing `Feature` objects.

#### Scenario: amp-knob arrays match the per-feature count when populated

- **WHEN** a strategy populates `theta_amps` (or any other amp-knob array) on a `KnobAssignmentResult` for `n` features
- **THEN** the list has length exactly `n`
- **AND** every entry is a finite float

#### Scenario: default `None` fields signal "use encoding defaults"

- **WHEN** a `KnobAssignmentResult` is constructed without populating the amp-knob arrays
- **THEN** all four amp-knob fields are `None`
- **AND** `from_sae_lens` builds `Feature` objects with the encoding's amp-knob defaults rather than overriding them
