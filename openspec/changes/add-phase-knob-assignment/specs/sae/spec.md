# add-phase-knob-assignment Specification

## Purpose

The `add-phase-knob-assignment` capability extends `from_sae_lens` so the loader can populate the MPS-substrate `α` and `φ` knobs from decoder geometry. With the default `assign_phase_knobs=False`, behaviour is byte-identical to the pre-change loader. With `assign_phase_knobs=True`, every encoding that has `α` and `φ` knobs (MPSRung1, Rung3, Rung4) receives per-feature values derived from the projection vectors' top PCA axes 2 and 3.

This is a parallel addition to PR #63's `assign_amp_knobs` (which targets amp-branch knobs on Rung3/Rung4). The two flags are orthogonal — users can enable either, both, or neither.

## ADDED Requirements

### Requirement: `from_sae_lens` accepts `assign_phase_knobs`

`polygram.sae_import.from_sae_lens` SHALL accept a new keyword argument `assign_phase_knobs: bool | None = None`. The precedence resolution SHALL mirror `assign_gamma` and `assign_amp_knobs`: explicit kwarg > `config.assign_phase_knobs` > default `False`.

When `assign_phase_knobs=False` (the default), the loader SHALL produce `Feature` objects with the encoding's default `α` and `φ` values (typically `0.0` for both). The resulting `Dictionary.gram()` SHALL be bit-identical to the pre-change loader on the same inputs.

When `assign_phase_knobs=True`, the loader SHALL populate each `Feature.alpha` from PCA axis 2 (PC2) of the centered projection vectors, and `Feature.phi` from PCA axis 3 (PC3). Both are rescaled linearly into `[0, 2π]`.

When `assign_phase_knobs=True` and the encoding has no `α` and `φ` knobs (currently only `HEA_Rung2`), the flag SHALL be a no-op. The loader SHOULD emit a `logging.debug` (or INFO-once) message naming the encoding.

#### Scenario: default `assign_phase_knobs=False` preserves byte-identity

- **WHEN** `from_sae_lens(records, encoding=MPSRung1())` is called without `assign_phase_knobs`
- **THEN** the resulting `Dictionary.gram()` is bit-identical to the pre-change loader on the same inputs
- **AND** every existing `from_sae_lens` test passes unchanged

#### Scenario: `assign_phase_knobs=True` materially reduces MPSRung1 gram saturation

- **WHEN** `from_sae_lens(records, encoding=MPSRung1(), assign_phase_knobs=True)` is called on a toy SAE fixture with 8 features
- **THEN** the resulting `Dictionary.gram()`'s squared modulus differs from the `assign_phase_knobs=False` path by Frobenius distance > 1.0
- **AND** the mean off-diagonal `|G|²` drops to below half its `assign_phase_knobs=False` value (sanity check: 0.76 → 0.28, a 63% drop)

#### Scenario: `assign_phase_knobs=True` works on Rung3 and Rung4 too

- **WHEN** `from_sae_lens(records, encoding=Rung3(), assign_phase_knobs=True)` is called on the same fixture
- **THEN** the resulting `Dictionary.gram()` differs from the `assign_phase_knobs=False` path by Frobenius distance > 1e-3
- **AND** the same invariant holds for `encoding=Rung4()`

#### Scenario: `assign_phase_knobs=True` is a no-op for HEA_Rung2

- **WHEN** `from_sae_lens(records, encoding=HEA_Rung2(depth=1, n_qubits=3), assign_phase_knobs=True)` is called
- **THEN** the resulting `Dictionary.gram()` is bit-identical to the same call with `assign_phase_knobs=False`

#### Scenario: degenerate PCA falls back to encoding defaults

- **WHEN** `from_sae_lens(records, encoding=MPSRung1(), assign_phase_knobs=True)` is called on a 2-feature SAE (insufficient non-zero PCA axes)
- **THEN** the call does not raise
- **AND** features default to `alpha=0`, `phi=0` (encoding defaults) for the unavailable PCA axes

### Requirement: Both phase and amp flags can be enabled together

When `assign_phase_knobs=True` and `assign_amp_knobs=True` are both passed, `from_sae_lens` SHALL populate phase knobs from PCA axes 2-3 AND amp knobs from PCA axes 4-7. The two flags compose additively; neither suppresses the other.

#### Scenario: both flags on for Rung4

- **WHEN** `from_sae_lens(records, encoding=Rung4(), assign_phase_knobs=True, assign_amp_knobs=True)` is called
- **THEN** each feature has non-default `alpha`, `phi`, `theta_amp`, `psi_aux`, `theta_amp_b`, `psi_amp_b` (six populated knob channels)
- **AND** the resulting gram differs from amp-only-on by a measurable Frobenius distance

### Requirement: `SAEImportConfig` propagates `assign_phase_knobs`

`polygram.config.SAEImportConfig` SHALL gain an `assign_phase_knobs: bool = False` field. `from_sae_lens` SHALL honor this field's value when the kwarg is not passed explicitly, mirroring the existing `assign_gamma` and `assign_amp_knobs` precedence pattern.

#### Scenario: SAEImportConfig field propagates to the loader

- **WHEN** `from_sae_lens(records, encoding=MPSRung1(), config=SAEImportConfig(assign_phase_knobs=True))` is called
- **THEN** the loader behaves as if `assign_phase_knobs=True` were passed directly

## MODIFIED Requirements

### Requirement: `KnobAssignment` strategy supports phase-knob assignment

`polygram.geometry.protocols.KnobAssignment.assign` SHALL accept a new keyword argument `assign_phase_knobs: bool = False`. The default preserves back-compat for any strategy that doesn't yet accept the new kwarg (via `**kwargs` or an explicit update).

Strategies that implement phase-knob assignment SHALL populate the result's `alphas` and `phis` fields with per-feature values derived from the projection geometry (PCA axes 2 and 3).

Strategies that do not implement phase-knob assignment SHALL leave the result's `alphas` and `phis` fields as `None`. The loader treats `None` as "use encoding defaults".

#### Scenario: ClusteredKnobAssignment honours `assign_phase_knobs=True`

- **WHEN** `ClusteredKnobAssignment().assign(projections, feature_names, ..., assign_phase_knobs=True, encoding=MPSRung1())` is called
- **THEN** the returned `KnobAssignmentResult` has non-`None` `alphas` and `phis` lists, each of length equal to `len(feature_names)`, with values in `[0, 2π]`

#### Scenario: UniformSphereKnobAssignment honours `assign_phase_knobs=True`

- **WHEN** `UniformSphereKnobAssignment().assign(projections, feature_names, ..., assign_phase_knobs=True, encoding=Rung4())` is called
- **THEN** the same shape + range invariants hold

### Requirement: `KnobAssignmentResult` carries optional phase-knob arrays

`polygram.geometry.protocols.KnobAssignmentResult` SHALL declare two new optional fields:

- `alphas: list[float] | None = None`
- `phis: list[float] | None = None`

When non-`None`, each list SHALL have length equal to `len(cluster_per_feature)`. `from_sae_lens` consumes these fields when constructing `Feature` objects.

#### Scenario: phase-knob arrays match the per-feature count when populated

- **WHEN** a strategy populates `alphas` on a `KnobAssignmentResult` for `n` features
- **THEN** the list has length exactly `n`
- **AND** every entry is a finite float in `[0, 2π]`

#### Scenario: default `None` fields signal "use encoding defaults"

- **WHEN** a `KnobAssignmentResult` is constructed without populating `alphas` and `phis`
- **THEN** both fields are `None`
- **AND** `from_sae_lens` builds `Feature` objects with the encoding's `alpha` and `phi` defaults (typically `0`) rather than overriding them

### Requirement: `assign_amp_knobs_pca` axis allocation shifts to axes 4-7

The helper `polygram.geometry.amp_assignment.assign_amp_knobs_pca` SHALL allocate amp-branch knobs to PCA axes 4-7 (zero-indexed 3-6) rather than the PR-#63-era axes 2-5 (zero-indexed 1-4). This shift makes room for phase knobs on axes 2-3.

This is a backward-incompat change for `assign_amp_knobs=True` callers — exact gram-condition numbers from PR-#63-era artifacts will not reproduce. The qualitative invariant (amp-on differs measurably from amp-off) is preserved.

#### Scenario: amp knobs after the shift use PCA axes 4-7

- **WHEN** `from_sae_lens(records, encoding=Rung4(), assign_amp_knobs=True)` is called on a SAE with at least 7 non-zero PCA axes
- **THEN** `Feature.theta_amp` derives from PCA axis 4 (PC4)
- **AND** `Feature.psi_aux` derives from PCA axis 5 (PC5)
- **AND** `Feature.theta_amp_b` derives from PCA axis 6 (PC6)
- **AND** `Feature.psi_amp_b` derives from PCA axis 7 (PC7)
