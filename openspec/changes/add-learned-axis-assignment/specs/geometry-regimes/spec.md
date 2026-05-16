## ADDED Requirements

### Requirement: `KnobAssignmentResult` carries an optional `axis_assignment` field

`polygram.geometry.protocols.KnobAssignmentResult` SHALL include
`axis_assignment: dict[str, int | list[float]] | None = None`. The
field surfaces *which PCA axis (or which linear combination of
axes)* fed each polygram knob slot during the import.

`ClusteredKnobAssignment` and `UniformSphereKnobAssignment` SHALL
leave the field at the default `None`. `LearnedAxisAssignment`
SHALL populate it per the
[`learned-axis-assignment` capability spec](../learned-axis-assignment/spec.md).

The field is optional rather than mandatory so adding it does not
churn every existing strategy implementation.

#### Scenario: existing strategies leave the field None

- **WHEN** `ClusteredKnobAssignment().assign(...)` or
  `UniformSphereKnobAssignment().assign(...)` returns
- **THEN** `result.axis_assignment is None`

#### Scenario: learned strategy populates the field

- **WHEN** `LearnedAxisAssignment().assign(...)` returns
- **THEN** `result.axis_assignment` is a non-empty dict whose keys
  are knob names

### Requirement: `LearnedAxisObjective` protocol formalises the objective surface

`polygram.geometry.LearnedAxisObjective` SHALL be a
`runtime_checkable` Protocol with the call signature
`__call__(analytic_gram: np.ndarray, decoder_geom: np.ndarray, *,
feature_names: list[str]) -> float`. The return value is a scalar
that the learned-assignment solver maximises.

Three built-in objectives SHALL be available in
`polygram.geometry.objectives`:

- `spearman_objective` — Spearman rank correlation on off-diagonal
  upper-triangle of `|analytic_gram|²` and `decoder_geom`.
- `pearson_objective` — Pearson correlation on the same entries.
- `behavioural_objective(reference_pair_sims)` — factory returning
  a closure that scores against a user-supplied square matrix.

User-defined objectives are accepted by the strategy as long as
they satisfy the protocol.

#### Scenario: built-in objectives satisfy the protocol

- **WHEN** `isinstance(spearman_objective, LearnedAxisObjective)`,
  `isinstance(pearson_objective, LearnedAxisObjective)`, and
  `isinstance(behavioural_objective(some_matrix), LearnedAxisObjective)`
  are evaluated
- **THEN** all three return `True`

#### Scenario: user callable accepted as objective

- **WHEN** a user defines `def my_obj(g, d, *, feature_names):
  return float(np.real(g).mean())` and passes it via
  `LearnedAxisAssignment(objective=my_obj)`
- **THEN** the strategy accepts it without complaint and uses it
  during the search
