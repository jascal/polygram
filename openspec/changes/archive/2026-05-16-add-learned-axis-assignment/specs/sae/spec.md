## ADDED Requirements

### Requirement: `from_sae_lens` accepts `learn_axis_assignment` kwarg

`polygram.sae_import.from_sae_lens` SHALL accept
`learn_axis_assignment: bool | LearnedKnobAssignment | None = None`
as a keyword-only argument.

- `None` or `False` (default): the import proceeds via the existing
  hardcoded `assign_amp_knobs_pca` + `assign_phase_knobs_pca`
  helpers — byte-identical to the pre-change behaviour.
- `True`: instantiate `LearnedKnobAssignment()` with default
  arguments (`solver="greedy"`,
  `objective=spearman_objective`) and use it for the import.
- A `LearnedKnobAssignment` instance: use it directly without
  modification.

When the learned strategy is in use, the per-feature knob arrays
(`alphas`, `phis`, `theta_amps`, `psi_auxes`, `theta_amp_bs`,
`psi_amp_bs`, `amp_knobs_list`) SHALL be populated via the
strategy's `assign()` return rather than via the hardcoded helpers.

#### Scenario: default behaviour byte-identical

- **WHEN** `from_sae_lens(records, ids, encoding=Rung4())` is called
  without `learn_axis_assignment`
- **THEN** the resulting Dictionary's gram matches the pre-change
  output to 1e-12 absolute tolerance on the toy SAE fixture

#### Scenario: True triggers default learned strategy

- **WHEN** `from_sae_lens(..., learn_axis_assignment=True)` is
  called
- **THEN** the import uses
  `LearnedKnobAssignment(solver="greedy", objective=spearman_objective)`

#### Scenario: explicit instance honoured

- **WHEN** `from_sae_lens(..., learn_axis_assignment=
  LearnedKnobAssignment(solver="scipy"))` is called
- **THEN** the import uses the scipy solver as configured

### Requirement: `SelectionReport.learned_axis_assignment` surfaces the learned map

`polygram.sae_import.SelectionReport` SHALL gain a
`learned_axis_assignment: dict[str, Any] | None = None` field.

When the import path runs `LearnedKnobAssignment`, the field SHALL
be populated with a dict containing at minimum:

- `axis_assignment`: copied from the result's `axis_assignment`
  field.
- `objective_name`: the name of the objective callable (e.g.,
  `"spearman_objective"`).
- `objective_value`: the achieved objective at the learned map.
- `objective_baseline`: the objective evaluated at the hardcoded
  baseline.
- `solver`: `"greedy"` or `"scipy"`.

When the import path uses the hardcoded helpers (default),
`learned_axis_assignment is None`.

The field SHALL serialise to JSON cleanly (no `numpy.ndarray`,
`numpy.float64`, or similar non-portable types — all values are
plain Python ints / floats / strs / dicts).

#### Scenario: field is None for hardcoded path

- **WHEN** `from_sae_lens(...)` is called without
  `learn_axis_assignment`
- **THEN** `report.learned_axis_assignment is None`

#### Scenario: field populated when strategy runs

- **WHEN** `from_sae_lens(..., learn_axis_assignment=True)` is
  called on a Rung4 fixture
- **THEN** `report.learned_axis_assignment` is a dict with keys
  `{"axis_assignment", "objective_name", "objective_value",
  "objective_baseline", "solver"}` and all values are JSON-safe

### Requirement: CLI surfaces the opt-in flag

The `polygram from-sae-lens` CLI subcommand SHALL accept
`--learn-axis-assignment` as a boolean flag. When set, the CLI
SHALL pass `learn_axis_assignment=True` to `from_sae_lens` and
SHALL include the resulting `SelectionReport.learned_axis_assignment`
in the emitted report file.

The flag SHALL be off by default — CLI invocations without the flag
get the hardcoded behaviour.

#### Scenario: CLI flag triggers learned import

- **WHEN** `polygram from-sae-lens --learn-axis-assignment
  <fixture> <ids>` is invoked
- **THEN** the emitted report file contains a
  `learned_axis_assignment` block with a populated
  `axis_assignment` field

#### Scenario: CLI flag default off

- **WHEN** `polygram from-sae-lens <fixture> <ids>` is invoked
  without the flag
- **THEN** the emitted report file's `learned_axis_assignment`
  field is `null`
