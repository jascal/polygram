## ADDED Requirements

### Requirement: Experiment declaration

Polygram SHALL expose an `Experiment` dataclass with fields `name: str`,
`dictionary: Dictionary`, `target_pair: tuple[str, str]`,
`sweep: dict[str, np.ndarray]`, `measures: list[str]`,
`assertions: list[str]`, and `seed: int = 0`.

#### Scenario: target pair must reference declared features

- **WHEN** an `Experiment` is constructed with a `target_pair` naming a
  feature absent from `dictionary.features`
- **THEN** `__post_init__` raises `ValueError` naming the missing feature

#### Scenario: unknown measure name rejected

- **WHEN** an `Experiment` is constructed with `measures=["bogus"]`
- **THEN** `__post_init__` raises `ValueError` listing the supported
  measure names (`overlap`, `gram_matrix`, `schmidt_rank`)

### Requirement: InterferenceSweep over phase parameters

`Experiment.run(backend="analytic")` SHALL walk the Cartesian product of
sweep values, recompute the analytic Gram at each point, and return an
`ExperimentResult` whose `gram_matrices` array has shape
`(*sweep_dims, N, N)` where `N = len(dictionary.features)`.

#### Scenario: 1D phi sweep produces correctly shaped result

- **WHEN** an `Experiment` with `sweep={"phi_cross": np.linspace(0, π, 5)}`
  on a 4-feature dictionary is `.run()` with `backend="analytic"`
- **THEN** `result.gram_matrices.shape == (5, 4, 4)` and
  `result.overlaps.shape == (5,)`, with `result.overlaps[0]` matching the
  `phi=0` Gram entry for the target pair within 1e-4

#### Scenario: shot-based backend not yet supported

- **WHEN** `Experiment.run(backend="qutip")` is called in v0
- **THEN** `NotImplementedError` is raised with a message pointing the
  user to the analytic backend

### Requirement: Built-in assertions

Polygram SHALL evaluate built-in assertions per sweep point and surface
the per-point results in `ExperimentResult.assertion_pass`.

#### Scenario: hierarchical_ordering_preserved reports per-point bools

- **WHEN** an experiment includes
  `assertions=["hierarchical_ordering_preserved"]`
- **THEN** `result.assertion_pass["hierarchical_ordering_preserved"]` is
  a boolean NumPy array of length equal to the number of sweep points

#### Scenario: target_pair_destructive_at_endpoint checks last point only

- **WHEN** an experiment includes
  `assertions=["target_pair_destructive_at_endpoint"]` and the target
  pair's overlap at the last sweep point is below the default threshold
  of 0.1
- **THEN** the corresponding entry in
  `result.assertion_pass["target_pair_destructive_at_endpoint"]` is `True`

## ADDED Requirements

### Requirement: Q-Orca file emission with provenance

Polygram SHALL expose `polygram.emit.write_qorca(dictionary, path)` that
writes a `.q.orca.md` file readable by `q_orca.parser.parse_q_orca_markdown`
and verifiable by `q_orca.verifier.verify`. The emitted file SHALL begin
with a comment block naming the source `Dictionary`, the generation
timestamp, and the git revision (or "unversioned" outside a repo).

#### Scenario: emitted file parses and verifies clean

- **WHEN** `write_qorca` is called for a 4-feature, 2-cluster dictionary
  and the result is round-tripped through `parse_q_orca_markdown` +
  `verifier.verify`
- **THEN** `verifier.verify` reports `valid == True`

#### Scenario: emitter never produces inverse-form when phi nonzero

- **WHEN** any feature has `phi != 0` and `write_qorca` is called
- **THEN** the emitted transitions table uses preparation-form call
  sites (`prepare_*` events into distinct `prepared_*` states), never
  inverse-form rollback transitions

### Requirement: Experiment.materialize bundles artifacts

`Experiment.materialize(output_dir)` SHALL create `output_dir`, write a
reference `<name>.q.orca.md`, write a self-contained `run_<name>.py`,
and (after a subsequent `.run()`) write `<name>_result.npz`.

#### Scenario: materialize creates expected files

- **WHEN** `experiment.materialize("out/")` is called for an experiment
  named `PoodleHawk`
- **THEN** `out/PoodleHawk.q.orca.md` and `out/run_PoodleHawk.py` exist
  on disk
