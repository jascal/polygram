## ADDED Requirements

### Requirement: `LearnedAxisAssignment` ships as an opt-in `KnobAssignment` strategy

`polygram.geometry.LearnedAxisAssignment` SHALL be a class
implementing the `KnobAssignment` protocol from
`polygram.geometry.protocols`. The class SHALL be importable from
`polygram.geometry` and re-exported from the package's
`__init__.py`.

The strategy SHALL NOT be invoked unless the caller explicitly opts
in — either by passing a `LearnedAxisAssignment` instance through
`from_sae_lens(learn_axis_assignment=...)` or via the matching
CLI flag.

#### Scenario: importable from polygram top-level

- **WHEN** `from polygram import LearnedAxisAssignment` is executed
- **THEN** the class is importable and is an instance of the
  `KnobAssignment` protocol per `runtime_checkable`

#### Scenario: not invoked by default

- **WHEN** `from_sae_lens(records, ids, encoding=Rung4())` is called
  without `learn_axis_assignment`
- **THEN** the resulting Dictionary is byte-identical to the
  pre-change behaviour (no `LearnedAxisAssignment` instantiation,
  no learned map applied)

### Requirement: Two solvers — greedy and scipy

`LearnedAxisAssignment(solver=...)` SHALL accept `"greedy"` (default)
and `"scipy"`.

- `solver="greedy"` SHALL perform deterministic permutation search:
  for each knob slot in canonical order (α, φ, then amp pairs in
  qubit-index ascending order), try every still-unused PCA axis and
  lock in the axis whose addition gives the best objective value.
  No external dependencies; ships in the base install.
- `solver="scipy"` SHALL perform continuous optimisation on a small
  linear map `W ∈ R^{n_knobs × n_axes}` initialised from the greedy
  result. SHALL use `scipy.optimize.minimize(method="Nelder-Mead")`
  for problems with fewer than 8 knobs and
  `scipy.optimize.differential_evolution` for 8+ knobs. Requires the
  `polygram[opt]` extra.

#### Scenario: greedy solver is deterministic

- **WHEN** `LearnedAxisAssignment(solver="greedy")` is invoked
  twice on the same projection matrix
- **THEN** both invocations produce the same `axis_assignment`
  bit-for-bit

#### Scenario: scipy solver requires the opt extra

- **WHEN** `LearnedAxisAssignment(solver="scipy")` is invoked in an
  environment without scipy installed
- **THEN** an `ImportError` is raised pointing at the
  `polygram[opt]` install hint

#### Scenario: scipy solver initialises from greedy result

- **WHEN** `LearnedAxisAssignment(solver="scipy")` is invoked
- **THEN** the strategy first computes the greedy assignment, then
  passes it as the initial point `x0` to scipy

### Requirement: Pluggable objective via `LearnedAxisObjective` protocol

`polygram.geometry.LearnedAxisObjective` SHALL be a
`runtime_checkable` Protocol whose `__call__(analytic_gram,
decoder_geom, *, feature_names) -> float` returns a scalar to
*maximise*. Three built-ins SHALL ship in
`polygram.geometry.objectives`:

- `spearman_objective` (the default): Spearman rank correlation on
  off-diagonal upper-triangle entries of `|analytic_gram|²` and
  `decoder_geom` (which the strategy populates with decoder
  cosine²).
- `pearson_objective`: Pearson correlation on the same entries.
- `behavioural_objective(reference_pair_sims) -> Callable`: factory
  returning a closure that correlates the analytic gram against a
  user-supplied ground-truth pair-similarity matrix instead of
  decoder cosine.

#### Scenario: default objective is Spearman

- **WHEN** `LearnedAxisAssignment()` is instantiated without an
  explicit `objective` kwarg
- **THEN** `strategy.objective is spearman_objective`

#### Scenario: pluggable objective accepted

- **WHEN** `LearnedAxisAssignment(objective=pearson_objective)` is
  instantiated
- **THEN** the strategy uses Pearson correlation as its objective
  during the search

#### Scenario: behavioural objective wraps user matrix

- **WHEN** `behavioural_objective(my_ground_truth)` is called with a
  user-supplied square matrix
- **THEN** the returned callable scores the analytic gram against
  `my_ground_truth` (ignoring the `decoder_geom` argument)

### Requirement: Result surfaces the learned map in `KnobAssignmentResult.axis_assignment`

`LearnedAxisAssignment.assign(...)` SHALL populate the
`axis_assignment` field of the returned `KnobAssignmentResult` with:

- A `dict[str, int]` mapping knob name to chosen PCA-axis index when
  the greedy solver runs.
- A `dict[str, list[float]]` mapping knob name to per-axis
  coefficient vector when the scipy solver runs.

Knob names SHALL be drawn from the set `{"alpha", "phi",
"amp_<i>_theta", "amp_<i>_psi"}` for `i ∈ [0, n_amp_qubits)`.

The result SHALL also include `objective_value` (the achieved
objective at the chosen map) and `objective_baseline` (the
objective evaluated at the hardcoded baseline map for comparison).

#### Scenario: greedy result is a knob → int map

- **WHEN** `LearnedAxisAssignment(solver="greedy").assign(...)` is
  called on a Rung4-encoded projection
- **THEN** the result's `axis_assignment` is a dict whose values are
  all integers, and whose keys cover at least `{"alpha", "phi",
  "amp_0_theta", "amp_0_psi", "amp_1_theta", "amp_1_psi"}`

#### Scenario: result includes baseline-vs-learned objective

- **WHEN** any `LearnedAxisAssignment.assign(...)` completes
- **THEN** the result carries `objective_value` and
  `objective_baseline` as floats, and `objective_value ≥
  objective_baseline - 1e-6` (the learned map is no worse than the
  baseline)

### Requirement: Reproduces the prototype's headline result

`LearnedAxisAssignment(solver="greedy")` SHALL reproduce the
prototype's published numbers on the synthetic 64-feature clustered
SAE described in `docs/research/rung5-pareto-scans.md` scan 4:

- At `Rung5(n_amp_qubits=3)`, the achieved Spearman SHALL be at
  least `+0.30` (prototype: `+0.3350`).
- At `Rung5(n_amp_qubits=4)`, the achieved Spearman SHALL be at
  least `+0.30` (prototype: `+0.3380`).

#### Scenario: reproduces scan-4 Spearman at k=3

- **WHEN** the production strategy is invoked on the scan-4
  synthetic SAE with `solver="greedy"`, seed `0`, k=3
- **THEN** `result.objective_value ≥ 0.30`

#### Scenario: reproduces scan-4 Spearman at k=4

- **WHEN** the production strategy is invoked on the scan-4
  synthetic SAE with `solver="greedy"`, seed `0`, k=4
- **THEN** `result.objective_value ≥ 0.30`

### Requirement: HEA_Rung2 falls back to the hardcoded helper

`LearnedAxisAssignment.assign(...)` SHALL detect
`isinstance(encoding, HEA_Rung2)` and SHALL fall back to
the hardcoded `assign_amp_knobs_pca` + `assign_phase_knobs_pca`
helpers (returning their result with `axis_assignment=None`).

The strategy SHALL log INFO-once when this fallback path triggers,
naming the encoding and explaining that HEA's per-feature θ tensor
shape is out of scope for v1 of the learned strategy.

#### Scenario: HEA encoding falls back cleanly

- **WHEN** `LearnedAxisAssignment().assign(projs, names,
  encoding=HEA_Rung2(n_qubits=3, depth=2), ...)` is called
- **THEN** the returned result populates per-feature knobs via the
  hardcoded helpers and `axis_assignment is None`

### Requirement: Validation split prevents objective overfitting

`LearnedAxisAssignment(validation_fraction=...)` SHALL accept a
fraction in `[0.0, 0.5]` (default `0.0` → no split) that holds out
that fraction of off-diagonal pairs as a validation set. The
training objective is computed on the remaining pairs; the result's
`objective_value` SHALL be the validation-set objective, with a
separate `training_objective_value` carrying the training score.

When `validation_fraction == 0.0`, the strategy SHALL compute the
objective on all off-diagonal pairs and SHALL set
`training_objective_value` equal to `objective_value`.

#### Scenario: validation fraction held out correctly

- **WHEN** `LearnedAxisAssignment(validation_fraction=0.2).assign(...)`
  is called on a 64-feature dictionary (2016 off-diagonal pairs)
- **THEN** the training objective is computed on ~1612 pairs and the
  validation objective on ~404 pairs, and the result carries both
  values

#### Scenario: zero validation fraction equals all-pairs objective

- **WHEN** `LearnedAxisAssignment(validation_fraction=0.0).assign(...)`
  is called
- **THEN** `result.objective_value == result.training_objective_value`
