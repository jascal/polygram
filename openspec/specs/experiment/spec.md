# experiment Specification

## Purpose

Polygram's experiment primitives: `InterferenceSweep` (landscape
exploration over φ ranges) and `Cancellation` (goal-directed φ-search
to drive a target-pair overlap below a tolerance, optionally
preserving cluster-tier ordering). Also defines q-orca file emission
with provenance (`polygram.emit.write_qorca`) and the structural-floor
diagnostic that bounds what φ-only search can achieve.

## Requirements
### Requirement: Q-Orca file emission with provenance

Polygram SHALL expose `polygram.emit.write_qorca(dictionary, path)` that
writes a `.q.orca.md` file readable by `q_orca.parser.parse_q_orca_markdown`
and verifiable by `q_orca.verifier.verify`. The emitted file SHALL begin
with a comment block naming the source `Dictionary`, the generation
timestamp, and the git revision (or "unversioned" outside a repo).

The shipped Animals example SHALL exercise this path end-to-end as part
of the test suite — closing the v0 milestone.

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

#### Scenario: animals example produces a valid q-orca artifact

- **WHEN** `tests/test_examples.py::test_animals_interference_runs`
  executes a coarsened version of `examples/animals_interference.py`
- **THEN** the emitted `.q.orca.md` parses and `verify(...).valid` is
  `True`, the hierarchical-ordering assertion holds at every sweep
  point, and the result tensor shapes match `(n_points, N, N)` /
  `(n_points,)`. Destructive interference is *not* asserted on this
  geometry — single-φ sweep on the bird_hawk side leaves the
  dog_poodle/bird_hawk overlap above baseline; antisymmetric two-side
  φ steering is the future `Cancellation` primitive's job.

### Requirement: Experiment.materialize bundles artifacts

`Experiment.materialize(output_dir)` SHALL create `output_dir`, write a
reference `<name>.q.orca.md`, write a self-contained `run_<name>.py`,
write `<name>_summary.md` describing the experiment configuration, and
(after a subsequent `.run()`) optionally write `<name>_result.npz` via
`result.save(...)`. The summary file SHALL include: dictionary name,
sweep axes with their value ranges, the target pair, and the list of
assertions.

#### Scenario: materialize creates summary alongside machine + runner

- **WHEN** `experiment.materialize("out/")` is called for an experiment
  named `PoodleHawk`
- **THEN** `out/PoodleHawk.q.orca.md`, `out/run_PoodleHawk.py`, and
  `out/PoodleHawk_summary.md` all exist on disk; the summary file
  contains the dictionary name, the target pair, and each sweep axis
  name with its `[min, max]` value range

### Requirement: Tier statistics in ExperimentResult

`ExperimentResult` SHALL expose `tier_stats: dict[str, np.ndarray]`
populated by `InterferenceSweep.run()`. Keys are `"self"`,
`"sibling"`, and `"cross_cluster"`. Each value is a real-valued
NumPy array of shape `(*sweep_dims,)` holding the *mean* of
`|<A|B>|²` over feature-pairs in that tier at each sweep point:

- `self` — diagonal entries, always 1.0 (kept for tier symmetry).
- `sibling` — pairs where both features share a cluster (excluding
  the diagonal). Equal to NaN if no cluster has ≥ 2 features.
- `cross_cluster` — pairs where features sit in different clusters.
  Equal to NaN if there is only one cluster.

#### Scenario: tier values match analytic baselines on matched-φ animals

- **WHEN** an `Experiment` over the 4-feature Animals dictionary
  (α=γ=0, β=±0.5 by cluster, all `phi=0`) is run
- **THEN** `result.tier_stats["self"][0] == 1.0`,
  `result.tier_stats["sibling"][0]` is within 1e-4 of `1.0` (siblings
  share α/β/γ → identical states under this dictionary), and
  `result.tier_stats["cross_cluster"][0]` is within 1e-4 of
  `cos(0.5) ** 4`

#### Scenario: tier shapes follow sweep dimensionality

- **WHEN** an `Experiment` with `sweep={"a.phi": linspace(0, π, 5),
  "b.phi": linspace(0, π, 7)}` is run
- **THEN** every value of `result.tier_stats` has shape `(5, 7)`

### Requirement: Default plot renderer

`ExperimentResult.plot(path, kind="overlap")` SHALL render a default
matplotlib figure to `path` (PNG by extension). Sweep dimensionality
selects the layout: 1D → line plot of target-pair overlap with
horizontal baselines for sibling and cross-cluster tiers; 2D →
heatmap of target-pair overlap with axes labeled by sweep keys.
Three or more sweep axes SHALL raise `NotImplementedError`.

`matplotlib` SHALL be an optional dependency. If unavailable,
`plot()` raises `ImportError` with a message naming the
`polygram[plot]` extra.

#### Scenario: 1D sweep writes a non-empty PNG

- **WHEN** a 1D sweep result calls `result.plot(tmp_path / "p.png")`
- **THEN** the returned path exists and is a non-empty file

#### Scenario: 2D sweep writes a heatmap PNG

- **WHEN** a 2D sweep result calls `result.plot(tmp_path / "p.png")`
- **THEN** the returned path exists and is a non-empty file

#### Scenario: 3D+ sweep refuses

- **WHEN** a result with three sweep axes calls `.plot(...)`
- **THEN** `NotImplementedError` is raised, naming the limitation

### Requirement: Multi-axis sweep is supported

`Experiment.run()` SHALL walk the Cartesian product of all keys in
`Experiment.sweep` and produce an `ExperimentResult` with arrays
shaped to match. Multiple sweep axes are a supported configuration,
not a side-effect.

#### Scenario: 2D sweep produces correctly shaped Gram tensor

- **WHEN** an `Experiment` over a 4-feature dictionary with two sweep
  axes of length 3 and 5 is run
- **THEN** `result.gram_matrices.shape == (3, 5, 4, 4)` and
  `result.overlaps.shape == (3, 5)`

### Requirement: Cancellation primitive

Polygram SHALL expose a `Cancellation` dataclass that searches for
phase values driving the `target_pair` overlap below `tolerance`
without breaking hierarchical-tier ordering when `preserve_tiers`
is True. `Cancellation.run()` returns a `CancellationResult`.

Fields: `dictionary: Dictionary`, `target_pair: tuple[str, str]`,
`tolerance: float = 0.05`, `preserve_tiers: bool = True`,
`optimize: dict = {"method": "grid", "max_steps": 50}`,
`optimize_all: bool = False`.

The search space in v0 is two-dimensional: the φ values of the two
target-pair features. `optimize_all=True` is reserved and SHALL
raise `NotImplementedError` in v0.

Optimization backends:

- `method="grid"` — deterministic resolution-`max_steps`-per-axis
  scan over `[0, 2π]²` (so `max_steps=50 → 2500` evaluations).
  Pure numpy. No extra dependency.
- `method="scipy"` — `scipy.optimize.differential_evolution` with
  bounds `[(0, 2π), (0, 2π)]`, `seed=0`, `maxiter=max_steps`.
  Lazy `import scipy`; if unavailable, `ImportError` names the
  `polygram[opt]` extra.

When `preserve_tiers=True`, candidates that violate
`hierarchical_ordering_preserved` are infeasible. Grid masks them
out before argmin. Scipy adds a large penalty (e.g., `+1.0`) to
the objective at infeasible candidates.

#### Scenario: target pair must reference declared features

- **WHEN** `Cancellation` is constructed with a `target_pair`
  naming a feature absent from `dictionary.features`
- **THEN** `__post_init__` raises `ValueError` naming the missing
  feature

#### Scenario: optimize_all=True refused in v0

- **WHEN** `Cancellation(..., optimize_all=True)` is constructed
- **THEN** `NotImplementedError` is raised, naming the v0 limit
  and the two-feature target search space

#### Scenario: unknown method rejected

- **WHEN** `Cancellation(..., optimize={"method": "bogus"})` is
  constructed
- **THEN** `__post_init__` raises `ValueError` listing the
  supported methods (`grid`, `scipy`)

#### Scenario: grid backend finds best feasible point on Animals

- **WHEN** a `Cancellation` is run on the 4-feature Animals
  Dictionary, starting from a mismatched-φ configuration (e.g.
  `dog_poodle.phi=π/2, bird_hawk.phi=0`), with target
  `(dog_poodle, bird_hawk)`, `preserve_tiers=True`,
  `optimize={"method": "grid", "max_steps": 30}`
- **THEN** `result.after_overlap <= result.before_overlap + 1e-9`,
  `result.feasible_count > 0`, and `result.dictionary_at_optimum`
  is a valid `Dictionary` whose target-pair Gram entry matches
  `result.after_overlap` to within 1e-6

### Requirement: CancellationResult fields

`CancellationResult` SHALL expose:

- `optimized_phis: dict[str, float]` — `{name_a: phi_a, name_b: phi_b}`
- `before_gram, after_gram: np.ndarray (N, N, complex)`
- `before_overlap, after_overlap: float`
- `tolerance_met: bool` — `after_overlap < tolerance`
- `method: str`
- `trajectory: np.ndarray (M, 3)` — every evaluation
  `(phi_a, phi_b, overlap)` in evaluation order; for grid this
  is a row-major flattening of the grid
- `feasible_count: int`
- `dictionary_at_optimum: Dictionary`
- `target_pair: tuple[str, str]`

#### Scenario: trajectory shape matches evaluation count

- **WHEN** a grid run with `max_steps=20` returns
- **THEN** `result.trajectory.shape == (400, 3)`

### Requirement: CancellationResult artifacts

`CancellationResult.plot(path)` and `.materialize(output_dir)` SHALL
produce useful researcher-facing artifacts:

- `.plot(path)` — for `method="grid"`: a heatmap of target-pair
  overlap on the `(φ_a, φ_b)` grid with the infeasible region
  masked and the optimum starred; for `method="scipy"`: line
  plot of objective vs evaluation count. `matplotlib` is the
  optional dependency.
- `.materialize(output_dir)` — writes `<name>.q.orca.md`
  (the optimized `Dictionary` emitted via `write_qorca`),
  `<name>_summary.md` (configuration, before/after overlap,
  tolerance met, method, feasible count), and
  `<name>_trajectory.csv` with columns `phi_a, phi_b,
  overlap, feasible`. Returns `dict[str, Path]`.

#### Scenario: materialized .q.orca.md verifies clean

- **WHEN** a grid `Cancellation` on a 4-feature Dictionary
  returns and `result.materialize(out)` is called
- **THEN** parsing `<out>/<name>.q.orca.md` with
  `q_orca.parser.markdown_parser.parse_q_orca_markdown` yields no
  errors and `q_orca.verify(machine)` reports `valid == True`

#### Scenario: plot writes a non-empty PNG

- **WHEN** a grid `CancellationResult.plot(tmp_path / "p.png")`
  is called
- **THEN** the returned path exists and is a non-empty file

### Requirement: Cancellation exposes structural_floor()

`Cancellation` SHALL expose `structural_floor() -> float` — the
analytic minimum of the target-pair `|<A|B>|²` reachable by varying
only `(φ_A, φ_B)`, holding all other features fixed at their current
configuration.

The implementation SHALL evaluate the target-pair overlap at exactly
two phase points: `(φ_anchor, φ_anchor)` (δ=0) and `(φ_anchor,
φ_anchor + π)` (δ=π), where `φ_anchor` is the current
`target_pair[0]` feature's φ value on the input dictionary. The
returned floor SHALL be `min(m_zero, m_pi)`, equivalent to
`M − |V|` for the decomposition
`|<A|B>|²(δ) = M + V·cos(δ)`.

`structural_floor()` SHALL NOT depend on `preserve_tiers` — it
reports the unconstrained phase-only floor.

#### Scenario: floor matches the empirical grid minimum

- **WHEN** `Cancellation(...).structural_floor()` is called on the
  Animals-4 fixture, then a separate `optimize={"method": "grid",
  "max_steps": 50}` run is performed
- **THEN** the returned floor matches `result.trajectory[:,2].min()`
  to within 1e-9 (in the unconstrained case where every grid cell
  is feasible)

### Requirement: CancellationResult exposes structural_floor and efficiency

`CancellationResult` SHALL expose two diagnostic fields:

- `structural_floor: float` — same value as
  `Cancellation.structural_floor()`, cached on the result so callers
  don't recompute it.
- `cancellation_efficiency: float | None` —
  `(before_overlap − after_overlap) / (before_overlap −
  structural_floor)`, clamped to `[0.0, 1.0]`. `None` when
  `before_overlap − structural_floor < 1e-9` (no cancellation gap
  to measure — already at the floor).

The fields SHALL be populated by `Cancellation.run()` from a single
floor computation; they SHALL NOT trigger a second optimization pass.

#### Scenario: efficiency is 1.0 when phase search reaches the floor

- **WHEN** a `Cancellation` is run starting from a mismatched-φ
  configuration where the optimum equals the structural floor
  (typical case under `preserve_tiers=False`, or when matched-φ is
  feasible)
- **THEN** `result.cancellation_efficiency` is `1.0` (within 1e-9)
  and `result.structural_floor == result.after_overlap` (within
  1e-9)

#### Scenario: efficiency is None when already at the floor

- **WHEN** a `Cancellation` is run on a Dictionary whose target pair
  is already at the structural floor (e.g., φ_A = φ_B = 0 with V
  negative)
- **THEN** `result.cancellation_efficiency is None` and
  `result.structural_floor == result.before_overlap` (within 1e-9)

### Requirement: Materialized summary reports the structural floor

`CancellationResult.materialize(output_dir)` SHALL append a
"Structural floor" section to `<name>_summary.md` reporting:

- the structural floor value
- `cancellation_efficiency` (or "no cancellation gap" if `None`)
- a one-line interpretation:
  - efficiency ≥ 0.99 → "phase search exhausted — encoding-bound"
  - 0.0 < efficiency < 0.99 → "phase search underutilized"
  - efficiency `None` → "no cancellation gap available"

#### Scenario: summary contains floor and efficiency lines

- **WHEN** `CancellationResult.materialize(output_dir)` is called
  after a grid run on the Animals-4 fixture
- **THEN** the produced `<name>_summary.md` contains a "Structural
  floor" section header and lines naming both the floor value and
  the efficiency value (or "no cancellation gap")

