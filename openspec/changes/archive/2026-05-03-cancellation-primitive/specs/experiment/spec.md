## ADDED Requirements

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
