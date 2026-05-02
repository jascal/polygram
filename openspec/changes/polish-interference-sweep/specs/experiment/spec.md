## ADDED Requirements

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

## MODIFIED Requirements

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
