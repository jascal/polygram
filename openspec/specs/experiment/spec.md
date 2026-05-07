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

The renderer SHALL dispatch on `dictionary.encoding`. For `MPSRung1`
the emitted body SHALL be the existing rung-1 staircase layout
unchanged. For `HEA_Rung2` the emitted body SHALL include three
extra sections in order:

1. A `## encoding` table with `kind: hea`, `depth`, `entangler`,
   and `rotations` matching `dictionary.encoding`.
2. A `## theta` table with three columns
   `| concept | tensor | cluster |`. The `concept` column carries
   the feature slug, `tensor` carries the literal-eval-able Python
   list form of each feature's θ tensor (using the encoding's
   default-tensor generator when `feature.theta is None`), and
   `cluster` carries the feature's `cluster` field verbatim.
3. A `## invariants` section declaring
   `- concept_gram_tier_separation >= <bound>` whenever
   `encoding.tier_separation_bound is not None`. When the field
   is `None`, the section SHALL be omitted from the HEA branch.

The shipped Animals example SHALL exercise this path end-to-end as part
of the example test (rung-1). A new `examples/animals_hea.py` SHALL
exercise the HEA branch, producing a file that
`q_orca.verifier.verify` accepts under default options (Stage 4b
including the tier-separation invariant).

#### Scenario: HEA dictionary emits encoding/theta/invariants

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` with three
  features grouped two-and-one across clusters `s1` and `s2`
- **WHEN** `polygram.emit.write_qorca(dictionary, path)` runs
- **THEN** the written file contains a `## encoding` section with
  `kind: hea`, a 3-column `## theta` table whose `cluster` column
  reads `s1, s1, s2`, and a `## invariants` section listing
  `concept_gram_tier_separation >= 0.025`

#### Scenario: HEA dictionary with bound=None omits invariants

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2,
  tier_separation_bound=None))`
- **WHEN** the emitter runs
- **THEN** the produced markdown does not contain a
  `## invariants` section

#### Scenario: HEA emission verifies clean

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` with
  features whose θ tensors satisfy the declared
  `tier_separation_bound`
- **WHEN** the emitted file is parsed and passed to
  `q_orca.verifier.verify`
- **THEN** `result.valid` is `True` and no error of code
  `HEA_GRAM_INVALID`, `HEA_TIER_INVARIANT_VIOLATED`, or
  `HEA_TIER_UNDEFINED` is reported

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
parameter values driving the `target_pair` overlap below
`tolerance` without breaking hierarchical-tier ordering when
`preserve_tiers` is True. `Cancellation.run()` returns a
`CancellationResult`.

Fields: `dictionary: Dictionary`, `target_pair: tuple[str, str]`,
`tolerance: float = 0.05`, `preserve_tiers: bool = True`,
`optimize: dict = {"method": "grid", "max_steps": 50}`,
`optimize_all: bool = False`, `knobs: list[str] | None = None`.

`knobs` declares the search space as a list of knob paths in the
same grammar as `Dictionary.with_knob`. Each path is one of:

- `<feature>.phi` (both encodings),
- `<feature>.theta[r,d,q]` (HEA only),
- `<cluster>.phi` (both encodings) — *cluster-shared* φ; one search
  axis whose value is applied to every feature in the named cluster,
- `<cluster>.theta[r,d,q]` (HEA only) — *cluster-shared* θ slot;
  applied to every feature in the named cluster.

When `knobs is None`, `__post_init__` SHALL resolve `knobs` to
`[f"{a}.phi", f"{b}.phi"]` for backwards compatibility, where
`(a, b) == target_pair`.

A cluster-shared path counts as a single search axis regardless of
how many features the cluster contains. `len(self.knobs)` after
resolution is therefore the search-space dimensionality.

Per-knob search bounds: `(0.0, 2π)` for `.phi` paths,
`(-π, π)` for `.theta[r,d,q]` paths — independent of whether the
leading identifier is a feature or a cluster.

`__post_init__` SHALL reject `.theta[...]` knobs on `MPSRung1`
dictionaries (`ValueError` naming the encoding) and forward
malformed paths to the same grammar errors that
`Dictionary.with_knob` raises (including unknown identifiers).

`optimize_all=True` is reserved and SHALL raise
`NotImplementedError` in v0.

Optimization backends:

- `method="grid"` — deterministic resolution-`max_steps`-per-axis
  scan over the per-knob bounds (so `max_steps=50` over two `.phi`
  knobs → 2500 evaluations). Pure numpy. No extra dependency.
  `len(knobs) > 4` SHALL raise `ValueError` recommending
  `method="scipy"` to keep grid total evaluations tractable.
- `method="scipy"` — `scipy.optimize.differential_evolution` with
  bounds derived per-knob, `seed=0`, `maxiter=max_steps`. Lazy
  `import scipy`; if unavailable, `ImportError` names the
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
  and pointing at the configurable `knobs` list as the supported
  multi-knob surface

#### Scenario: unknown method rejected

- **WHEN** `Cancellation(..., optimize={"method": "bogus"})` is
  constructed
- **THEN** `__post_init__` raises `ValueError` listing the
  supported methods (`grid`, `scipy`)

#### Scenario: default knobs preserve the 2-φ search

- **GIVEN** `Cancellation(dictionary=..., target_pair=("a", "b"))`
  with `knobs=None`
- **WHEN** `__post_init__` runs
- **THEN** `self.knobs == ["a.phi", "b.phi"]` and the resulting
  trajectory has shape `(max_steps**2, 3)` (one column per knob
  plus the overlap)

#### Scenario: theta-slot knobs accepted on HEA dictionary

- **GIVEN** an HEA-encoded dictionary and
  `Cancellation(..., knobs=["a.theta[0,0,1]", "b.theta[0,0,1]"])`
- **WHEN** the cancellation runs
- **THEN** `result.trajectory.shape == (max_steps**2, 3)` and each
  trajectory row's first two columns are values within `[-π, π]`

#### Scenario: cluster-shared knobs accepted

- **GIVEN** an HEA-encoded dictionary with two clusters
  `"dogs"` (size 2) and `"birds"` (size 2), and
  `Cancellation(..., knobs=["dogs.theta[0,0,0]",
  "birds.theta[0,0,0]"], optimize={"method": "grid", "max_steps":
  8})`
- **WHEN** the cancellation runs
- **THEN** `result.trajectory.shape == (64, 3)` (two search axes,
  one column per cluster-shared knob plus overlap), and
  `result.optimized_knobs` keys are `{"dogs.theta[0,0,0]",
  "birds.theta[0,0,0]"}`

#### Scenario: MPS cluster-shared phi preserves sibling overlaps

- **GIVEN** an `MPSRung1`-encoded dictionary with cluster `"dogs"` of
  size 2 whose siblings share the same pre-mutation `phi`, and
  `Cancellation` on `("dog_poodle", "bird_hawk")` with cluster-shared
  `<cluster>.phi` knobs on both clusters
- **WHEN** the cancellation runs and produces
  `result.dictionary_at_optimum`
- **THEN** `abs(result.before_gram[i_poodle, i_beagle] -
  result.after_gram[i_poodle, i_beagle]) < 1e-9` (and likewise for
  every pair within `"birds"`); this is the bit-for-bit case
  guaranteed by the final-Rz factorization

#### Scenario: HEA cluster-shared sibling overlaps may drift

- **GIVEN** an HEA-encoded dictionary with diverse sibling baselines
  (any per-feature `alpha`, `gamma`, or explicit `theta` variation)
  and `Cancellation` with cluster-shared θ knobs
- **WHEN** the cancellation runs
- **THEN** the run SHALL complete and the within-cluster Gram entries
  MAY differ from the pre-mutation values. The cluster-shared regime
  bounds optimizer leverage (one axis per cluster instead of one per
  feature) but does not zero the drift; bit-for-bit preservation is
  reserved for the MPS phi case

#### Scenario: mixed cluster + feature knob list accepted but not invariant-preserving

- **GIVEN** a `Cancellation` whose `knobs` list mixes
  `<feature>.theta[r,d,q]` and `<cluster>.theta[r,d,q]` paths
- **WHEN** the cancellation runs
- **THEN** the run completes (mixed lists are valid input), but the
  cluster-shared invariant on within-cluster Gram entries does NOT
  apply (the per-feature mutations on one branch break the matched
  unitarity)

### Requirement: CancellationResult fields

`CancellationResult` SHALL expose:

- `optimized_knobs: dict[str, float]` — keyed by knob path
  (`<feature>.phi` or `<feature>.theta[r,d,q]`), holding the value
  at the optimum
- `before_gram, after_gram: np.ndarray (N, N, complex)`
- `before_overlap, after_overlap: float`
- `tolerance_met: bool` — `after_overlap < tolerance`
- `method: str`
- `trajectory: np.ndarray (M, len(knobs) + 1)` — every evaluation
  with one column per knob (in declaration order) plus the
  target-pair overlap. For grid this is a row-major flattening of
  the per-axis grid.
- `feasible_count: int`
- `dictionary_at_optimum: Dictionary`
- `target_pair: tuple[str, str]`

#### Scenario: trajectory shape matches knobs and evaluation count

- **WHEN** a grid run with `max_steps=20` over the default 2-φ
  knobs returns
- **THEN** `result.trajectory.shape == (400, 3)`

#### Scenario: trajectory shape widens for multi-knob runs

- **WHEN** a grid run with `max_steps=12` over three HEA
  `.theta[r,d,q]` knobs returns
- **THEN** `result.trajectory.shape == (12 ** 3, 4)`

### Requirement: CancellationResult artifacts

`CancellationResult` SHALL produce researcher-facing artifacts via
`plot(path, kind=None)` and `materialize(output_dir)`.

`plot(path, kind=None)` SHALL select a renderer per `kind`. When
`kind is None`, the method dispatches on the result's `method`
(`"grid"` → existing grid heatmap, `"scipy"` → existing line plot).
Recognized explicit kinds:

- `"grid"` — heatmap of target-pair overlap on the `(φ_a, φ_b)`
  grid with the infeasible region masked and the optimum starred.
  Defined only when `len(knobs) == 2` (the heatmap surface is
  intrinsically 2D); `NotImplementedError` otherwise.
- `"scipy"` — line plot of objective vs evaluation count.
- `"before_after"` — three-panel figure: before Gram heatmap, after
  Gram heatmap (shared colorbar), and a bar chart with
  `before_overlap`, `after_overlap`, and (when defined) the
  structural floor for the target pair. The before/after panels
  highlight the `target_pair` cell with a marker. Defined for any
  `knobs` length.

`matplotlib` SHALL remain the optional dependency.

`.materialize(output_dir)` writes `<name>.q.orca.md` (the optimized
`Dictionary` emitted via `write_qorca`), `<name>_summary.md`
(configuration, before/after overlap, tolerance met, method,
feasible count, structural-floor section per the relevant
requirement), and `<name>_trajectory.csv` whose columns are the
declared knob paths in order plus `overlap, feasible`. Returns
`dict[str, Path]`.

#### Scenario: materialized .q.orca.md verifies clean

- **WHEN** a grid `Cancellation` on a 4-feature Dictionary
  returns and `result.materialize(out)` is called
- **THEN** parsing `<out>/<name>.q.orca.md` with
  `q_orca.parser.markdown_parser.parse_q_orca_markdown` yields no
  errors and `q_orca.verify(machine)` reports `valid == True`

#### Scenario: default plot writes a non-empty PNG

- **WHEN** a grid `CancellationResult.plot(tmp_path / "p.png")`
  is called with `kind=None`
- **THEN** the returned path exists and is a non-empty file

#### Scenario: before/after plot writes a non-empty PNG

- **WHEN** any `CancellationResult.plot(tmp_path / "ba.png",
  kind="before_after")` is called
- **THEN** the returned path exists and is a non-empty file

#### Scenario: grid plot kind refused on multi-knob runs

- **WHEN** a 3-knob `CancellationResult.plot(...,
  kind="grid")` is called
- **THEN** `NotImplementedError` is raised, naming the limitation

### Requirement: Cancellation exposes structural_floor()

`Cancellation` SHALL expose `structural_floor() -> float` — the
analytic minimum of the target-pair `|<A|B>|²` reachable by varying
only `(φ_A, φ_B)` on a rung-1 `MPSRung1` dictionary.

`structural_floor()` is well-defined exactly when:

1. `dictionary.encoding` is an `MPSRung1` instance, AND
2. `self.knobs` equals the canonical pair
   `[f"{target_pair[0]}.phi", f"{target_pair[1]}.phi"]`.

Inside that shape, the implementation SHALL evaluate the target-pair
overlap at exactly two phase points: `(φ_anchor, φ_anchor)` (δ=0)
and `(φ_anchor, φ_anchor + π)` (δ=π), where `φ_anchor` is the
current `target_pair[0]` feature's φ value on the input
dictionary. The returned floor SHALL be `min(m_zero, m_pi)`,
equivalent to `M − |V|` for the decomposition
`|<A|B>|²(δ) = M + V·cos(δ)`.

Outside that shape — every multi-knob configuration, every
non-canonical knob list, and every HEA-encoded dictionary regardless
of knob list — `structural_floor()` SHALL raise
`NotImplementedError`. The error message SHALL name the
configuration (encoding kind plus declared knobs) and identify the
absent analytic bound as a deferred research question.

`structural_floor()` SHALL NOT depend on `preserve_tiers` — when
defined, it reports the unconstrained phase-only floor.

#### Scenario: floor matches the empirical grid minimum on MPS

- **WHEN** `Cancellation(...).structural_floor()` is called on the
  Animals-4 `MPSRung1` fixture with default 2-φ knobs, then a
  separate `optimize={"method": "grid", "max_steps": 50}` run is
  performed
- **THEN** the returned floor matches `result.trajectory[:,2].min()`
  to within 1e-9 (in the unconstrained case where every grid cell
  is feasible)

#### Scenario: floor refused on HEA dictionary

- **GIVEN** a `Cancellation` whose dictionary uses `HEA_Rung2`
- **WHEN** `cancellation.structural_floor()` is called
- **THEN** `NotImplementedError` is raised, naming `HEA_Rung2` and
  pointing at the deferred analytic-bound research question

#### Scenario: floor refused on MPS with non-canonical knobs

- **GIVEN** an `MPSRung1`-encoded `Cancellation` constructed with
  `knobs=["a.phi"]` (a single-feature knob list, not the canonical
  pair)
- **WHEN** `cancellation.structural_floor()` is called
- **THEN** `NotImplementedError` is raised, naming the non-canonical
  knob list

### Requirement: CancellationResult exposes structural_floor and efficiency

`CancellationResult` SHALL expose two diagnostic fields:

- `structural_floor: float` — the analytic floor when defined per
  the `Cancellation exposes structural_floor()` requirement, or
  `float("nan")` when the floor is undefined for the configuration.
  Cached on the result so callers don't recompute it.
- `cancellation_efficiency: float | None` —
  `(before_overlap − after_overlap) / (before_overlap −
  structural_floor)`, clamped to `[0.0, 1.0]`. `None` when (a)
  `structural_floor` is undefined (NaN), or (b)
  `before_overlap − structural_floor < 1e-9` (no cancellation gap
  to measure — already at the floor).

`Cancellation.run()` SHALL catch the `NotImplementedError` raised by
`structural_floor()` for unsupported configurations and store
`structural_floor=float("nan")`,
`cancellation_efficiency=None` rather than propagating the
exception. The fields SHALL be populated from a single floor
computation; they SHALL NOT trigger a second optimization pass.

#### Scenario: efficiency is 1.0 when phase search reaches the floor

- **WHEN** an MPS-encoded `Cancellation` is run starting from a
  mismatched-φ configuration where the optimum equals the
  structural floor (typical case under `preserve_tiers=False`, or
  when matched-φ is feasible)
- **THEN** `result.cancellation_efficiency` is `1.0` (within 1e-9)
  and `result.structural_floor == result.after_overlap` (within
  1e-9)

#### Scenario: efficiency is None when already at the floor

- **WHEN** an MPS-encoded `Cancellation` is run on a Dictionary
  whose target pair is already at the structural floor
- **THEN** `result.cancellation_efficiency is None` and
  `result.structural_floor == result.before_overlap` (within 1e-9)

#### Scenario: floor undefined on HEA produces NaN floor and None efficiency

- **WHEN** a `Cancellation` is run on an HEA-encoded dictionary
- **THEN** `math.isnan(result.structural_floor) is True` and
  `result.cancellation_efficiency is None`

### Requirement: Materialized summary reports the structural floor

`CancellationResult.materialize(output_dir)` SHALL append a
"Structural floor" section to `<name>_summary.md` reporting:

- the structural floor value, or the literal string "undefined for
  this configuration" when `structural_floor` is NaN
- `cancellation_efficiency` (or "no cancellation gap" if `None`,
  or "not applicable" when the floor is undefined)
- a one-line interpretation:
  - floor undefined → "structural floor is encoding-bound; not yet
    defined for this configuration"
  - efficiency ≥ 0.99 → "phase search exhausted — encoding-bound"
  - 0.0 < efficiency < 0.99 → "phase search underutilized"
  - efficiency `None` and floor defined → "no cancellation gap
    available"

When `structural_floor` is undefined (NaN), the materialized summary
SHALL also append a `## Caveat` section whose body depends on the
shape of the knob list:

- **Pure cluster-shared** (every path's leading identifier is a
  cluster name): the caveat SHALL describe the cluster-shared regime
  honestly. For `MPSRung1` `<cluster>.phi` knobs the caveat SHALL
  state that within-cluster Gram entries are preserved bit-for-bit
  by the final-Rz factorization (when sibling pre-mutation phi
  values agree). For HEA cluster-shared paths the caveat SHALL
  describe the regime as a search-space dimensionality reduction
  (one axis per cluster) and warn that within-cluster Gram MAY drift
  on diverse-sibling fixtures, recommending verification of
  `concept_gram_tier_separation` on the materialized optimum.
- **Mixed** (knob list contains both per-feature and cluster-shared
  paths): the caveat SHALL fire the multi-knob warning *and* an
  explicit note that mixed lists do not inherit the cluster-shared
  invariant.
- **Per-feature** (no cluster-shared paths): the caveat SHALL fire
  the existing multi-knob warning that the reported `after` is the
  best value found by the optimizer, not a guaranteed bound; if any
  knob is a `.theta[...]` path, the caveat SHALL also recommend
  hand-checking `concept_gram_tier_separation` on the optimum.

#### Scenario: summary contains floor and efficiency lines on MPS

- **WHEN** `CancellationResult.materialize(output_dir)` is called
  after a grid run on the Animals-4 MPS fixture
- **THEN** the produced `<name>_summary.md` contains a "Structural
  floor" section header and lines naming both the floor value and
  the efficiency value (or "no cancellation gap")

#### Scenario: summary marks floor undefined on HEA

- **WHEN** `CancellationResult.materialize(output_dir)` is called
  after a run on an HEA-encoded dictionary
- **THEN** the produced `<name>_summary.md` "Structural floor"
  section reports "undefined for this configuration" and the
  one-line interpretation matches the floor-undefined branch

#### Scenario: pure-cluster caveat on MPS phi names the unitarity guarantee

- **WHEN** the knob list contains only `<cluster>.phi` paths on an
  `MPSRung1`-encoded dictionary and `materialize` is called
- **THEN** the produced `<name>_summary.md` `## Caveat` section text
  mentions that within-cluster Gram entries are preserved bit-for-bit
  by the final-Rz factorization, and does NOT use the per-feature
  "best value found" language

#### Scenario: pure-cluster caveat on HEA names the search-space framing

- **WHEN** the knob list contains only cluster-shared paths on an
  `HEA_Rung2`-encoded dictionary and `materialize` is called
- **THEN** the produced `<name>_summary.md` `## Caveat` section text
  describes the cluster-shared regime as a search-space dimensionality
  reduction (one axis per cluster), warns that within-cluster Gram
  MAY drift on diverse-sibling fixtures, and recommends verifying
  `concept_gram_tier_separation` on the optimum

#### Scenario: mixed caveat names both warnings

- **WHEN** the knob list mixes per-feature and cluster-shared paths
  and `materialize` is called
- **THEN** the produced `<name>_summary.md` `## Caveat` section
  contains both the per-feature "best value found" warning and an
  explicit note that the cluster-shared invariant does NOT apply to
  mixed lists

### Requirement: Sweep keys accept HEA θ slot syntax

`Experiment.sweep` keys SHALL accept the form
`<feature_name>.theta[r,d,q]` in addition to the existing
`<feature_name>.phi`. `.theta[...]` keys are valid only when
`Experiment.dictionary.encoding` is `HEA_Rung2` and `(r, d, q)` is
within the encoding's `theta_shape`. Each sweep axis value SHALL be
applied via `Dictionary.with_knob(key, value)`.

`Experiment.__post_init__` SHALL raise `ValueError` for:

- `.theta[...]` keys on `MPSRung1` dictionaries,
- `(r, d, q)` triples outside `encoding.theta_shape`,
- malformed keys that match neither `<feature>.phi` nor
  `<feature>.theta[r,d,q]`.

#### Scenario: phi axis works on HEA dictionary

- **GIVEN** an `Experiment` whose `dictionary.encoding` is
  `HEA_Rung2(depth=2)` and whose `sweep` is `{"a.phi":
  np.linspace(0, π, 5)}`
- **WHEN** the experiment runs
- **THEN** `result.overlaps.shape == (5,)` and the `tier_separation`
  array (where defined) has matching shape

#### Scenario: theta-slot axis works on HEA dictionary

- **GIVEN** an `Experiment` whose `dictionary.encoding` is
  `HEA_Rung2(depth=2)` and whose `sweep` is `{"a.theta[1,0,1]":
  np.linspace(-π, π, 5)}`
- **WHEN** the experiment runs
- **THEN** `result.overlaps.shape == (5,)` and the materialized
  midpoint `.q.orca.md` carries a `## theta` row whose `(1, 0, 1)`
  slot equals the sweep midpoint value

#### Scenario: theta-slot axis rejected on MPS dictionary

- **GIVEN** an `Experiment` whose `dictionary.encoding` is
  `MPSRung1()` and whose `sweep` keys include
  `"a.theta[0,0,1]"`
- **WHEN** the `Experiment` is constructed
- **THEN** `__post_init__` raises `ValueError` naming the encoding

### Requirement: ExperimentResult exposes per-sweep-point tier separation

`ExperimentResult` SHALL expose a `tier_separation: np.ndarray |
None` field. When non-`None`, its shape matches `overlaps` and each
entry is the `concept_gram_tier_separation` value for the dictionary
configured at that sweep point, computed by
`q_orca.compiler.concept_gram_hea.compute_tier_separation` against
`gram_matrices[idx]` and the per-feature `cluster` labels. The field
SHALL be `None` exactly when every cluster in
`Experiment.dictionary.hierarchy` is a singleton (matching
`Dictionary.tier_separation()`'s contract).

`ExperimentResult.save(...)` SHALL persist `tier_separation` as the
`tier_separation` key in the produced `.npz` (skipped when `None`).
`ExperimentResult.to_csv(...)` SHALL append a `tier_separation`
column when the field is non-`None`.

#### Scenario: tier_separation populated for tiered HEA dictionary

- **GIVEN** a clearly-tiered HEA dictionary (two-feature `s1`
  cluster, one-feature `s2` cluster) and a 5-point φ sweep
- **WHEN** `experiment.run()` returns
- **THEN** `result.tier_separation` is a `(5,)` real-valued array
  with every entry > 0 and the CSV produced by `to_csv` has a
  `tier_separation` column

#### Scenario: tier_separation None for all-singleton dictionary

- **GIVEN** a dictionary in which every cluster has exactly one
  feature
- **WHEN** `experiment.run()` returns
- **THEN** `result.tier_separation is None` and `to_csv` does not
  emit a `tier_separation` column

### Requirement: concept_gram_tier_separation_bound_holds assertion

`Experiment.assertions` SHALL accept a new entry
`"concept_gram_tier_separation_bound_holds"`. When present, the
checker
`polygram._assertions.concept_gram_tier_separation_bound_holds(gram,
dictionary)` is invoked at every sweep point and the result is
recorded in `ExperimentResult.assertion_pass` under the same key.

`Experiment.__post_init__` SHALL raise `ValueError` when the
assertion is requested on a dictionary whose encoding lacks a
non-`None` `tier_separation_bound` (either `MPSRung1`, or
`HEA_Rung2(tier_separation_bound=None)`). The error message SHALL
name the dictionary's encoding and explain why no bound is
available.

#### Scenario: assertion passes on clearly-tiered HEA fixture

- **GIVEN** an `Experiment` whose dictionary is HEA-encoded with
  `tier_separation_bound=0.025` and a φ sweep that keeps every
  per-point `tier_separation >= 0.025`
- **WHEN** the experiment runs with
  `assertions=["concept_gram_tier_separation_bound_holds"]`
- **THEN** every entry of
  `result.assertion_pass["concept_gram_tier_separation_bound_holds"]`
  is `True`

#### Scenario: assertion rejected on dictionary lacking a bound

- **GIVEN** an `Experiment` whose dictionary is `MPSRung1`-encoded,
  with `assertions=["concept_gram_tier_separation_bound_holds"]`
- **WHEN** the `Experiment` is constructed
- **THEN** `__post_init__` raises `ValueError` naming the encoding

### Requirement: Cancellation accepts CancellationConfig

`Cancellation` SHALL accept an optional keyword argument
`config: CancellationConfig | None = None`. When supplied,
fields from `config` (`tolerance`, `preserve_tiers`, `optimize`,
`grid_outer`, `min_amp_overlap`) SHALL provide values for the
corresponding constructor fields. An explicit per-field keyword
argument SHALL override the config; the config SHALL override the
existing per-field defaults declared on `Cancellation`. When
`config is None`, behaviour SHALL be identical to today's
per-field-default behaviour.

`config` SHALL NOT replace the `dictionary`, `target_pair`,
`knobs`, or `optimize_all` fields — those remain explicit
constructor inputs because they describe the *target* of the
search, not its *tuning*.

The per-field defaults on `Cancellation` itself SHALL remain
unchanged in this change (`tolerance=0.05`,
`preserve_tiers=True`, `optimize={"method": "grid",
"max_steps": 50}`); the dataclass values in `CancellationConfig`
mirror them.

#### Scenario: config supplies tolerance and preserve_tiers

- **GIVEN** `cfg = CancellationConfig(tolerance=0.01,
  preserve_tiers=False)` and a valid HEA dictionary `d` with
  features `("a", "b")`
- **WHEN** `Cancellation(dictionary=d, target_pair=("a", "b"),
  config=cfg)` is constructed
- **THEN** the resulting instance has `tolerance == 0.01` and
  `preserve_tiers is False`

#### Scenario: per-field kwarg overrides config

- **GIVEN** `cfg = CancellationConfig(tolerance=0.01)`
- **WHEN** `Cancellation(dictionary=d, target_pair=("a", "b"),
  config=cfg, tolerance=0.001)` is constructed
- **THEN** `tolerance == 0.001`

#### Scenario: no-config call preserves legacy behaviour

- **WHEN** `Cancellation(dictionary=d, target_pair=("a", "b"))`
  is constructed with no `config`
- **THEN** `tolerance == 0.05`, `preserve_tiers is True`, and
  `optimize == {"method": "grid", "max_steps": 50}` — exactly
  matching the legacy defaults

