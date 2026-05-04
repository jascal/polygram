## MODIFIED Requirements

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

#### Scenario: cluster-shared mutation preserves sibling overlaps

- **GIVEN** an HEA-encoded dictionary `D` with cluster `"dogs"` of
  size 2 (members `dog_poodle`, `dog_beagle`) and `Cancellation`
  on `("dog_poodle", "bird_hawk")` with cluster-shared θ knobs on
  both clusters
- **WHEN** the cancellation runs and produces
  `result.dictionary_at_optimum`
- **THEN** `abs(result.before_gram[i_poodle, i_beagle] -
  result.after_gram[i_poodle, i_beagle]) < 1e-9` (and likewise for
  every pair within `"birds"`)

#### Scenario: mixed cluster + feature knob list accepted but not invariant-preserving

- **GIVEN** a `Cancellation` whose `knobs` list mixes
  `<feature>.theta[r,d,q]` and `<cluster>.theta[r,d,q]` paths
- **WHEN** the cancellation runs
- **THEN** the run completes (mixed lists are valid input), but the
  cluster-shared invariant on within-cluster Gram entries does NOT
  apply (the per-feature mutations on one branch break the matched
  unitarity)

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
  cluster name): the caveat SHALL state that within-cluster Gram
  entries are preserved exactly by unitarity, and recommend
  verifying `concept_gram_tier_separation` on the materialized
  optimum.
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

#### Scenario: cluster-shared caveat names the unitarity guarantee

- **WHEN** the knob list contains only cluster-shared paths and
  `materialize` is called
- **THEN** the produced `<name>_summary.md` `## Caveat` section
  text mentions that within-cluster Gram entries are preserved
  exactly (and does NOT use the per-feature "best value found"
  language)

#### Scenario: mixed caveat names both warnings

- **WHEN** the knob list mixes per-feature and cluster-shared paths
  and `materialize` is called
- **THEN** the produced `<name>_summary.md` `## Caveat` section
  contains both the per-feature "best value found" warning and an
  explicit note that the cluster-shared invariant does NOT apply to
  mixed lists
