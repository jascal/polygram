## ADDED Requirements

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
