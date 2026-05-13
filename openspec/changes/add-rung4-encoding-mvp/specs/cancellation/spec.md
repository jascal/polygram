## MODIFIED Requirements

### Requirement: Cancellation accepts encoding="rung4"

`polygram.cancellation.Cancellation` SHALL accept `encoding="rung4"`
as a value parallel to the existing `"mps"`, `"hea"`, and `"rung3"`.
`_infer_encoding_string` SHALL return `"rung4"` for `Rung4` instances.

The default canonical knob list for `encoding="rung4"` SHALL be the
6-knob fixed layout:

```
[f"{a}.phi", f"{b}.phi",
 f"{b}.theta_amp", f"{b}.psi_aux",
 f"{b}.theta_amp_b", f"{b}.psi_amp_b"]
```

where `a` and `b` are the target pair feature names. Custom knob
lists SHALL be rejected in v0 with a clear error message matching
the existing Rung3 stance.

#### Scenario: Rung4 cancellation accepts the canonical knob list

- **WHEN** `Cancellation(dictionary=<Rung4 dictionary>, target_pair=(a, b), encoding="rung4")`
  is constructed without an explicit `knobs` argument
- **THEN** `self.knobs` equals the canonical 6-knob list

#### Scenario: Rung4 cancellation rejects non-canonical knob lists

- **WHEN** `Cancellation(... , encoding="rung4", knobs=["a.phi", "b.phi"])`
  is constructed
- **THEN** a `ValueError` is raised whose message references the
  canonical 6-knob list and notes that custom lists are not supported
  in v0

### Requirement: Rung4 joint optimiser handles 4-dim amp outer grid

The Rung4 joint optimiser SHALL follow the three-stage pipeline
shipped for Rung3: outer grid over feature B's amp knobs, inner 2-φ
MPS-equivalent grid at every outer cell, scipy Nelder-Mead refine
over the full knob set starting from the best outer cell.

The outer grid for Rung4 SHALL be 4-dimensional
(θ_amp, ψ_aux, θ_amp_b, ψ_amp_b) rather than Rung3's 2-dimensional
(θ_amp, ψ_aux). The default `grid_outer` value for Rung4 SHALL be
chosen to keep the total outer cell count comparable to Rung3's
(within an order of magnitude); the recommended default is
`(3, 3, 3, 3)` = 81 cells (vs Rung3's `(5, 5)` = 25).

#### Scenario: Rung4 joint optimiser lowers target overlap

- **WHEN** `Cancellation(...).run()` is called on a 2-feature Rung4
  dictionary with non-trivial initial overlap
- **THEN** `result.after_overlap < result.before_overlap`

#### Scenario: Rung4 outer grid is 4-dimensional

- **WHEN** the Rung4 joint optimiser materialises its outer grid
- **THEN** the grid has shape `(grid_outer[0], grid_outer[1], grid_outer[2], grid_outer[3])`

### Requirement: Rung4 min_amp_overlap applies to the product-amp overlap

The `min_amp_overlap` constraint SHALL be applied to
`rung4_amp_overlap_squared(theta_a, psi_a, theta_a_b, psi_a_b,
theta_b, psi_b, theta_b_b, psi_b_b)` (computed from feature A's
anchored amp knobs and the candidate feature B's amp knobs). Outer-
grid cells and Nelder-Mead candidates whose amp factor falls below
the threshold SHALL be marked infeasible (same penalty pattern as
Rung3).

#### Scenario: min_amp_overlap rejects amp-zeroing candidates

- **WHEN** `Cancellation(... , min_amp_overlap=0.5)` runs on a Rung4
  pair where the unconstrained optimum has amp overlap ≈ 0
- **THEN** the constrained run's `feasible_count` excludes that
  candidate and the chosen optimum has amp factor ≥ 0.5

### Requirement: Rung4 structural_floor reduces to the MPS-phase-only floor

`Cancellation.structural_floor()` on a Rung4 dictionary SHALL return
the MPS-phase-only floor `M − |V|` evaluated on the (α, β, γ, φ)
subset, using the same `_mps_equivalent_floor` helper Rung3 uses.

The returned value is the baseline the Rung4 optimiser is trying to
break, NOT a bound the optimiser is constrained by.

#### Scenario: Rung4 floor equals the Rung3 floor on shared (α, β, γ, φ)

- **WHEN** a 2-feature Rung4 and Rung3 dictionary with the same
  (α, β, γ, φ) compute `structural_floor()`
- **THEN** both return the same value to 1e-12 absolute

### Requirement: Rung4 plot kind="grid" raises NotImplementedError

`CancellationResult.plot(kind="grid")` on a Rung4 result SHALL raise
`NotImplementedError` because the 6-dimensional search space cannot
be visualised as a 2D grid. Other plot kinds (`"before_after"`,
`"scipy"`) SHALL continue to work.

#### Scenario: grid plot rejected on Rung4

- **WHEN** `result.plot("/tmp/grid.png", kind="grid")` is called on
  a Rung4 result
- **THEN** `NotImplementedError` is raised with a message naming the
  6-knob dimensionality

#### Scenario: before_after plot works on Rung4

- **WHEN** `result.plot("/tmp/before_after.png", kind="before_after")`
  is called on a Rung4 result
- **THEN** the plot file is written successfully
