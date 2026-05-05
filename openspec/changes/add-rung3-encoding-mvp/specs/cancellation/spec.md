## ADDED Requirements

### Requirement: Cancellation supports encoding="rung3" with joint amp + aux optimization

`polygram.cancellation.Cancellation` SHALL accept `encoding="rung3"` as a value parallel to the existing `"mps"` and `"hea"` modes. Under `encoding="rung3"`, the optimizer SHALL jointly search over (φ_a, φ_b, theta_amp, psi_aux) per pair via the following pipeline:

1. **Outer grid** over (theta_amp, psi_aux). Default 5×5 cells:
   `theta_amp ∈ linspace(0, π/2, 5)` and
   `psi_aux ∈ linspace(0, 2π, 5, endpoint=False)`. Configurable
   via `Cancellation(grid_outer=(M, N))`.
2. **Inner phase optimization** at every outer cell — reuse the
   existing 2-φ MPSRung1 phase optimizer to convergence on
   (φ_a, φ_b) with (theta_amp, psi_aux) frozen at the cell's
   value. Cache the cell's best (φ_a, φ_b, post-overlap).
3. **Scipy refine** — at the best outer cell, run
   `scipy.optimize.minimize` over all four knobs starting from
   that cell's best.

The optimizer SHALL return the global minimum across the outer × inner search.

### Requirement: Cancellation Rung3 result reports MPS phase-only floor as the residual baseline

A `Cancellation(encoding="rung3").run()` result's `structural_floor` field SHALL carry the *MPS-phase-only floor* `M − |V|` of the same (α, β, γ) tuple — i.e., the floor that the corresponding `MPSRung1` cancellation on the same dictionary geometry would have hit.

This is implemented by the Rung3 cancellation internally constructing the MPSRung1 instance with the same (α, β, γ) parameters and calling its existing `structural_floor()` helper. The Rung3 implementation does NOT add a new `structural_floor` definition for Rung3 itself.

The result's docstring SHALL clearly state that for Rung3 results, `structural_floor` is the MPS-phase-only baseline the Rung3 optimizer was *trying to break*, not a floor the Rung3 optimizer was bounded by.

### Requirement: Cancellation Rung3 reports per-pair amp + aux residuals

The `CancellationResult` for `encoding="rung3"` SHALL expose, in addition to the existing fields:

- `theta_amp_optimum: float` — the θ_amp value at the global
  minimum.
- `psi_aux_optimum: float` — the ψ_aux value at the global
  minimum.

Both fields SHALL be populated for Rung3 results and SHALL be `float("nan")` for `encoding ∈ {"mps", "hea"}` results (the existing modes).

### Requirement: Cancellation Rung3 mode is torch-free

The `encoding="rung3"` cancellation path SHALL NOT import torch or transformers. The optimizer uses numpy + scipy only (existing dependencies). No new optional extra is introduced for cancellation.

### Requirement: Rung3 cancellation cost is bounded

For an N-cell outer grid with default 5×5 = 25 cells, K-cell inner phase grid (the existing 2-φ optimizer), and a single scipy refine, the total Gram evaluations per pair SHALL be bounded by `outer × inner + scipy_refine_evals` where `scipy_refine_evals` is the scipy minimizer's evaluation budget (typically ≤ 200 for `Nelder-Mead`).

Actual runtime against the §4.4 8-feature panel (28 pairs) under default settings SHALL fit within a 30-minute wall-clock budget on commodity laptop hardware. The implementation SHOULD verify this empirically as part of the worked-example smoke test.
