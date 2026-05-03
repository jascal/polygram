## Why

Polygram's first experiment primitive — `InterferenceSweep` — answers
"how does φ-steering change overlaps across this Dictionary?" The
natural follow-on question is "what φ values *minimize* a target-pair
overlap without breaking the cluster hierarchy?" That's a different
shape of work: an optimizer on top of the Gram, not a sweep through it.

This change introduces `Cancellation` — Polygram's second experiment
primitive. Given a target pair, it finds φ values that drive
`|<A|B>|²` below a tolerance, optionally constrained to preserve the
hierarchical-ordering tier structure, and returns a result that
includes the before/after Gram, the optimization trajectory, the
optimized Dictionary, and the verifiable Q-Orca artifact baking the
optimized φs in.

Two optimization backends ship: a deterministic 2D grid scan over
`(φ_A, φ_B)` (no dependencies, default), and a `scipy.optimize`
backend behind a new `[opt]` extra. The grid backend is what the
2D Animals heatmap (last release) revealed: the antidiagonal
direction in `(φ_dog, φ_bird)` space is where overlap collapses.
Cancellation makes that observation actionable.

Bundled with this primitive: small SAE-import polish — γ assignment
from in-cluster projection-vector PCA (the secondary axis after β),
per-feature reconstruction error in `SelectionReport`, and a
tier-preservation estimate (Pearson correlation of original
projection-space cosine overlaps against the analytic Polygram Gram).
These are the three deltas the prior round flagged but didn't ship.

After this change, Polygram has two complementary primitives —
`InterferenceSweep` for landscape exploration and `Cancellation` for
goal-directed φ-search — plus an SAE bridge that surfaces enough
fidelity stats for researchers to judge what they're losing.

## What Changes

- **NEW** `experiment` capability — `Cancellation` primitive:
  - `polygram.cancellation.Cancellation` dataclass with fields
    `dictionary`, `target_pair`, `tolerance: float = 0.05`,
    `preserve_tiers: bool = True`,
    `optimize: dict = {"method": "grid", "max_steps": 50}`,
    `optimize_all: bool = False` (reserved future flag — must be
    False in v0; True raises `NotImplementedError`).
  - `Cancellation.run()` returns a `CancellationResult`:
    `optimized_phis: dict[str, float]`,
    `before_gram, after_gram: np.ndarray`,
    `before_overlap, after_overlap: float`,
    `tolerance_met: bool`,
    `method: str`, `trajectory: np.ndarray`,
    `feasible_count: int`,
    `dictionary_at_optimum: Dictionary`.
  - `CancellationResult.plot(path)` — for `method="grid"`: heatmap
    of overlap on the (φ_A, φ_B) grid with feasible region masked
    and optimum starred; for `method="scipy"`: line plot of
    objective vs evaluation count. Lazy `matplotlib` import.
  - `CancellationResult.materialize(output_dir)` — emits
    `<name>.q.orca.md` (Dictionary at optimum φs), `<name>_summary.md`
    (config + before/after + tolerance met), `<name>_trajectory.csv`
    (every evaluation: φ_A, φ_B, overlap, feasible).
  - Grid backend: `max_steps` is the resolution **per axis** (default
    50 → 2,500 evaluations on the (φ_A, φ_B) grid in `[0, 2π]²`).
    Pure numpy, deterministic, no extra deps.
  - Scipy backend: `from scipy.optimize import differential_evolution`,
    bounds `[(0, 2π), (0, 2π)]`, `maxiter=optimize["max_steps"]`,
    `seed=0`. `preserve_tiers` constraint enforced via large penalty
    on infeasible candidates. Lazy import; clear `ImportError`
    pointing to `polygram[opt]` if absent.
  - When `preserve_tiers=True` and no feasible point is found, the
    result still returns the best infeasible candidate but with
    `tolerance_met=False` and `feasible_count=0` so the caller can
    see what happened.

- **MODIFIED** `sae` capability — finer-grained fidelity:
  - `SelectionReport` gains three fields:
    `reconstruction_error: dict[str, float]` (per-feature Euclidean
    distance from each feature's projection vector to its assigned
    cluster centroid), `tier_preservation: float | None` (Pearson
    correlation between off-diagonal `|G|²` entries of the
    projection-space cosine-overlap matrix and the analytic
    Polygram Gram of the constructed Dictionary; `None` if there
    is only one selected feature so no off-diagonals exist),
    and `gamma_method: str` (`"zero"` if γ defaults are kept, or
    `"projection_pca"` if the new auto-γ path is used).
  - `from_sae_lens` gains keyword `assign_gamma: bool = False`.
    When True, γ for each feature is its projection vector's
    coefficient on the first PCA component of its assigned
    cluster's centered projections, rescaled into `gamma_range`
    (default `(-0.25, 0.25)`). When False (default), γ stays 0.
  - The new `gamma_range: tuple[float, float] = (-0.25, 0.25)`
    parameter on `from_sae_lens`.

- `polygram/cancellation.py` — new module hosting `Cancellation`
  and `CancellationResult`.
- `polygram/__init__.py` — re-export `Cancellation`,
  `CancellationResult`.
- `pyproject.toml` — `[project.optional-dependencies] opt = ["scipy"]`.
- `examples/cancellation_example.py` — new combined walk:
  load toy SAE → 4-feature Dictionary → run an `InterferenceSweep`
  AND a `Cancellation` → save all artifacts including the optimized
  `.q.orca.md` and the trajectory plot.
- `tests/test_cancellation.py` — coverage for grid backend
  end-to-end, scipy backend (skipped if scipy absent),
  preserve_tiers feasibility filtering, materialized .q.orca.md
  parses + verifies, plot writes a non-empty PNG, optimize_all
  not-yet-implemented refusal.
- `tests/test_sae_import.py` — extend with γ assignment, per-feature
  reconstruction error, and tier preservation tests.
- README — quickstart adds `Cancellation`, SAE Integration section
  documents `assign_gamma` + new report fields, capacity callout
  unchanged.

## Capabilities

### New Capabilities

*(none — `Cancellation` lives in the existing `experiment` capability)*

### Modified Capabilities

- `experiment` — adds the `Cancellation` primitive and its result
  type alongside `InterferenceSweep`.
- `sae` — extends `SelectionReport` with reconstruction error,
  tier-preservation correlation, gamma method; extends
  `from_sae_lens` with optional auto-γ assignment.

## Impact

- `polygram/cancellation.py` — new (~250 LOC)
- `polygram/sae_import.py` — extend (~80 LOC delta)
- `polygram/__init__.py` — two new re-exports
- `pyproject.toml` — `[opt]` extra (`scipy`)
- `tests/test_cancellation.py` — new (~10 tests)
- `tests/test_sae_import.py` — +3 tests
- `examples/cancellation_example.py` — new combined walk
- `README.md` — quickstart + SAE Integration extension

No q-orca version bump. No physics changes. No breaking changes to
existing classes. Default behavior of `from_sae_lens` is unchanged
(γ stays 0 unless `assign_gamma=True`). `Cancellation` does not
touch `Experiment` or `InterferenceSweep`.
