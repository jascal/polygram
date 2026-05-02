## Why

The `cancellation-primitive` change shipped `Cancellation` with a φ-only
search over the target pair. While implementing it we discovered a
fundamental limit: the squared overlap as a function of the phase
difference δ = φ_A − φ_B factors as `|⟨A|B⟩|² = M + V·cos(δ)`, where M
is set by β/α/γ alignment and V is the part φ modulates. Pure-phase
search bounds the overlap to `[M − |V|, M + |V|]` — there is a
**structural floor at M − |V|** that φ cannot pierce. To drive overlap
below the floor you need amplitude matching (vary β/α/γ, or coordinate
phases across multiple features in one cluster).

Today this limit is invisible at runtime. A researcher running
`Cancellation` on an Animals-4 dictionary sees `after_overlap = 0.59`
and a `tolerance_met = False` and has no way to tell whether the
optimizer underperformed or whether it hit the structural floor and
nothing better is reachable. Honest diagnostics for this case are
load-bearing: they turn a footgun ("did Cancellation just fail?")
into a legible result ("you got 100% of the available cancellation
gap; the rest of the residue is encoding-bound").

This change adds two diagnostics — `structural_floor` and
`cancellation_efficiency` — and a method on `Cancellation` that
computes the floor analytically (two Gram evaluations, backend-free)
so users can query the floor *before* running. The full
`Cancellation.run()` path then caches the floor on the result and
derives the efficiency.

## What Changes

- **MODIFIED** `experiment` capability:
  - `Cancellation.structural_floor() -> float` — analytic minimum
    overlap reachable by varying only `(φ_A, φ_B)` on the target pair
    while holding all other features fixed. Computed by evaluating
    the target-pair overlap at two known δ values: `(φ, φ)` (δ=0)
    and `(φ, φ+π)` (δ=π), giving `M = (m+M)/2`, `|V| = (M−m)/2`,
    floor = `min(m, M)` (i.e., `M − |V|`). Uses
    `dictionary.features[a].phi` as the anchor `φ` so the floor
    reflects the *current* feature configuration; not affected by
    `preserve_tiers`.
  - `CancellationResult` gains `structural_floor: float` and
    `cancellation_efficiency: float | None`:
    - `structural_floor` — same value as
      `Cancellation.structural_floor()`, cached on the result.
    - `cancellation_efficiency` — `(before_overlap −
      after_overlap) / (before_overlap − structural_floor)`, clamped
      to `[0.0, 1.0]`. `None` when `before_overlap −
      structural_floor < 1e-9` (no cancellation gap to measure —
      already at the floor).
  - `CancellationResult.materialize()` summary appends a
    "Structural floor" section reporting `structural_floor`,
    `cancellation_efficiency`, and a one-line interpretation
    ("phase search exhausted — encoding-bound" /
    "phase search underutilized" / "no cancellation gap available").
  - Docstrings on `Cancellation` and `CancellationResult` honestly
    describe φ-only search as a **constraint solver bounded by the
    encoding's structural floor**, not a destructive-interference
    oracle.

- `polygram/cancellation.py` — extend (~80 LOC delta).
- `tests/test_cancellation.py` — +4 tests:
  `test_structural_floor_matches_grid_minimum`,
  `test_efficiency_one_when_floor_reached`,
  `test_efficiency_none_when_already_at_floor`,
  `test_summary_includes_floor_section`.
- `README.md` — Cancellation section gains a "Structural floor"
  callout explaining the limit and what `cancellation_efficiency`
  reports.

## Capabilities

### New Capabilities

*(none)*

### Modified Capabilities

- `experiment` — `Cancellation` exposes a `structural_floor()`
  diagnostic; `CancellationResult` gains `structural_floor` and
  `cancellation_efficiency` fields, materialized summary reports
  them.

## Impact

- `polygram/cancellation.py` — ~80 LOC delta
- `tests/test_cancellation.py` — +4 tests
- `README.md` — small extension to existing Cancellation section
- `docs/research/cancellation-phase-floor.md` — already drafted;
  promoted to a doc reference

No breaking changes. Existing `CancellationResult` consumers ignore
the new fields. No new dependencies. No q-orca version bump. The
`Cancellation.structural_floor()` method costs two Gram evaluations
regardless of backend — negligible compared to a 2,500-cell grid scan.
