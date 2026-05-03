# cancellation-floor-diagnostic — tasks

## 1. Cancellation.structural_floor()

- [x] 1.1 `Cancellation.structural_floor() -> float` — evaluate
      target-pair overlap at `(φ, φ)` (δ=0) and `(φ, φ+π)` (δ=π)
      using the current `target_pair[0]` φ as anchor; return
      `min(m_zero, m_pi)` (i.e., `M − |V|`)
- [x] 1.2 Helper `_floor_terms()` returning `(M, V)` for tests /
      future diagnostics; called by both `structural_floor()` and
      `run()` so the value is cached without recomputation

## 2. CancellationResult fields

- [x] 2.1 Add `structural_floor: float`,
      `cancellation_efficiency: float | None` to `CancellationResult`
- [x] 2.2 In `Cancellation.run()`, compute the floor once and pass
      it to the constructed result; derive
      `cancellation_efficiency` from `(before, after, floor)` with
      the no-gap guard returning `None`
- [x] 2.3 `_render_summary` appends a "Structural floor" section
      with floor value, efficiency, and a one-line interpretation
- [x] 2.4 Docstrings on `Cancellation` and `CancellationResult`
      honestly describe φ-only search as encoding-bound

## 3. Tests

- [x] 3.1 `test_structural_floor_matches_grid_minimum` —
      `Cancellation.structural_floor()` matches `traj[:,2].min()`
      to within 1e-9 on the Animals-4 fixture
- [x] 3.2 `test_efficiency_one_when_floor_reached` — running on
      a mismatched-φ start, after `run()` returns
      `cancellation_efficiency == 1.0` (within 1e-9) when the
      optimum equals the floor
- [x] 3.3 `test_efficiency_none_when_already_at_floor` — when
      `before_overlap == structural_floor`, efficiency is `None`
- [x] 3.4 `test_summary_includes_floor_section` — materialized
      `<name>_summary.md` contains the floor and efficiency
      values in a "Structural floor" section

## 4. README + docs

- [x] 4.1 README — Cancellation section gains a "Structural floor"
      callout explaining what `cancellation_efficiency` reports
- [x] 4.2 README — link to `docs/research/cancellation-phase-floor.md`
      for the full derivation

## 5. Validate + commit

- [x] 5.1 `openspec validate cancellation-floor-diagnostic --strict` ✓
- [x] 5.2 All tests pass; ruff clean
- [x] 5.3 Commit + push
