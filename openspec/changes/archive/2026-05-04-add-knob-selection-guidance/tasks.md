# add-knob-selection-guidance — tasks

## 0. Proposal

- [x] 0.1 `proposal.md` — Why / What Changes / Capabilities / Out of
      Scope / Impact (replaces the rejected `suggest_safe_knobs`
      helper with documentation in the existing report).
- [x] 0.2 `specs/analysis/spec.md` — MODIFIED `render_report` adds
      a `## Choosing knobs` section quoting
      `KNOB_SELECTION_GUIDANCE`.
- [x] 0.3 `openspec validate add-knob-selection-guidance --strict` ✓.

## 1. Implementation

- [x] 1.1 `polygram/analysis/triage.py` — add
      `KNOB_SELECTION_GUIDANCE` module-level string constant.
      Content covers the five empirical findings named in the
      spec: default phi knob, cluster-shared grammar preference,
      cluster-shatterer hazard, Rz-depth-0 zero-leverage, structural
      floor.
- [x] 1.2 `render_report` — emit `## Choosing knobs` section
      between `## Per-feature sensitivity` and
      `## Encoding suitability`. Section body quotes
      `KNOB_SELECTION_GUIDANCE`.
- [x] 1.3 `polygram/analysis/__init__.py` — re-export
      `KNOB_SELECTION_GUIDANCE`.

## 2. Tests

- [x] 2.1 `tests/test_analysis.py` — extend the existing
      `render_report` test (or add a sibling) to assert the new
      `## Choosing knobs` heading is present and that a stable
      substring (`"cluster-shatterer"`) from the guidance text
      appears in the rendered report.
- [x] 2.2 New `test_knob_selection_guidance_constant_exposed` —
      `polygram.analysis.KNOB_SELECTION_GUIDANCE` is a non-empty
      string and is the same content as appears in the rendered
      report.

## 3. Validate + commit

- [x] 3.1 Full pytest suite green; ruff clean.
- [x] 3.2 `openspec validate add-knob-selection-guidance --strict` ✓.
- [ ] 3.3 Commit + push, open PR (proposal + impl in one PR — the
      change is small enough to bundle).

## 4. Archive

- [ ] 4.1 `openspec archive add-knob-selection-guidance` after
      merge — propagate the modified `render_report` requirement
      into `openspec/specs/analysis/spec.md`.
