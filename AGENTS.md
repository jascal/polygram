# Polygram — Agent Orientation

## What this repo is

Polygram is a researcher-friendly Python frontend that emits verifiable
[Q-Orca](https://github.com/jascal/q-orca-lang) `.q.orca.md` machines for
mechanistic-interpretability experiments on hierarchical polysemantic feature
dictionaries. It does **not** define a new file format — it generates standard
Q-Orca artifacts and uses Q-Orca for verification, simulation, and the analytic
Gram helper.

## Workflow: OpenSpec-driven

All non-trivial work lands through OpenSpec changes in `openspec/changes/`.
The v0 milestone is staged as four changes:

1. `bootstrap-package` — pyproject, smoke test, CI, this file. Process-only.
2. `core-dictionary-mpsrung1` — `Feature`, `Dictionary`, `MPSRung1`,
   `Dictionary.gram()`.
3. `experiment-interference-sweep` — `Experiment`, `InterferenceSweep`,
   `polygram.emit.write_qorca`, built-in assertions.
4. `animals-example` — worked example + integration test that closes v0.

Each change has `proposal.md` (why + scope), `tasks.md` (checklist), and
`specs/<capability>/spec.md` (delta requirements + scenarios). Validate with
`openspec validate <change-name>` before working on it; archive via
`openspec archive` when done.

## Q-Orca dependency contract

- Pinned at `q-orca>=0.9.0` — the first PyPI release with first-class rung-2
  HEA encoding (`## encoding` + `## theta` grammar, optional `cluster` column)
  and the declarative `concept_gram_tier_separation` invariant enforced at
  Stage 4b (`HEA_TIER_INVARIANT_VIOLATED` / `HEA_TIER_UNDEFINED` /
  `HEA_TIER_INVARIANT_NOT_APPLICABLE`).
- Treat `q_orca.compiler.concept_gram_mps.compute_concept_gram_mps` as a
  stable API surface. Always call it with `form="preparation"` — the
  inverse-form `Rz` symmetry break documented in q-orca-lang's
  `examples/larql-animals-interference.q.orca.md` is a hard error
  (`MpsGramConfigurationError(kind="rz_in_inverse_form")`), but stay on the
  preparation form regardless.
- For HEA dictionaries, the stable surface is
  `q_orca.compiler.concept_gram_hea.compute_concept_gram_hea` plus the
  `compute_tier_separation` analysis helper. Both are part of v0.9.0's
  public API.
- Q-Orca is **not** vendored. If you need a fix in Q-Orca, open it there.

## File layout

```
polygram/                  Python package
  __init__.py              version + (later) public re-exports
  dictionary.py            Feature, Dictionary               (added in change 2)
  encoding.py              MPSRung1                          (added in change 2)
  experiment.py            Experiment, InterferenceSweep     (added in change 3)
  emit.py                  write_qorca                       (added in change 3)
  _qorca_emit.py           internal QMachineDef builder      (added in change 2)
  _assertions.py           built-in assertion checkers       (added in change 3)
tests/                     pytest suite (mirrors package layout)
examples/                  worked examples                   (added in change 4)
openspec/                  spec-driven changes + capability specs
.github/workflows/         CI
```

## Local dev

```bash
pip install -e ".[dev]"
pytest
ruff check polygram tests
```

## Conventions

- Only the `numpy` and `q-orca` runtime deps. Plotting and Jupyter are
  optional extras.
- Generated `.q.orca.md` files include a header comment block with the source
  Polygram dictionary name, a timestamp, and the git rev (or `unversioned`).
- No emojis in code or generated artifacts.
- Default to no comments unless the *why* is non-obvious.
