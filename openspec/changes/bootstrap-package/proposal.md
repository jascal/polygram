## Why

Polygram is a new Python package that sits on top of Q-Orca and provides a
researcher-friendly frontend for declaring polysemantic feature dictionaries
and running quantum interference experiments. The repo currently has only a
README, `pyproject.toml`, and a stubbed `polygram/__init__.py`; before any
behavior can be added, the package needs a stable foundation: an importable
module skeleton, a passing test suite, lint config, and a CI workflow that
exercises both.

This change does **not** add user-facing functionality — it establishes the
floor that subsequent OpenSpec changes (`core-dictionary-mpsrung1`,
`experiment-interference-sweep`, `animals-example`) will build on.

## What Changes

- Pin a `q-orca` minimum version that includes the safe-`Rz` matcher
  (q-orca-lang PR #51, squash-merged as `a92c9e1`) and document the dep.
- Add a tiny smoke-test (`tests/test_smoke.py`) that imports `polygram`
  and asserts `__version__` is a non-empty string — guarantees the package
  installs and is importable.
- Add a GitHub Actions workflow (`.github/workflows/test.yml`) that runs
  `pytest` and `ruff check` on Python 3.11 + 3.12.
- Add a top-level `polygram/_version.py` shim is **not** introduced —
  hatchling already reads `__version__` from `polygram/__init__.py`.
- Add `AGENTS.md` at the repo root with a one-screen orientation for any
  agent (human or AI) picking up work: project goal, OpenSpec workflow,
  q-orca dependency contract.

## Capabilities

### New Capabilities

*(none — bootstrap only; no behavior yet)*

### Modified Capabilities

*(none)*

## Impact

- `polygram/` — package init only, no logic.
- `tests/test_smoke.py` — new.
- `.github/workflows/test.yml` — new.
- `AGENTS.md` — new.
- No public API. No q-orca interaction yet — that lands in
  `core-dictionary-mpsrung1`.
