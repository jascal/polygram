# bootstrap-package — tasks

## 1. Package skeleton

- [x] 1.1 `pyproject.toml` with hatchling backend, Python ≥3.10, `q-orca`
      pinned at the post-PR-#51 minimum
- [x] 1.2 `polygram/__init__.py` exposes `__version__`
- [x] 1.3 `.gitignore` for Python build artifacts and notebooks
- [x] 1.4 `README.md` with project intent, layout, and dev quickstart

## 2. Tests + lint

- [ ] 2.1 `tests/test_smoke.py` — imports `polygram`, asserts version is
      a non-empty string and `q_orca` is importable
- [ ] 2.2 Confirm `pytest` passes locally with `pip install -e ".[dev]"`
- [ ] 2.3 Confirm `ruff check polygram tests` is clean

## 3. CI

- [ ] 3.1 `.github/workflows/test.yml` runs `pytest` + `ruff check`
      on Python 3.11 and 3.12
- [ ] 3.2 First successful CI run on `main`

## 4. Agent orientation

- [ ] 4.1 `AGENTS.md` covers: project goal, OpenSpec workflow, q-orca
      dep contract (do not vendor; treat `compute_concept_gram_mps` as
      stable API), file-layout map
