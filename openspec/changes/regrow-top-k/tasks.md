## 1. RegrowConfig field

- [ ] 1.1 Add `top_k: int | None = None` field to `RegrowConfig`. Place it after `n_init` and before `device` to keep config-construction call sites readable
- [ ] 1.2 `RegrowConfig.__post_init__`: when `top_k is not None` and `top_k < 0`, raise `ValueError` naming the field and value
- [ ] 1.3 Confirm `RegrowConfig.to_dict()` round-trips the new field. Add a smoke assertion in `tests/test_regrow_config.py` (or wherever the existing config tests live)

## 2. Regrower.run gating

- [ ] 2.1 In `Regrower.run(self, output_checkpoint)`, after `plan = self.plan()`, slice `plan.populations[:self.top_k]` when `self.top_k is not None and self.top_k < len(plan.populations)`. Use `dataclasses.replace(plan, populations=...)` to avoid mutating the plan in place
- [ ] 2.2 The remaining slots stay zero — no special handling needed because the regrower iterates `plan.populations` and only writes those rows
- [ ] 2.3 The resulting `RegrowReport.populations` reflects only the regrown subset (no extra entries for un-regrown slots)

## 3. from_compression_report kwarg

- [ ] 3.1 Add `top_k: int | None = None` to `Regrower.from_compression_report(...)` signature, after the existing `device` kwarg
- [ ] 3.2 In the `if config is not None: ...` precedence block, add `if top_k is None: top_k = config.top_k`. Per-field kwarg wins when both are set
- [ ] 3.3 Pass `top_k` to the `Regrower(...)` constructor

## 4. Byte-equivalence acceptance gate

- [ ] 4.1 New test `test_top_k_none_is_byte_identical_to_pre_change` in `tests/test_regrow_top_k.py`. Build a `Regrower` with `top_k=None`, run against a fixture compression report, hash the output checkpoint. Assert SHA matches the pre-change baseline (captured by running the test once on `main` before the change lands and pinning the SHA in the test fixture)
- [ ] 4.2 If 4.1 fails: do not rebaseline. The slicing logic is wrong; fix it

## 5. Functional tests

- [ ] 5.1 `test_top_k_caps_population_count`: 10-zeroed-slot fixture; `top_k=3` regrows 3 slots; remaining 7 slots stay zero (assert via numpy `(rows == 0).all(axis=1)` on the un-regrown rows)
- [ ] 5.2 `test_top_k_above_zeroed_count_is_no_op_cap`: 5-slot fixture; `top_k=999` regrows all 5; no error
- [ ] 5.3 `test_top_k_zero_is_no_regrow`: 5-slot fixture; `top_k=0` produces a checkpoint where no rows changed vs the input compression's output
- [ ] 5.4 `test_top_k_deterministic`: two runs with identical config (`top_k=3`, same seed) produce byte-identical output checkpoints
- [ ] 5.5 `test_top_k_negative_raises`: `RegrowConfig(top_k=-1)` raises `ValueError` at construction

## 6. CompressionReport / RegrowPlan fields are unchanged

- [ ] 6.1 Confirm by inspection: this change touches only `RegrowConfig` and the slicing site in `Regrower.run`. No fields added to `CompressionReport`, `RegrowPlan`, or `RegrowReport`. No JSON schema changes
- [ ] 6.2 The `RegrowReport.populations` list naturally reflects only the regrown subset because it's built during iteration

## 7. Documentation

- [ ] 7.1 `docs/regrowth.md` (or whichever doc covers the regrower API) — new "Capping regrowth with `top_k`" subsection. Keep it short: when to use it, what it does, the byte-equiv guarantee under `None`, the current selection strategy (plan order)
- [ ] 7.2 `CHANGELOG.md` `## [Unreleased]` `### Added` entry: "RegrowConfig.top_k optional cap on per-call regrow count. Default None preserves byte-equivalence with pre-change behavior. Selection is plan-order; richer strategies tracked as `regrow-selection-strategies`"
- [ ] 7.3 The `### Added` entry MAY mention the downstream sae-forge use case ("unblocks sae-forge adaptive-regrow") for context, but is not required to

## 8. Coordination

- [ ] 8.1 After landing this change and cutting a polygram release, ping the sae-forge maintainer to bump the polygram floor in `pyproject.toml` (`polygram>=X.Y.Z` where X.Y.Z is the release containing this change)
- [ ] 8.2 sae-forge resumes its parked `adaptive-regrow` work — controller scaffolding (`saeforge.basis.RegrowController` + `ForgePipeline.adaptive_regrow` knobs) lives on a stash on the impl branch, ready to thread through `ctx["regrow"]["top_k"] = effective_regrow_count`

## 9. OpenSpec scaffolding

- [x] 9.1 `openspec/changes/regrow-top-k/proposal.md`
- [x] 9.2 `openspec/changes/regrow-top-k/design.md`
- [x] 9.3 `openspec/changes/regrow-top-k/tasks.md` (this file)
- [x] 9.4 `openspec/changes/regrow-top-k/specs/tuning-config/spec.md` (MODIFIED — adds two ADDED requirements + one MODIFIED requirement)
- [ ] 9.5 Run `openspec validate regrow-top-k --strict` locally; resolve any structural complaints before opening the PR

## 10. Validation matrix

- [ ] 10.1 Full `pytest` suite passes (existing + new) on the polygram CI matrix
- [ ] 10.2 Byte-equivalence gate (4.1) passes
- [ ] 10.3 Determinism gate (5.4) passes
- [ ] 10.4 No new lint errors (existing `ruff` / `mypy` gates pass)
