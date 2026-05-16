## 1. `LearnedAxisObjective` protocol + built-ins

- [ ] 1.1 Add `LearnedAxisObjective` Protocol to `polygram/geometry/protocols.py`. `runtime_checkable`. Signature `__call__(analytic_gram: np.ndarray, decoder_geom: np.ndarray, *, feature_names: list[str]) -> float`.
- [ ] 1.2 New module `polygram/geometry/objectives.py` with `spearman_objective`, `pearson_objective`, and `behavioural_objective(reference_pair_sims) -> Callable`. Unit tests in `tests/test_learned_axis_assignment.py::TestObjectives`.
- [ ] 1.3 Extract the existing off-diagonal-triangle Spearman helper from `examples/rung5_pareto_scans.py::_spearman` into `polygram/geometry/objectives.py` (rename to `_spearman_off_diagonal`) and have `spearman_objective` consume it. Update the example to import the canonical helper.

## 2. `KnobAssignmentResult.axis_assignment` field

- [ ] 2.1 Add `axis_assignment: dict[str, int | list[float]] | None = None` to `KnobAssignmentResult` in `polygram/geometry/protocols.py`.
- [ ] 2.2 Add `objective_value: float | None = None` and `objective_baseline: float | None = None` and `training_objective_value: float | None = None` to the same dataclass.
- [ ] 2.3 Verify `ClusteredKnobAssignment` and `UniformSphereKnobAssignment` continue to construct `KnobAssignmentResult` without these new fields (they default to `None`). No code changes expected; add a regression test.

## 3. `LearnedAxisAssignment` strategy class

- [ ] 3.1 Add `polygram/geometry/learned_axis_assignment.py` with the `LearnedAxisAssignment` dataclass. Fields: `solver: str = "greedy"`, `objective: LearnedAxisObjective = spearman_objective`, `max_axes: int = 32`, `validation_fraction: float = 0.0`, `scipy_restarts: int = 1`, `seed: int = 0`.
- [ ] 3.2 `__post_init__` validates `solver in {"greedy", "scipy"}`, `0.0 <= validation_fraction <= 0.5`, `max_axes >= 1`, `scipy_restarts >= 1`.
- [ ] 3.3 `assign(projections, feature_names, *, n_clusters, gamma_range, assign_gamma, seed, assign_amp_knobs=False, assign_phase_knobs=False, encoding=None) -> KnobAssignmentResult`. Implements the `KnobAssignment` protocol.
- [ ] 3.4 Internal helper `_build_axis_map_greedy(projs, encoding, objective, max_axes) -> dict[str, int]` performing the prototype's greedy permutation search.
- [ ] 3.5 Internal helper `_build_axis_map_scipy(projs, encoding, objective, max_axes, scipy_restarts) -> dict[str, list[float]]` performing continuous optimisation via `scipy.optimize.minimize(method="Nelder-Mead")` for k<3 and `differential_evolution` for k>=3. Initialises from the greedy result. Raises `ImportError` (with the existing `_SCIPY_INSTALL_HINT`-style message) when scipy is unavailable.
- [ ] 3.6 Internal helper `_apply_axis_map(projs, encoding, axis_map) -> tuple[list[float], list[float], list[float], list[tuple[tuple[float, float], ...]]]` that consumes a chosen map and returns per-feature `(alphas, phis, theta_amps, psi_auxes, theta_amp_bs, psi_amp_bs, amp_knobs_list)` consistent with the existing helpers' contracts.
- [ ] 3.7 HEA_Rung2 fallback: when `isinstance(encoding, HEA_Rung2)`, log INFO-once and delegate to `assign_amp_knobs_pca` + `assign_phase_knobs_pca`; set `axis_assignment=None` and copy `objective_value = objective_baseline = NaN` to flag the bypass.
- [ ] 3.8 Validation split: when `validation_fraction > 0`, hold out the configured fraction of off-diagonal pairs as a validation set; train objective on the remaining pairs; report validation score in `objective_value` and training score in `training_objective_value`.
- [ ] 3.9 Re-export `LearnedAxisAssignment`, `LearnedAxisObjective`, `spearman_objective`, `pearson_objective`, `behavioural_objective` from `polygram/geometry/__init__.py`.
- [ ] 3.10 Re-export `LearnedAxisAssignment` from `polygram/__init__.py` (top-level).

## 4. `from_sae_lens` opt-in plumbing

- [ ] 4.1 Add `learn_axis_assignment: bool | LearnedAxisAssignment | None = None` kwarg to `from_sae_lens` in `polygram/sae_import.py`.
- [ ] 4.2 Resolve the kwarg at function entry: `None`/`False` → keep current behaviour; `True` → instantiate `LearnedAxisAssignment()`; an instance → use directly. Document the resolution rule in the docstring.
- [ ] 4.3 In the dispatch flow, when `learn_axis_assignment` is populated AND the path is not `cluster_assignments` / `from_labels`: route through `LearnedAxisAssignment.assign(...)` instead of the strategy from `resolved_profile.knob_assignment`. (For the `cluster_assignments` / `from_labels` paths, we already bypass the strategy entirely; the learned-axis-assignment opt-in is also bypassed there in v1 — document this as a known limitation and a follow-up if real callers hit it.)
- [ ] 4.4 Populate per-feature knob arrays from `result.axis_assignment` (greedy: integer-indexed PC slice + rescale to range; scipy: linear-combination of PCs + rescale).
- [ ] 4.5 Surface the learned map in `SelectionReport.learned_axis_assignment` as a JSON-safe dict (cast numpy values to plain Python).

## 5. `SelectionReport.learned_axis_assignment` field

- [ ] 5.1 Add `learned_axis_assignment: dict[str, Any] | None = None` to `SelectionReport` in `polygram/sae_import.py`.
- [ ] 5.2 Round-trip the new field through `SelectionReport.to_dict()` / `from_dict()` (cast nested dicts cleanly; ensure no numpy types leak).
- [ ] 5.3 Pretty-print stanza in `SelectionReport.__str__` that surfaces the learned map when populated. Skipped (no output) when `None`.

## 6. `SAEImportConfig.learn_axis_assignment`

- [ ] 6.1 Add `learn_axis_assignment: bool | None = None` to `SAEImportConfig` in `polygram/config.py`.
- [ ] 6.2 Plumb through `from_sae_lens`'s `config=` kwarg handling so the config field maps to the function kwarg.
- [ ] 6.3 Round-trip through `SAEImportConfig.to_dict()` / `from_dict()`.

## 7. CLI flag

- [ ] 7.1 Add `--learn-axis-assignment` boolean flag to the `from-sae-lens` subcommand in `polygram/cli.py`.
- [ ] 7.2 Wire the flag through to `from_sae_lens(learn_axis_assignment=...)`.
- [ ] 7.3 Include the resulting `report.learned_axis_assignment` in the emitted report file.

## 8. Tests

- [ ] 8.1 `tests/test_learned_axis_assignment.py::TestObjectives` — Spearman / Pearson / behavioural built-ins return sane values on a hand-rolled fixture; protocol checks pass.
- [ ] 8.2 `tests/test_learned_axis_assignment.py::TestGreedySolver::test_deterministic` — twice-run same seed produces bit-identical `axis_assignment`.
- [ ] 8.3 `tests/test_learned_axis_assignment.py::TestGreedySolver::test_reproduces_scan4_k3_spearman_above_0_30`.
- [ ] 8.4 `tests/test_learned_axis_assignment.py::TestGreedySolver::test_reproduces_scan4_k4_spearman_above_0_30`.
- [ ] 8.5 `tests/test_learned_axis_assignment.py::TestGreedySolver::test_objective_value_no_worse_than_baseline` — assert `objective_value >= objective_baseline - 1e-6`.
- [ ] 8.6 `tests/test_learned_axis_assignment.py::TestScipySolver::test_initialises_from_greedy` — scipy result starts from greedy, doesn't regress below it.
- [ ] 8.7 `tests/test_learned_axis_assignment.py::TestScipySolver::test_requires_opt_extra` — `pytest.importorskip("scipy")`-gated; without scipy, ImportError with the install hint.
- [ ] 8.8 `tests/test_learned_axis_assignment.py::TestValidationSplit::test_train_vs_validation_objectives_separate` — validation-fraction split surfaces both scores.
- [ ] 8.9 `tests/test_learned_axis_assignment.py::TestHEAFallback` — HEA_Rung2 encoding falls through to hardcoded helpers, sets `axis_assignment=None`, logs INFO-once.
- [ ] 8.10 `tests/test_sae_import.py::TestLearnedAxisAssignment::test_default_behaviour_byte_identical` — without `learn_axis_assignment`, gram bit-identical to pre-change baseline.
- [ ] 8.11 `tests/test_sae_import.py::TestLearnedAxisAssignment::test_true_triggers_default_strategy`.
- [ ] 8.12 `tests/test_sae_import.py::TestLearnedAxisAssignment::test_instance_honoured`.
- [ ] 8.13 `tests/test_sae_import.py::TestLearnedAxisAssignment::test_report_field_populated` — `SelectionReport.learned_axis_assignment` is JSON-safe and has the expected keys.
- [ ] 8.14 `tests/test_config.py::TestSAEImportConfig::test_learn_axis_assignment_round_trip`.
- [ ] 8.15 `tests/test_cli.py::TestFromSaeLens::test_learn_axis_assignment_flag` — CLI smoke test.

## 9. Worked example

- [ ] 9.1 `examples/learned_axis_assignment_demo.py` — reproduces scan 4's headline (k=3 and k=4 Spearman lift) via the production strategy, emits JSON to `docs/research/data/learned_axis_assignment_demo.json`.
- [ ] 9.2 Smoke test in `tests/test_examples.py::test_learned_axis_assignment_demo`.

## 10. Documentation

- [ ] 10.1 `docs/research/learned-axis-assignment.md` — design note. Sections: motivation (scan 4 numbers), production design (solvers + objectives), API surface, expected real-SAE follow-up, link to the theoretical treatment §9 / §11 entries.
- [ ] 10.2 Update `docs/research/rung5-pareto-scans.md` scan 4 to cite the production strategy (replace "prototype" framing with "shipped in `add-learned-axis-assignment`").
- [ ] 10.3 Add a paragraph to the theoretical treatment `docs/research/theory/polygram.tex` §9 (Algorithms) introducing learned axis-to-knob calibration as a first-class algorithm; add a corresponding §11 (Open problems) entry on sample-complexity bounds for axis recovery.

## 11. Closing

- [ ] 11.1 Bump version in `pyproject.toml` to `0.8.0` (additive new strategy; new public API surface).
- [ ] 11.2 Update `polygram/__init__.py` `__all__` to include `LearnedAxisAssignment`, `LearnedAxisObjective`, and the three built-in objective functions.
- [ ] 11.3 CHANGELOG entry under `0.8.0`.
- [ ] 11.4 Run the full test suite — confirm no regressions in MPSRung1 / HEA / Rung3 / Rung4 / Rung5 paths.
- [ ] 11.5 Manual smoke: load the toy SAE fixture, import with `learn_axis_assignment=True`, confirm `report.learned_axis_assignment` is well-formed.
