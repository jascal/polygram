## 1. Helper module

- [ ] 1.1 Create `polygram/geometry/phase_assignment.py` with `assign_phase_knobs_pca(projections, encoding) -> dict[str, list[float] | None]`. Returns `{"alphas": ..., "phis": ...}`.
- [ ] 1.2 PCA via SVD on centered projections. Use PCA axis 2 (index 1) for `alphas`, PCA axis 3 (index 2) for `phis`. Rescale linearly into `[0, 2π]` for both.
- [ ] 1.3 INFO-once log message for `HEA_Rung2` (different knob structure → returns `{None, None}` cleanly).
- [ ] 1.4 Degenerate-PCA fallback: when fewer than the requested non-zero PCA axes exist, return `None` for the affected knob arrays (loader uses encoding default).
- [ ] 1.5 Pure-numpy, deterministic, no RNG.

## 2. `KnobAssignmentResult` extension

- [ ] 2.1 Add `alphas: list[float] | None = None`, `phis: list[float] | None = None` to `KnobAssignmentResult` in `polygram/geometry/protocols.py`.
- [ ] 2.2 `KnobAssignment.assign` signature extended with `assign_phase_knobs: bool = False` kwarg.

## 3. Shift `assign_amp_knobs_pca` axis allocation

- [ ] 3.1 In `polygram/geometry/amp_assignment.py`, update the `knob_slots` table so amp knobs read from PCA axes 4-7 (indices 3-6) instead of 2-5 (indices 1-4). Phase knobs now own axes 2-3.
- [ ] 3.2 Update the docstring's "PC2 / PC3 / PC4 / PC5" comment to "PC4 / PC5 / PC6 / PC7".
- [ ] 3.3 NOTE: this is a backward-incompat change for any consumer relying on PR-#63-era exact gram-condition numbers with `assign_amp_knobs=True`. The PR-#63 v2.1 results note will need a one-line regenerate-and-update.

## 4. Strategy updates

- [ ] 4.1 `polygram/geometry/clustered.py::ClusteredKnobAssignment.assign` — accept `assign_phase_knobs` kwarg, call the helper, populate the result's `alphas` and `phis` fields.
- [ ] 4.2 `polygram/geometry/uniform_sphere.py::UniformSphereKnobAssignment.assign` — same.

## 5. `from_sae_lens` plumbing

- [ ] 5.1 `polygram/sae_import.py::from_sae_lens` gains `assign_phase_knobs: bool | None = None` kwarg.
- [ ] 5.2 Precedence: explicit kwarg > `config.assign_phase_knobs` > default False (mirrors `assign_gamma` and `assign_amp_knobs`).
- [ ] 5.3 Thread `assign_phase_knobs` and `encoding` into the `KnobAssignment.assign` call.
- [ ] 5.4 After the assignment, when `result.alphas` / `result.phis` are non-`None`, override `Feature.alpha` / `Feature.phi` per-feature; otherwise use encoding defaults.
- [ ] 5.5 The bypass-strategy path (cluster_assignments / from_labels) calls `assign_phase_knobs_pca` directly when the flag is set, mirroring the bypass-path pattern in PR #63.

## 6. `SAEImportConfig` field

- [ ] 6.1 Add `assign_phase_knobs: bool = False` to `SAEImportConfig` in `polygram/config.py`. Document mirror of `assign_amp_knobs`.

## 7. Compression plumbing (mirrors PR #64)

- [ ] 7.1 `EpochCompressor` gains `assign_phase_knobs: bool = False` field. Thread through to `_validate_panel_inline` and the final-rebuild call.
- [ ] 7.2 `Compressor` gains `assign_phase_knobs: bool = False` field. Thread into `Compressor.apply`'s `from_sae_lens` rebuild.
- [ ] 7.3 `EpochCompressor` passes `assign_phase_knobs=self.assign_phase_knobs` into `Compressor(...)`.
- [ ] 7.4 The captured-kwargs monkeypatch test pattern (`tests/compression/test_epoch_encoding_configurable.py::test_assign_amp_knobs_plumbed_into_per_panel_from_sae_lens`) gets a sibling for `assign_phase_knobs` — catches missing call sites the same way.

## 8. Example CLI flags

- [ ] 8.1 `examples/rung_gram_condition.py` gains `--assign-phase-knobs` flag.
- [ ] 8.2 `examples/rung_compression_coverage.py` gains `--assign-phase-knobs` flag.
- [ ] 8.3 Both record the flag's value in the output JSON's top-level `assign_phase_knobs` field.

## 9. Tests — falsifying invariant

- [ ] 9.1 Create `tests/test_phase_knob_assignment.py`.
- [ ] 9.2 `test_phase_knobs_activate_mpsrung1_capacity`: the cornerstone. With `encoding=MPSRung1()`, gram with `assign_phase_knobs=True` differs from `assign_phase_knobs=False` by Frobenius > 1.0; mean off-diagonal drops to < 0.5× the default value (calibrated against the toy-fixture sanity check: 0.76 → 0.28).
- [ ] 9.3 `test_phase_knobs_activate_rung3`: same for Rung3.
- [ ] 9.4 `test_phase_knobs_activate_rung4`: same for Rung4. Plus a both-flags-on test: combined gram differs from amp-only-on gram.
- [ ] 9.5 `test_phase_knobs_default_false_byte_identical`: pin the default-path invariance.
- [ ] 9.6 `test_phase_knobs_deterministic`: same inputs → same outputs.
- [ ] 9.7 `test_phase_knobs_degenerate_pca_falls_back`: 2-feature SAE doesn't crash; α/φ fall back to encoding defaults.
- [ ] 9.8 `test_sae_import_config_propagates_assign_phase_knobs`: `SAEImportConfig(assign_phase_knobs=True)` matches passing the kwarg directly.

## 10. Tests — compression plumbing

- [ ] 10.1 `tests/compression/test_epoch_encoding_configurable.py::test_assign_phase_knobs_plumbed_into_per_panel_from_sae_lens` — captured-kwargs monkeypatch confirms `assign_phase_knobs=True` reaches every `from_sae_lens` call during a compression run.
- [ ] 10.2 `test_assign_phase_knobs_false_default_keeps_from_sae_lens_default` — pins the byte-identical default.

## 11. Tests — backward-incompat axis shift

- [ ] 11.1 Update the existing PR-#63 falsifying tests in `tests/test_amp_knob_assignment.py` if their assertions depend on specific values that change with the axis shift. The qualitative assertions (Frobenius > 1e-3, off-diagonal differs) should still pass — re-verify and adjust thresholds if needed.

## 12. Research artifact regeneration

- [ ] 12.1 Re-run `examples/rung_gram_condition.py --encoding rung3 --assign-amp-knobs ...` on the real SAE and update `docs/research/data/rung_gram_condition_rung3_amp_on.json`.
- [ ] 12.2 Same for `..._rung4_amp_on.json`.
- [ ] 12.3 `docs/research/rung4-viability-spike-v2.md` — add a one-paragraph note in the "Resolved" section: the PR-#63 axis shift means exact numbers in the v2.1 table changed. Qualitative findings preserved; new numbers reflect the cleaner phase-then-amp allocation.

## 13. CHANGELOG + closing

- [ ] 13.1 CHANGELOG entry under unreleased: "**Encoding-aware phase-knob assignment in `from_sae_lens`** — new `assign_phase_knobs: bool = False` kwarg. When True, the loader populates MPS-substrate `α` and `φ` knobs from decoder geometry (PCA axes 2 and 3), un-dormanting the second half of MPSRung1's state space. Resolves the gram-saturation finding from the 2026-05-15 GPT-2 bug report (MPSRung1 off-diagonal `|G|²` mean drops 0.76 → 0.28 on the toy fixture with the flag on). `assign_amp_knobs`'s PCA-axis allocation shifts from 2-5 to 4-7 to make room; qualitative findings from the PR-#63 v2.1 results note survive, exact numbers change."
- [ ] 13.2 `openspec validate add-phase-knob-assignment --strict`.
- [ ] 13.3 `pytest` full suite green; `ruff check` clean.

## 14. What this change explicitly defers

- [ ] 14.1 Other 3 issues from the bug report (`plan_pareto` ignoring `gate_pass`; EpochCompressor cross-panel leak into ForgePipeline; uniform-sphere docstring) — separate scopes, separate PRs.
- [ ] 14.2 Auto-default-flip for higher-rung encodings — opt-in stays.
- [ ] 14.3 Sinusoidal rescale variant — open question carried from PR #63's design.md.
- [ ] 14.4 HEA_Rung2 phase-knob assignment — different knob structure; out of scope.
