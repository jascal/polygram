## 1. Helper module

- [ ] 1.1 Create `polygram/geometry/phase_assignment.py` with `assign_phase_knobs_pca(projections, encoding) -> dict[str, list[float] | None]`. Returns `{"alphas": ..., "phis": ...}`.
- [ ] 1.2 PCA via SVD on centered projections. Use PC2 (`vt[1]`) for `alphas`, PC3 (`vt[2]`) for `phis`. Rescale linearly into `[0, 2π]` for both.
- [ ] 1.3 INFO-once log message for `HEA_Rung2` (different knob structure → returns `{None, None}` cleanly). Message MUST explicitly name `HEA_Rung2` and state "its per-feature θ tensor has a different shape (`rotations × depth × n_qubits`) and is not compatible with the phase-knob assignment pattern; use `theta` directly via `HEA_Rung2State` instead". Mirrors PR #63's clarity for the same encoding.
- [ ] 1.4 Degenerate-PCA fallback: when fewer than the requested non-zero PCA axes exist, return `None` for the affected knob arrays (loader uses encoding default).
- [ ] 1.5 Pure-numpy, deterministic, no RNG.

## 2. `KnobAssignmentResult` extension

- [ ] 2.1 Add `alphas: list[float] | None = None`, `phis: list[float] | None = None` to `KnobAssignmentResult` in `polygram/geometry/protocols.py`.
- [ ] 2.2 `KnobAssignment.assign` signature extended with `assign_phase_knobs: bool = False` kwarg.

## 3. Shift `assign_amp_knobs_pca` axis allocation

- [ ] 3.1 In `polygram/geometry/amp_assignment.py`, update the `knob_slots` table so amp knobs read from PC4-PC7 (code: `vt[3]` through `vt[6]`) instead of PC2-PC5 (code: `vt[1]` through `vt[4]`). Phase knobs now own PC2-PC3.
- [ ] 3.2 Update the docstring's "PC2 / PC3 / PC4 / PC5" comment to "PC4 / PC5 / PC6 / PC7" and update the inline `vt[N]` references accordingly.
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
- [ ] 9.4a `test_both_flags_on_rung4_populates_all_six_knob_channels`: explicit combinatorics check — `from_sae_lens(records, encoding=Rung4(), assign_phase_knobs=True, assign_amp_knobs=True)` produces features where all six MPS-substrate + amp-branch knob channels (`alpha`, `phi`, `theta_amp`, `psi_aux`, `theta_amp_b`, `psi_amp_b`) are non-default (i.e., distinct from the encoding's `0` / `π/4` / `0` defaults). Mirrors the "FULL on Rung4" row of design.md's two-flag interaction matrix.
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
- [ ] 12.4 **Re-run Axis 1 C3 (Rung4 amp-on)** on the same fixture used for v2.2. The amp-axis shift changes the per-feature amp-knob values, so exact `features_zeroed_total` / `cumulative_cross_entropy_delta` numbers will differ. Confirm PASS verdict holds qualitatively (Rung4 amp-on > MPS baseline by ≥10% features, ≤−20% CE). Update the v2.2 "Axis 1 result" table or add a v2.3 supplement. **This is more load-bearing than 12.1/12.2** — v2.2 is the empirical evidence supporting the future default-flip path; v2.1's gram-condition numbers were already known to be loader-dormant artifacts.

## 13. CHANGELOG + closing

- [ ] 13.1 CHANGELOG entry under unreleased: "**Encoding-aware phase-knob assignment in `from_sae_lens`** — new `assign_phase_knobs: bool = False` kwarg. When True, the loader populates MPS-substrate `α` and `φ` knobs from decoder geometry (PC2 and PC3), un-dormanting the second half of MPSRung1's state space. Resolves the gram-saturation finding from the 2026-05-15 GPT-2 bug report (MPSRung1 off-diagonal `|G|²` mean drops 0.76 → 0.28 on the toy fixture with the flag on). `assign_amp_knobs`'s PCA-component allocation shifts from PC2-PC5 to PC4-PC7 to make room; qualitative findings from the PR-#63 v2.1 results note survive, exact numbers change."
- [ ] 13.1a `README.md` "SAE import" section — add a one-paragraph note under the existing `assign_gamma` discussion: "After `add-phase-knob-assignment` (release 0.5.0+), `from_sae_lens` also accepts `assign_phase_knobs` and `assign_amp_knobs` to populate α/φ and amp-branch knobs respectively from decoder PCA. See [`docs/research/rung4-viability-spike-v2.md`](docs/research/rung4-viability-spike-v2.md) for the empirical motivation."
- [ ] 13.1b Bump `polygram/__init__.py::__version__` to `0.6.0` (minor) — new opt-in capability, backward-compat default. Reviewer flagged this as part of the merge gating; the PR-#63-era amp-axis shift is the load-bearing reason the version must bump (callers depending on exact gram-condition numbers see different output even with the same flag value). Note: `0.5.0` was claimed by `add-kl-attribution-rep-selection` between this openspec's draft and impl, so the bump target shifts from 0.5.0 → 0.6.0.
- [ ] 13.2 `openspec validate add-phase-knob-assignment --strict`.
- [ ] 13.3 `pytest` full suite green; `ruff check` clean.

## 14. What this change explicitly defers

- [ ] 14.1 Other 3 issues from the bug report (`plan_pareto` ignoring `gate_pass`; EpochCompressor cross-panel leak into ForgePipeline; uniform-sphere docstring) — separate scopes, separate PRs.
- [ ] 14.2 Auto-default-flip for higher-rung encodings — opt-in stays.
- [ ] 14.3 Sinusoidal rescale variant — open question carried from PR #63's design.md.
- [ ] 14.4 HEA_Rung2 phase-knob assignment — different knob structure; out of scope.
