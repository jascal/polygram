## 1. `KnobAssignmentResult` extension

- [x] 1.1 Add four optional fields to `polygram.geometry.protocols.KnobAssignmentResult`: `theta_amps: list[float] | None = None`, `psi_auxes: list[float] | None = None`, `theta_amp_bs: list[float] | None = None`, `psi_amp_bs: list[float] | None = None`.
- [x] 1.2 The defaults (`None`) preserve byte-identity with the pre-change result construction — every existing strategy that doesn't pass the new fields produces an identical result.

## 2. `KnobAssignment` protocol extension

- [x] 2.1 Extend `polygram.geometry.protocols.KnobAssignment.assign(...)` signature with `assign_amp_knobs: bool = False` and `encoding: object = None` kwargs.
- [x] 2.2 Document in the protocol docstring that strategies SHOULD honour `assign_amp_knobs=True` by populating the result's amp-knob fields. Strategies that don't implement amp assignment leave the fields as `None`.

## 3. PCA-axis amp-knob assignment helper

- [x] 3.1 Add `_assign_amp_knobs_pca(projections, encoding) -> dict[str, list[float] | None]` to `polygram/geometry/` (location TBD — likely a new `polygram/geometry/amp_assignment.py` module).
- [x] 3.2 The function:
  - Returns `{"theta_amps": None, "psi_auxes": None, "theta_amp_bs": None, "psi_amp_bs": None}` for `MPSRung1` (no amp branch) or `HEA_Rung2` (different knob structure). Emits a `logging.debug` message naming the encoding so callers can verify.
  - For `Rung3`: computes top-3 PCA axes of `projections`. Returns `theta_amps` from axis-2 coords rescaled to `[0, π/2]`, `psi_auxes` from axis-3 coords rescaled to `[0, 2π]`. `theta_amp_bs` and `psi_amp_bs` remain `None`.
  - For `Rung4`: computes top-5 PCA axes. Returns all four amp arrays from axes 2-5, with the appropriate per-knob rescaling.
  - For features with degenerate PCA (rank deficient), falls back to encoding defaults for the affected axes.
- [x] 3.3 The helper is pure-numpy, deterministic, no RNG.

## 4. `ClusteredKnobAssignment.assign` extension

- [x] 4.1 Update `polygram/geometry/clustered.py::ClusteredKnobAssignment.assign` to accept `assign_amp_knobs` and `encoding` kwargs.
- [x] 4.2 When `assign_amp_knobs=False` (default), behaviour is byte-identical to today.
- [x] 4.3 When `assign_amp_knobs=True`, call `_assign_amp_knobs_pca(projections, encoding)` and populate the result's amp-knob fields.

## 5. `UniformSphereKnobAssignment.assign` extension

- [x] 5.1 Mirror task 4 for `UniformSphereKnobAssignment.assign` in `polygram/geometry/uniform_sphere.py`.
- [x] 5.2 Use the same `_assign_amp_knobs_pca` helper.

## 6. `from_sae_lens` plumbing

- [x] 6.1 Add `assign_amp_knobs: bool = False` kwarg to `polygram.sae_import.from_sae_lens`.
- [x] 6.2 Thread `assign_amp_knobs` and the resolved `encoding` into the `KnobAssignment.assign` call.
- [x] 6.3 After the assignment, when the result's amp-knob fields are non-`None`, override the `Feature` constructor's per-knob defaults with the assigned values. When `None`, leave the encoding's defaults (current behaviour).
- [x] 6.4 `SAEImportConfig` gains an `assign_amp_knobs: bool = False` field with the same semantics (matches the existing `assign_gamma` pattern).

## 7. Tests — byte-identity at default

- [x] 7.1 The existing `tests/test_sae_import.py` suite passes unchanged (every existing test uses `assign_amp_knobs=False` by omission).
- [x] 7.2 Add `test_assign_amp_knobs_false_is_default_byte_identical`: construct a Rung4 dictionary with and without explicit `assign_amp_knobs=False`. The two grams are bit-equal.

## 8. Tests — falsifiable invariant

- [x] 8.1 Add `tests/test_amp_knob_assignment.py` (new file).
- [x] 8.2 `test_assign_amp_knobs_true_changes_rung4_gram`: same toy SAE fixture, encoding=`Rung4()`, compare `assign_amp_knobs=False` vs `True`. Assert Frobenius distance between `|gram|²` matrices > 1e-3 (well above FP noise). Off-diagonal-only Frobenius distance also > 1e-3.
- [x] 8.3 `test_assign_amp_knobs_true_changes_rung3_gram`: same as 8.2 but `Rung3()`.
- [x] 8.4 `test_assign_amp_knobs_true_no_op_for_mpsrung1`: with `MPSRung1()`, the flag is a no-op — `|gram|²` is unchanged whether the flag is `True` or `False`.
- [x] 8.5 `test_amp_knob_assignment_values_are_deterministic`: same input → same output across multiple calls.
- [x] 8.6 `test_amp_knob_assignment_pca_uses_higher_axes`: instrument the helper, confirm Rung3 path uses PCA axes 2-3 and Rung4 uses axes 2-5. (Implementation detail test; can use a mock or just sanity check by running a 2-feature SAE where higher axes don't exist and seeing the helper fall back to defaults.)
- [x] 8.7 `test_amp_knob_assignment_handles_degenerate_pca`: feed a 2-feature SAE (only 1 non-zero PCA axis available for amp assignment); assert no crash, partial assignment uses encoding defaults for missing axes.

## 9. Tests — re-run the gram-condition spike with amp assignment on

- [x] 9.1 Run `examples/rung_gram_condition.py --encoding rung4 --sae <real SAE>` with `--assign-amp-knobs` flag (new). Verify the resulting metrics DIFFER from the v2 results note's Rung4 K=32 row.
- [x] 9.2 Add the `--assign-amp-knobs` flag to `examples/rung_gram_condition.py`.
- [x] 9.3 Optionally re-publish the gram-condition table in a v2.1 results note showing both paths side by side, so the un-dormanting effect is visible in the headline numbers.

## 10. CHANGELOG + closing

- [x] 10.1 Add entry under unreleased: `**Encoding-aware knob assignment in from_sae_lens** — new \`assign_amp_knobs: bool = False\` kwarg. When True, the loader populates higher-rung amp-branch knobs from decoder geometry (PCA-axis extension), un-dormanting Rung3 and Rung4's larger state-space. Default False preserves byte-identical behaviour. Closes the "higher rungs dormant in loader" finding from \`docs/research/rung4-viability-spike-v2.md\`.`
- [x] 10.2 Run `openspec validate encoding-aware-knob-assignment --strict`.
- [x] 10.3 Run full `pytest` suite + `ruff check`.
- [x] 10.4 Update `docs/research/rung4-viability-spike-v2.md` with a "Resolved" section pointing at this change.

## 11. What this change explicitly defers

- [x] 11.1 HEA_Rung2 amp assignment. Different knob structure (`theta_shape = (|rotations|, depth, n_qubits)`). Separate change if a consumer needs it.
- [x] 11.2 Optimization of amp knobs (search for "best" values). Single-forward-pass PCA-axis extension only.
- [x] 11.3 Auto-on default for higher-rung encodings. Stays opt-in via the explicit flag; future change can flip the default once Axis 1 / Axis 4 measurements support it.
- [x] 11.4 Alternative amp-assignment strategies (random, cluster-ordinal, decoder-cosine-pattern). API leaves room for them; v1 hardcodes PCA-axis.
- [x] 11.5 Encoding-aware plumbing through `EpochCompressor` (it currently always uses `from_sae_lens` defaults). Following PR could pass `assign_amp_knobs=True` automatically when `encoding != MPSRung1`. Out of scope here.
