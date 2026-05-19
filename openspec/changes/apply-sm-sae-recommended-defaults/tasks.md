## 1. SAEImportConfig defaults

- [x] 1.1 In `polygram/config.py`, change `SAEImportConfig.assign_amp_knobs: bool = False` to `True`. Rewrite the preceding comment block to cite the sm-sae measured recommendation rather than the legacy "preserves byte-identical behaviour" framing.
- [x] 1.2 Change `SAEImportConfig.assign_phase_knobs: bool = False` to `True`. Same comment rewrite.
- [x] 1.3 Update the class docstring's "Note: ``assign_gamma`` defaults to ``True``..." paragraph to call out the new `True` defaults for the two knob fields as well, citing the sm-sae source.

## 2. Test audit

- [x] 2.1 Find every test that previously relied on the `False` default for either knob field. The expected sites are tests that construct `SAEImportConfig()` (or call `from_sae_lens(...)` without the kwargs) and then assert that knobs were *not* populated.
- [x] 2.2 For each such test, pass `assign_amp_knobs=False` / `assign_phase_knobs=False` explicitly to preserve the assertion's intent. Tests that exercise the True path should keep their explicit `True` and continue to assert post-conditions on populated knobs.

## 3. Validation

- [x] 3.1 `openspec validate apply-sm-sae-recommended-defaults --strict`.
- [x] 3.2 Full `pytest` run; verify no regressions outside the pre-existing sklearn-missing failures in the compression/regrow suites.
