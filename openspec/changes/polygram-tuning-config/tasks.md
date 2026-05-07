## 1. Config module

- [x] 1.1 Create `polygram/config.py` with module docstring documenting the
      `config=` + per-field-kwarg precedence rule.
- [x] 1.2 Implement `ValidationConfig` (frozen dataclass + `__post_init__`
      range checks for `polygram_overlap_threshold ∈ [0,1]`,
      `jaccard_threshold ∈ [0,1]`, `min_firing_rate ∈ [0,1]`,
      `min_both_fire >= 0`).
- [x] 1.3 Implement `CancellationConfig` (range checks for
      `tolerance ∈ [0,1]`, `grid_outer` both ints `>= 1`,
      `optimize["method"] ∈ {"grid","scipy"}`).
- [x] 1.4 Implement `CompressionConfig` (validates
      `strategy ∈ {"merge","zero"}`, `rep_selection ∈
      {"n_fires","scale_aware"}`, `merge_mode ∈
      {"freq_weighted","simple_mean"}`).
- [x] 1.5 Implement `EpochCompressionConfig` (range checks
      mirroring `EpochCompressor.__post_init__` at
      `polygram/compression/epoch.py:114-142`; embeds optional
      `validation: ValidationConfig`).
- [x] 1.6 Implement `RegrowConfig` with `model_name` and `layer`
      as required keyword-only fields (no defaults).
- [x] 1.7 Implement `SAEImportConfig` with `assign_gamma=True`
      default (the new behaviour).
- [x] 1.8 Add `to_dict()` / `from_dict()` to a shared mixin or
      base; implement tuple↔list coercion and `UserWarning` on
      unknown keys; recurse into composed configs.
- [x] 1.9 Re-export the six config dataclasses from
      `polygram/__init__.py` under `polygram.config` and at the
      top level (`from polygram import CompressionConfig, ...`).

## 2. Tests for config module

- [x] 2.1 Add `tests/test_config.py` covering frozen-ness,
      `__post_init__` range failures, default values, and the
      preset-vs-default equality checks.
- [x] 2.2 Round-trip test: every config survives
      `to_dict → from_dict` equal to the original (including
      nested `validation`).
- [x] 2.3 Unknown-key test: `from_dict` warns and ignores; legacy
      dicts with future keys keep loading.
- [x] 2.4 Tuple coercion test: list inputs to `grid_outer`,
      `gamma_range` deserialise as tuples.

## 3. Wire config= into Cancellation and BehaviouralValidator

- [x] 3.1 Add `config: CancellationConfig | None = None` kwarg to
      `Cancellation` (`polygram/cancellation.py`); apply
      precedence rule in `__post_init__` before existing
      validation runs.
- [x] 3.2 Add `config: ValidationConfig | None = None` kwarg to
      `BehaviouralValidator`
      (`polygram/behavioural/validator.py`).
- [x] 3.3 Add tests for kwarg-overrides-config and config-only
      construction paths in `tests/test_cancellation*.py` and
      `tests/test_behavioural_validator*.py`.

## 4. Wire config= into Compressor and EpochCompressor

- [x] 4.1 Add `config: CompressionConfig | None = None` to
      `Compressor` (`polygram/compression/compressor.py`).
- [x] 4.2 Add `config: EpochCompressionConfig | None = None` to
      `EpochCompressor` (`polygram/compression/epoch.py`); when
      `config.validation` is supplied and no explicit
      `polygram_overlap_threshold` / `jaccard_threshold` /
      `min_both_fire` kwarg is given, source those from
      `config.validation` (the embedded `ValidationConfig`).
- [x] 4.3 Add tests in `tests/test_compression*.py`.

## 5. Add EpochCompressor presets

- [x] 5.1 Implement `EpochCompressor.fast(**overrides)` and
      `EpochCompressor.thorough(**overrides)` classmethods.
- [x] 5.2 Test that `.fast()` matches the dataclass defaults and
      `.thorough()` matches the legacy values
      (`coverage_target=0.95`, `n_visits_per_feature=3`,
      `max_iterations=5`).
- [x] 5.3 Test override pass-through: `fast(coverage_target=0.6)`.

## 6. Flip EpochCompressor field defaults

- [x] 6.1 Change defaults at `polygram/compression/epoch.py:78-87`
      to `coverage_target=0.5`, `n_visits_per_feature=1`,
      `max_iterations=1` (other fields unchanged).
- [x] 6.2 Audit existing tests under `tests/compression/`
      that constructed `EpochCompressor()` with no args; pin the
      old defaults explicitly via `EpochCompressor.thorough()`
      where the test depends on them. **Audit result:** every
      in-tree caller (`tests/compression/test_epoch_convergence.py`,
      `examples/compress_epoch_validated.py`) already passes
      `max_iterations` / `coverage_target` / `n_visits_per_feature`
      explicitly, so none rely on the changing defaults.
- [x] 6.3 Update `examples/forge_gpt2_real_sae.py:125-138` to
      `EpochCompressor.fast()`. **N/A here:** the
      `forge_gpt2_real_sae.py` file lives in the sae-forge repo,
      not polygram. The migration is tracked in the companion
      change `forge-polygram-tuning-passthrough` (sae-forge
      side).
- [x] 6.4 Add CHANGELOG entry under "Breaking" calling out the
      default flip and the `.thorough()` migration path.
      **Bundled with §7.4 / §8.5** in a single CHANGELOG.md
      created at the end of this change.

## 7. Wire config= into from_sae_lens and flip assign_gamma

- [x] 7.1 Add `config: SAEImportConfig | None = None` kwarg to
      `from_sae_lens` (`polygram/sae_import.py:466`).
- [x] 7.2 Flip `assign_gamma` default at
      `polygram/sae_import.py:475` from `False` to `True`.
- [x] 7.3 Audit `tests/test_sae_import*.py` and
      `tests/test_from_sae_lens*.py`; update assertions that
      expected `gamma == 0` from a defaulted call, OR pin those
      tests to `assign_gamma=False` if that was the intent.
- [x] 7.4 Update CHANGELOG with the assign_gamma default flip.

## 8. Wire config= into Regrower and remove GPT-2 defaults

- [x] 8.1 Add `config: RegrowConfig | None = None` to
      `Regrower.from_compression_report`
      (`polygram/compression/regrow.py:190`).
- [x] 8.2 Remove the `model_name: str = "gpt2"` and `layer: int =
      10` defaults at `polygram/compression/regrow.py:200-201`;
      keep keyword-only.
- [x] 8.3 Update every in-tree caller (grep
      `from_compression_report(`) to pass `model_name` and
      `layer` explicitly.
- [x] 8.4 Add tests for the missing-kwarg `TypeError` and for
      `config=`-supplied values.
- [x] 8.5 Update CHANGELOG with the required-kwarg change.

## 9. Documentation

- [x] 9.1 Add a "Configuration" section to README pointing at
      `polygram.config`, the precedence rule, and the
      `EpochCompressor.fast() / .thorough()` presets.
- [x] 9.2 Update docstrings on each affected constructor to
      reference the corresponding config dataclass and the
      precedence rule.
- [x] 9.3 Note in the README that downstream callers (e.g.
      sae-forge) can store `cfg.to_dict()` on their FSM context
      and rebuild via `from_dict`.

## 10. Verification

- [x] 10.1 Run `pytest` end-to-end; all pre-existing tests pass
      with the new defaults (or with explicit pin to old
      defaults where intentional).
- [x] 10.2 Run `examples/forge_gpt2_real_sae.py` to completion
      and verify it produces equivalent output to a baseline run.
- [x] 10.3 Run `openspec validate polygram-tuning-config
      --strict` and confirm no errors.
