## Why

Polygram's tuning knobs are scattered across half a dozen `__init__` signatures (`Cancellation`, `BehaviouralValidator`, `EpochCompressor`, `Regrower`, `from_sae_lens`, `MPSRung1`), and several defaults are wrong for the use cases we actually run:

- `EpochCompressor` defaults (`coverage_target=0.95`, `n_visits_per_feature=3`, `max_iterations=5`) are tuned for one-shot offline runs — every iterative caller (e.g. `examples/forge_gpt2_real_sae.py:125`) silently overrides them with `0.5 / 1 / 1`. The defaults are 5–10× too expensive for the iterative case.
- `from_sae_lens(assign_gamma=False)` collapses every feature inside a cluster onto an identical γ — the README explicitly warns this is "almost always wrong" on real SAEs, yet it's the default.
- `Regrower.from_compression_report(layer=10, model_name="gpt2")` hardcodes a GPT-2-specific layer index. Any non-GPT-2 host model is silently wrong.

Downstream callers (sae-forge in particular) want to thread these knobs end-to-end through their own FSM context, but today they have to either accept brittle defaults or pass each kwarg individually with no shared schema.

## What Changes

- Introduce centralised, importable config dataclasses in `polygram.config`: `CompressionConfig`, `CancellationConfig`, `ValidationConfig`, `RegrowConfig`, `SAEImportConfig`. Each carries the existing knobs plus their validation logic.
- Re-target `EpochCompressor` defaults to the iterative-use values (`coverage_target=0.5`, `n_visits_per_feature=1`, `max_iterations=1`). Document the rationale on the dataclass.
- Provide named presets — `EpochCompressor.fast()` (current iterative defaults) and `EpochCompressor.thorough()` (the old offline defaults) — so the offline-run path is one method call rather than four kwargs.
- **BREAKING**: Flip `from_sae_lens(assign_gamma=...)` default from `False` to `True`. README guidance becomes the default; callers who actually want γ=0 set `assign_gamma=False` explicitly.
- **BREAKING**: Remove the `model_name="gpt2"` and `layer=10` defaults from `Regrower.from_compression_report`; both become required keyword arguments. Silent GPT-2 assumptions go away.
- Each constructor (`Compressor`, `EpochCompressor`, `Cancellation`, `BehaviouralValidator`, `Regrower`, `from_sae_lens`) accepts an optional `config: <Config>` kwarg in addition to the existing per-field kwargs. Per-field kwargs continue to work and override config values when both are supplied.
- Add a `polygram.config.from_dict(...) / to_dict(...)` round-trip so callers (sae-forge FSM ctx, YAML configs) can serialise tuning bundles.

## Capabilities

### New Capabilities

- `tuning-config`: Centralised config dataclasses (`CompressionConfig`, `CancellationConfig`, `ValidationConfig`, `RegrowConfig`, `SAEImportConfig`) with validation, dict round-trip, and named presets. Each public constructor accepts a `config=` kwarg whose fields are layered under per-field kwargs.

### Modified Capabilities

- `sae`: `from_sae_lens` `assign_gamma` default changes from `False` to `True`; new `config: SAEImportConfig | None` kwarg.
- `dictionary`: `Cancellation` and `BehaviouralValidator` accept `config=` kwargs; their per-field defaults are unchanged but now sourced from `CancellationConfig` / `ValidationConfig`.
- `experiment`: `Compressor` and `EpochCompressor` accept `config=` kwargs. `EpochCompressor` field defaults change to the iterative preset; the previous defaults are reachable via `EpochCompressor.thorough()`. `Regrower.from_compression_report` removes default values for `model_name` and `layer` (now required).

## Impact

- `polygram/config.py` — new module with the five config dataclasses, validation, presets, and dict serialization helpers.
- `polygram/__init__.py` — export the config dataclasses.
- `polygram/compression/epoch.py` — re-target defaults, add `.fast()` / `.thorough()` classmethods, accept `config=`.
- `polygram/compression/compressor.py` — accept `config: CompressionConfig | None`.
- `polygram/compression/regrow.py` — drop GPT-2 defaults, accept `config: RegrowConfig | None`.
- `polygram/cancellation.py` — accept `config: CancellationConfig | None`.
- `polygram/behavioural/validator.py` — accept `config: ValidationConfig | None`.
- `polygram/sae_import.py` — flip `assign_gamma` default, accept `config: SAEImportConfig | None`.
- Tests: new `tests/test_config.py` for round-trip and override-precedence; existing tests touching the renamed defaults updated.
- `examples/forge_gpt2_real_sae.py` (and any other in-tree caller passing the override quartet) simplified to `EpochCompressor.fast()`.
- Downstream: sae-forge's `compress_with_polygram` / `perform_regrowth` actions can now accept a single `CompressionConfig` / `RegrowConfig` from FSM context instead of marshalling per-field kwargs. (Tracked separately in sae-forge's `forge-polygram-tuning-passthrough` change.)
- Migration: callers relying on the old `EpochCompressor` defaults switch to `EpochCompressor.thorough()`; callers that omit `model_name`/`layer` to `from_compression_report` get a clear `TypeError` at call time. Both surface area changes show up at construction, not at runtime.
