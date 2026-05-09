# Changelog

## 0.2.0 (unreleased)

### Added

- **`polygram.geometry` module** — named `GeometricProfile` registry
  for SAE projection-space regimes. Two built-in profiles ship:
  `clustered` (the v0.1.0 default — small dense LM SAEs at GPT-2-small
  scale) and `uniform-sphere` (audio + large LM SAEs with
  `d_model ≥ ~1K`, `n_features ≥ ~16K`). Empirically validated on a
  five-SAE panel spanning Whisper, Qwen-Scope, and Llama-Scope; see
  [`docs/research/sae-geometry-regimes.md`](docs/research/sae-geometry-regimes.md).
- **`profile=` kwarg on `from_sae_lens`** — selects which
  `GeometricProfile` governs knob assignment + fidelity. Resolution
  order: per-field kwarg > `SAEImportConfig.profile` > registry
  default (`clustered`). Omitting the kwarg is byte-for-byte identical
  to passing `profile="clustered"` (locked in by a golden fixture).
- **`SelectionReport.profile` and `SelectionReport.geometric_fidelity`** —
  the active profile name and its headline scalar. The existing
  `tier_preservation` field is retained but populated only by the
  `clustered` profile (and any third-party profiles that opt to reuse
  `TierPreservationFidelity`).
- **`SAEImportConfig.profile`** — optional string field (default
  `None`); resolved against the registry at `from_sae_lens` call time.
- **`GeometricProfile`, `register_profile`, `get_profile`,
  `available_profiles`, `clustered`, `uniform_sphere`** — re-exported
  from the top-level `polygram` namespace.

### Fixed

- **bf16 slice path in `load_sae_safetensors(feature_ids=...)`** —
  used to crash with `TypeError: data type 'bfloat16' not understood`
  on bf16 checkpoints. Llama-Scope L0R surfaced this; modern LLM SAEs
  ship bf16 by default. The slice path now reads bf16 row bytes
  directly off disk for the common (non-transposed) case and falls
  through to the eager bf16 conversion for the `decoder.weight`
  PyTorch-orientation case. New `_safe_to_float32` helper centralises
  the conversion so future loaders don't re-discover the quirk.

## 0.1.0 (2026-05-07)

### Added

- **`polygram.config` module** — frozen dataclasses for every tuning
  surface: `CompressionConfig`, `EpochCompressionConfig`,
  `CancellationConfig`, `ValidationConfig`, `RegrowConfig`,
  `SAEImportConfig`. Each implements `to_dict()` / `from_dict(data)`
  with JSON-friendly tuple↔list coercion, nested-config recursion, and
  forward-compatible `UserWarning` on unknown keys. Re-exported from
  the top-level `polygram` namespace.
- **`config=` keyword on every public constructor** — `Compressor`,
  `EpochCompressor`, `Cancellation`, `BehaviouralValidator`,
  `Regrower.from_compression_report`, and `from_sae_lens` now accept
  an optional `config: <Config> | None = None`. Resolution rule is
  **per-field kwargs > config > dataclass defaults**.
- **`EpochCompressor.fast()` / `EpochCompressor.thorough()` named
  presets** — bundle the iterative-loop defaults (the new behaviour)
  and the pre-change exhaustive-offline-run defaults respectively.
  Both accept `**overrides` for any constructor kwarg.

### Changed (Breaking)

- **`EpochCompressor` defaults flipped to iterative-preset values.**
  `coverage_target`: 0.95 → 0.5; `n_visits_per_feature`: 3 → 1;
  `max_iterations`: 5 → 1. Callers depending on the prior values can
  switch in one line: `EpochCompressor.thorough(...)` (or pass the
  three kwargs explicitly).
- **`from_sae_lens(assign_gamma=...)` default flipped from `False` to
  `True`.** The README has long noted that γ=0 collapses every
  in-cluster feature onto the same encoded state and is "almost always
  wrong" on real SAEs. Callers that genuinely want γ=0 (toy fixtures,
  reproducing the legacy demo numbers) pass `assign_gamma=False`
  explicitly. The polygram `analyze` CLI still defaults its
  `--assign-gamma` flag off — the CLI surface contract is unchanged.
- **`Regrower.from_compression_report` no longer defaults `model_name`
  or `layer`.** The previous defaults (`model_name="gpt2"`,
  `layer=10`) silently bound the regrower to a GPT-2-shaped host
  model; an incorrect layer index on a non-GPT-2 host produces
  nonsense regrowth. Both arguments are now required keyword inputs;
  omitting either raises `TypeError` at the call site. The
  `Regrower(...)` direct constructor's defaults are unchanged for
  back-compat. Callers who want a typed bundle can pass
  `config=RegrowConfig(model_name=..., layer=...)`.

### Migration

- **Iterative-loop callers** (e.g. sae-forge's outer-loop FSM):
  replace any hardcoded `EpochCompressor(coverage_target=0.5,
  cosine_threshold=0.30, n_visits_per_feature=1, max_iterations=1,
  ...)` with `EpochCompressor.fast(...)`. The `coverage_target`,
  `n_visits_per_feature`, and `max_iterations` defaults already match.
- **Offline-run callers**: `EpochCompressor()` →
  `EpochCompressor.thorough()`. Same constructor surface, same return
  type; only the preset of tuning fields differs.
- **`from_sae_lens` callers** that relied on γ=0: pass
  `assign_gamma=False`. Tests that asserted `gamma_method == "zero"`
  on a defaulted call need to either flip the assertion to
  `"projection_pca"` or pin `assign_gamma=False` explicitly.
- **`Regrower.from_compression_report` callers**: pass `model_name`
  and `layer` explicitly (or via `config=RegrowConfig(...)`). The
  in-tree polygram CLI and example already do.

### Internal

- `tests/test_config.py` — 45 tests covering frozen-ness, range
  validation, default values, dict round-trip, unknown-key warning,
  tuple↔list coercion, top-level re-export, required-field
  enforcement on `RegrowConfig`.
- Constructor `__post_init__` methods updated to perform the
  precedence resolution before existing range checks.
- `polygram analyze` CLI now passes `assign_gamma=bool(args.assign_gamma)`
  explicitly so the CLI's documented "without --assign-gamma → γ=0"
  contract is preserved across the library default flip.
