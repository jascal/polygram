# Changelog

## Unreleased

### Added

- **`Rung4` encoding** — new 5-qubit encoding with a **product**
  amplitude branch on qubits 3 and 4: two independent single-qubit
  amps (vs Rung3's Bell-pattern entangled amp). Per-feature Hilbert
  dim **32** (vs Rung3's 16, MPSRung1's 8) — empirically verified
  by `examples/rung4_rank_verification.py` (rank-32 saturation at
  N ∈ {32, 40} across two seeds). Adds `polygram.encoding.Rung4`,
  `Rung4State`, `rung4_amp_overlap`, `rung4_amp_overlap_squared`,
  plus the shared `_single_qubit_overlap` helper (Rung3 now
  delegates to it; behaviour unchanged). `Feature` gains two new
  fields `theta_amp_b` and `psi_amp_b` (default 0.0; Rung3 ignores
  them); `Dictionary.gram()` dispatches Rung4 through the same
  elementwise-product factorisation as Rung3 with the product-amp
  formula. **Default-knob Rung4 reduces to MPSRung1-equivalent gram**
  on the same (α, β, γ, φ).
- **`Cancellation(encoding="rung4")`** — 6-knob canonical joint
  optimiser (`[a.phi, b.phi, b.theta_amp, b.psi_aux, b.theta_amp_b,
  b.psi_amp_b]`). 4D outer grid over feature B's product-amp knobs +
  inner 2-φ MPS-equivalent grid + scipy Nelder-Mead refine, mirroring
  the Rung3 pipeline shape. `min_amp_overlap` constraint applies to
  `rung4_amp_overlap_squared`. `structural_floor` reduces to the
  MPS-phase-only floor of (α, β, γ) — the baseline the optimiser
  tries to break.
- **Q-OrCA emit awareness for Rung4** — emitted `.q.orca.md` machine
  is the same MPS-substrate that Rung3 emits today (the amp branch
  lives in the analytic `Dictionary.gram()` path, not in q-orca);
  the markdown header now explicitly labels the encoding
  (`rung-4 MPS-substrate`) and notes where the amp factor lives.
  Default-knob Rung4 round-trips through q-orca's
  `compute_concept_gram_mps` to the same gram as the analytic path
  (verified at 1e-10).
- **`examples/rung4_rank_verification.py`** — reproducible probe
  confirming Rung4's 32-feature ceiling; default sweep
  N ∈ {4, 8, 16, 24, 32, 40} across two seeds.
  `docs/research/data/rung4_rank_verification.json` is the
  committed artifact.

### Deferred (follow-up PR)

- **Rung4 viability spike** (§7 of `add-rung4-encoding-mvp/tasks.md`)
  is deferred to a research-track follow-up PR. The methodology
  mirrors `examples/rung3_viability_spike.py`'s A/B/C/D bucket
  analysis on a real GPT-2-small SAE; the result decides whether
  Rung4 becomes the default encoding or stays opt-in like Rung3.

### Changed (Internal — no observable behaviour drift)

- **`EpochCompressor.run` consumes `ClusteredDictionary` internally** —
  `_validate_panels` and `_synthesize_validation_report` now take a
  `ClusteredDictionary` as their primary panel data structure
  (renamed from `panels: list[Panel]`). `EpochCompressor.run` builds
  the clustered view per iteration via
  `ClusteredDictionary.from_compression_panels` and threads it
  through. The pre-refactor `_select_panels` algorithm is
  **untouched**; the per-iteration `EpochResult` is bit-identical
  to the pre-refactor output (modulo `wall_seconds` + tempfile
  paths) on the deterministic fixture, gated by
  `tests/compression/test_epoch_clustered_consume.py` against the
  frozen reference at
  `tests/compression/data/epoch_result_reference.json`. The
  encoding stays implicit `MPSRung1()` at the call site; future
  work (issue #48) would plumb a configurable encoding through.

### Added

- **`encoding.max_features` per-encoding cap** — each encoding class
  declares a `max_features` attribute matching its reachable
  Hilbert-space dimension: `MPSRung1.max_features = 8`,
  `Rung3.max_features = 16` (corrected from the previous universal
  8 cap; see `docs/research/rung3-rank-bound.md` for the empirical
  basis), `HEA_Rung2.max_features = 2 ** n_qubits` (scales with the
  existing `n_qubits` knob). `from_sae_lens` and
  `BehaviouralValidator` now query the encoding's cap rather than
  the `MAX_FEATURES_PER_DICTIONARY` module constant. The constant
  is retained as a back-compat alias at the `MPSRung1` value (8).
  The error message names the encoding and suggests larger-cap
  alternatives. **`Rung3` users can now load 9–16 features** without
  hitting the cap.

- **`polygram.clustered_dictionary` module** (renamed from
  `polygram.clustered` to avoid a namespace collision with the
  geometry-registered `clustered` factory) — `ClusteredDictionary`
  primitive for SAE-scale analyses. Holds a list of `Dictionary`
  blocks (each
  ≤ `encoding.max_features`) plus a sparse cross-block adjacency.
  Block formation via cosine clustering (default), user-declared
  hierarchy, or reserved co-firing API.
  `ClusteredDictionary.gram()` returns a `BlockSparseGram` (per-block
  dense Gram + sparse cross-block edges).
  `cross_block_redundant_pairs(threshold)` surfaces high-cosine
  pairs spanning cluster boundaries with a
  `CrossBlockRedundancyReport`.
  `emit_qorca(output_dir)` writes one verifiable `.q.orca.md` per
  block plus a `manifest.json` describing the topology.
  Killer-experiment fixture confirms recall = 1.0 vs flat baseline on
  a real GPT-2-small SAE at N=2k and N=8k; speedup story reframed
  per `docs/research/clustered-dictionary-recall-vs-flat.md`.
- **`from_sae_lens(..., clustered=True, block_formation=...)`** —
  opt-in clustered loader path returning a `ClusteredDictionary`
  instead of a single `Dictionary`. Default `clustered=False`
  preserves byte-identical behaviour.
- **`SelectionReport.{n_blocks, mean_block_size, n_cross_block_edges}`** —
  per-clustering stats populated when the clustered loader is used.
- **`compute_cosine_pair_graph`** in `polygram.clustered_dictionary` — public
  helper extracted from `polygram.compression.epoch._compute_cosine_graph`.
  The epoch-side function continues to call it as a thin wrapper
  (back-compat preserved; all existing compression tests pass
  unchanged).

## 0.3.0 (2026-05-10)

### Added

- **`RegrowConfig.top_k`** — optional cap on the per-call regrow
  count. Default `None` preserves byte-equivalence with pre-change
  behavior (every zeroed slot regrown). When set to a non-negative
  integer, the regrower regrows only the first `top_k` zeroed slots
  in plan order; remaining slots stay zero. `Regrower.from_compression_report`
  also accepts a `top_k=` kwarg with the standard kwarg-wins-over-config
  precedence. Selection is plan-order; richer strategies tracked
  as `regrow-selection-strategies`. Unblocks sae-forge's
  `adaptive-regrow` controller.

## 0.2.0 (2026-05-08)

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
