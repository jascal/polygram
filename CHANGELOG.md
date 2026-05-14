# Changelog

## Unreleased

### Added

- **Axis 1 (compression coverage) measurement landed (v2.2).** Ran
  the 4-cell battery (MPS baseline, Rung4 amp-off control, Rung4
  amp-on load-bearing, Rung3 amp-on generality) on the 2019 MBP
  against the real GPT-2-small SAE, plus 10-iteration extensions
  for Rung4 amp-on and MPS as control. Rung4 amp-on zeros 2.65×
  more features than MPS at 28% *lower* cumulative CE budget at
  the default 3-iter operating point — **PASS** for Axis 1.
  Rung3 amp-on confirms the lift generalises across rungs. The
  MPS 10-iter control disambiguates a Rung4-amp-on iter-9 CE
  spike as encoding-specific cluster-exhaustion behaviour rather
  than universal late-stage compression. Features-per-CE-budget
  ratio: Rung4 amp-on stays 1.71× MPS even at iter 10. Results +
  per-iter trajectories captured in
  `docs/research/rung4-viability-spike-v2.md` under "Axis 1 result
  (v2.2)"; raw JSON + console captures under `docs/research/data/`.
  Closes the "Axis 1 / 4 pending a torch-enabled host" TODO for
  Axis 1; Axis 4 remains separate work.
- **`EpochCompressor(assign_amp_knobs=True)` + `Compressor(assign_amp_knobs=True)`** —
  threads the encoding-aware-knob-assignment flag through the
  compression pipeline. Both the per-panel `from_sae_lens` call in
  `_validate_panel_inline`, the final dictionary rebuild in
  `EpochCompressor.run`, and the post-compression `from_sae_lens`
  rebuild in `Compressor.apply` now honour the flag. Without this,
  setting `encoding=Rung4()` on the compressor would still result
  in MPS-equivalent per-block dictionaries during validation —
  defeating the un-dormanting work from PR #63. Default `False`
  preserves byte-identical behaviour. Unblocks Axis 1 (compression
  coverage) measurements on real SAEs.
- **Encoding-aware knob assignment in `from_sae_lens`** — new
  `assign_amp_knobs: bool = False` kwarg. When True, the loader
  populates higher-rung amp-branch knobs (`theta_amp`, `psi_aux`,
  `theta_amp_b`, `psi_amp_b`) from decoder geometry via PCA-axis
  extension of the existing β strategy. Resolves the load-bearing
  finding in `docs/research/rung4-viability-spike-v2.md` that
  higher-rung dictionaries were structurally dormant in the loader
  path (gram-equivalent to MPSRung1 at default knobs). Default
  `False` preserves byte-identical behaviour. `SAEImportConfig`
  gains the same field. Both shipped profiles (`clustered`,
  `uniform-sphere`) honour the flag. Falsifying tests in
  `tests/test_amp_knob_assignment.py` pin the invariant that
  Rung4 gram with amp-on differs from amp-off by Frobenius >
  1e-3 on the toy fixture (off-diagonal-only > 1e-3 too).
  Real-SAE measurements: Rung4 mean off-diagonal drops 0.82 →
  0.32 with amp-on; the encoding's state space is no longer
  collapsing to MPSRung1.
- **Rung-viability v2 methodology — measurement scripts and partial
  results.** Implements axes 1 + 2 of the v2 methodology proposed
  in `docs/research/rung-viability-methodology.md`:
  - `examples/rung_compression_coverage.py` (Axis 1) —
    `EpochCompressor(encoding=X)` runs across encodings on the
    same SAE; reports per-iteration zeroing trajectory + cumulative
    cross-entropy delta. Graceful skip when torch / SAE is missing.
  - `examples/rung_gram_condition.py` (Axis 2) — builds a
    `Dictionary` at K=max_features on the top-cosine subset of an
    SAE; reports λ_min(|gram|²), off-diagonal Frobenius mass,
    condition number. Torch-free.
  - `docs/research/rung4-viability-spike-v2.md` — captures the
    Axis 2 finding: at default knobs, Rung3 and Rung4 reduce to
    MPSRung1-equivalent gram (designed property of the encodings);
    Axis 2 as designed can't discriminate without knob optimization
    or encoding-specific knob assignment in `from_sae_lens`. Axes 1
    + 4 pending a torch-enabled host. Decision: **inconclusive — v1
    opt-in verdict stands.**

### Fixed

- **`examples/rung_compression_coverage.py`** — script referenced a
  nonexistent `EpochIteration.cumulative_cross_entropy_delta` field
  and crashed after the compressor finished, before writing JSON
  output. Replaced with a cumulative sum derived from the real
  per-iter `cross_entropy_delta` field; both per-iter and final
  CE deltas now appear in the output payload. The smoke test
  (`test_rung_compression_coverage_smoke`) was insufficient — it
  exercises only the SAE-missing skip path and never reaches the
  report-render code where the bug lived.

### Changed (Performance)

- **`build_clustered_dictionary`** now shares the cosine pair graph
  between block formation and cross-block adjacency rather than
  recomputing it twice. Wall-clock at N=8192 K=8 drops from 12.4 s
  to 7.0 s (1.8× faster); clustered speedup vs flat-cosine baseline
  improves from 0.48× to 0.94× across all measured K values. No
  behavior change — recall remains 1.000 on the killer-experiment
  fixture; cross-block adjacency is bit-equal. Closes issue #58.
  Research note `docs/research/clustered-dictionary-recall-vs-flat.md`
  updated with post-fix measurements and a side-by-side speedup row.

### Added

- **`examples/clustered_dictionary_walkthrough.py --encoding`** —
  new CLI flag (`mps` / `rung3` / `rung4` / `hea`) selecting the
  per-block encoding. The encoding's `max_features` becomes the
  default `--block-size`. Closes issue #47. K=16 / K=32 rows added
  to `docs/research/clustered-dictionary-recall-vs-flat.md`'s
  headline table — recall stays 1.000 at all three K values; block
  count drops 25-41% from K=8 to K=32 at N=8192 but wall-clock is
  K-invariant (cosine-pair-graph O(N²) dominates over per-block
  Python overhead at this scale).
- **`examples/rung4_viability_spike.py`** — research-track follow-up
  to `add-rung4-encoding-mvp` §7. Mirrors the Rung3 viability spike:
  runs all 28 pairs of the §4.4 8-feature panel through
  `Cancellation(encoding="rung4")` (6-knob joint optimizer, 4D outer
  grid over feature B's product-amp knobs), reports the four-criterion
  A/B/C/D decision banner, and writes a partial JSON with criterion A
  when the `[behavioural]` extra (torch + transformers) is missing.
  See `docs/research/rung4-viability-spike.md` for the findings
  (**decision: Rung4 stays opt-in**; the constrained spike's residual
  is structurally identical to Rung3's because both encodings hit the
  `min_amp_overlap` constraint boundary tight).
- **`polygram.Rung4` / `polygram.Rung4State`** are now exported from
  the top-level `polygram` namespace (was previously only available
  via `polygram.encoding.Rung4`).
- **`EpochCompressor(encoding=...)`** — new constructor parameter
  (defaults to `MPSRung1()`) plumbs the configured encoding through
  the compression pipeline. Compression runs can now opt into Rung3
  (K=16) / Rung4 (K=32) / HEA_Rung2 (K=2^n_qubits) to exploit the
  larger per-encoding feature caps. The internal `_select_panels`
  neighbour cap now scales as `encoding.max_features - 1` (was
  hardcoded 7). Byte-identical behaviour preserved at the
  `MPSRung1()` default — load-bearing differential regression
  (`test_byte_identical_epoch_result_against_frozen_reference`)
  passes unchanged. Closes issue #48.
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
