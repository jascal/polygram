# Changelog

## Unreleased

(nothing yet)

## 0.13.0 — 2026-05-21

### Added

- **`add-encoding-partition` Phase 1** — locks the API surface for
  per-block heterogeneous encoding (the gating prerequisite for
  sae-forge's `add-block-structured-sae`). 33 new tests, full
  suite green (1037 → 1070).
  - `BlockSpec` dataclass + per-family validator registry
    (`_BLOCK_SPEC_KWARG_VALIDATORS`) for clean Rung6+ extensibility.
    Hashable (frozen + custom `__hash__` since `encoding_kwargs` is
    a dict). `block_id`, `encoding_class`, `encoding_kwargs`,
    `learn_axis_assignment`, `feature_ids`.
  - `PartitionCoverageError` (subclasses `ValueError`) +
    `validate_partition_coverage(partition, n_features_input=...)`
    helper. Names offending feature ids (capped at first 10).
  - `make_default_block(...)` convenience constructor for the
    "default + heavy override" partition pattern.
  - `CompressionConfig.encoding_partition: tuple[BlockSpec, ...] | None`
    field. Type/membership/non-empty validation at config
    construction. Default `None` preserves the single-encoding path
    byte-equivalently.
  - `CompressionReport.blocks: tuple[BlockReport, ...] | None` +
    new `BlockReport` dataclass mirroring per-block diagnostics.
    `CompressionReport.SCHEMA_VERSION` bumped 2 → 3 with back-compat
    loader for v2 payloads (defaults `blocks=None`).
  - `MAX_CLUSTERS_PER_BLOCK = 10_000` module-level constant for the
    global cluster-id namespace.
  - `from_sae_lens(..., encoding_partition=...)` kwarg accepted +
    type-validated.
  - Public exports from `polygram.compression`: `BlockSpec`,
    `BlockReport`, `PartitionCoverageError`, `make_default_block`,
    `validate_partition_coverage`.

  Phase 2 (the actual per-block dispatch in `Compressor.apply`) is
  filed as a follow-up impl. The current `Compressor.apply` raises
  `NotImplementedError` with a clear pointer when a partitioned
  config is supplied — downstream consumers don't silently get a
  single-encoding compression where a partitioned one was requested.

## 0.12.0 — 2026-05-20

### Added

- **`emit-cluster-metadata-from-epoch-compressor` shipped.**
  `EpochCompressor.run(out_path)` now writes a sidecar
  `<out_path>_compression_report.json` alongside the compressed
  safetensors, matching the convention `Compressor.apply()` already
  uses. `EpochReport` schema bumped 3 → 4 with two new fields:
  - `n_clusters: int` — count of distinct non-negative cluster ids
    in `cluster_assignments`.
  - `cluster_assignments: tuple[int, ...] | None` — per-feature
    cluster id (length `n_features_input`); `-1` for fully-zeroed
    features.

  Closes the asymmetry between the two compression entry points
  surfaced by sae-forge PR #69's §8.4 falsifiable smoke: previously
  `FeatureBasis.from_polygram_checkpoint` on an EpochCompressor
  output saw `n_clusters=0` (defensive default) and the
  `add-concept-anchored-finetune` polygram-clusters backend refused
  the basis, forcing the smoke to inject synthetic
  cluster_assignments. After this change a production
  EpochCompressor output is fully self-describing.

  Loader is back-compat: pre-v4 payloads load without error and
  default the new fields to `0` / `None`. Frozen-fixture tests
  (`test_byte_identical_epoch_result_against_frozen_reference` and
  the multi-iter variant) regenerated against the v4 schema. See
  `openspec/changes/emit-cluster-metadata-from-epoch-compressor/`.

## 0.11.0 — 2026-05-20

### Changed (behavior — load-bearing for downstream `polygram_bridge.py` scripts)

- **`CancellationResult.cancellation_efficiency` semantics narrowed.**
  When `before_overlap ≈ structural_floor ≈ after_overlap` within
  `1e-6` (the new at-floor case), `cancellation_efficiency` is now
  coerced to `0.0` instead of `None`. **`None` is now reserved for
  the case where `structural_floor` itself is undefined** (e.g.
  `NaN` floor on a non-canonical knob configuration). Pair this
  with the new `at_structural_floor: bool` field on
  `CancellationResult` to distinguish the two cases programmatically.

  **Downstream migration:** `sm-sae/smsae/polygram_bridge.py`,
  `econ-sae/econsae/polygram_bridge.py`, and
  `bio-sae/biosae/polygram_bridge.py` all currently use
  `result.cancellation_efficiency is None` to mean "N/A / at floor".
  After bumping the polygram pin, the at-floor case will print
  `0.00%` instead of `N/A`. Update those scripts to branch on
  `result.at_structural_floor` (True → "at floor / N/A") rather
  than (or in addition to) `cancellation_efficiency is None`.

  **At-floor result JSON, before/after (same `Cancellation.run()`):**

  ```jsonc
  // polygram 0.10.x (legacy):
  {
    "before_overlap": 0.593133,
    "after_overlap":  0.593133,
    "structural_floor": 0.593133,
    "cancellation_efficiency": null   // ambiguous: at-floor OR floor-undefined
  }

  // polygram 0.11.0:
  {
    "before_overlap": 0.593133,
    "after_overlap":  0.593133,
    "structural_floor": 0.593133,
    "cancellation_efficiency": 0.0,   // legibly "no gap consumed because no gap existed"
    "at_structural_floor": true       // new — distinguishes at-floor from floor-undefined
  }

  // Floor-undefined case (e.g. HEA with preserve_tiers=True on non-canonical knobs):
  {
    "before_overlap": 0.412,
    "after_overlap":  0.118,
    "structural_floor": NaN,
    "cancellation_efficiency": null,  // unchanged from 0.10.x: floor itself unmeasurable
    "at_structural_floor": false
  }
  ```

### Added

- **`CancellationResult.at_structural_floor: bool`** field plus a
  `UserWarning` emitted by `Cancellation.run()` (and the rung3 /
  rung4 / MPSRung1 variants) when `before ≈ floor ≈ after` within
  `1e-6`. Makes the at-floor case observable without parsing
  trajectories by eye. Closes P1 of the 2026-05-20 joint feedback
  doc from bio-sae / sm-sae / econ-sae. README gains a
  "Which tool should I use?" pivot table separating
  `Cancellation` (encoding expressiveness) from `cluster_experts`
  (feature coherence). See
  `openspec/changes/add-cancellation-and-compression-diagnostics/`.

- **Convergence-test diagnostics on `CompressionReport`,
  `EpochReport`, and `RegrowReport`.** Four new optional fields per
  report — `rank_ratio: float | None`, `post_A: float | None`,
  `forge_mse: float | None`, and the derived
  `informative_metric: Literal["post_A", "both", "forge_mse"] | None`.
  `Compressor.apply()` computes `rank_ratio` (basis_rank of the
  kept-features' decoder rows divided by `d_model`) and `post_A`
  (subspace input-variance preservation); `forge_mse` is
  caller-supplied (the host repo populates it post-hoc).
  Schema versions bump (`CompressionReport` 1→2, `EpochReport`
  2→3, `RegrowReport` 1→2); loaders accept prior-version
  payloads with the new fields defaulting to `None`. Shared
  helpers (`json_finite`, `floats_eq`, `informative_metric`,
  `compute_rank_ratio`) live in
  `polygram/compression/_helpers.py`.

- **`compressor-partial-key-sae` shipped.** `Compressor.apply()` now
  loads its input via a strategy-dependent required-key set plus a
  permissive optional-key probe. Shipping strategies (`"zero"` /
  `"merge"`) require only `W_dec`; `W_enc` / `b_enc` / `b_dec` are
  loaded when present and silently omitted when absent. The output
  safetensors mirrors the input's key set (no placeholder synthesis):
  a `W_dec`-only input produces a `W_dec`-only output; a full-SAE
  input produces a byte-equivalent full-SAE output. Driven by
  sae-forge's synth-basis use case which writes a decoder-only
  safetensors and previously forced placeholder-encoder workarounds.
  New helper: `polygram.sae_import._load_sae_checkpoint_optional`.
  See `openspec/changes/compressor-partial-key-sae/`.

## 0.10.0 — 2026-05-19

### Added

- **`add-polygram-010-diagnostics` shipped (impl track).**
  `EpochReport` carries new first-class fields `redundancy_ratio:
  float` and `n_features_input: int`. The ratio (observed in the
  60–74% range across recent Gemma-2 runs, correlating strongly
  with downstream KL outcomes) is now readable directly off the
  report instead of being a downstream derivation. Schema bumped
  to v2; legacy v1 payloads load with a forward-compat shim that
  computes the ratio from any available divisor or sets the
  defensive `0.0` sentinel (not `nan`, which downstream consumers
  consistently misread as a real measurement). `polygram analyze`
  surfaces the ratio alongside `n_features_zeroed_total`.
- **`BlockFormation` degenerate-partition warning.**
  `build_clustered_dictionary` now emits a `UserWarning` when the
  cosine partition has zero multi-feature blocks — every block is
  a singleton because no decoder pair cleared `cosine_threshold`.
  The warning names the configured threshold, the observed
  maximum off-diagonal cosine, and a recommended fallback
  threshold (`max_observed_cosine * 0.8`, clipped). When invoked
  via `from_sae_lens(..., clustered=True)`, the same text appears
  in `SelectionReport.warnings`. Cosine-strategy-only;
  co_firing / user_declared are suppressed.

### Added

- **`BlockView` public surface.** New `polygram.clustered_dictionary.BlockView`
  frozen dataclass carrying per-block metadata: `indices`,
  `decoder_slice`, `encoding`, `feature_names`, `feature_clusters`.
  `ClusteredDictionary` exposes the parallel `block_views: tuple[BlockView, ...]`
  property and `block_view(idx) -> BlockView` accessor. Populated by
  both canonical builders (`build_clustered_dictionary` populates
  indices + metadata with `decoder_slice=None` to avoid the per-block
  copy overhead; consumers compute the slice on demand from
  `view.indices` against the parent matrix). `from_compression_panels`
  populates the same fields with `decoder_slice=None` since the
  compression panel path doesn't carry raw decoder vectors.
  See `polygram/clustered_dictionary.py:BlockView` for the rationale
  and the deferred-lazy-property scope note.

### Performance

- **`_materialise_blocks` eliminates per-feature `dataclasses.replace`.**
  The per-block hierarchy is now built from each feature's
  original `cluster` value rather than synthesised via a
  per-feature `replace(f, cluster=…)`. Object-allocation reduction
  measured at full N=24,576: MPS 60%, Rung3 69%, Rung4 77%
  (post-fix vs pre-fix in
  `docs/research/data/clustered_amortised_benchmark_full_*_v2.json`).
  Build wall-clock improves 2.55× on MPS, 1.16–1.18× on Rung3/Rung4.
  Does NOT meet the proposal's gram-cross-over success criterion —
  see `docs/research/clustered-amortised-benchmark.md`'s
  post-lighter-container sweep section for the honest diagnosis.
  Behaviour change: blocks built via cosine / co_firing strategies
  now preserve each feature's *original* cluster name in the
  per-block hierarchy (previously they were all rewritten to a
  synthetic `<parent>_b<idx>` name); user_declared blocks already
  preserved the original cluster values and are unaffected.
- **`ClusteredDictionary.cross_block_edges_tuple` cache.**
  `cross_block_pairs` is now also exposed as a precomputed tuple
  at build time so sampled cross-block walks avoid the
  `list(dict.items())` re-materialisation surfaced by the
  amortised benchmark. Measured per-op cost on `cross_block_overlap`
  at full N=24,576 dropped 51.6 ms → 0.18 ms (287×).

## 0.9.0 — 2026-05-19

### Added

- **`add-cluster-experts-mvp` shipped.** New `polygram.experts`
  module with `ExpertDictionary` (frozen dataclass partitioning a
  flat `Dictionary` into routable expert blocks plus a precomputed
  feature→expert map) and the `cluster_experts(dictionary,
  decoder_vectors, *, method="cosine", coherence_threshold,
  max_features_per_expert, activations)` factory. Routing via
  `ExpertDictionary.route(activations, top_k)` returns top-k expert
  indices by summed per-expert activation — numpy-only, no torch
  dependency. `method="coactivation"` is reserved (raises
  `NotImplementedError`) and will light up when
  `clustered_dictionary._form_blocks_co_firing` is implemented.
  Both symbols re-exported from `polygram`. See OpenSpec change
  `add-cluster-experts-mvp` for scope and explicit follow-ups
  (Louvain/HDBSCAN, MLP router, bio metrics, min/max
  cluster-size post-processing).

### Behaviour changes

- **`from_sae_lens` auto-promotes to clustered on cap overflow.**
  The `clustered` kwarg is now tri-state: `clustered=None` (the new
  default) auto-promotes to `ClusteredDictionary` when
  `len(feature_ids) > encoding.max_features`, appending an
  `"auto-promoted to clustered: …"` entry to
  `SelectionReport.warnings`. `clustered=False` keeps the legacy
  strict `ValueError`; `clustered=True` is unchanged. Callers that
  caught the previous default `ValueError` should either expect a
  `ClusteredDictionary` or pass `clustered=False` explicitly. See
  OpenSpec change `auto-promote-clustered-on-cap-overflow`.

- **`SAEImportConfig.assign_amp_knobs` and `assign_phase_knobs`
  default to `True`.** Per the
  [sm-sae](https://jascal.github.io/sm-sae/) benchmark's measured
  Recommended-defaults table — without these, cancellation on Rung3+
  dictionaries is stuck at the structural floor. The legacy `False`
  behaviour is reachable by passing the kwargs explicitly. See
  OpenSpec change `apply-sm-sae-recommended-defaults`.

### Removed

- Dead `_LEGACY_MAX_FEATURES = 8` fallback in
  `polygram/clustered_dictionary.py` — every encoding has exposed
  `max_features` natively since the archived
  `per-encoding-feature-cap` change.

## 0.8.0 — 2026-05-16

### Added

- **`add-learned-axis-assignment` shipped.** New
  `polygram.geometry.LearnedKnobAssignment` strategy that calibrates
  the decoder-PCA-to-polygram-knob projection from data instead of
  using the hardcoded `PC2 → α / PC3 → φ / PC4..→ amp_knobs`
  permutation. Greedy per-knob permutation search (default,
  pure-numpy) or scipy continuous-optimisation (`polygram[opt]`)
  initialised from the greedy result. Pluggable
  `LearnedAxisObjective` protocol with three built-ins:
  `spearman_objective` (default), `pearson_objective`,
  `behavioural_objective(ref_matrix)` factory for sae-forge
  co-activation matrices.
  - `polygram.geometry`: new module
    `polygram.geometry.learned_axis_assignment` with
    `LearnedKnobAssignment`; new
    `polygram.geometry.objectives` module with the three built-in
    objectives.
  - `polygram.geometry.protocols`: new `LearnedAxisObjective`
    protocol; new optional fields on `KnobAssignmentResult`
    (`axis_assignment`, `objective_value`, `objective_baseline`,
    `training_objective_value`).
  - `polygram.sae_import.from_sae_lens`: new
    `learn_axis_assignment` kwarg (`bool | LearnedKnobAssignment |
    None = None`). Strict opt-in; `None`/`False` preserves
    byte-identical behaviour. `SelectionReport.learned_axis_assignment`
    surfaces the chosen map (JSON-safe).
  - `polygram.config.SAEImportConfig`: new
    `learn_axis_assignment: bool | None = None` field; round-trips
    through `to_dict()` / `from_dict()`.
  - `polygram analyze --learn-axis-assignment` CLI flag wires the
    opt-in through `predict_cancellation_depth`.
  - Empirical motivation: `docs/research/rung5-pareto-scans.md`
    scan 4 (greedy prototype ~3×s decoder-Gram Spearman on a
    synthetic SAE).
  - Production demo: `examples/learned_axis_assignment_demo.py` +
    committed `docs/research/data/learned_axis_assignment_demo.json`
    confirming the headline result on the same fixture.
  - Research note: `docs/research/learned-axis-assignment.md`
    documents API, promote-to-default gate (≥ 2 real SAEs,
    downstream metric, calibration cost budget), and the §9 / §11
    follow-ups in the theoretical treatment.

### Notes

Strict default-off. `HEA_Rung2` falls back to
`ClusteredKnobAssignment` with an INFO-once log — learned
assignment for HEA's per-feature θ tensor is out of scope for v1.
Real-SAE replication (Gemma-Scope / GPT-2-small) is the next
research-track follow-up; promote-to-default gate documented in
the openspec proposal and `learned-axis-assignment.md`.

## 0.7.0 — 2026-05-16

### Added

- **`add-rung5-encoding` shipped.** New `Rung5(n_amp_qubits=k)`
  encoding generalises Rung4's fixed `k=2` product amp branch to a
  configurable amp-register width. Per-feature Hilbert dim is
  `8 · 2^k`; `max_features` scales accordingly (cap at
  `RUNG5_MAX_N_AMP_QUBITS = 16` → 524288). Targets sae-forge pareto
  sweeps over feature counts above Rung4's 32-feature cap.
  - `polygram.encoding`: `Rung5`, `Rung5State`,
    `rung5_amp_overlap`, `rung5_amp_overlap_squared`,
    `RUNG5_MAX_N_AMP_QUBITS`.
  - `polygram.dictionary.Feature`: new `amp_knobs` field
    (`tuple[tuple[float, float], ...]`); new
    `Feature.with_default_amp_knobs(encoding)` helper.
  - `polygram.dictionary.Dictionary`: Rung5 length validation +
    gram dispatch; extended `with_knob` grammar with
    `<name>.amp_knobs[i].{theta,psi}` paths (per-feature and
    cluster-shared).
  - `polygram.cancellation.Cancellation`: `"rung5"` SUPPORTED;
    `(2 + 2k)`-knob canonical list assembled from
    `dictionary.encoding.n_amp_qubits`; new `_run_rung5_joint`
    using scipy `differential_evolution`; k-independent
    `structural_floor` via `_mps_equivalent_floor`.
  - `polygram.emit.write_qorca`: Rung5 dictionaries write a
    `## amp branch` table with indexed `(theta_i, psi_i)` columns
    per amp qubit.
  - `polygram.sae_import.from_sae_lens`: encoding type hint
    broadened to accept every encoding class; new
    `amp_knobs_list` plumbing through `KnobAssignmentResult` and
    the `from_sae_lens` consumer site.
  - `polygram.geometry.amp_assignment.assign_amp_knobs_pca`: new
    Rung5 branch consumes PCA axes 4..(4 + 2k − 1) to populate
    per-feature `amp_knobs` from decoder geometry.
  - `examples/rung5_rank_verification.py` + committed
    `docs/research/data/rung5_rank_verification.json` confirm
    saturation at `8 · 2^k` for k ∈ {2, 3, 4} × seeds {0, 42}.
  - `docs/research/rung5-encoding.md` — math, design choices,
    forward-looking note on a potential unified
    `RungMPS(n_mps_qubits, n_amp_qubits)`.

No breaking changes. `Rung3`, `Rung4`, `MPSRung1`, `HEA_Rung2`
encodings unchanged. Default encoding remains `MPSRung1`.
`Feature.amp_knobs` defaults to `()` and is ignored by every
non-Rung5 dispatch.

## 0.6.0 — 2026-05-15

### Added

- **`add-phase-knob-assignment` shipped.** Resolves the root cause
  of the 2026-05-15 GPT-2 bug report (MPSRung1.gram() saturating on
  activation-uncorrelated features). New opt-in
  `from_sae_lens(..., assign_phase_knobs=True)` populates
  MPS-substrate α (PC2) and φ (PC3) per-feature from decoder PCA,
  un-dormanting the second half of MPSRung1's state space.
  - **Sanity-check effect** (toy fixture): mean off-diagonal |G|²
    drops 0.76 → 0.28 (63% drop); pairs ≥ 0.9 drop 12 → 1.
  - **Applies to all MPS-substrate encodings**: MPSRung1, Rung3,
    Rung4. HEA_Rung2 is a structural no-op (different knob shape).
  - **`SAEImportConfig.assign_phase_knobs`** parallel field with
    the same precedence as `assign_gamma` / `assign_amp_knobs`.
  - **`EpochCompressor` + `Compressor`** plumb the flag through
    all three `from_sae_lens` call sites (same monkeypatch test
    pattern catches missed sites).
  - **`--assign-phase-knobs` CLI flag** on
    `examples/rung_gram_condition.py` and
    `examples/rung_compression_coverage.py`.
  - **Two-flag orthogonality**: `assign_phase_knobs` and
    `assign_amp_knobs` compose additively. With both on for Rung4,
    every feature has non-default values across all six knob
    channels (α, φ, theta_amp, psi_aux, theta_amp_b, psi_amp_b).
  - **Default unchanged.** `assign_phase_knobs=False` is the
    default; existing call sites byte-identical.
- **`tests/test_phase_knob_assignment.py`** — 12 tests including
  the cornerstone falsifying invariant (gram with phase-on must
  differ from phase-off by Frobenius > 1.0 AND mean off-diagonal
  drops below half).

### Changed (Behavioural)

- **`assign_amp_knobs` PCA-component allocation shifted PC2-PC5 → PC4-PC7.**
  Phase knobs now own PC2-PC3 (they apply to all MPS-substrate
  encodings, so they get the lowest available components after β).
  Amp knobs (Rung3/Rung4) shift up. **This is a backward-incompat
  change** for callers depending on exact PR-#63-era
  gram-condition numbers with `assign_amp_knobs=True`: qualitative
  invariants hold (amp-on differs measurably from amp-off, the
  cornerstone test of PR #63 still passes), but exact values
  change. v2.1 results note's `_amp_on` data files would need
  regeneration if exact reproducibility matters.
- **v2.2 Axis 1 results regeneration deferred to a follow-up.**
  The Axis 1 PASS verdict was measured with the pre-shift amp-knob
  allocation. The verdict's qualitative claim (Rung4 amp-on ≫ MPS
  baseline) is expected to hold under the new allocation but
  exact `n_features_zeroed` / CE-delta numbers will differ. A v2.3
  re-run on a torch host is the recommended follow-up.

## 0.5.0 — 2026-05-15

### Added

- **`add-kl-attribution-rep-selection` shipped.** New opt-in third
  value for `CompressionConfig.rep_selection`: `"kl_attribution"`.
  Picks cluster representatives by **behavioural-ablation importance**
  (existing `CandidatePair.kl_ablate_*` fields from
  `BehaviouralValidator`) instead of by the geometric proxies
  (`n_fires`, `scale_aware`). See
  [`openspec/changes/add-kl-attribution-rep-selection/`](openspec/changes/add-kl-attribution-rep-selection/).
  - **Algorithm**: per-feature mean `kl_ablate` across pairs
    containing the feature; tiebreak `n_fires_total` descending, then
    feature id ascending.
  - **Per-feature NaN fallback**: a feature whose mean `kl_ablate`
    is NaN (very-low-firing feature where validator KL is
    statistically unreliable) competes via a geometric proxy
    (50% norm proximity + 50% log firing count) normalised within
    the NaN-only subset. Finite-kl features normalise within their
    own subset; both compete on the [0, 1] axis.
  - **All-NaN cluster**: raises `ValueError` with an actionable
    message naming the supported alternatives. Surfaces caller
    mis-configuration (e.g. `DecoderGeometryConfirmer`-produced
    report fed to `kl_attribution`) loudly rather than silently
    degrading.
  - **Flows transparently through `plan()`, `plan_with_target()`,
    and `plan_pareto()`** — rep_selection is a knob for any planning
    path, not pareto-specific.
  - **Default unchanged.** `CompressionConfig.rep_selection` remains
    `"scale_aware"`; existing callers and tests byte-identical (866
    pre-existing tests still pass).
  - **No interface widening.** The behavioural signal is already in
    `CandidatePair`; Compressor does not accept activations,
    forward-pass machinery, or new dependencies. Numpy-only.
  - **Capability**: new `recon-aware-rep-selection` capability spec
    documenting the algorithm, NaN contract, tiebreak rule.

### Corrected

- The "deferred — requires interface widening" note in
  `add-pareto-target-compression`'s proposal was wrong about a
  recon-aware `rep_selection` needing Compressor to accept
  activations. The required signal (`kl_ablate_*`) was already on
  `CandidatePair` in 0.3.0; `add-kl-attribution-rep-selection` ships
  it without any interface change.

### Empirical motivation status

- **Conceptually motivated, not empirically.** The proposal ships
  the option so the question can be asked. Pareto-dominance over
  `scale_aware` on real forge runs is an open research question. The
  natural test bed is a sae-forge Axis-4 sweep with
  `--rep-selection kl_attribution` filtered to
  `quality_tier in {"good", "saturated"}` rows (sae-forge's
  `add-forge-quality-diagnostics` capability).

## 0.4.0 — 2026-05-14

### Added

- **`add-pareto-target-compression` shipped (all three phases).**
  Target-K and Pareto-path planning for `Compressor`. Threshold mode
  is byte-identical to 0.3.0; new modes are opt-in. See
  [`openspec/changes/add-pareto-target-compression/`](openspec/changes/add-pareto-target-compression/).
  - **Public Python API**: `Compressor.plan_with_target(target_n_features_kept=None)`
    and `Compressor.plan_pareto(targets) -> ParetoReport`.
    `ParetoReport` and `ParetoOutcome` are new frozen dataclasses
    exported from [`polygram`](polygram/__init__.py) with JSON
    round-trip via `to_json` / `from_json`.
    `CompressionPlan.n_features_kept` is a new derived `@property`
    (`= len(self.clusters)`) mirroring the existing
    `CompressionReport.n_features_kept` semantic.
  - **Config**: `CompressionConfig` gains
    `target_n_features_kept: int | None = None` and
    `score_field: str = "polygram_overlap"` with `__post_init__`
    validation. Only the three bounded `[0, 1]` `CandidatePair`
    score fields are accepted (`polygram_overlap`, `jaccard`,
    `decoder_overlap`); KL- and count-based fields are excluded by
    Decision 3 of the change.
  - **Algorithm**: greedy union-find over score-sorted pairs;
    `(−score, min(i,j), max(i,j))` deterministic tiebreak; "must
    exceed then drop back" stop rule per K so very-high targets
    don't return trivial empty plans. `plan_pareto` performs
    **one shared sort plus one shared union-find walk** regardless
    of `len(targets)` (sort-once invariant tested via spy);
    snapshots `parent` per K and materialises plans through a
    shared `_materialise_plan_from_parent` helper.
  - **CLI** (`polygram compress`): four new flags —
    `--target-features N`, `--pareto K1,K2,...` (mutually
    exclusive), `--pareto-materialize` (opt-in SAE rewrite gate),
    `--score-field {polygram_overlap,jaccard,decoder_overlap}`.
    In `--pareto` mode, `--output` is treated as a directory
    receiving `pareto.json` plus (with `--pareto-materialize`)
    `pareto/k_{K}.safetensors` per K.
  - **Tests**: 47 new tests across
    `tests/compression/test_compressor_plan_with_target.py`
    (15), `tests/compression/test_compressor_plan_pareto.py` (15),
    `tests/compression/test_cli_compress_pareto.py` (12), and 5
    new entries in `tests/test_config.py`. Full suite (850) green.
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
  [`docs/research/rung4-viability-spike-v2.md`](docs/research/rung4-viability-spike-v2.md)
  under "Axis 1 result (v2.2)"; raw JSON + console captures under
  `docs/research/data/`.
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
