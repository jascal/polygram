# add-compression-epoch — tasks

## 1. Subpackage scaffold + report types

- [ ] 1.1 New module `polygram/compression/epoch.py` for the
      orchestrator (kept separate from `compressor.py` to preserve
      the latter's torch-free, single-panel-only contract).
- [ ] 1.2 New module `polygram/compression/epoch_report.py` for the
      orchestrator's report types (`Panel`, `EpochIteration`,
      `EpochReport`, `EpochResult`). Mirrors the single-panel
      `report.py` JSON-shape conventions (six-sigfig floats,
      sorted-key serialization, `format(v, ".6g")` rounding).
- [ ] 1.3 `Panel` frozen dataclass with `panel_id: int`, `anchor:
      int`, `feature_ids: tuple[int, ...]` (length 1–8),
      `cosines_to_anchor: tuple[float, ...]` (length 7 — anchor
      excluded). `feature_ids` is sorted ascending for
      deterministic panel hashing.
- [ ] 1.4 `EpochIteration` frozen dataclass with the field set
      named in `design.md` Decision 8 (`iteration: int`, `panels:
      tuple[Panel, ...]`, `validation_report_paths:
      tuple[Path, ...]`, `confirmed_pair_count: int`,
      `clusters_compressed: int`, `features_zeroed_this_iteration:
      tuple[int, ...]`, `cross_entropy_delta: float`,
      `convergence_state: str`).
- [ ] 1.5 `EpochReport` frozen dataclass with the field set named
      in Decision 8 (`schema_version`, `source_checkpoint`,
      `source_checkpoint_sha256`, `output_checkpoint`,
      `output_checkpoint_sha256`, `convergence_reason: str`,
      `n_features_zeroed_total: int`, `n_panels_total: int`,
      `coverage_achieved: float`, `wall_seconds: float`,
      `iterations: tuple[EpochIteration, ...]`).
- [ ] 1.6 `EpochResult` frozen dataclass (`report: EpochReport`,
      `output_checkpoint: Path`, `final_dictionary: Dictionary`).
- [ ] 1.7 `EpochReport.to_json(path)` / `from_json(path)` round-trip
      matching `CompressionReport`'s schema discipline.
- [ ] 1.8 Public exports added to
      `polygram/compression/__init__.py` and
      `polygram/__init__.py` (`EpochCompressor`, `EpochReport`,
      `EpochIteration`, `EpochResult`, `Panel`).

## 2. Pre-pass: firing rates + cosine graph

- [ ] 2.1 Helper `_compute_firing_rates(sae_checkpoint, prompts,
      model_name, layer, device) -> np.ndarray` — runs ONE forward
      pass per prompt through GPT-2, captures residuals at
      `model.transformer.h[layer]`, encodes through the SAE for
      every feature in the SAE (vectorized: `f =
      np.maximum((residuals - b_dec) @ W_enc + b_enc, 0)`), returns
      per-feature firing rate over all tokens. Lazy-imports torch
      via `_import_torch_and_transformers` and resolves device via
      `_resolve_device`.
- [ ] 2.2 Helper `_compute_cosine_graph(W_dec, eligible,
      threshold) -> set[tuple[int, int]]` — returns the set of
      `(i, j)` pairs (i < j, both in `eligible`) with
      `cos(W_dec[i], W_dec[j]) ≥ threshold`. Vectorized via
      `unit @ unit.T` then upper-triangular mask. Caps memory
      with a chunked path when `len(eligible) > 8192`.
- [ ] 2.3 Caching: the firing-rates pre-pass runs once per
      epoch run (NOT once per iteration). The cosine graph is
      recomputed per iteration (cheap; depends on `W_dec` which
      changes between iterations) — but `_compute_cosine_graph`
      respects the `eligible` set (which excludes `zeroed`) so
      zeroed features don't contribute spurious cosine edges
      against zero-decoder rows.

## 3. Panel selection

- [ ] 3.1 `_select_panels` function implementing Decision 2's
      greedy seeded coverage algorithm:
      - sort eligible features by `firing_rate × decoder_norm`
        descending (priority queue);
      - pop highest-priority anchor not yet at `n_visits_per_feature`
        cap;
      - build panel = anchor + 7 nearest cosine-similar features
        from eligible (excluding zeroed and excluding features at
        their visit cap);
      - update per-feature visit counter and `pairs_covered` set;
      - terminate on `coverage ≥ coverage_target` OR `len(panels)
        ≥ n_panels_max` OR priority queue exhausted.
- [ ] 3.2 Panel size invariant: every emitted panel has exactly 8
      `feature_ids` (1 anchor + 7 neighbours) UNLESS the eligible
      pool has fewer than 8 features (then panel = full eligible
      set; emit at most one such panel and warn).
- [ ] 3.3 Skip-zeroed: panel selection MUST exclude any feature in
      `self.zeroed`. Asserted in tests.
- [ ] 3.4 Determinism: two `_select_panels` calls with the same
      inputs (firing rates, cosine graph, parameters) produce
      identical panel sequences. Tiebreaks on equal priority go to
      lower fid; tiebreaks on equal cosine in neighbour selection
      go to lower fid.

## 4. Multi-panel aggregation

- [ ] 4.1 `_synthesize_validation_report(panels: list[Panel],
      per_panel_reports: list[ValidationReport]) -> ValidationReport`
      — emits the synthetic multi-panel report per Decision 3:
      union of confirmed pairs across panels; per-pair statistics
      aggregated per the table in design.md (max for
      polygram_overlap / jaccard, sum for n_fires_*, weighted mean
      for KL ratios).
- [ ] 4.2 `_compute_global_n_fires(panels: list[Panel],
      firing_rates: np.ndarray, n_tokens: int) ->
      dict[int, int]` — per Decision 4. The orchestrator passes
      this to a new `representatives` builder that uses the
      cluster-global counts.
- [ ] 4.3 `_pick_representatives_global(synthetic_report,
      global_n_fires) -> dict[int, int]` — runs union-find on
      `synthetic_report.confirmed`, picks representative per
      cluster as `argmax(global_n_fires)` with lowest-fid
      tiebreak. Returns the `{cluster_id: fid}` map for
      `Compressor.representatives`.

## 5. Quality bound: cross-entropy delta

- [ ] 5.1 Helper `_reconstruct_residuals(residuals, W_enc, b_enc,
      W_dec, b_dec) -> np.ndarray` — applies the SAE
      encode→decode loop once on cached residuals (no GPT-2
      forward needed). Pure numpy.
- [ ] 5.2 Helper `_token_cross_entropy_delta(residuals,
      sae_state_before, sae_state_after) -> float` — computes the
      mean per-token cross-entropy between the two reconstructions.
      Uses softmax-normalized squared distance as a tractable
      proxy (proper next-token CE would require a full GPT-2
      forward pass per iteration; the proxy is computed in numpy
      from cached residuals and is monotonic in actual
      reconstruction error).
- [ ] 5.3 Quality-bound check inside `EpochCompressor.run()`:
      after iteration `k > 1`, compare `delta_k` to
      `quality_delta_multiplier × delta_1`; on breach, set
      convergence_reason and revert to iteration `k-1`'s
      checkpoint.

## 6. Iteration loop

- [ ] 6.1 `EpochCompressor` dataclass with the field set named in
      proposal.md (sae_checkpoint, prompts, layer, model_name,
      strategy, device, coverage_target, cosine_threshold,
      n_visits_per_feature, n_panels_max, min_firing_rate,
      max_iterations, quality_delta_multiplier,
      polygram_overlap_threshold, jaccard_threshold,
      min_both_fire, save_intermediate_reports).
- [ ] 6.2 `__post_init__` validation:
      - sae_checkpoint exists;
      - strategy is supported by `Compressor`;
      - all numeric thresholds in valid ranges (coverage_target ∈
        (0, 1], cosine_threshold ∈ [-1, 1], n_visits_per_feature
        ≥ 1, max_iterations ≥ 1, quality_delta_multiplier > 0,
        etc.);
      - layer ≥ 1 unless `allow_layer_zero=True` (delegated to
        `BehaviouralValidator`'s same check).
- [ ] 6.3 `run(output_checkpoint) -> EpochResult` — the main
      loop:
      1. compute firing rates (one full forward);
      2. for iteration k in 0..max_iterations:
         - select panels;
         - if no panels selected: terminate
           (`convergence_reason='no_more_priority_candidates'`);
         - run validator on each panel sequentially;
         - synthesize multi-panel report;
         - if no confirmed pairs: terminate
           (`convergence_reason='stable_clusters'` if k > 0 with
           same cluster set; `'no_more_priority_candidates'` if k =
           0);
         - run `Compressor.apply()` with the synthetic report and
           orchestrator-built `representatives`;
         - update zeroed set;
         - compute cross-entropy delta vs original;
         - if k = 0: store delta_1; if k > 0 and breach: revert,
           terminate;
         - if cluster fingerprint matches previous iteration's:
           terminate ('stable_clusters');
      3. write final EpochReport.
- [ ] 6.4 Atomic final-checkpoint write: each iteration writes to
      a temp path, then `os.replace`s to the final
      `output_checkpoint` only after the iteration succeeds and
      the quality bound holds. On revert, the prior temp is the
      final.
- [ ] 6.5 Cleanup: all per-iteration temp files are deleted after
      successful completion; on exception, leave behind for
      forensics with a clear log message.

## 7. CLI — `polygram compress-epoch` subcommand

- [ ] 7.1 New subparser registered in `polygram/cli.py`:
      `polygram compress-epoch` with the flag set named in
      proposal.md.
- [ ] 7.2 `--prompts` reads via the same helper as `polygram
      validate` (one prompt per non-empty, non-`#`-prefixed line).
- [ ] 7.3 `--strategy` validates against `Compressor`'s supported
      set; exits 2 on unknown value.
- [ ] 7.4 Stage progress to stderr: per-iteration progress line
      (`epoch_compress: iter 1 / 5 — 312 panels selected,
      coverage 0.94`), per-panel progress within an iteration
      (`epoch_compress: iter 1 panel 47 / 312`), final
      summary line with sha256s and `n_features_zeroed_total`.
- [ ] 7.5 Exit code 2 on missing input files, output ==
      sae-checkpoint path collision, malformed flag values,
      `BehaviouralValidator` post-init failures.

## 8. Tests

- [ ] 8.1 `tests/compression/test_epoch_panel_selection.py` —
      `_select_panels` unit tests:
      - small synthetic (eligible = 16 features), assert greedy
        priority order matches firing × norm sort;
      - assert each feature appears at most `n_visits_per_feature`
        times;
      - assert coverage_achieved is monotonic in n_panels;
      - assert zeroed features never appear in any panel;
      - determinism: two calls with same seed → identical panels.
- [ ] 8.2 `tests/compression/test_epoch_aggregation.py` —
      multi-panel aggregation:
      - hand-build 3 overlapping panels with known confirmed
        pairs, assert the synthetic report's confirmed list is
        the union;
      - assert max-aggregation for `polygram_overlap` / `jaccard`;
      - assert weighted-mean aggregation for KL ratios;
      - assert orchestrator-built `representatives` reflects
        global n_fires, not intra-pair sums.
- [ ] 8.3 `tests/compression/test_epoch_convergence.py` —
      iteration loop:
      - synthetic SAE + hand-injected redundancies; assert
        convergence in 1 iteration when first pass exhausts all
        confirmed pairs;
      - assert convergence on `stable_clusters` when iteration k
        and k-1 produce the same cluster fingerprint;
      - assert termination on `max_iterations` cap (set to 2 for
        a workload that would otherwise run further);
      - assert quality-bound revert: hand-injected reconstruction
        error in iteration 2 → revert to iteration 1's
        checkpoint.
- [ ] 8.4 `tests/compression/test_epoch_postinit.py` —
      `EpochCompressor.__post_init__` rejection paths: missing
      checkpoint, unsupported strategy, bad threshold ranges,
      layer = 0 without override.
- [ ] 8.5 `tests/compression/test_epoch_report_roundtrip.py` —
      `EpochReport.from_json(report.to_json()) == report` on a
      hand-built fixture with 2 iterations, 5 panels, all
      convergence_reason values.
- [ ] 8.6 `tests/cli/test_compress_epoch_cli.py` — argument
      parsing + skip-path:
      - missing prompts file → exit 2;
      - output == source path collision → exit 2;
      - unknown strategy → argparse exit 2;
      - end-to-end synthetic SAE smoke run with
        `--n-panels-max 2 --max-iterations 1`.
- [ ] 8.7 `tests/test_examples.py` gains
      `test_compress_epoch_validated_smoke` mirroring the
      validator / compress smoke patterns: tiny configuration,
      success path asserts the epoch banner, skip path asserts a
      clear message.

## 9. Worked example + research note

- [ ] 9.1 New `examples/compress_epoch_validated.py` showing the
      full workflow: load the §4.4 SAE, run `EpochCompressor` with
      default flags, dump the `EpochReport`. Same skip-path
      semantics as `examples/compress_validated.py` (SAE absent →
      exit 0 with a clear hint).
- [ ] 9.2 New `docs/research/compression-epoch-design.md`
      (one page) — pointer to this change's `design.md` plus a
      paragraph contrasting the GC / defrag analogy with the
      statistical-clustering reality (sampled signal, no
      deterministic reference graph).
- [ ] 9.3 Update `tech-debt-backlog/tasks.md` §5: append the
      epoch orchestrator as the loop's *third* component (after
      validator + compressor).

## 10. Closing

- [ ] 10.1 README "Library tour" gains a one-paragraph entry for
      the `EpochCompressor` (between the compression-action
      section and Development).
- [ ] 10.2 Run `openspec validate add-compression-epoch` — clean.
- [ ] 10.3 Run the full test suite end-to-end. CI green.
- [ ] 10.4 Squash-merge to main; archive this change directory
      under `openspec/changes/archive/`.
