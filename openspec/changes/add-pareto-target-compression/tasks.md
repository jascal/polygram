## Phase 1 — Target-K compression

### 1. CompressionPlan derived `n_features_kept` property

- [x] 1.1 Add `@property def n_features_kept(self) -> int` to
  `CompressionPlan` returning `len(self.clusters)`.
- [x] 1.2 Verify `CompressionReport.to_json()` output is unchanged
  (existing report serializer tests stay green).
- [x] 1.3 Regression test:
  `plan().n_features_kept == post-apply report.n_features_kept`.

### 2. CompressionConfig field additions

- [x] 2.1 Add `target_n_features_kept: int | None = None` with the
  cluster-representative-count docstring.
- [x] 2.2 Add `score_field: str = "polygram_overlap"`.
- [x] 2.3 `_SUPPORTED_SCORE_FIELDS` constant added.
- [x] 2.4 `__post_init__` validates both fields with explicit
  error messages.
- [x] 2.5 Round-trip + missing-fields-from_dict tests added in
  `tests/test_config.py`.

### 3. Compressor target-K planning

- [x] 3.1 `Compressor._filter_pairs_for_score` static helper drops
  NaN-scored pairs and raises `ValueError` with the
  `DecoderGeometryConfirmer`-vs-`BehaviouralValidator` hint when
  the filter empties the list.
- [x] 3.2 `Compressor._greedy_union_to_target` sorts by the
  Decision-6 key, walks union-find tracking `n_clusters`, and
  stops once the trajectory crosses back to `<= target_k` after
  previously exceeding it. (Phase 1 chose the "must exceed first
  then drop" stop rather than the bare `<= target_k` literal so
  large target_k values don't return a trivial empty plan; see
  `_greedy_union_to_target` docstring.)
- [x] 3.3 `Compressor.plan_with_target(target_n_features_kept=None)`
  reads the config when omitted; raises if both are None.
- [x] 3.4 Uses `_pick_representative` for cluster reps; respects
  `rep_selection` / `representatives` overrides.
- [x] 3.5 `Compressor.apply()` already accepted an optional `plan`
  argument prior to this work; verified its signature suffices
  for plumbing `plan_with_target` output without re-planning.

### 4. Phase 1 tests

All 15 tests in
`tests/compression/test_compressor_plan_with_target.py`, plus
5 new tests in `tests/test_config.py`. Full suite (823 tests)
passes.

- [x] 4.1 `plan_with_target(target_k=N)` returns
  `n_features_kept <= N` on a feasible fixture.
- [x] 4.2 Infeasible target returns the most-compressed reachable
  plan; `n_features_kept > target_k`, no exception.
- [x] 4.3 Huge `target_k` returns the most-compressed reachable
  plan (algorithm processes all pairs).
- [x] 4.4 Determinism via `(−score, min(i,j), max(i,j))`.
- [x] 4.5 All three `score_field` axes (`polygram_overlap`,
  `jaccard`, `decoder_overlap`) work end-to-end.
- [x] 4.6 NaN-only behavioural fields raise `ValueError`; per-pair
  NaN entries are filtered out individually.
- [x] 4.7 Byte-identity regression covered by the existing
  threshold-path tests (still green; no change to `_build_plan`).
- [x] 4.8 `CompressionConfig(target_n_features_kept=0)` raises
  (also `-5`).
- [x] 4.9 `CompressionConfig(score_field="bogus")` and
  `score_field="kl_log_ratio_abs"` both raise.
- [x] 4.10 `Compressor.plan_with_target()` with no argument and no
  config setting raises `ValueError`.
- [x] 4.11 `Compressor.apply(plan=plan_with_target_output)`
  produces a `CompressionReport` whose
  `n_features_kept == plan.n_features_kept` and whose
  `plan.clusters` shape matches the input plan
  (identity check skipped because `apply` patches scale fields
  onto a fresh plan instance).

## Phase 2 — Pareto path artifact

### 5. ParetoOutcome + ParetoReport dataclasses

- [x] 5.1 Created `polygram/compression/pareto.py`.
- [x] 5.2 `@dataclass(frozen=True) class ParetoOutcome` with
  fields `target_k`, `reached_target`, `plan`.
- [x] 5.3 `@dataclass(frozen=True) class ParetoReport` with
  `schema_version`, `sae_checkpoint`, `sae_checkpoint_sha256`,
  `score_field`, `targets`, `outcomes`.
- [x] 5.4 `ParetoReport.to_json(path=None) -> str` and
  `ParetoReport.from_json(source)` mirror
  `CompressionReport`'s hand-coded serializer. Reuse
  `_cluster_to_dict` / `_cluster_from_dict`.
- [x] 5.5 Exported from `polygram/__init__.py` and
  `polygram/compression/__init__.py`.

### 6. Compressor.plan_pareto

- [x] 6.1 `Compressor.plan_pareto(targets) -> ParetoReport`.
  Empty / `None` / non-positive entries raise `ValueError` with a
  usage hint naming the offending input. Duplicates dedup;
  `ParetoReport.targets` is sorted descending.
- [x] 6.2 Single sort + single union-find walk via
  `_walk_pairs_for_pareto_snapshots`. Each K's `parent` is
  snapshotted at the moment its Phase 1 stop condition fires
  (must-exceed-then-drop); K values whose trajectory never
  matches the rule fall back to the final parent state. Sort
  invocation verified exactly-once via test-suite spy.
- [x] 6.3 `ParetoOutcome.reached_target = plan.n_features_kept <= target_k`.

### 7. Phase 2 tests

15 tests in `tests/compression/test_compressor_plan_pareto.py`.

- [x] 7.1 Nested-plans invariant verified across
  `plan_pareto([3, 2, 1])` on a 6-feature chain fixture; every
  cluster at higher K is `⊆` some cluster at lower K, feature
  counts weakly decrease.
- [x] 7.2 `plan_pareto([K])` matches
  `plan_with_target(K)` cluster-for-cluster and
  feature_ids-for-feature_ids.
- [x] 7.3 Sort-once instrumented via `unittest.mock.patch.object`
  on `Compressor._sort_pairs_by_score`: exactly 1 call per
  `plan_pareto` invocation regardless of `len(targets)`.
- [x] 7.4 `ParetoReport.to_json` / `from_json` round-trip via
  string AND via path; missing-key payload raises; non-string,
  non-path source raises `TypeError`.
- [x] 7.5 Per-K `reached_target`: on a 3-disjoint-pairs fixture
  reachable only to `n_features_kept = 3`,
  `plan_pareto([10, 2])` yields `reached_target = True` for K=10
  and `False` for K=2.
- [x] 7.6 `plan_pareto([])` and `plan_pareto(None)` both raise
  `ValueError`.
- [x] 7.7 `plan_pareto([1, 3, 2, 1, 2])` returns
  `targets == (3, 2, 1)` with 3 outcomes ordered to match.
  Also: `plan_pareto([0, 2])` and `plan_pareto([-1])` raise.

## Phase 3 — CLI + integration + release

### 8. CLI surface

- [ ] 8.1 `polygram compress` gains `--target-features N` (mutually
  exclusive with `--pareto`); plumbs through to
  `Compressor(..., config=CompressionConfig(target_n_features_kept=N, ...))`
  and invokes `Compressor.apply(plan=Compressor.plan_with_target())`.
- [ ] 8.2 `polygram compress` gains `--pareto K1,K2,K3,...`
  (comma-separated ints); always emits `<out>/pareto.json`. SAE
  materialisation is gated by `--pareto-materialize` (next task).
- [ ] 8.3 `polygram compress` gains `--pareto-materialize`. When
  passed alongside `--pareto`, the CLI writes
  `<out>/pareto/k_{K}.safetensors` for every K via
  `Compressor.apply(plan=outcome.plan)`. Without the flag, no
  safetensors are written.
- [ ] 8.4 `--score-field {polygram_overlap,jaccard,decoder_overlap}`
  flag with default `polygram_overlap`. Honoured by both
  `--target-features` and `--pareto` modes.
- [ ] 8.5 `polygram compress --help` documents all four flags and
  mentions:
  - the byte-identity guarantee for the threshold path,
  - the cluster-representative-count semantic for
    `--target-features`,
  - the materialisation-cost note for `--pareto`.

### 9. Integration test

- [ ] 9.1 End-to-end test: toy SAE fixture →
  `BehaviouralValidator` (or `DecoderGeometryConfirmer`) →
  `Compressor(config=CompressionConfig(target_n_features_kept=K))`
  → `plan_with_target()` → `apply(plan=...)` → reload compressed
  checkpoint → assert `CompressionReport.n_features_kept <= K`
  (or report `reached_target == False` if infeasible).
- [ ] 9.2 End-to-end Pareto test: same fixture →
  `plan_pareto([4, 2, 1])` → `apply(plan=outcome.plan)` for each
  outcome → assert nested compression (more zeroed rows at lower
  K).
- [ ] 9.3 CLI smoke test (subprocess via `pytest tmp_path`):
  - `polygram compress --target-features K` writes one
    safetensors + one report.
  - `polygram compress --pareto K1,K2` writes only
    `pareto.json` (no `pareto/` subdir).
  - `polygram compress --pareto K1,K2 --pareto-materialize`
    writes `pareto.json` + `pareto/k_{K1,K2}.safetensors`.
  - `polygram compress --target-features K --pareto K1,K2`
    exits non-zero with a mutual-exclusion error.

### 10. Spec validation & release

- [ ] 10.1 `openspec validate add-pareto-target-compression --strict`
  is green.
- [ ] 10.2 Bump `polygram.__version__` to `0.4.0`
  (`polygram/__init__.py:77` plus `pyproject.toml` if it
  duplicates the value). Minor bump: purely additive,
  byte-identity preserved.
- [ ] 10.3 `CHANGELOG.md` entry under a new `0.4.0` heading
  summarising target-K + Pareto path additions, the
  `CompressionPlan.n_features_kept` property, and the CLI flag
  set. Link the
  [`docs/research/rung4-viability-spike-v2.md`](../../../docs/research/rung4-viability-spike-v2.md)
  v2.2 follow-up call-out.
- [ ] 10.4 `openspec archive add-pareto-target-compression` after
  merge.

## Phase 4 — Out of scope (recorded so future work has a pointer)

- [ ] 11.1 **`rep_selection="recon_proxy"`** — picks cluster reps
  by reconstruction-loss attribution. Requires `Compressor` to
  accept activations or a per-feature attribution vector; real
  interface widening. Deferred.
- [ ] 11.2 **Iterative target-K in `EpochCompressor`** — needs a
  decision about whether the target should be enforced per
  iteration or only on the final dictionary. Out of scope here.
- [ ] 11.3 **KL-based score fields** — possible after a documented
  monotone transformation (e.g. `−kl_log_ratio_abs`). Deferred
  pending real consumer demand.
- [ ] 11.4 **SAE-wide feature-count target** (counts singletons
  too) — separate semantic; could be added as a second field if a
  real caller asks for it. See Decision 1.
- [ ] 11.5 **Automatic K selection** (elbow-point / knee
  detection on the Pareto curve). Deferred.
- [ ] 11.6 **sae-forge integration** — sae-forge bumps
  `polygram>=0.4.0` and consumes the new API in its own change.
  Not part of this proposal.
