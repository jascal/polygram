## Phase 1 — Target-K compression

### 1. CompressionPlan derived `n_features_kept` property

- [ ] 1.1 Add `@property def n_features_kept(self) -> int` to
  `CompressionPlan` (`polygram/compression/report.py`) returning
  `len(self.clusters)`. Property must be on the frozen dataclass
  without triggering `eq` / `hash` recomputation.
- [ ] 1.2 Verify `CompressionReport.to_json()` output is unchanged
  (the report-level `n_features_kept` is already serialized;
  adding the plan-level property MUST NOT change report payload
  shape).
- [ ] 1.3 Regression test: `Compressor(report, ckpt).plan().n_features_kept`
  equals the post-`apply()` `CompressionReport.n_features_kept`
  on the existing toy fixture.

### 2. CompressionConfig field additions

- [ ] 2.1 Add `target_n_features_kept: int | None = None` to
  `CompressionConfig` (`polygram/config.py:251`). Docstring SHALL
  explicitly state the cluster-representative-count semantic from
  Decision 1.
- [ ] 2.2 Add `score_field: str = "polygram_overlap"` to
  `CompressionConfig`.
- [ ] 2.3 Extend `_SUPPORTED_SCORE_FIELDS = ("polygram_overlap", "jaccard", "decoder_overlap")`
  module constant (or local equivalent inside `config.py`).
- [ ] 2.4 Extend `__post_init__` range/value validation:
  `target_n_features_kept` is `None` or `>= 1`; `score_field` in
  `_SUPPORTED_SCORE_FIELDS`. Error messages name the field and
  list valid values.
- [ ] 2.5 Tests: round-trip through `_ConfigMixin.to_dict` /
  `_ConfigMixin.from_dict` covers new fields automatically; add
  explicit round-trip assertion + a `from_dict` with-missing-keys
  test.

### 3. Compressor target-K planning

- [ ] 3.1 Add a private `_filter_pairs_for_score(pairs, score_field) -> tuple[CandidatePair, ...]`
  helper in `compression/compressor.py` that drops pairs whose
  `getattr(pair, score_field)` is NaN (per Decision 5). Raises
  `ValueError` if the filtered list is empty, naming the score
  field and noting the `DecoderGeometryConfirmer`-vs-`BehaviouralValidator`
  distinction.
- [ ] 3.2 Add a private `_greedy_union_to_target(pairs, score_field, target_k) -> CompressionPlan`
  helper. Sort the filtered pair list descending by
  `(−getattr(pair, score_field), min(i, j), max(i, j))` (Decision 6).
  Walk the sorted list through union-find; after each union,
  recompute the distinct-component count and stop when it first
  reaches `<= target_k`. Materialise `ClusterPlan` objects exactly
  like the existing `_build_plan` (same `_pick_representative`
  call site).
- [ ] 3.3 Add public
  `Compressor.plan_with_target(target_n_features_kept: int | None = None) -> CompressionPlan`.
  Reads from `self.config.target_n_features_kept` if the argument
  is omitted; raises `ValueError` if neither source provides one.
- [ ] 3.4 `Compressor.plan_with_target()` uses the same
  `_pick_representative` path as `plan()` so `rep_selection` and
  `representatives` overrides behave identically.
- [ ] 3.5 `Compressor.apply()` gains an optional
  `plan: CompressionPlan | None = None` argument. When supplied,
  `apply` skips the internal `plan()` call. Existing arity is
  preserved (no breaking change).

### 4. Phase 1 tests

- [ ] 4.1 `plan_with_target(target_k=N)` on a fixture with `M > N`
  reachable components returns a plan with `n_features_kept <= N`.
- [ ] 4.2 `plan_with_target(target_k=1)` on a fixture whose
  components can't all be merged returns the most-compressed
  reachable plan; `plan.n_features_kept > 1` is OK (no exception).
- [ ] 4.3 `plan_with_target(target_k=len(feature_ids))` returns an
  empty-clusters plan (no compression).
- [ ] 4.4 Ordering determinism: pairs with identical scores
  tiebreak on `(min(i, j), max(i, j))` and produce reproducible
  cluster ids across runs.
- [ ] 4.5 `score_field="jaccard"` and `score_field="decoder_overlap"`
  produce different but valid plans on a fixture where the score
  columns diverge.
- [ ] 4.6 **NaN-only behavioural fields** (decoder-only report from
  `DecoderGeometryConfirmer`):
  `plan_with_target(score_field="polygram_overlap")` raises
  `ValueError` naming the score field. The same call with
  `score_field="decoder_overlap"` succeeds.
- [ ] 4.7 Byte-identity regression:
  `Compressor(report, ckpt).plan().apply()` output is unchanged
  for the existing toy fixture; assert via
  `CompressionReport.to_json()` against a frozen reference string.
- [ ] 4.8 `CompressionConfig(target_n_features_kept=0)` raises
  `ValueError`.
- [ ] 4.9 `CompressionConfig(score_field="bogus")` raises
  `ValueError`; also test `score_field="kl_log_ratio_abs"`
  (a real CandidatePair field that is deliberately excluded).
- [ ] 4.10 `Compressor.plan_with_target()` with no argument and no
  config setting raises `ValueError`.
- [ ] 4.11 `Compressor.apply(plan=...)` with a `plan_with_target`
  output produces a `CompressionReport` whose `plan` is the
  supplied plan and whose `n_features_kept` equals
  `plan.n_features_kept`.

## Phase 2 — Pareto path artifact

### 5. ParetoOutcome + ParetoReport dataclasses

- [ ] 5.1 Create `polygram/compression/pareto.py`.
- [ ] 5.2 Define `@dataclass(frozen=True) class ParetoOutcome`
  with fields `target_k: int`, `reached_target: bool`,
  `plan: CompressionPlan`.
- [ ] 5.3 Define `@dataclass(frozen=True) class ParetoReport` with
  fields `schema_version: int`, `sae_checkpoint: Path`,
  `sae_checkpoint_sha256: str`, `score_field: str`,
  `targets: tuple[int, ...]`, `outcomes: tuple[ParetoOutcome, ...]`.
- [ ] 5.4 `ParetoReport.to_json(path=None) -> str` and
  `ParetoReport.from_json(source) -> ParetoReport` mirroring
  `CompressionReport`'s hand-coded serializer. Reuse
  `_cluster_to_dict` / `_cluster_from_dict` from
  `polygram/compression/report.py`.
- [ ] 5.5 Export `ParetoReport`, `ParetoOutcome` from
  `polygram/__init__.py` and `polygram/compression/__init__.py`.

### 6. Compressor.plan_pareto

- [ ] 6.1 Add
  `Compressor.plan_pareto(targets: Sequence[int]) -> ParetoReport`.
  Empty / `None` `targets` raises `ValueError` with a usage hint.
  Duplicate K values in `targets` are deduplicated; output
  `ParetoReport.targets` is sorted descending for stable
  iteration.
- [ ] 6.2 The method SHALL sort pairs once (same NaN filter,
  same tiebreak rule as Phase 1) and walk the prefix for each K
  in sorted-descending order, sharing union-find state.
- [ ] 6.3 Per-K `ParetoOutcome.reached_target` is `True` iff the
  resulting plan's `n_features_kept <= target_k`.

### 7. Phase 2 tests

- [ ] 7.1 `plan_pareto([2000, 1000, 500, 200])` on a fixture returns
  4 outcomes whose feature counts are weakly decreasing and whose
  cluster structures are nested (every cluster at K=1000 is a
  subset of some cluster at K=500).
- [ ] 7.2 `plan_pareto([K])` matches `plan_with_target(K)` cluster-
  for-cluster (verify via `plan.clusters == outcomes[0].plan.clusters`
  and same `feature_ids`).
- [ ] 7.3 Sort-once efficiency: instrument with a counter / spy
  that verifies the pair sort is invoked exactly once across an
  N-target call (no per-K resort).
- [ ] 7.4 Round-trip: `ParetoReport.from_json(report.to_json()) == report`.
- [ ] 7.5 `reached_target` reflects per-K outcome: on a fixture
  reachable to K=100 but not K=50,
  `plan_pareto([2000, 50])` produces
  `outcomes[0].reached_target = True`,
  `outcomes[1].reached_target = False`.
- [ ] 7.6 `plan_pareto([])` raises `ValueError`.
- [ ] 7.7 `plan_pareto([500, 200, 1000, 500])` returns
  `targets == (1000, 500, 200)` with 3 outcomes.

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
