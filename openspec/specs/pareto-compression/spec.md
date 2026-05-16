# pareto-compression Specification

## Purpose
TBD - created by archiving change add-pareto-target-compression. Update Purpose after archive.
## Requirements
### Requirement: CompressionPlan exposes n_features_kept

`CompressionPlan` SHALL expose `n_features_kept: int` as a
`@property` whose value equals `len(self.clusters)`. The property is
derived, not a stored field, so it MUST NOT affect serialization,
`__eq__`, or `__hash__` of `CompressionPlan` or
`CompressionReport`.

#### Scenario: n_features_kept on a threshold-mode plan

- **WHEN** `Compressor(report, ckpt).plan()` is called against
  a fixture that produces 5 clusters
- **THEN** the returned plan's `n_features_kept` is `5`, and the
  post-`apply()` `CompressionReport.n_features_kept` is also `5`

#### Scenario: n_features_kept on an empty-clusters plan

- **WHEN** `plan_with_target` returns a plan with no clusters
  (target K ≥ trivial no-compression count)
- **THEN** `plan.n_features_kept == 0`

### Requirement: Compressor exposes plan_with_target

`Compressor` SHALL expose
`plan_with_target(target_n_features_kept: int | None = None) -> CompressionPlan`.

- When the argument is `None`, the method reads
  `self.config.target_n_features_kept`. If both are `None`, it
  raises `ValueError`.
- The method SHALL ignore `validation_report.confirmed` entirely.
- Planning consumes `validation_report.pairs`, filtered to drop
  any pair whose `self.config.score_field` value is NaN
  (see Decision 5 in `design.md`), then sorted descending by that
  score field with `(min(i, j), max(i, j))` as the canonical
  tiebreak (Decision 6).
- The sorted list is processed via greedy union-find. After each
  union, the method computes the new representative count and
  stops when it first reaches
  `<= target_n_features_kept`. If the list is exhausted first,
  the most-compressed reachable plan is returned (Decision 10).
- If the NaN filter leaves zero pairs, `ValueError` is raised
  naming the chosen `score_field` and the report's score-field
  populations (Decision 5).
- Cluster representatives SHALL be picked via the same
  `_pick_representative` path used by `plan()`, so `rep_selection`
  and `representatives` overrides behave identically.

#### Scenario: plan_with_target reaches the requested K

- **WHEN** `Compressor(report, ckpt, config=CompressionConfig(target_n_features_kept=200)).plan_with_target()`
  runs against a `ValidationReport` whose pair list can union-find
  down to at most 150 components
- **THEN** the returned plan has `n_features_kept <= 200`

#### Scenario: plan_with_target reports infeasible targets

- **WHEN** `target_k` is lower than the minimum reachable
  representative count for the supplied pair list
- **THEN** `plan_with_target` returns the most-compressed
  reachable plan with `n_features_kept > target_k` rather than
  raising

#### Scenario: plan_with_target on a decoder-only report with behavioural score_field

- **WHEN** `plan_with_target` is called with
  `score_field="polygram_overlap"` on a `ValidationReport` whose
  pairs all have NaN `polygram_overlap` (produced by
  `DecoderGeometryConfirmer`)
- **THEN** `ValueError` is raised with a message that names
  `polygram_overlap` and explains that the report is decoder-only

#### Scenario: plan_with_target preserves byte-identity for the threshold path

- **WHEN** `Compressor(report, ckpt)` is constructed with default
  `CompressionConfig()` and `.plan().apply()` is called against the
  existing toy fixture
- **THEN** the resulting `CompressionReport.to_json()` output is
  byte-identical to the pre-change reference

#### Scenario: plan_with_target tiebreaks deterministically

- **WHEN** two candidate pairs share an identical `score_field`
  value
- **THEN** they are processed in `(min(i, j), max(i, j))` order so
  the resulting cluster ids are reproducible across runs

#### Scenario: plan_with_target raises when no target is provided

- **WHEN** `Compressor(report, ckpt).plan_with_target()` is called
  with neither an argument nor `config.target_n_features_kept` set
- **THEN** `ValueError` is raised naming both candidate sources

### Requirement: Compressor.apply accepts a plan override

`Compressor.apply` SHALL accept an optional
`plan: CompressionPlan | None = None` argument. When provided,
`apply` SHALL use the supplied plan instead of invoking `plan()`
internally. When omitted, behaviour is unchanged.

#### Scenario: apply with a plan_with_target output

- **WHEN** `compressor.apply(plan=compressor.plan_with_target(K))`
  is called
- **THEN** the resulting `CompressionReport.plan` is the supplied
  plan and `CompressionReport.n_features_kept == plan.n_features_kept`

#### Scenario: apply without plan (back-compat)

- **WHEN** `compressor.apply()` is called with no arguments
- **THEN** behaviour matches the pre-change implementation
  byte-for-byte on the existing toy fixture

### Requirement: Compressor exposes plan_pareto

`Compressor` SHALL expose
`plan_pareto(targets: Sequence[int]) -> ParetoReport`.

- `targets` SHALL be deduplicated and sorted descending in the
  returned `ParetoReport` (Decision 11).
- The method SHALL sort `validation_report.pairs` once by
  `self.config.score_field` descending (with the same NaN filter
  and tiebreak rule as `plan_with_target`) and, for each K in the
  deduplicated/sorted list, walk the prefix of the sorted list and
  materialise a `CompressionPlan`.
- The shared sort and shared union-find state amortise across all
  K, so total cost is `O(N log N)` for the sort plus `O(N α(N))`
  for the union-find walk, independent of `len(targets)`.

#### Scenario: plan_pareto yields nested plans

- **WHEN** `plan_pareto([2000, 1000, 500, 200])` is called on a
  fixture whose pair list can reach K=200
- **THEN** the returned `ParetoReport.outcomes` has length 4,
  feature counts are weakly decreasing across outcomes, and every
  cluster at a higher K is a subset of some cluster at a lower K

#### Scenario: plan_pareto with one target matches plan_with_target

- **WHEN** `plan_pareto([K])` and `plan_with_target(K)` are called
  with otherwise identical config
- **THEN** the single plan in the returned `ParetoReport.outcomes`
  equals (byte-identical via cluster comparison) the plan returned
  by `plan_with_target(K)`

#### Scenario: plan_pareto deduplicates and sorts targets

- **WHEN** `plan_pareto([500, 200, 1000, 500])` is called
- **THEN** the returned `ParetoReport.targets` is `(1000, 500, 200)`
  and `outcomes` has length 3

#### Scenario: plan_pareto rejects empty targets

- **WHEN** `plan_pareto([])` is called
- **THEN** `ValueError` is raised naming the empty `targets`
  argument

#### Scenario: plan_pareto sorts pairs exactly once

- **WHEN** `plan_pareto([1000, 500, 200])` is called with an
  instrumented sort spy
- **THEN** the spy records exactly one invocation of the pair
  sort, regardless of `len(targets)`

### Requirement: ParetoReport dataclass

The `polygram.compression.pareto.ParetoReport` SHALL be a frozen
dataclass with fields:

- `schema_version: int` (currently `1`)
- `sae_checkpoint: pathlib.Path`
- `sae_checkpoint_sha256: str`
- `score_field: str`
- `targets: tuple[int, ...]` (deduplicated, sorted descending)
- `outcomes: tuple[ParetoOutcome, ...]` (one per `targets[i]`)

The `ParetoOutcome` SHALL be a frozen dataclass with fields:

- `target_k: int`
- `reached_target: bool` — `True` iff the greedy walk produced a
  plan whose `n_features_kept <= target_k` *via* the per-K stop
  rule (i.e. the trajectory exceeded `target_k` and then dropped
  back). When `target_k` exceeds the observed peak `n_clusters`,
  the walk never had to compress down to K; the returned plan is
  the *peak-state* snapshot (most-decomposed observed) and
  `reached_target` is `False` so callers can distinguish
  "target unreachable from above" from a true greedy stop. When
  `target_k` is below the trajectory's terminal `n_clusters`
  (infeasibly small), the returned plan is the final-state
  snapshot and `reached_target` is `False`.
- `plan: CompressionPlan`

`ParetoReport` SHALL expose `.to_json(path: str | os.PathLike | None = None) -> str`
and `ParetoReport.from_json(source: str | os.PathLike) -> ParetoReport`
mirroring `CompressionReport`'s hand-coded serializer
(Decision 8). The cluster encoding SHALL reuse
`_cluster_to_dict` / `_cluster_from_dict` from
`polygram/compression/report.py`. Both `ParetoReport` and
`ParetoOutcome` SHALL be exported from `polygram.__init__` and
`polygram.compression.__init__`.

#### Scenario: ParetoReport round-trips via to_json / from_json

- **WHEN** a `ParetoReport` produced by
  `plan_pareto([K1, K2, K3])` is serialised via `.to_json()` and
  reconstructed via `ParetoReport.from_json(...)`
- **THEN** the reconstructed instance equals the original

#### Scenario: ParetoReport is importable from polygram

- **WHEN** `from polygram import ParetoReport, ParetoOutcome` is
  executed
- **THEN** both names resolve to the dataclasses defined in
  `polygram.compression.pareto`

#### Scenario: reached_target reflects per-K outcome

- **WHEN** `plan_pareto([2000, 150, 50])` runs on a fixture whose
  trajectory peaks at `n_clusters == 200` and terminates at
  `n_clusters == 100`
- **THEN** `outcomes[0].reached_target` is `False` (K=2000 exceeds
  the observed peak — the returned plan is the peak-state snapshot,
  not a greedy stop), `outcomes[1].reached_target` is `True`
  (K=150 reached via the per-K stop rule), and
  `outcomes[2].reached_target` is `False` (K=50 below terminal
  cluster count — infeasible)

#### Scenario: target_k above observed peak returns peak-state plan

- **WHEN** `plan_pareto([K])` is called with `K` strictly greater
  than the observed peak `n_clusters` during the union-find walk
- **THEN** the returned `outcomes[0].plan` is the snapshot of
  `parent` taken at the iteration where `n_clusters` first reached
  its peak (not the final post-walk state, which may have merged
  further), and `outcomes[0].reached_target` is `False`

### Requirement: CLI exposes target-features and pareto flags

The `polygram compress` CLI subcommand SHALL accept four new
flags:

- `--target-features N` — single-shot target-K compression.
  Plumbs to `CompressionConfig(target_n_features_kept=N, ...)` and
  invokes `Compressor.plan_with_target()` followed by
  `Compressor.apply(plan=...)`.
- `--pareto K1,K2,K3,...` — comma-separated integer K list. Invokes
  `Compressor.plan_pareto(...)` and writes a `pareto.json` artifact
  under the output directory. SAE materialisation is gated by
  `--pareto-materialize` (below).
- `--pareto-materialize` — opt-in. When passed alongside
  `--pareto`, the CLI also writes one materialised SAE per K under
  `<out>/pareto/k_{K}.safetensors` (Decision 9).
- `--score-field {polygram_overlap,jaccard,decoder_overlap}` —
  defaults to `polygram_overlap`. Honoured by both
  `--target-features` and `--pareto` modes.

`--target-features` and `--pareto` SHALL be mutually exclusive.

#### Scenario: --target-features produces a single compressed SAE

- **WHEN** `polygram compress --sae-checkpoint <ckpt> --validation-report <report> --target-features 200 --output <dir>`
  is invoked
- **THEN** the CLI exits 0 and writes a compressed SAE checkpoint
  plus a `CompressionReport` JSON whose `n_features_kept <= 200`

#### Scenario: --pareto without --pareto-materialize writes plan only

- **WHEN** `polygram compress --sae-checkpoint <ckpt> --validation-report <report> --pareto 4,2,1 --output <dir>`
  is invoked
- **THEN** the CLI exits 0 and writes `<dir>/pareto.json` and no
  `<dir>/pareto/k_*.safetensors` files

#### Scenario: --pareto with --pareto-materialize writes nested artifacts

- **WHEN** `polygram compress --sae-checkpoint <ckpt> --validation-report <report> --pareto 4,2,1 --pareto-materialize --output <dir>`
  is invoked
- **THEN** the CLI exits 0 and writes `<dir>/pareto.json`,
  `<dir>/pareto/k_4.safetensors`, `<dir>/pareto/k_2.safetensors`,
  `<dir>/pareto/k_1.safetensors`

#### Scenario: --target-features and --pareto are mutually exclusive

- **WHEN** both `--target-features` and `--pareto` are supplied
- **THEN** the CLI exits non-zero with an error message naming the
  conflict

