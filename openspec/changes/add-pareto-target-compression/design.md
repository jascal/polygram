## Context

`Compressor.plan()`
([`polygram/compression/compressor.py:187`](../../../polygram/compression/compressor.py))
consumes `validation_report.confirmed` — a tuple of `(i, j)` pairs
already filtered by `BehaviouralValidator`'s gate. Union-find over
`confirmed` produces clusters; one representative per cluster
survives, the rest are zeroed (or merged via
`strategy="merge"`). The `Compressor` reports
`n_features_kept = len(plan.clusters)` (number of cluster
representatives, not "total surviving SAE features" — see Decision 1).

The pairs that *did not* pass the gate are still present in
`validation_report.pairs` with their full 14 score fields
(`polygram_overlap`, `jaccard`, `n_fires_i/j`, `kl_*`, etc. — see
[`polygram/behavioural/report.py:62`](../../../polygram/behavioural/report.py)).
They are simply unused at compression time.

A target-K caller wants to say: *"I don't care about the threshold.
Give me clusters that leave exactly K representatives standing."*
The data is there; we just need a different reducer over it.

The key structural observation is **nestedness**: if you sort pairs
by descending score and process them greedily through union-find,
the cluster structure at any prefix length is a refinement of the
structure at any longer prefix. So K=2000, K=1000, K=500, K=200 all
share the high-confidence pairs at the top of the sorted list. One
sort suffices for the whole Pareto path.

## Goals / Non-Goals

**Goals:**
- Caller can request a specific `n_features_kept` and get back a
  deterministic `CompressionPlan` that achieves it (or first crosses
  it from above).
- Caller can request a Pareto path over a list of K values and get
  back a nested sequence of plans from a single sort.
- The historical threshold-driven path is byte-identical when the
  new fields are unset.
- The score axis is configurable
  (`polygram_overlap` default; `jaccard` and `decoder_overlap` as
  alternatives).
- CLI exposes both modes (`--target-features`, `--pareto`), with
  a separate flag to gate materialisation cost.

**Non-Goals:**
- Changing `ValidationReport`, `BehaviouralValidator`, `Confirmer`,
  `EpochCompressor`, or `Regrower`.
- Recon-aware `rep_selection` (deferred — requires activation
  pass-through into Compressor).
- Automatic K selection (e.g. "elbow-point" detection). The caller
  chooses K.
- Coordinating with `EpochCompressor`'s iterative loop. Target-K
  mode is a single-shot path; `EpochCompressor` is unchanged and
  continues to drive threshold-mode iterations.

## Decisions

### Decision 1 — `target_n_features_kept` matches the existing report semantic

The proposal's `target_n_features_kept` is compared against
`CompressionReport.n_features_kept`, which is currently computed as
`sum(1 for _ in plan.clusters)` (see
[`compressor.py:385`](../../../polygram/compression/compressor.py)) — i.e.
the count of cluster *representatives*, not the count of features
that remain in the SAE checkpoint after `apply()`.

A typical real SAE has N=24576 features, of which only a small
fraction appear in confirmed pairs. If 200 clusters cover 1000
features (zeroing 800), then:
- `n_features_zeroed = 800`
- `n_features_kept = 200` (the existing report)
- Total surviving features in the SAE = N − 800 = 23776 (23576
  singletons + 200 representatives).

A user-friendly "compress to ~K features" target might naturally
mean the SAE-wide surviving count (23776 above), but that's
**dominated by singletons** and gives the user almost no control:
the only knob is the zeroed count. Matching the existing
`n_features_kept` semantic — count of cluster representatives —
gives the caller the actually-controllable dial and stays
consistent with the field they already read from
`CompressionReport`. This is the choice made here.

**Documentation requirement**: the `CompressionConfig.target_n_features_kept`
docstring SHALL explicitly state this semantic so callers don't
miscount.

**Alternative considered**: define a new SAE-wide
`n_features_surviving = len(plan.feature_ids) - n_features_zeroed
+ (N − len(plan.feature_ids))` and target on that. Rejected —
adds a second metric, callers would have to know which they
want, and it doesn't reflect a controllable degree of freedom.

### Decision 2 — Add a new method, don't overload `plan()`

`plan()` keeps consuming `confirmed`. Target-K mode is a new
method `plan_with_target()`, dispatched from a thin wrapper that
reads `config.target_n_features_kept`. This keeps the
byte-identity guarantee trivial — if you don't set the field, you
never enter the new code path. It also keeps the two algorithms
readable: one consumes a filtered set, the other consumes a
sorted list with a cut.

**Alternative considered**: overload `plan()` to branch internally.
Rejected — invisible behaviour change for callers who supply both
`confirmed` and a target.

### Decision 3 — Three score axes; KL fields deliberately excluded

`CandidatePair` has 14 fields. Only three are exposed as valid
`score_field` values:

- `polygram_overlap` (default) — primary behavioural-gate signal
  used by `BehaviouralValidator`
  (`ValidationConfig.polygram_overlap_threshold = 0.7`). Bounded
  `[0, 1]`, larger = more similar.
- `jaccard` — co-firing set similarity. Bounded `[0, 1]`. Cheaper
  to compute; useful when polygram overlap is noisy on
  very-sparse SAEs.
- `decoder_overlap` — decoder cosine². Bounded `[0, 1]`. Populated
  by `DecoderGeometryConfirmer` for behavioural-free runs (where
  the behavioural fields land as NaN — see Decision 5).

Excluded:
- `n_fires_i`, `n_fires_j`, `n_both_fire`, `n_either_fire` — counts,
  not similarity scores.
- `i`, `j` — feature ids.
- `kl_ablate_i`, `kl_ablate_j`, `kl_ratio_paired`,
  `kl_log_ratio_abs` — KL-based fields are unbounded above and
  *smaller* = more similar. They could plausibly be added later
  as `-kl_log_ratio_abs` after a documented sign-flip, but the
  surface complexity isn't worth it pre-evidence.
- `pearson_activation` — `[-1, 1]` range; negative values would
  flip the greedy ordering. Could be added later with an explicit
  monotone transformation.
- `gate_pass` — boolean.

**Alternative considered**: weighted-sum score. Rejected — adds a
hyperparameter; the three single-field options are interpretable
and cover the cases.

### Decision 4 — Greedy union-find on a sorted pair list

The greedy reducer is deterministic, `O(N log N)` for the sort
plus `O(N α(N))` for union-find, and produces a nested family of
cluster structures over prefix lengths. A search (e.g. "find
thresholds that yield K=200") would be slower per query,
non-deterministic across score tiebreaks, and would not give
nestedness for free.

**Stopping rule**: process pairs in descending score order; after
each union, compute the new representative count as

```
n_features_kept = number_of_distinct_components_seen_so_far
```

where "components seen so far" counts each connected component
exactly once. Stop when `n_features_kept <= target_n_features_kept`.
The returned plan is the first one that achieves *at most* the
target — this is the natural Pareto convention (closer-to-the-frontier
wins).

**Alternative considered**: bisect on score thresholds. Rejected —
slower, non-nested, and doesn't compose for `plan_pareto`.

### Decision 5 — NaN-safety in the score sort

`DecoderGeometryConfirmer`
([`polygram/confirmation/decoder_geometry.py:30`](../../../polygram/confirmation/decoder_geometry.py))
populates only `decoder_overlap` and leaves all behavioural score
fields as NaN. If a caller asks
`Compressor.plan_with_target(score_field="polygram_overlap")` on
a decoder-only report, the sort would land in undefined-behaviour
territory (Python sorts NaN inconsistently and the result depends
on prior list order).

**Policy**: `plan_with_target` and `plan_pareto` SHALL filter out
pairs whose chosen `score_field` is NaN *before* sorting. If the
filter leaves zero pairs, both methods raise `ValueError` naming
the chosen score field and pointing at the
`DecoderGeometryConfirmer`-vs-`BehaviouralValidator` distinction.

**Alternative considered**: silently treat NaN as `−∞` and process
last. Rejected — masks misconfiguration. The explicit error makes
the caller pick a compatible score field.

### Decision 6 — Determinism via canonical tiebreak

Two candidate pairs with identical `score_field` value need a
stable tiebreak so cluster ids are reproducible across runs.

**Rule**: sort by `(−score, min(i, j), max(i, j))`. All three score
axes are symmetric in `(i, j)` (polygram_overlap, jaccard, and
decoder_overlap are defined over unordered pairs), so the
`(min, max)` canonicalisation gives a total order without
ambiguity. If a future asymmetric score is added (e.g. a KL
variant), this decision is revisited.

### Decision 7 — `pareto_reached_target` belongs to `ParetoReport`, not `CompressionPlan`

The proposal's earlier draft added a `pareto_reached_target: bool`
field directly to `CompressionPlan`. This has two costs:
- It changes the shape of `CompressionPlan`, which `CompressionReport`
  embeds and serializes by-hand in `_serialize` / `from_json`.
  Adding a field with a `True` default would require either editing
  the serializer to write/read it or accepting that the field is
  invisible across serialization (silent data loss).
- The "did we reach the target?" question is a property of a
  *target-K planning operation*, not of the plan itself. A
  `CompressionPlan` from threshold mode has no target to reach.

**Resolution**:
- `CompressionPlan` gets a `@property def n_features_kept(self) -> int`
  (`= len(self.clusters)`). This is a derived value, no impact on
  storage or serialization.
- `plan_with_target()` returns just a `CompressionPlan`. Callers can
  trivially check `plan.n_features_kept <= target_k`.
- `ParetoReport.outcomes: tuple[ParetoOutcome, ...]` — one
  `ParetoOutcome(target_k, plan, reached_target)` per requested K.
  This is the place "reached the target" lives, because the report
  knows the targets.

### Decision 8 — `ParetoReport` serialization mirrors `CompressionReport`

The first draft claimed `ParetoReport` round-trips "via the existing
`CompressionPlan.to_dict()` machinery". That machinery doesn't
exist — `CompressionPlan` has no `to_dict` /`from_dict`, only the
parent `CompressionReport` has a hand-coded `to_json` / `from_json`
serializer.

**Resolution**: `ParetoReport` implements its own `to_json` /
`from_json` mirroring `CompressionReport`'s pattern. The payload
shape:

```json
{
  "schema_version": 1,
  "sae_checkpoint": "...",
  "sae_checkpoint_sha256": "...",
  "score_field": "polygram_overlap",
  "outcomes": [
    {
      "target_k": 2000,
      "reached_target": true,
      "clusters": [...],
      "feature_ids": [...]
    },
    ...
  ]
}
```

The cluster / feature_ids encoding reuses the existing
`_cluster_to_dict` / `_cluster_from_dict` helpers from
`polygram/compression/report.py`.

### Decision 9 — CLI `--pareto-materialize` is opt-in

Writing one safetensors file per K in a Pareto sweep is expensive
(hundreds of MB to multiple GB per K for production SAEs) and
most callers want to inspect the curve before picking a K.

`polygram compress --pareto K1,K2,K3` writes only the `pareto.json`
artifact by default. `--pareto-materialize` opts in to writing
`<out>/pareto/k_{K}.safetensors` for every K. This separates the
cheap operation (planning, ~ms) from the expensive operation
(SAE rewrite + disk I/O, multiple seconds per K).

**Alternative considered**: always materialise. Rejected — easy
to footgun callers into a multi-GB disk write they didn't intend.

### Decision 10 — `target_n_features_kept` semantics under infeasible targets

If the target K is **higher** than the minimum reachable
representative count for the supplied pair list (i.e. if the
algorithm finishes processing all pairs before the cluster count
drops to `target_k`), the algorithm returns the most-compressed
plan it can reach. The resulting `n_features_kept` is `> target_k`.

For `plan_with_target()`, callers detect this by
`plan.n_features_kept > target_k`. No exception is raised — this
is a common case in sweeps.

For `plan_pareto()`, the `ParetoOutcome.reached_target` flag
records the result per K.

If the target K is **higher** than the trivial-no-compression
count (`len(feature_ids)` — every feature is its own component),
the algorithm short-circuits and returns a plan with zero
clusters. `reached_target = True` (trivially).

**Alternative considered**: raise on infeasible target. Rejected —
common in sweeps; better to return the closest achievable plan and
let the caller decide.

### Decision 11 — Phased delivery

The work is split into three phases (see [`tasks.md`](tasks.md)):

- **Phase 1** — Config fields, `CompressionPlan.n_features_kept`
  property, `plan_with_target()`, byte-identity tests. Self-contained,
  no new module.
- **Phase 2** — `pareto.py` module with `ParetoReport` + `ParetoOutcome`
  + `plan_pareto()`. Builds on Phase 1's greedy reducer.
- **Phase 3** — CLI flags (`--target-features`, `--pareto`,
  `--pareto-materialize`, `--score-field`) + integration tests +
  release prep.

Each phase ends with a green test suite; the version bump and
changelog entry happen at the end of Phase 3. Downstream consumers
(sae-forge) can adopt after Phase 3 lands.

## Risks / Trade-offs

- **Aggressive K at the bottom of the curve**: target-K mode
  unions pairs that threshold-mode would reject. At very low K
  (high compression ratios), this *will* be noisier than threshold
  mode. Documented in the `CompressionConfig.target_n_features_kept`
  docstring; callers can use threshold mode as a floor by feeding
  the threshold-filtered subset of `pairs` if they care.
- **Rep selection unchanged**: clusters discovered greedily may
  not be ideally represented by the existing `n_fires` /
  `scale_aware` reps (see
  [`polygram/config.py:50`](../../../polygram/config.py) for the
  current options). This is the motivation for the deferred
  `recon_proxy` rep_selection; for now, target-K mode inherits
  whatever rep_selection the caller chose.
- **Non-monotonic faithfulness**: KL vs. K is empirically
  non-monotonic at default knobs (sae-forge observation). This
  change makes the non-monotonicity *visible* — it does not fix
  it. Fixing it is the `rep_selection` follow-up.
- **`n_features_kept` semantic surprises**: the
  cluster-representative-count semantic (Decision 1) may surprise
  callers expecting SAE-wide feature counts. Docstring + a CLI
  `--help` note are the mitigation; a future change could add a
  SAE-wide target if real callers ask for it.
- **NaN-only pair sets**: a caller pointing target-K at a
  decoder-only report and choosing
  `score_field="polygram_overlap"` gets a `ValueError` (Decision 5).
  Mitigation: error message names both the score field and the
  report's score-field populations.
