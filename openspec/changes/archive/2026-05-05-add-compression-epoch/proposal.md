## Why

PR #27 (validator) and PR #28 / commit `7bdc7e7` (compression action)
closed the redundancy-elimination loop for one panel of ≤ 8 SAE
features. Live runs on GPT-2 small `blocks.10` reproduce the §4.4
calibration cleanly:

- §4.4 panel: Spearman(Polygram, Jaccard) = +0.6371, 8 confirmed pairs
  (2 near-pairs + 4-clique), 5/8 features zeroed (62.5%); post-
  compression confirmed = 0.
- Fresh anchor `feat_7836` panel: 13 confirmed pairs (3-triangle +
  5-clique), 6/8 features zeroed (75%); post-compression confirmed = 0.

The gap, named explicitly in `add-compression-action`'s §5.2 closure
block: the SAE has 24,576 features but the validator can only examine
**8 per panel** (`MAX_FEATURES_PER_DICTIONARY = 8` in
`polygram/sae_import.py:23`, a structural property of the rung-1 MPS
encoding). Disjoint partitions miss every redundancy whose members
fall in different panels; exhaustive C(24576, 8) surveying is
infeasible. Zeroing is non-local — silencing one cluster member can
expose new redundancies among the survivors. The right shape is
**budgeted statistical clustering with fixed-point iteration**,
structurally analogous to incremental garbage collection but driven
by sampled behavioural signals over a held-out prompt set rather than
deterministic reference graphs.

This change ships the *epoch orchestrator* that scales the validator +
compressor loop across many panels while respecting the structural
8-feature cap, the non-local mutation semantics, the compute budget,
and the existing `Compressor`'s component-first contract.

## What Changes

### `compression` capability — extended

Add `polygram.compression.EpochCompressor`, a torch-free orchestrator
that composes existing primitives:

- **Two-stage API**: `select_panels() -> list[Panel]` (cheap; reads
  decoder weights + a single prompt forward pass to compute firing
  rates) and `run(output_checkpoint=...) -> EpochResult` (expensive;
  invokes `BehaviouralValidator.run()` per panel, aggregates
  confirmed pairs, hands a synthetic multi-panel `ValidationReport`
  to the existing `Compressor`, iterates to fixed point, writes one
  final compressed checkpoint).
- **Panel selection**: greedy seeded coverage. Sort eligible features
  (firing rate ≥ `min_firing_rate`) by `firing_rate × decoder_norm`
  descending; iterate this priority queue. Each anchor builds a
  panel = anchor + 7 nearest-cosine neighbours drawn from the same
  eligible pool (excluding already-zeroed features). Stop when the
  coverage metric — fraction of `(i, j)` pairs with
  `cos(W_dec[i], W_dec[j]) ≥ cosine_threshold` that share at least
  one panel — reaches `coverage_target`, OR the priority queue is
  exhausted, OR `n_panels_max` is hit. Each feature appears in at
  most `n_visits_per_feature` panels.
- **Cross-panel aggregation**: a pair `(i, j)` confirmed in *any*
  panel is confirmed in the synthetic report. Per-pair statistics
  use the maximum Polygram overlap and maximum Jaccard across panels
  (panel-composition-dependent), summed `n_both_fire` and
  `n_either_fire` (counts), and weighted-mean ablation-KL ratio
  (weighted by `n_both_fire`). Per-cluster representative selection
  sums `n_fires` over *every* panel that touched each cluster member,
  not just from pairs naming that member — a refinement on top of
  `Compressor`'s within-cluster aggregation that would otherwise
  systematically under-count members appearing in many panels.
- **Fixed-point iteration**: after one compression apply, re-run
  panel selection and validation against the rewritten checkpoint;
  iterate. Convergence: same set of compressed feature ids across
  two consecutive iterations (`convergence_reason='stable_clusters'`).
  Hard cap: `max_iterations` (default 5). Quality guard: per-token
  cross-entropy on the prompt set must not increase by more than
  `quality_delta_multiplier × first_iteration_delta` (default
  multiplier 2.0); breach triggers
  `convergence_reason='quality_bound_breached'` and reverts to the
  prior iteration's checkpoint.
- **Skip-zeroed bookkeeping**: the orchestrator maintains a
  `zeroed: set[int]` updated after each `Compressor.apply()`. Panel
  selection excludes those features (saves the validator's per-
  feature ablation budget on features that fire on 0 tokens).

### Report types — new

- **`Panel`** dataclass: `panel_id`, `feature_ids: tuple[int, ...]`
  (length ≤ 8), `anchor: int`, `cosines_to_anchor:
  tuple[float, ...]`.
- **`EpochIteration`** dataclass: `iteration: int`, `panels:
  tuple[Panel, ...]`, `validation_report_paths:
  tuple[Path, ...]`, `confirmed_pair_count: int`, `clusters_compressed:
  int`, `features_zeroed_this_iteration: tuple[int, ...]`,
  `cross_entropy_delta: float`, `convergence_state: str` (one of
  `'continuing'`, `'stable_clusters'`, `'max_iterations'`,
  `'quality_bound_breached'`, `'no_more_priority_candidates'`).
- **`EpochReport`** dataclass: `schema_version`,
  `source_checkpoint_sha256`, `output_checkpoint`,
  `output_checkpoint_sha256`, `iterations: tuple[EpochIteration, ...]`,
  `convergence_reason: str`, `n_features_zeroed_total: int`,
  `n_panels_total: int`, `coverage_achieved: float`,
  `wall_seconds: float`. JSON round-trip via the same
  `format(v, ".6g")` pattern as `ValidationReport` /
  `CompressionReport`.
- **`EpochResult`** dataclass (process bundle): `report`,
  `output_checkpoint`, `final_dictionary` (rebuilt via
  `from_sae_lens` on the rewritten checkpoint).

### `cli` capability — new `polygram compress-epoch` subcommand

`polygram compress-epoch` wraps `EpochCompressor.run()`:

```
polygram compress-epoch \
  --sae-checkpoint path/to/sae_weights.safetensors \
  --prompts path/to/prompts.txt \
  --output-checkpoint path/to/sae_weights.epoch-compressed.safetensors \
  --output path/to/epoch_report.json \
  [--layer 10] \
  [--model gpt2] \
  [--strategy zero] \
  [--device auto] \
  [--coverage-target 0.95] \
  [--cosine-threshold 0.30] \
  [--n-visits-per-feature 3] \
  [--n-panels-max 1000] \
  [--min-firing-rate 0.01] \
  [--max-iterations 5] \
  [--quality-delta-multiplier 2.0] \
  [--polygram-threshold 0.7] \
  [--jaccard-threshold 0.30] \
  [--min-both-fire 5] \
  [--save-intermediate-reports]
```

`--save-intermediate-reports` writes each iteration's per-panel
`ValidationReport`s alongside the `EpochReport` (default off; reports
linger only as paths in the `EpochReport.iterations[k].validation_report_paths`).

### No new optional extra

Reuses `[behavioural]` (validator) + base `safetensors` (compressor +
orchestrator). The orchestrator itself is torch-free; only the
delegated `BehaviouralValidator.validate()` calls pull torch in.

## What this proposal explicitly does NOT do

- **Change `MAX_FEATURES_PER_DICTIONARY`.** The 8-feature cap is a
  structural property of the rung-1 MPS encoding, not a knob.
- **Bundle a `merge` strategy.** Epoch is strategy-agnostic and
  passes `--strategy` through to `Compressor`. `merge` will land in
  its own change.
- **Manage cross-process parallelism.** The orchestrator runs panels
  sequentially in-process. Sequential is honest about the bottleneck
  (a single GPT-2 model in memory, dominated by per-prompt forward
  passes), and process-pool parallelism would force re-loading the
  model per worker — defeating the win. Users wanting concurrency
  shard at the CLI layer: invoke `compress-epoch` multiple times
  with disjoint `--n-panels-max` or anchor ranges, merge the
  emitted `ValidationReport`s with a separate aggregator. That
  separate aggregator can be a follow-up change if the workload
  ever justifies it.
- **Auto-recalibrate gates.** The §4.4 thresholds (Polygram 0.7,
  Jaccard 0.30, min_both_fire 5) are surfaced as flags but default
  to the calibrated values. Per-workload threshold tuning is the
  user's job.
- **Filter out 2-element clusters.** The §4.4 / §5.1 evidence shows
  2-pairs are real redundancies; both `Compressor` and `EpochCompressor`
  treat any cluster of size ≥ 2 as compressible.
- **Run a "validate the compressed SAE end-to-end on a held-out
  benchmark" stage.** The cross-entropy quality guard inside the
  loop is a *bound*, not a benchmark. Comprehensive post-epoch
  evaluation (e.g., perplexity on WikiText, downstream task metrics)
  is the user's downstream call.
- **Persist intermediate compressed checkpoints by default.** Only
  the final checkpoint is written. Each iteration rewrites in-place
  on a temp path, then `os.replace`s to the final path on the last
  successful iteration. Users wanting per-iteration artifacts pass
  `--save-intermediate-reports`, which retains per-iteration
  `ValidationReport`s but still emits only the final checkpoint —
  per-iteration *checkpoints* are deferred to a follow-up if needed.

## Discussion

### Why coverage over the cosine graph, not the pair graph

The redundancy graph (which pairs are "really" redundant) is exactly
what the validator computes — it requires forward passes. The cosine
graph (which pairs share decoder direction) is computable from the
checkpoint alone, in milliseconds. PR #18 (§4.1) measured Spearman
0.94 between Polygram-predicted overlap and the *real-decoder cosine*
Gram on the §4.4-class SAE — i.e., decoder cosine is a strong proxy
for which pairs the validator will eventually flag. The coverage
metric is therefore: every pair in the cosine-similar graph
(`cos ≥ cosine_threshold`) appears in at least one panel. This is
a cheap, observable, well-defined target that bounds how many
panels we need without forcing the orchestrator to predict its own
ground truth.

The default `cosine_threshold = 0.30` is the lower bound of the §4.4
mid-overlap bucket (0.30–0.70 in `behavioural-scaleup-probe.md`),
where Polygram's ranking signal first becomes informative. Below
that threshold, the §4.4 evidence shows Polygram's correlation with
behavioural Jaccard collapses; pairs at `cos < 0.30` aren't worth
panel-budget.

### Why fixed-point with stable-clusters convergence

A first compression pass collapses the panels' confirmed clusters.
The survivors include each cluster's representative — which now
absorbs the cluster's semantic role. If the representative is itself
projection-similar to *another* cluster's representative (or a
non-cluster survivor), a second iteration will find that emergent
redundancy. PR #28's §5.1 closure block flagged this directly:
"Re-running the validator against the new checkpoint should show
those 8 pairs no longer gate-pass." The natural extension is to
re-survey the survivors — and keep going until no new confirmed
clusters appear.

Convergence on **stable cluster sets** (the same feature ids get
compressed in two consecutive iterations) is the cleanest convergence
test: it directly measures "we've stopped finding new redundancies."
Convergence on confirmed-pair *count* would be noisier — a single
small pair appearing and disappearing across iterations could prevent
termination indefinitely.

The hard `max_iterations = 5` cap is a budget guard, not a
convergence target. The §4.4 panel converges in 1 iteration (8 → 3
features, post-compression confirmed = 0). The fresh-anchor panel
converges in 1 iteration (8 → 2 features, post-compression
confirmed = 0). Multi-panel epochs may need more iterations as
cross-cluster emergence surfaces; 5 is a generous ceiling for the
current calibration. Hitting it indicates either a design gap (the
panel selection isn't seeing the emergent clique) or a genuinely
stubborn redundancy structure worth investigating manually.

### Why a quality bound, and why relative not absolute

A sufficiently aggressive epoch could compress the SAE down to a
small "non-redundant" core, but at some point the survivors absorb
too much semantic load from silenced neighbours. The right guard is
a per-token cross-entropy delta on the prompt set:
`ce(compressed_sae) - ce(original_sae)`, computed by re-running the
SAE encoder + decoder on the captured residuals at hook-time and
measuring how much the reconstructed activations diverge from the
original. (Implementation note: this can be done without a second
GPT-2 forward pass — the validator already captures residuals at
hook time; the compressed-SAE reconstruction runs in numpy.)

The bound is **relative** to the first iteration's delta because the
first compression pass establishes a natural reference: "this much
quality loss is what one round of validator-confirmed cluster
collapse costs." Allowing 2× that delta lets later iterations make
incremental progress without unbounded degradation. An absolute
nat-bound would require calibration we don't have — PR #23 measured
*per-feature* ablation-KL at blocks.10 in the 0.5–2 nats range, but
those are single-feature ablations, not whole-checkpoint
reconstruction deltas. The two metrics aren't comparable, and
inventing a number now would be calibration theatre.

The bound is configurable via `--quality-delta-multiplier`. The
EpochReport carries the per-iteration deltas so a user can post-hoc
audit whether the loop should have stopped earlier.

### Why representatives need orchestrator-level n_fires aggregation

`Compressor`'s representative selection sums `n_fires_i` and
`n_fires_j` from the `pairs` table for pairs whose endpoints both
belong to the cluster. A feature appearing in many panels accumulates
`n_fires` only from the pairs that named it — and pairs naming a
feature are a strict subset of the panels containing it. Concretely:
if feature `A` is in panels P1, P2, P3 and only pairs `(A, X)` from
P1 and `(A, Y)` from P2 enter the synthetic report's confirmed list,
the per-pair sum sees `A`'s n_fires from P1 and P2 only — missing
P3's contribution. For a single panel this is the same number; for
the multi-panel synthetic report it under-counts.

The orchestrator therefore aggregates `n_fires` per (cluster member)
across every panel that contained the member (not just from confirmed
pairs naming it), and surfaces this aggregate via the
`representatives` override on `Compressor`. `Compressor`'s default
selection rule still applies; the override just supplies the
right number to drive that rule.

### Why panels are sequential, not parallel

The validator's per-panel cost is dominated by GPT-2 forward passes.
GPT-2 small fits comfortably in a single process's memory; running
panels sequentially against a single in-process model amortizes the
weights across all panels. Process-pool parallelism would force
re-loading the model in each worker — for a 1000-panel run with
~500MB GPT-2 small, that's ~500GB of redundant model loading. The
math gets worse at Gemma-2-2B (~10GB per worker).

The honest concurrency unit is therefore *the CLI invocation*. The
spec surfaces `--n-panels-max` and (later, if needed) anchor-range
flags so users can shard at the invocation level: `invocation 1`
handles anchors with priority rank 0–999, `invocation 2` handles
1000–1999, etc. Their per-panel `ValidationReport`s land on disk;
a separate aggregation utility (deferred to a follow-up if real
demand exists) merges them into a final compressed checkpoint.

This is structurally the same shape that scientific computing
clusters use for embarrassingly-parallel workloads — sharded
independent invocations, central reduction — and it's the right shape
*because* the validator is the bottleneck, not because the
orchestrator can't manage threads.
