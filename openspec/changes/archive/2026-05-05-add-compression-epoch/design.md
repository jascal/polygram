## Context

`add-compression-action` (PR #28 spec, commit `7bdc7e7` impl) closed
the per-panel compression loop. Live runs reproduce §4.4 byte-for-
byte and demonstrate the action handles a 4-clique cleanly via
union-find. But the SAE has 24,576 features and the validator caps
each panel at 8 features (`MAX_FEATURES_PER_DICTIONARY = 8`).
Compressing the full SAE requires running many panels and
aggregating their confirmed-pair lists.

This change adds the orchestrator. The hard questions are panel
selection, cross-panel aggregation, fixed-point iteration, and the
quality bound. Each decision below commits to a concrete answer
calibrated against existing findings.

## Goals / Non-Goals

**Goals:**

- An `EpochCompressor` that scales the per-panel loop to the full SAE.
- Concrete, computable panel selection with a defined coverage metric.
- Cross-panel pair-list aggregation that preserves the existing
  `Compressor`'s component-first contract.
- Fixed-point iteration with a defensible convergence criterion and
  a relative quality bound.
- Skip-zeroed bookkeeping that respects the validator's per-feature
  ablation cost.
- An `EpochReport` carrying full provenance: source/output sha256s,
  every iteration's per-panel `ValidationReport` paths, convergence
  reason, per-iteration cross-entropy deltas.
- A torch-free orchestrator surface (lazy-imports inside the
  delegated `BehaviouralValidator.validate()` only).
- A `polygram compress-epoch` CLI subcommand with file-based inputs.
- One end-to-end smoke test against a synthetic SAE on the
  `tests/_synth_sae.py` fixture.

**Non-Goals:**

- Changing `MAX_FEATURES_PER_DICTIONARY`.
- Implementing a `merge` strategy (deferred; epoch passes through to
  whatever `Compressor.strategy` supports).
- Cross-process parallelism (deferred; users shard at CLI level).
- Auto-recalibrating gate thresholds.
- Filtering 2-element clusters.
- Persisting intermediate compressed checkpoints (only final).

## Decisions

### Decision 1 — Coverage metric: cosine-graph pair coverage at a threshold

**Definition:** `coverage = |{(i, j) ∈ S : ∃ panel containing both i and j}| / |S|`,
where `S = {(i, j) : i < j, cos(W_dec[i], W_dec[j]) ≥ cosine_threshold}`.

**Default threshold:** `cosine_threshold = 0.30`. Justification: the
§4.4 mid-overlap bucket (`docs/research/behavioural-scaleup-probe.md`)
starts at Polygram-overlap 0.40, and PR #18 (§4.1) measured
`Spearman(Polygram, decoder_cosine_squared) = 0.94` on the §4.4-class
SAE. The cosine-cutoff at 0.30 corresponds to the lower tail of the
bucket where Polygram's ranking signal first becomes informative;
below this threshold the §4.4 evidence shows the correlation with
behavioural Jaccard collapses.

**Default coverage target:** `coverage_target = 0.95`. Justification:
the eligible pair set `S` is bounded by `|S| ≤ C(n_eligible, 2)`
where `n_eligible` is the count of features with firing rate
≥ `min_firing_rate`. For GPT-2 small blocks.10 this is roughly
~3000–5000 features (per the §4.4 selector's fire-filter rate). At
n_visits = 3 panels per feature and 8 features per panel, the
expected `|S|`-coverage of a greedy seeded selector saturates well
before 100% (some pairs are between low-priority features that never
get an anchor slot). 0.95 is the honest "best effort" target.

**Rejected alternatives:**

- *Coverage over the redundancy graph.* Defining coverage in terms
  of which pairs the validator will eventually flag is circular —
  computing the redundancy graph is exactly the work the loop is
  trying to do.
- *Coverage over the full pair graph.* `|S| = C(24576, 2) ≈ 302M`
  is the wrong denominator. Most of those pairs have negligible
  cosine and will never be redundant; counting them dilutes the
  metric.
- *Per-feature visit-count target.* "Every feature visited K times"
  is checkable but doesn't bound which *pairs* get co-located.

### Decision 2 — Panel selection: greedy seeded coverage

**Algorithm:**

1. **Pre-pass:** one full forward over the prompt set; encode through
   the SAE; compute per-feature firing rates. (Reused from
   `BehaviouralValidator.predict()` but materialized once at the
   epoch level so all panels share the result.)
2. **Filter:** `eligible = {fid : firing_rate[fid] ≥ min_firing_rate
   AND fid ∉ zeroed}`.
3. **Priority queue:** sort eligible by
   `firing_rate[fid] × ‖W_dec[fid]‖` descending.
4. **Coverage tracking:** maintain `pairs_covered: set[(int, int)]`
   over the cosine-similar pair set `S`.
5. **Iterate:** pop highest-priority anchor that hasn't yet appeared
   in `n_visits_per_feature` panels. Build panel = anchor + 7 nearest
   cosine-similar features from the eligible pool (excluding `zeroed`
   and excluding features already at their `n_visits_per_feature`
   cap). Update `pairs_covered`. Stop when:
   - `coverage ≥ coverage_target`, OR
   - `n_panels_max` panels have been built, OR
   - the priority queue is exhausted.

**Default `n_visits_per_feature = 3`:** lets a feature appear in its
own anchor panel + two neighbour panels (giving it a chance to be
paired with features anchored elsewhere) without runaway over-
visiting.

**Default `n_panels_max = 1000`:** at ~25 s/panel on GPT-2 small +
MPS (per the live-run timing), 1000 panels = ~7 hours. A user
running on a slower CPU-only setup or a larger model can lower this;
the orchestrator surfaces `coverage_achieved` in the `EpochReport`
so they can see what they're trading.

**Rejected alternatives:**

- *Disjoint partitioning* (`24576 / 8 = 3072 panels`). Misses every
  cross-partition redundancy by construction. The §4.4 / §5.1
  observed cliques span features that disjoint partitioning would
  separate.
- *Random panels.* No coverage guarantee; in expectation a random
  panel set covers `~1 − exp(−n × p)` of the cosine-similar pairs
  where `p = 28/n_eligible². For n_eligible = 3000, achieving 0.95
  coverage random would require ~1M panels.
- *Exhaustive K-cliques in the cosine-similar graph.* Equivalent
  to clique enumeration on a sparse graph; bounded above by the
  fewer-pivot variant of Bron–Kerbosch but still exponential in the
  worst case. Greedy seeded is much cheaper and reaches 95%+
  coverage on the empirical pair distributions seen in the live runs.

### Decision 3 — Multi-panel aggregation: union-of-confirmed with max-statistics

**Confirmed:** a pair `(i, j)` is in the synthetic report's
`confirmed` list iff at least one panel that contains both endpoints
flagged `gate_pass=True`.

**Per-pair statistics in the synthetic report:**

| Field | Aggregation rule |
|-------|------------------|
| `polygram_overlap` | `max` across panels containing the pair |
| `decoder_overlap` | first panel's value (panel-independent: depends only on `W_dec[i]`, `W_dec[j]` — same across all panels containing the pair) |
| `jaccard` | `max` across panels containing the pair |
| `pearson_activation` | weighted mean by `n_either_fire` |
| `kl_ablate_i` / `kl_ablate_j` | weighted mean by panels' `n_fires` for that endpoint |
| `kl_ratio_paired` | weighted mean by `n_both_fire` |
| `kl_log_ratio_abs` | weighted mean by `n_both_fire` |
| `n_fires_i` / `n_fires_j` | sum across panels (a feature appearing in multiple panels fires the same set of tokens; we do NOT double-count tokens — see Decision 3a below) |
| `n_both_fire` | the same: sum if from disjoint forward passes, but our forward passes are deterministic so all panels containing `(i, j)` see the same firing pattern → take any single panel's value |
| `n_either_fire` | same as `n_both_fire` rule |
| `gate_pass` | `True` iff at least one panel had it `True` |

**Decision 3a — n_fires aggregation across deterministic forwards:**
Each panel runs its own forward pass over the same prompt set, and
GPT-2 + the SAE encoder are deterministic given fixed weights. So a
feature's firing pattern is identical across every panel that
contains it. `n_fires_i` for the synthetic report is therefore the
**same value as in any single panel**, not a sum. This is a free win
the orchestrator gets from determinism. (If we ever introduce
non-deterministic per-panel sampling — e.g., randomized prompt
subsets — this rule changes; flagged in the implementation as an
invariant assertion.)

**Rationale for max on `polygram_overlap` and `jaccard`:** the
Polygram overlap depends on the panel's k-means cluster assignments,
which depend on the panel's full feature composition. Two panels
containing both `(i, j)` but with different other 6 features can
produce different Polygram overlaps for the same pair. Taking the
max is the conservative claim — "this pair plausibly has high
predicted overlap because at least one panel composition surfaced it
that way." Jaccard is *also* technically panel-independent (depends
only on the deterministic firing patterns), so per Decision 3a it'll
be the same across panels — but we compute the max anyway as a
defence against any future non-determinism.

### Decision 4 — Cross-panel representative selection: orchestrator aggregation

**Problem:** `Compressor._pick_representative` sums `n_fires_i` and
`n_fires_j` from confirmed pairs whose both endpoints belong to the
cluster. For multi-panel input, a feature `A` may appear in many
panels, but only some of them generate confirmed pairs naming `A`.
The intra-pair sum then under-counts `A`'s true firing-cost relative
to other cluster members.

**Fix:** the orchestrator computes per-feature panel coverage and
passes the rep choice via `Compressor.representatives` override.
Specifically:

```
for each cluster from union-find on the synthetic report:
    for each member in cluster:
        n_fires_global[member] = firing_rate[member] * n_tokens
        # firing_rate is the panel-independent value from the
        # epoch-level pre-pass; identical across every panel that
        # contained the member.
    rep = argmax_over_cluster(n_fires_global)
    # tiebreak: lowest fid (matches Compressor's rule)
    representatives[cluster_id] = rep
```

The `Compressor`'s default selection rule still applies as a fallback
for clusters where the orchestrator's override doesn't fire — but in
practice the orchestrator always supplies the override.

### Decision 5 — Fixed-point iteration: stable-clusters convergence + relative quality bound

**Convergence on cluster set stability:**

After each `Compressor.apply()`, compute `cluster_fingerprint =
frozenset(frozenset(c.members) for c in plan.clusters)`. Compare to
the previous iteration's fingerprint. If equal:
`convergence_reason = 'stable_clusters'`, terminate.

**Hard cap:** `max_iterations = 5`. Justification: the live single-
panel runs converge in 1 iteration. Multi-panel epochs may need more
as cross-cluster emergent redundancies surface; 5 is generous against
the empirical baseline. Hitting the cap without stable clusters
indicates either a panel-selection gap (the orchestrator isn't
seeing emergent cliques) or a genuinely stubborn redundancy structure
worth investigating manually. `convergence_reason =
'max_iterations'`.

**Relative quality bound:**

After each iteration, compute the per-token cross-entropy delta on
the prompt set:

```
delta_k = mean_token(H(p_baseline_residual_reconstruction,
                       p_compressed_residual_reconstruction))
```

where the "reconstruction" is the SAE's decode-from-encoded loop,
applied to the residuals captured at hook time. **No second GPT-2
forward pass needed** — residuals are cached from the validator's
predict-stage forward.

Bound: `delta_k ≤ quality_delta_multiplier × delta_1`. Default
`quality_delta_multiplier = 2.0`. On breach:
`convergence_reason = 'quality_bound_breached'`. The orchestrator
discards iteration `k`'s checkpoint and uses iteration `k-1`'s as
the final.

**Why relative not absolute:** PR #23 (§4.3) measured *per-feature
single-feature* ablation-KL at blocks.10 in the 0.5–2 nats range.
Those are single-feature ablations on next-token logits, not
whole-checkpoint reconstruction deltas. The two metrics aren't
comparable; inventing an absolute nat-bound now would be calibration
theatre. The first iteration's delta IS the natural reference: it
measures what one round of confirmed-cluster collapse costs in
quality. Allowing 2× lets later iterations make incremental progress
without unbounded degradation.

The user can disable the guard with
`--quality-delta-multiplier inf`; the EpochReport will still carry
the per-iteration deltas for post-hoc audit.

### Decision 6 — Skip-zeroed: required at panel-selection layer

The orchestrator MUST exclude features in `zeroed` from panel
selection (Decision 2 step 2). The `BehaviouralValidator` itself is
unchanged — it sees only feature_ids the orchestrator hands to it.

Rationale: a zeroed feature fires on 0 tokens, so its ablation pass
emits zero `kl_per_token` and contributes nothing to any pair's
gate_pass evaluation. But the validator's contract is "exactly
`len(feature_ids)` ablation forward-pass-batches" — so passing a
zeroed feature burns 1/8 of the panel's ablation budget on dead
weight. Filtering at panel selection saves the validator's cost
budget; modifying the validator to short-circuit zeroed features
would be defensible too but introduces a coupling between validator
and compressor state that's better avoided.

### Decision 7 — Parallelism: out of scope for the orchestrator

The orchestrator runs panels sequentially in a single process,
sharing one in-memory GPT-2 instance across all panels. Process-pool
parallelism would force re-loading the model per worker — for a
1000-panel run on GPT-2 small that's ~500GB of redundant model
loading; on Gemma-2-2B (~10GB per worker) it's an order of magnitude
worse.

Users wanting concurrency shard at the CLI invocation level: invoke
`compress-epoch` multiple times with non-overlapping anchor priority
ranges. Each invocation writes its own per-panel `ValidationReport`s
to disk; a follow-up aggregation utility (deferred; introduce only if
a real workload demands it) merges them into a final compressed
checkpoint.

### Decision 8 — `EpochReport` carries every iteration's pointer

JSON layout:

```json
{
  "schema_version": 1,
  "source_checkpoint": "...",
  "source_checkpoint_sha256": "...",
  "output_checkpoint": "...",
  "output_checkpoint_sha256": "...",
  "convergence_reason": "stable_clusters",
  "n_features_zeroed_total": 47,
  "n_panels_total": 312,
  "coverage_achieved": 0.954,
  "wall_seconds": 6420.7,
  "iterations": [
    {
      "iteration": 0,
      "panels": [
        {"panel_id": 0, "anchor": 12999, "feature_ids": [...],
         "cosines_to_anchor": [...]},
        ...
      ],
      "validation_report_paths": ["epoch_iter0_panel0.json", ...],
      "confirmed_pair_count": 134,
      "clusters_compressed": 24,
      "features_zeroed_this_iteration": [42, 100, 256, ...],
      "cross_entropy_delta": 0.003421,
      "convergence_state": "continuing"
    },
    ...
  ]
}
```

Per-iteration `ValidationReport`s are written to disk only when
`--save-intermediate-reports` is set. By default,
`validation_report_paths` is an empty tuple and the per-panel reports
are constructed in-memory and discarded after aggregation.

`EpochReport.from_json` round-trips `to_json` exactly, matching the
`CompressionReport` contract. SHA256s use the same hashing flow.
