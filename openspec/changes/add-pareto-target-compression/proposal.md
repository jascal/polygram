## Why

`Compressor` today is threshold-driven and single-shot: how many
clusters fall out depends entirely on whatever `ValidationReport.confirmed`
contains, which in turn depends on `ValidationConfig`'s
`polygram_overlap_threshold` / `jaccard_threshold` cutoffs. A caller
who wants to ask *"compress this SAE to ~K cluster representatives
and tell me the cost"* has no API for it — they have to bisect on
thresholds, re-running the validator each time, which is the
expensive step.

This also blocks downstream callers (e.g. sae-forge) from sweeping
a feature-count vs. faithfulness Pareto frontier. Experimental
evidence from sae-forge runs shows the KL/feature-count curve is
**non-monotonic** in the kept count (e.g. 25→211 features got
*worse* on GPT-2 layer-8 at default knobs), so the curve genuinely
needs sampling, not bisection. A one-validator-run-per-point cost
makes that impractical.

The data needed to do this cheaply is already in
`ValidationReport.pairs` (a tuple of `CandidatePair`, see
`polygram/behavioural/report.py:62`). Each candidate carries 14
score fields; the threshold gate just discards the ordering. By
exposing a target-K cut over the sorted pair list, the full Pareto
path becomes **one sort plus K constant-time cuts**, not K
validator runs.

This proposal also addresses Axis 1 of the rung-viability work
([`docs/research/rung4-viability-spike-v2.md`](../../../docs/research/rung4-viability-spike-v2.md)):
the v2.2 result showed a Rung4-amp-on iter-9 CE spike consistent
with cluster-exhaustion, and called out
"Pareto curve over `(max_iterations, quality_delta_multiplier)`"
as the natural follow-up. Target-K planning is the load-bearing
primitive for that follow-up.

## What Changes

### Public API

- **`CompressionConfig` grows two fields** (`polygram/config.py:251`):
  - `target_n_features_kept: int | None = None` — when set, requests
    that target-K planning produce a plan whose `n_features_kept` is
    ≤ this value. *Semantic*: matches the existing
    `CompressionReport.n_features_kept`, which counts cluster
    representatives (one survivor per cluster), **not** the
    SAE's total surviving feature count. See Decision 1 in
    [`design.md`](design.md) for the rationale.
  - `score_field: str = "polygram_overlap"` — one of
    `polygram_overlap`, `jaccard`, `decoder_overlap`. These three
    are picked from the 14 `CandidatePair` fields because they are
    bounded `[0, 1]` similarity-like quantities suitable for a
    monotonic greedy sort. KL-based fields are deliberately
    excluded (see Decision 3 in [`design.md`](design.md)).

- **New `Compressor.plan_with_target()` method** that ignores
  `validation_report.confirmed`, instead sorts
  `validation_report.pairs` by `score_field` descending and greedily
  unions pairs via union-find until the cluster count first drops
  to `target_n_features_kept` (or until pairs are exhausted, whichever
  comes first). Returns a `CompressionPlan` shaped exactly like
  today's `plan()` (same `clusters`, `feature_ids` tuple).

- **New `Compressor.plan_pareto(targets)` method + `ParetoReport`
  artifact** — returns one `CompressionPlan` per requested K,
  sharing the high-confidence pairs by construction (Decision 4 in
  [`design.md`](design.md): nestedness). The single sort cost is
  amortised across all K.

- **`Compressor.apply()` accepts a `plan` override** so callers can
  pass either `plan()` or `plan_with_target()` output (or any
  individual plan from a `ParetoReport`) without re-running planning.

- **`CompressionPlan` gains a `@property def n_features_kept`** that
  computes the existing report-level semantic (`= len(self.clusters)`)
  so target-K assertions can read the same value off either the plan
  or the post-`apply()` report. *Not a stored field* — no impact on
  serialization or `__eq__`.

- **CLI** — `polygram compress` gains:
  - `--target-features K` — single-shot target-K compression
  - `--pareto K1,K2,K3,...` — multi-K planning; default writes a
    `pareto.json` only
  - `--pareto-materialize` — opt-in to materialise one SAE per K
    under `<out>/pareto/k_{K}.safetensors` (without this flag, only
    the plan JSON is written)
  - `--score-field {polygram_overlap,jaccard,decoder_overlap}`
  - `--target-features` and `--pareto` are mutually exclusive

### Out of scope, deliberately

- **Target-K mode does not interact with `EpochCompressor`.**
  `EpochCompressor`'s iterative loop drives its own
  `Compressor(...).plan()` call (threshold-mode). Target-K mode is
  single-shot. If a future change wants iterative target-K
  compression, that's a separate proposal.
- **`rep_selection="recon_proxy"`** — picks cluster reps by
  reconstruction-loss attribution. Requires Compressor to accept
  activations or a per-feature attribution vector; real interface
  widening; deferred.
- **sae-forge integration** — callers bump `polygram>=0.4.0` and
  wire the new fields through their own configs; not part of this
  change.
- **Automatic K selection** (e.g. elbow-point detection) — caller
  chooses K.

### Byte-identity guarantee

The existing call path is byte-identical when
`target_n_features_kept is None`:
- `Compressor(report, ckpt).plan()` continues to consume
  `confirmed` and is unchanged.
- `CompressionConfig()` defaults to threshold mode.
- `CompressionReport.to_json()` output for the existing toy
  fixture is bit-equal to the pre-change reference. The new
  `@property n_features_kept` on `CompressionPlan` doesn't affect
  the report JSON since the value is already serialized at the
  report level.

## Capabilities

### New Capabilities

- `pareto-compression`: `Compressor.plan_with_target()`,
  `Compressor.plan_pareto()`, `ParetoReport` dataclass. Public API,
  exported from `polygram`.

### Modified Capabilities

- `tuning-config`: `CompressionConfig` gains
  `target_n_features_kept` and `score_field` with documented
  defaults and `__post_init__` validation. Existing fields and
  defaults unchanged.
- `cli`: `polygram compress` gains `--target-features`, `--pareto`,
  `--pareto-materialize`, and `--score-field` flags. Existing
  flags unchanged.

## Impact

- **New module**: `polygram/compression/pareto.py` (`ParetoReport`
  dataclass with `to_json` / `from_json` mirroring
  `CompressionReport`'s hand-coded serializer; greedy-union
  helper).
- **Modified**:
  - `polygram/compression/compressor.py` — `plan_with_target`,
    `plan_pareto`, internal `_greedy_union_to_target` helper;
    `apply()` gains optional `plan` override.
  - `polygram/compression/report.py` — `CompressionPlan` gains
    `@property n_features_kept` (derived, not stored).
  - `polygram/config.py` — `CompressionConfig` field additions +
    validation.
  - `polygram/cli.py` — new flags on the `compress` subcommand.
  - `polygram/__init__.py` — export `ParetoReport`.
- **No breaking changes**: existing
  `Compressor(validation_report, sae_checkpoint).plan().apply()`
  path is byte-identical. `ValidationReport`,
  `BehaviouralValidator`, `Confirmer` protocol, `EpochCompressor`,
  `Regrower` are untouched.
- **Dependencies**: no new dependencies; numpy + dataclasses only.
- **Version bump**: `0.3.0 → 0.4.0` (minor; additive, byte-identity
  preserved).
