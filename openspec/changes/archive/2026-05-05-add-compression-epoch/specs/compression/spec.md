## ADDED Requirements

### Requirement: EpochCompressor consumes an SAE checkpoint and produces a fixed-point compressed checkpoint

`polygram.compression.EpochCompressor` SHALL be a dataclass that
consumes an SAE checkpoint, prompt set, and panel-selection
parameters; runs `BehaviouralValidator` over many panels;
synthesizes one cross-panel `ValidationReport`; hands it to
`Compressor` with orchestrator-supplied `representatives`; iterates
to a stable-clusters fixed point; and emits a final compressed
`.safetensors` plus an `EpochReport`. The orchestrator itself MUST
be torch-free; torch lazy-imports happen only inside the delegated
`BehaviouralValidator.validate()` calls.

The dataclass SHALL expose at least the following fields:

- `sae_checkpoint: Path` (required)
- `prompts: Sequence[str]` (required, non-empty)
- `layer: int` (required)
- `model_name: str = "gpt2"`
- `strategy: str = "zero"`
- `device: str | None = None` (auto-resolves per `BehaviouralValidator`)
- `coverage_target: float = 0.95`
- `cosine_threshold: float = 0.30`
- `n_visits_per_feature: int = 3`
- `n_panels_max: int = 1000`
- `min_firing_rate: float = 0.01`
- `max_iterations: int = 5`
- `quality_delta_multiplier: float = 2.0`
- `polygram_overlap_threshold: float = 0.7`
- `jaccard_threshold: float = 0.30`
- `min_both_fire: int = 5`
- `save_intermediate_reports: bool = False`
- `allow_layer_zero: bool = False`

`__post_init__` SHALL validate every field and raise `ValueError`
(with field name and offending value) on any violation.

#### Scenario: end-to-end run on a synthetic SAE with one redundant cluster

- **GIVEN** a synthetic SAE checkpoint at `tests/_synth_sae.synth_sae`
  output with 32 features × 8 d_model
- **AND** a 4-feature subset whose decoder rows are deliberately
  similar (cosines ≥ 0.95)
- **AND** a 2-prompt held-out set that produces co-firing on those
  4 features
- **WHEN** `EpochCompressor.run(output_checkpoint=tmp_path /
  "out.safetensors")` is called with `n_panels_max=4`,
  `max_iterations=2`
- **THEN** the call SHALL succeed and emit an `EpochResult`
- **AND** `result.report.n_features_zeroed_total` SHALL be ≥ 3
  (the 4-feature redundant cluster collapses to 1 representative)
- **AND** `result.report.convergence_reason` SHALL be one of
  `'stable_clusters'`, `'no_more_priority_candidates'`
- **AND** the rewritten checkpoint SHALL exist at the named path
- **AND** `result.final_dictionary` SHALL be a `Dictionary`
  rebuilt via `from_sae_lens` on the rewritten checkpoint

#### Scenario: post_init rejects bad parameter values

- **GIVEN** an existing SAE checkpoint
- **WHEN** `EpochCompressor(sae_checkpoint=path, prompts=["x"],
  layer=10, coverage_target=1.5)` is constructed
- **THEN** `__post_init__` SHALL raise `ValueError` whose message
  names the `coverage_target` field and the rejected value `1.5`

### Requirement: Panel selection is greedy seeded coverage over the cosine-similar pair graph

`EpochCompressor._select_panels` SHALL implement greedy seeded
coverage per `design.md` Decision 2:

1. Sort the eligible-feature set (firing rate ≥ `min_firing_rate`
   AND not in `zeroed`) by `firing_rate × decoder_norm` descending.
2. Iterate this priority queue: each anchor that has not yet
   appeared in `n_visits_per_feature` panels builds a panel
   consisting of the anchor + 7 nearest cosine-similar neighbours
   from the eligible pool, excluding any feature already at its
   visit cap.
3. Maintain `pairs_covered ⊆ S` where `S = {(i, j) : i < j,
   cos(W_dec[i], W_dec[j]) ≥ cosine_threshold, i and j eligible}`.
4. Terminate when any of: `|pairs_covered| / |S| ≥ coverage_target`;
   `n_panels_max` panels emitted; priority queue exhausted.

Each emitted `Panel` SHALL have exactly 8 `feature_ids` UNLESS the
eligible pool has fewer than 8 features at construction time, in
which case the orchestrator SHALL emit at most one panel covering
all eligible features and SHALL log a warning.

Panel selection SHALL be deterministic: two `_select_panels` calls
with identical inputs MUST produce identical panel sequences.
Tiebreaks on equal priority go to lower fid; tiebreaks on equal
cosine in neighbour selection go to lower fid.

#### Scenario: priority queue ordering matches firing × norm

- **GIVEN** an eligible set of 16 features with hand-set firing
  rates and decoder norms such that
  `priority(fid) = firing_rate[fid] × ‖W_dec[fid]‖`
- **WHEN** `_select_panels` runs with `n_panels_max=4`,
  `n_visits_per_feature=1`, `coverage_target=1.0`
- **THEN** the first 4 panels' anchors SHALL match the top-4
  features by `priority` in descending order
- **AND** ties SHALL break on lowest fid

#### Scenario: zeroed features are never selected

- **GIVEN** an eligible set of 24 features
- **AND** a `zeroed` set containing 8 of them
- **WHEN** `_select_panels` runs
- **THEN** no emitted `Panel` SHALL contain any feature from
  `zeroed` (asserted by intersection check)

#### Scenario: visit cap is respected

- **GIVEN** an eligible set of 32 features
- **WHEN** `_select_panels` runs with `n_visits_per_feature=2`
  and a coverage target that would otherwise drive more visits
- **THEN** every emitted panel's `feature_ids` SHALL show each
  feature appearing in at most 2 panels

### Requirement: Cross-panel synthetic ValidationReport unions confirmed pairs and aggregates per-pair statistics

`EpochCompressor._synthesize_validation_report` SHALL accept a list
of `Panel`s and their per-panel `ValidationReport`s and emit one
synthetic `ValidationReport` whose `confirmed` field is the union
of every panel's confirmed pairs. Per-pair statistics in the
synthetic report SHALL aggregate across panels containing the pair
per `design.md` Decision 3's aggregation table:

- `polygram_overlap`, `jaccard`: max
- `decoder_overlap`: any single panel's value (panel-independent)
- `pearson_activation`, `kl_*`: weighted mean per the table
- `n_fires_i`, `n_fires_j`, `n_both_fire`, `n_either_fire`:
  identical-across-panels invariant from deterministic forwards
  (Decision 3a) — assert and use any panel's value
- `gate_pass`: True iff at least one panel had it True

The synthetic report's `feature_ids` field SHALL be the union of
every panel's `feature_ids`, sorted ascending. Its
`dictionary_name` SHALL be the orchestrator's input
`dictionary_name` (or auto-generated from the SAE checkpoint
filename).

#### Scenario: union of confirmed across overlapping panels

- **GIVEN** three panels P1, P2, P3 with overlapping feature_ids
- **AND** P1 confirms pairs `{(0, 1), (0, 2)}`
- **AND** P2 confirms pairs `{(0, 1), (3, 4)}`
- **AND** P3 confirms pairs `{(5, 6)}`
- **WHEN** `_synthesize_validation_report` runs
- **THEN** the synthetic report's `confirmed` SHALL be exactly
  `((0, 1), (0, 2), (3, 4), (5, 6))` (sorted ascending by min then max fid)

#### Scenario: max-aggregation on polygram_overlap

- **GIVEN** pair `(0, 1)` appearing in panel P1 with
  `polygram_overlap=0.71` and panel P2 with `polygram_overlap=0.83`
- **WHEN** `_synthesize_validation_report` runs
- **THEN** the synthetic report's pair `(0, 1)` SHALL have
  `polygram_overlap == 0.83`

### Requirement: Cross-panel representative selection uses orchestrator-aggregated n_fires

`EpochCompressor._pick_representatives_global` SHALL run union-find
on the synthetic report's `confirmed` list and SHALL pick each
cluster's representative as the member with the highest panel-
independent firing count, NOT the per-pair sum that
`Compressor._pick_representative` uses internally. The orchestrator
passes the resulting `{cluster_id: fid}` map to `Compressor` via
its `representatives` field.

The panel-independent firing count is the global `n_fires` from
the epoch's pre-pass: `firing_rate[fid] × n_tokens` (an integer or
near-integer; the `firing_rates` array is itself a count divided by
`n_tokens`, so the product round-trips). Tiebreak: lowest fid.

#### Scenario: cross-panel rep selection picks the globally-most-firing member

- **GIVEN** a cluster `{A=10, B=20, C=30}` formed via union-find
- **AND** A appears in 3 panels and fires on 100 tokens
- **AND** B appears in 1 panel and fires on 50 tokens
- **AND** C appears in 4 panels and fires on 60 tokens
- **WHEN** `_pick_representatives_global` runs
- **THEN** the chosen representative for the cluster SHALL be A
  (highest panel-independent firing count)
- **AND** B and C SHALL be in the `Compressor`-emitted `zeroed`
  list for that cluster

### Requirement: Iteration converges on stable cluster sets or honors hard caps

`EpochCompressor.run` SHALL iterate panel selection + validation +
compression up to `max_iterations` times. After each iteration's
compression, the orchestrator SHALL compute
`cluster_fingerprint = frozenset(frozenset(c.members) for c in
plan.clusters)` and compare to the previous iteration's fingerprint.
On equality, the orchestrator SHALL terminate with
`convergence_reason='stable_clusters'`.

The orchestrator SHALL also terminate on:

- `max_iterations` reached without stable clusters
  (`convergence_reason='max_iterations'`);
- panel selection produced zero panels in iteration 0
  (`convergence_reason='no_more_priority_candidates'`);
- the quality bound is breached (per the next requirement).

Each iteration's `EpochIteration.convergence_state` field SHALL
record the state observed at the END of that iteration:
`'continuing'` while iterating; one of the terminal reasons on the
final iteration.

#### Scenario: stable cluster fingerprint terminates the loop

- **GIVEN** an SAE workload where iterations 0 and 1 both produce
  the same set of compressed clusters
  `{{0, 1}, {3, 4, 5}}`
- **WHEN** `EpochCompressor.run` executes
- **THEN** the loop SHALL terminate after iteration 1
- **AND** `report.convergence_reason` SHALL be `'stable_clusters'`
- **AND** `len(report.iterations)` SHALL be 2

#### Scenario: max_iterations terminates a non-converging loop

- **GIVEN** an `EpochCompressor` with `max_iterations=2`
- **AND** a workload that produces different cluster fingerprints
  on each iteration
- **WHEN** `EpochCompressor.run` executes
- **THEN** the loop SHALL terminate after iteration 1 (zero-indexed,
  so iterations 0 and 1 ran)
- **AND** `report.convergence_reason` SHALL be `'max_iterations'`

### Requirement: Quality bound reverts on cross-entropy delta breach

After each iteration `k > 0`, `EpochCompressor.run` SHALL compute
`delta_k`, the mean per-token cross-entropy between the
SAE-reconstructed residuals before and after iteration `k`'s
compression apply. If `delta_k > quality_delta_multiplier × delta_1`,
the orchestrator SHALL:

- discard iteration `k`'s compressed checkpoint and revert to
  iteration `k-1`'s as the final;
- set `convergence_reason='quality_bound_breached'`;
- record `delta_k` on the iteration that breached so a post-hoc
  audit of the EpochReport can see the boundary.

#### Scenario: quality breach reverts to the prior iteration's checkpoint

- **GIVEN** `quality_delta_multiplier=2.0`
- **AND** iteration 1 produces `delta_1 = 0.01`
- **AND** iteration 2 produces `delta_2 = 0.05` (5× delta_1, breaches
  the 2× bound)
- **WHEN** `EpochCompressor.run` executes
- **THEN** the final `output_checkpoint` SHALL byte-equal iteration
  1's compressed checkpoint
- **AND** `report.convergence_reason` SHALL be
  `'quality_bound_breached'`
- **AND** `report.iterations[2].cross_entropy_delta` SHALL equal
  `0.05`

### Requirement: Zeroed features are excluded from panel selection

The orchestrator SHALL maintain a `zeroed: set[int]` initialized
empty before iteration 0 and updated after each iteration's
compression apply (union with the iteration's `features_zeroed_this_iteration`).
Panel selection (Decision 2) SHALL exclude any feature in `zeroed`.
This is required, not optional — to save the validator's per-
feature ablation budget on features that fire on 0 tokens.

#### Scenario: zeroed features do not appear in panels of subsequent iterations

- **GIVEN** an SAE where iteration 0 zeroes features `{42, 100}`
- **WHEN** iteration 1's panel selection runs
- **THEN** no panel emitted in iteration 1 SHALL contain feature
  42 or feature 100
- **AND** iteration 1's `Panel.feature_ids` lists SHALL all be
  disjoint from `{42, 100}`

### Requirement: EpochReport carries provenance and is JSON round-trippable

The orchestrator's `EpochReport` SHALL serialize to and deserialize
from JSON exactly: `EpochReport.from_json(report.to_json()) ==
report` MUST hold for any well-formed `EpochReport` instance. The
JSON layout SHALL match `design.md` Decision 8 and SHALL carry at
least: `schema_version`, `source_checkpoint`,
`source_checkpoint_sha256`, `output_checkpoint`,
`output_checkpoint_sha256`, `convergence_reason`,
`n_features_zeroed_total`, `n_panels_total`, `coverage_achieved`,
`wall_seconds`, and an `iterations` array with one
`EpochIteration` per iteration that ran.

Float fields SHALL be rounded to six significant figures via the
same `format(v, ".6g")` discipline `CompressionReport` uses.

#### Scenario: round-trip preserves report state

- **GIVEN** a hand-built `EpochReport` with 2 iterations, 5 panels
  total, `convergence_reason='stable_clusters'`
- **WHEN** `r2 = EpochReport.from_json(r.to_json())`
- **THEN** `r2 == r` (using `EpochReport.__eq__` with NaN-aware
  float comparison)
- **AND** `r2.iterations[0].panels[0].feature_ids ==
  r.iterations[0].panels[0].feature_ids`

#### Scenario: required keys are all present in serialized JSON

- **GIVEN** any `EpochReport` instance `r`
- **WHEN** `payload = json.loads(r.to_json())` is parsed
- **THEN** `payload` SHALL contain every required key listed in
  `design.md` Decision 8 (`schema_version`, `source_checkpoint`,
  `source_checkpoint_sha256`, `output_checkpoint`,
  `output_checkpoint_sha256`, `convergence_reason`,
  `n_features_zeroed_total`, `n_panels_total`,
  `coverage_achieved`, `wall_seconds`, `iterations`)

### Requirement: Final checkpoint write is atomic and source is immutable

`EpochCompressor.run` SHALL write the final compressed checkpoint
atomically: each iteration writes to a sibling temp file, then
`os.replace`s to the final `output_checkpoint` only after the
iteration succeeds AND the quality bound holds (or the iteration
is the final converging one). The source SAE checkpoint MUST never
be modified; the orchestrator SHALL refuse to run if
`output_checkpoint` resolves to the same path as `sae_checkpoint`
(the underlying `Compressor` already enforces this, but the
orchestrator SHALL also check at `run` entry).

#### Scenario: source checkpoint bytes unchanged after run

- **GIVEN** a source SAE checkpoint at `path`
- **AND** `before = sha256(path.read_bytes())`
- **WHEN** `EpochCompressor.run(output_checkpoint=other_path)`
  completes successfully
- **THEN** `sha256(path.read_bytes()) == before`

#### Scenario: output equal to source raises

- **GIVEN** an `EpochCompressor` with `sae_checkpoint=path`
- **WHEN** `run(output_checkpoint=path)` is called
- **THEN** `ValueError` SHALL be raised before any work begins,
  with a message naming both paths
