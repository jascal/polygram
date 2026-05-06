## Why

PR #27 shipped the read-only `BehaviouralValidator` — the loop's
*upstream* half. End-to-end run on the §4.4 selection (12999 +
near-cluster {19398, 4192, 23625} + far-cluster {8371, 2287, 68,
13737}, blocks.10) reproduced Spearman = +0.6371 byte-for-byte,
emitted 8 confirmed candidates, and surfaced one structural fact the
prose §4.4 closure had not flagged: **the four far-cluster features
all fire on the same 12 tokens.** Every far × far pair (6 of them)
gate-passes with Jaccard ≥ 0.92. They are not three independent
redundancies — they are a single redundancy *clique*.

The compression action is the loop's *downstream* half: it consumes
the validator's `confirmed` list and acts on the SAE weights so the
flagged redundancies become inert. The clique observation forces a
specific shape: compression must operate on **connected components
of `confirmed`**, not on individual pairs. Pair-by-pair compression
of the far-cluster would zero each member three times (once per pair
it appears in) and the order of operations would matter. Component-
first compression zeros each redundant member exactly once, against
a single representative.

The behavioural-validator design note (`docs/research/behavioural-
validator-design.md`) named compression as the natural follow-up;
the §4.4 closure block named the action's contract as "only act on
pairs that pass both Polygram and Jaccard thresholds, verify
ablation-KL changes match expectations." This change ships that
action.

## What Changes

### `compression` capability — new subpackage

Add `polygram.compression`, a new subpackage hosting:

- **`Compressor`** dataclass: takes a `ValidationReport` (or just
  its `confirmed` pairs), a source SAE checkpoint path, an output
  checkpoint path, a `strategy` (initial release: `"zero"` only),
  and an optional `representatives` override dict.
- **`CompressionPlan`** dataclass: per-cluster description of what
  *would* change — cluster id, member feature ids, representative,
  zero-list. Computed cheaply by `Compressor.plan()`; no I/O, no
  torch, no safetensors writes.
- **`CompressionResult`** dataclass: outcome of `apply()` — the
  new `Dictionary` rebuilt from the rewritten checkpoint, the
  `CompressionPlan` actually applied, and a `CompressionReport`
  carrying provenance + the source `ValidationReport`'s
  `schema_version` + `dictionary_name` so a reader can verify the
  upstream chain.
- **Two-stage API**: `plan() -> CompressionPlan` (cheap, no I/O);
  `apply(plan=None, output_checkpoint=...) -> CompressionResult`
  (writes one new `.safetensors` file; never mutates the source);
  `run(output_checkpoint=...)` is the convenience wrapper.
- **Round-trip**: `CompressionReport.to_json(path)` /
  `from_json(path)` matches the `ValidationReport` JSON-shape
  contract.

### `zero` strategy — initial release

The `zero` strategy:

1. Builds connected components from
   `validation_report.confirmed` via Union-Find.
2. Per cluster, picks a representative — the member with the
   highest summed `n_fires` across the cluster's pairs (tiebreak:
   lowest feature id), unless the user supplied an explicit
   `representatives[cluster_id] = fid` override.
3. For every non-representative member of every cluster, zeros
   `W_enc[:, fid]`, `b_enc[fid]`, `W_dec[fid, :]` (and `b_dec`
   stays unchanged — it is global).
4. Writes the rewritten weights to a new `.safetensors` at
   `output_checkpoint`. The source checkpoint is never modified.

Rationale: zeroing both encoder and decoder makes the redundant
feature *fully inert* — downstream interpretability tools that
inspect SAE activations see zero, not a phantom encoder reading.
Decoder-only zeroing leaves a "ghost" activation pattern that
reads as alive but writes nothing.

The `merge` strategy (decoder centroid: replace cluster
`W_dec[fid, :]` rows with their magnitude-weighted mean) is
deliberately deferred to a follow-up change; it creates a
synthetic decode direction not present in any source feature
and merits its own spec.

### `cli` capability — new `polygram compress` subcommand

`polygram compress` wraps `Compressor.run()` with file-based
inputs:

```
polygram compress \
  --validation-report path/to/validation_report.json \
  --sae-checkpoint path/to/sae_weights.safetensors \
  --output-checkpoint path/to/sae_weights.compressed.safetensors \
  --strategy zero \
  --output path/to/compression_report.json
```

The CLI reads the validation report, loads the source checkpoint,
runs `plan() + apply()`, writes the rewritten checkpoint, and
emits a `CompressionReport` JSON next to it.

### No new optional extra

The compression action is `safetensors`-only (already a base
dependency). It does **not** import `torch` or `transformers` —
the only weight operations needed are slice-and-zero on numpy
arrays loaded via `safetensors.numpy.load_file`. Validation
*after* compression (re-running `polygram validate` against the
rewritten checkpoint) needs the existing `[behavioural]` extra,
but that is the user's downstream choice.

## What this proposal explicitly does NOT do

- **Run validation on the compressed checkpoint.** Verifying
  that ablation-KL on confirmed pairs collapses post-compression
  is a separate workflow the user runs by re-invoking the
  validator. The compression action emits a report, not a
  validation.
- **Implement `merge` strategy.** Centroid-merge produces a
  decode direction that wasn't in any source feature; it
  deserves its own spec with explicit acceptance criteria
  (e.g., "centroid magnitude preserved within X%", "encoder
  remains aligned"). Deferred.
- **Auto-pick representatives across cluster boundaries.** The
  representative selection rule is per-cluster (highest summed
  `n_fires` within the cluster). It does not reach across
  clusters or consult the global SAE.
- **Modify the source checkpoint.** `apply()` always writes a
  new file. The path *must* differ from the source path; the
  validator raises if it doesn't.
- **Compress without a `ValidationReport`.** Users cannot
  hand-construct "I think these features are redundant" inputs
  and skip the four-constraint pipeline. The validator is the
  required upstream; compressing without it bypasses the
  empirical gate the loop was built to enforce.
- **Touch the Polygram Dictionary's β / γ assignments.** The
  compressed `Dictionary` is rebuilt by re-running
  `from_sae_lens` on the rewritten checkpoint with the same
  feature-id list. Cluster assignments are derived afresh from
  the new geometry, not transferred from the source dictionary.

## Discussion

### Why component-first instead of pair-first

The live validator run on the §4.4 selection produced 6 far × far
pairs all sharing the same 12 firing tokens. Treating each pair
independently would mean: zero feature B against feature A; then
zero feature C against feature B (which was just zeroed); then
zero feature D against feature C (zeroed); and so on. The
ordering matters and the result is order-dependent.

Component-first compression makes the operation order-
*independent*: the union-find collapses the clique to a single
component {8371, 2287, 68, 13737}; the representative gets picked
once; the other three get zeroed exactly once. Idempotent under
re-application; deterministic across pair-list orderings.

### Why `zero` first instead of `merge`

`zero` has a single non-controversial semantics: redundant
features become silent. `merge` (decoder-row centroid) writes
a synthetic direction back to the SAE; "magnitude-weighted mean
across cluster members" is one of several plausible centroid
definitions, and the choice has empirical implications worth
measuring before locking in. Shipping `zero` first lets the
loop *close end-to-end* (validator → compression → re-validation
shows reduced gate_pass) and creates a baseline against which a
future `merge` strategy can be benchmarked.

### Why no in-place compression

The `apply()` contract is "produce a new checkpoint." Three
reasons:

- **Reproducibility.** Anyone with the source checkpoint can
  replay compression deterministically; in-place would silently
  mutate shared state.
- **A/B comparison.** The validator-after-compression workflow
  needs both checkpoints simultaneously to demonstrate the
  redundancy-collapse effect. In-place would make the comparison
  destructive.
- **Audit trail.** The `CompressionReport` lists exactly which
  feature ids were zeroed against which representative. Combined
  with the source-checkpoint hash, that uniquely identifies the
  operation; in-place would hide the hash drift.

### Defaults the spec deliberately keeps user-visible

- Representative selection by `n_fires` sum is *one* defensible
  rule; another is "feature with highest activation peak across
  the prompt panel" or "feature with the lowest decoder norm."
  The spec encodes the firing-count rule because it matches the
  validator's per-pair counters directly (no extra model passes
  needed) — but the `representatives` override exists precisely
  because researchers may want a different rule per cluster.
- The strategy name (`zero`) is required, not defaulted, even
  though it is the only currently-implemented option. Forcing
  the user to write it out makes future-proofing for `merge`
  trivial: when `merge` lands, no existing call sites silently
  switch behavior.
