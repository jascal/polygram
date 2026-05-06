## Context

The validator (PR #27) emits a `ValidationReport` whose `confirmed`
field is a deterministic list of `(i, j)` feature-id pairs that
passed all three gates (Polygram ≥ 0.7, Jaccard ≥ 0.30, both-fire
≥ 5). On the §4.4 selection at `blocks.10`, that list contains:

- 2 near-cluster pairs: `(12999, 19398)`, `(4192, 23625)`
- 6 far-cluster pairs: every pair from `{8371, 2287, 68, 13737}`

The 6 far-cluster pairs collapse under union-find to a single
4-element component. So the validator's output describes
**3 redundancy clusters**: two 2-element pairs and one 4-element
clique.

The compression action's job is to take that 3-cluster description,
pick one representative per cluster, and zero the encoder + decoder
rows of the non-representatives. After compression, re-running the
validator against the new checkpoint should show those 8 pairs no
longer gate-pass (or at minimum: the non-representative members no
longer have nonzero `n_fires`).

## Goals / Non-Goals

**Goals:**

- A `polygram.compression.Compressor` that consumes a
  `ValidationReport` and produces a rewritten SAE checkpoint with
  redundant features zeroed.
- Two-stage API: `plan()` (cheap, no I/O) and `apply()` (writes
  one new `.safetensors`) with `run()` as the convenience wrapper.
- Order-independent semantics: connected-component analysis, not
  pair-by-pair sequential rewriting.
- Round-trippable `CompressionReport` (JSON) that links back to
  the source `ValidationReport` for full provenance.
- A `polygram compress` CLI subcommand wrapping `run()`.
- One end-to-end test that exercises `plan() → apply()` on a
  synthetic SAE checkpoint and confirms the rewritten weights have
  the expected zero pattern.

**Non-Goals:**

- Implementing the `merge` (decoder-centroid) strategy. Deferred
  to its own spec.
- Re-running the validator inside `apply()`. The validator-after-
  compression sanity check is the user's downstream call.
- Modifying β / γ cluster assignments on the rebuilt Dictionary.
  `from_sae_lens` is re-run on the new checkpoint and produces
  cluster assignments afresh from the new geometry.
- Compressing across multiple Dictionaries simultaneously.
  Single-Dictionary scope.

## Decisions

### Decision 1 — New `polygram.compression` subpackage; no new extra

Rejected alternatives:

- **Add to `polygram.behavioural`.** Conflates the read-only
  validation surface with weight-rewriting. The validator's
  bounded-blast-radius property is exactly what made it
  shippable first; merging compression into the same subpackage
  obscures that.
- **Top-level `polygram.compress` module.** Subpackage scales
  better when `merge` arrives — `polygram.compression.zero`
  vs `polygram.compression.merge` is the natural split.

Choice: new `polygram.compression/` subpackage with `compressor.py`
(the `Compressor` dataclass), `report.py` (`CompressionPlan`,
`CompressionReport`, `CompressionResult`), and `strategies/zero.py`
(the strategy implementation). No new optional extra:
`safetensors` is already a base dependency, and the action is
torch-free.

### Decision 2 — Two-stage API (`plan` then `apply`)

Public methods:

```python
def plan(self) -> CompressionPlan:
    "Build clusters from confirmed pairs; pick representatives."

def apply(
    self,
    plan: CompressionPlan | None = None,
    output_checkpoint: Path | None = None,
) -> CompressionResult:
    "Rewrite weights to a new checkpoint; rebuild Dictionary."

def run(self, output_checkpoint: Path) -> CompressionResult:
    "plan() + apply() in one call."
```

`plan()` returns a `CompressionPlan` listing every cluster, its
members, and its representative. No I/O, no torch. Calling
`plan()` on a system without the source checkpoint readable still
succeeds — it only consumes the in-memory `ValidationReport`.

`apply()` accepts an optional `plan` argument so the user can
hand-edit cluster representatives between stages (e.g., "I want
8371 as the rep for the far cluster, not 68"). When `plan` is
None, `apply()` calls `self.plan()` first.

### Decision 3 — `output_checkpoint` is required and must differ from source

The `apply()` contract:

- `output_checkpoint` is a required argument. There is no default.
  No "in-place mode."
- `output_checkpoint != self.sae_checkpoint` — `apply()` raises
  `ValueError` if they resolve to the same path (`Path.resolve()`
  comparison).
- The output file is written atomically: rewrite a sibling temp
  file, then `os.replace()` to the final path. (Matches the
  pattern `save_dictionary` already uses for JSON output.)

This is a hard contract, not a knob. In-place compression would
break the A/B-comparison workflow the validator-after-compression
sanity check depends on.

### Decision 4 — Representative selection: highest summed n_fires within the cluster

For each connected component (cluster), the default representative
is the feature id with the highest summed `n_fires` across the
cluster's confirmed pairs:

```python
def _default_representative(cluster: set[int], pairs: list[CandidatePair]) -> int:
    n_fires_total: dict[int, int] = defaultdict(int)
    for p in pairs:
        if p.i in cluster and p.j in cluster:
            n_fires_total[p.i] += p.n_fires_i
            n_fires_total[p.j] += p.n_fires_j
    # Tiebreak: lowest feature id.
    return min(cluster, key=lambda fid: (-n_fires_total[fid], fid))
```

Rationale: the most-active feature is most likely to be the
"original" of a redundancy clique — the one others duplicated.
Zeroing the duplicates against it preserves the most signal.
Tiebreak by lowest fid is deterministic and reproducible.

Override: `representatives: dict[int, int] | None` maps cluster
ids (assigned in `plan()` by ascending min-fid order) to a
chosen feature id. The override must name a fid that is actually
in the cluster, else `apply()` raises.

### Decision 5 — `zero` strategy: zero encoder AND decoder rows

For every non-representative member of every cluster:

```python
W_enc[:, fid] = 0
b_enc[fid]    = 0
W_dec[fid, :] = 0
# b_dec stays unchanged — it is global.
```

Rationale for both encoder and decoder:

- **Decoder-only zeroing** leaves the encoder reading nonzero
  activations downstream interpretability tools see, but the
  feature contributes nothing to the residual. That is a "ghost"
  state — confusing for debugging and inconsistent with the
  "this feature is redundant" semantics the validator emitted.
- **Encoder-only zeroing** would make the feature silent in
  activations but its decoder direction would still be addressable
  via direct injection (uncommon but possible in
  interpretability harnesses).
- **Both zeroed** is the cleanest "this feature is gone"
  semantics. The feature still exists at its index (so feature-id
  references in downstream notebooks don't dangle); it just
  produces zero activations and zero contribution.

`b_dec` (the decoder bias, applied globally to the residual sum)
is untouched. It is not feature-specific.

### Decision 6 — `CompressionReport.to_json` carries source-validation provenance

JSON layout:

```json
{
  "schema_version": 1,
  "source_checkpoint": "scratch/real-sae/.../sae_weights.safetensors",
  "source_checkpoint_sha256": "abcd...",
  "output_checkpoint": "scratch/real-sae/.../sae_weights.compressed.safetensors",
  "output_checkpoint_sha256": "wxyz...",
  "validation_report_dictionary_name": "ScaleupBlocks10",
  "validation_report_schema_version": 1,
  "strategy": "zero",
  "feature_ids": [12999, 19398, ...],
  "clusters": [
    {
      "cluster_id": 0,
      "members": [12999, 19398],
      "representative": 12999,
      "zeroed": [19398]
    },
    {
      "cluster_id": 1,
      "members": [4192, 23625],
      "representative": 4192,
      "zeroed": [23625]
    },
    {
      "cluster_id": 2,
      "members": [68, 2287, 8371, 13737],
      "representative": 8371,
      "zeroed": [68, 2287, 13737]
    }
  ],
  "n_features_zeroed": 5,
  "n_features_kept": 3,
  "n_clusters": 3
}
```

Round-trip: `CompressionReport.from_json(report.to_json(...))` is
equality-true. Floats follow the same `format(v, ".6g")` pattern
as `ValidationReport`. SHA256 of both checkpoints is included
because the operation's identity depends on what was read and
what was written; reproducing the operation requires the source
hash to verify the input was the same.

### Decision 7 — CLI: `polygram compress` with file-based inputs

```
polygram compress \
  --validation-report path/to/validation_report.json \
  --sae-checkpoint path/to/sae_weights.safetensors \
  --output-checkpoint path/to/sae_weights.compressed.safetensors \
  --strategy zero \
  --output path/to/compression_report.json \
  [--representatives 0=12999,1=4192,2=8371]
```

`--strategy` is required (no default; future-proofs for `merge`).
`--representatives` is comma-separated `cluster_id=fid` pairs;
omitting any cluster keeps the default selection for that
cluster. `--output` writes the `CompressionReport` JSON next to
the rewritten checkpoint.

### Decision 8 — One end-to-end test, synthetic-checkpoint-only

The compression action is fully deterministic and torch-free, so
a real SAE is not needed for the smoke test. Mirroring
`tests/behavioural/test_validator_predict.py`:

- Build a synthetic SAE (`_synth_sae` helper, already in the
  validator tests) with N=8 features.
- Build a synthetic `ValidationReport` with hand-picked
  `confirmed` pairs that form 3 clusters (two singletons +
  one clique).
- Run `Compressor.run(...)`; assert the output checkpoint has
  the expected zero pattern in `W_enc / b_enc / W_dec`.
- Round-trip the `CompressionReport` through JSON; assert
  equality.

A separate smoke test in `tests/test_examples.py` exercises a
worked example `examples/compress_validated.py` that takes the
fixture validation report from PR #27's worked example and
emits the compressed checkpoint. Skip path: source checkpoint
or validation report missing.
