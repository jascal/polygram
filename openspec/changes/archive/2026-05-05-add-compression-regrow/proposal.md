## Why

`add-compression-action` (PR #28 spec, commit `7bdc7e7` impl) and
`add-compression-epoch` (commit `c8fbb52` spec) close the
*identification + collapse* half of the loop: the validator finds
behavioural redundancies, the compressor zeroes the encoder column,
encoder bias, and decoder row of every non-representative member.
Each compressed checkpoint carries the same total slot count as the
source (24,576 features for GPT-2 small layer 0; 16,384 for
Gemma-Scope 2B layer 12) — but a fraction of those slots are now
mathematically pristine zeros. The §4.4 panel zeros 5/8 (62.5%);
the fresh anchor panel zeros 6/8 (75%); a full-SAE epoch on the
calibration distribution would zero some smaller global rate.

Those slots are usable. The four tensors backing each slot
(`W_enc[:, fid]`, `b_enc[fid]`, `W_dec[fid, :]`, the global
`b_dec` is shared) are independent of every other slot's tensors —
the SAE's per-feature semantics live entirely in those rows. A
fresh decoder direction populated into a zeroed slot is, by
construction, a feature that fires somewhere in the activation
distribution but wasn't represented before compression. The slots
become *capacity*.

This change ships the regrow primitive — the smallest operation
that turns a compressed checkpoint back into a fully-populated one
by repopulating zeroed slots with new directions chosen from the
SAE's own activation residuals. It does **not** ship fine-tuning,
training infrastructure, or any concept of "improvement." It ships
a deterministic, audit-trail-heavy transform: `(compressed
checkpoint, zeroed feature ids, activation stream) → primed
checkpoint with the zeroed slots populated`.

The primed checkpoint is the contract handoff. Downstream consumers
— a workflow node in orca-lang, a fine-tune script outside this
repo, or a research notebook measuring whether the primed SAE moves
closer to or further from the validator's gates — receive a
checkpoint with the same shape, the same shared `b_dec`, and the
same untouched representative weights, but with previously-silent
slots now carrying directions extracted from the residual
distribution. What happens after that handoff is not Polygram's
concern.

## What Changes

### `compression` capability — extended

Add `polygram.compression.Regrower`, a torch-free primitive that
composes existing pieces:

- **Two-stage API**: `plan() -> RegrowPlan` (cheap; reads the
  source checkpoint and the prompt-set residuals to decide which
  zeroed slots to populate and what directions to use) and
  `apply(plan, output_checkpoint=...) -> RegrowResult` (writes the
  new checkpoint atomically). `run(output_checkpoint=...)` is the
  convenience wrapper.
- **Two construction modes** (both supported, neither preferred):
  - `Regrower(source_checkpoint=..., zeroed=set[int], ...)` —
    direct: caller supplies the zeroed feature-id set explicitly.
    The orca-lang demo path and the isolation-test path use this.
  - `Regrower.from_compression_report(report: CompressionReport,
    sae_checkpoint=..., ...)` — provenance-chained: the zeroed
    set comes from the report's `clusters[*].zeroed`, and the
    `RegrowReport` carries the upstream `CompressionReport`'s
    sha256s for full provenance.
  - (A symmetric `from_epoch_report` will arrive once
    `add-compression-epoch` is implemented; this change explicitly
    leaves that as a follow-up surface to add when `EpochReport`
    exists.)
- **One initial strategy**: `residual_kmeans`. Other strategies
  (`high_decoder_norm_random`, `orthogonal_noise_scaled`) are
  named in the design's strategy enum but are deliberately not
  implemented in v0 — they will land as separate changes if
  evidence justifies them. The strategy field is required, not
  defaulted, matching the `Compressor.strategy` discipline so that
  call sites are explicit and future-proof against new strategies
  changing default behaviour.
- **Activation-stream input**: the strategy needs SAE input
  activations for the prompt set, not raw text. The `Regrower`
  accepts either:
  - `prompts: Sequence[str]` + `model_name: str` + `layer: int`
    (the validator-style path: lazy-import torch, run one forward
    pass to capture residuals at the named hook), OR
  - `cached_residuals: np.ndarray` (`shape = (n_tokens, d_model)`,
    pre-captured by an upstream caller — the orca-lang workflow
    path, where residuals come from a prior workflow node).

### `RegrowPlan`, `RegrowReport`, `RegrowResult` — new

- **`SlotPopulation`** dataclass: `feature_id: int`,
  `cluster_size: int` (number of residual tokens that fed this
  slot's centroid; 0 for non-residual strategies),
  `decoder_norm: float`, `encoder_norm: float`. One per repopulated
  slot.
- **`RegrowPlan`** dataclass: `slots:
  tuple[SlotPopulation, ...]`, `zeroed_input:
  tuple[int, ...]` (the zeroed-set the plan was built against;
  for audit), `strategy: str`, `n_residual_tokens: int`,
  `feature_ids: tuple[int, ...]` (the SAE-wide feature-id list,
  matching `CompressionReport`'s convention).
- **`RegrowReport`** dataclass: `schema_version`,
  `source_checkpoint`, `source_checkpoint_sha256`,
  `output_checkpoint`, `output_checkpoint_sha256`, `strategy`,
  `plan: RegrowPlan`, `n_slots_repopulated: int`,
  `n_slots_left_zero: int` (slots that were in `zeroed_input`
  but the strategy chose not to repopulate — e.g., insufficient
  residual signal), `provenance:
  dict[str, str]` (a free-form sha256 + path map carrying any
  upstream `CompressionReport`'s identifying hashes; populated
  when constructed via `from_compression_report`, empty when
  constructed directly).
- **`RegrowResult`**: `plan`, `report`, `output_checkpoint:
  Path`, `dictionary: Dictionary` (rebuilt via `from_sae_lens`
  on the rewritten checkpoint).

### `cli` capability — new `polygram regrow` subcommand

`polygram regrow` wraps `Regrower.run()`:

```
polygram regrow \
  --sae-checkpoint path/to/sae_weights.compressed.safetensors \
  --output-checkpoint path/to/sae_weights.primed.safetensors \
  --output path/to/regrow_report.json \
  --strategy residual_kmeans \
  ( --zeroed-list 42,100,256 | --compression-report path/to/compression_report.json ) \
  ( --prompts path/to/prompts.txt --layer 10 --model gpt2 [--device auto]
  | --cached-residuals path/to/residuals.npy ) \
  [--seed 0]
```

The `--zeroed-list` and `--compression-report` flags are mutually
exclusive (CLI rejects supplying both). Same shape for `--prompts`
vs `--cached-residuals`.

### No new optional extra

Reuses `[behavioural]` (only when `--prompts` is supplied; lazy
torch import inside `Regrower.plan()` then) plus base
`safetensors` + `numpy`. With `--cached-residuals`, the path stays
torch-free.

## What this proposal explicitly does NOT do

- **Train, fine-tune, or update any SAE weights via gradient
  descent.** Regrow is a one-shot population: "fill the zero with
  this direction." Subsequent training is the consumer's problem.
- **Add training infrastructure to Polygram.** No optimizer, no
  loss, no training data loader, no epoch counter. The regrown
  checkpoint is the artifact; downstream tools (orca-lang
  workflows, external trainers) handle the rest.
- **Ship `high_decoder_norm_random` or `orthogonal_noise_scaled`
  strategies.** Named in the strategy enum so the dispatcher is
  open for extension; bodies raise `NotImplementedError`. Future
  changes that need them will land them on evidence (e.g., "we
  measured residual_kmeans collapses too often on Gemma-Scope —
  here's the comparison").
- **Validate the regrown SAE.** A user running `polygram validate`
  on the primed checkpoint is the standard post-regrow audit;
  Regrower itself does not invoke the validator. Its output is a
  checkpoint, not a quality claim.
- **Modify the source checkpoint.** `apply()` writes a new file;
  `output_checkpoint == sae_checkpoint` is rejected at
  construction. Same hard contract `Compressor` ships.
- **Inherit `MAX_FEATURES_PER_DICTIONARY` constraints.** The 8-
  feature cap applies to `Dictionary` construction (the rung-1
  MPS encoding cap). Regrow operates on raw tensors; it can
  populate any number of zeroed slots in one apply, including
  cases where the SAE has thousands of zeroed slots from many
  epoch iterations.
- **Touch `b_dec`.** The decoder bias is global; compression left
  it untouched and regrow leaves it untouched. The two operations
  are symmetric in this respect.
- **Auto-pick the strategy.** The user supplies `--strategy`
  explicitly. There is no inference of "best strategy for this
  SAE" — a future change can ship a heuristic if real workloads
  push for it.
- **Iterate or compose with `Compressor`/`EpochCompressor`.**
  Each `Regrower.run()` is one shot. Loops that interleave
  compression and regrowth are the orchestrator's concern (and
  defer to a future change once we have evidence about whether
  iterated compress→regrow→compress pays).

## Discussion

### Why `residual_kmeans` first

The post-compression residual stream — `activation -
SAE_reconstruct(activation)` — is exactly the signal the SAE was
*supposed* to reconstruct but didn't. Components that consistently
appear in the residual across many tokens are the directions the
current dictionary fails to represent. K-means on those residuals
finds clustered failure modes; cluster centroids are reasonable
candidate decoder directions for "what's missing."

This is the smallest defensible answer to "where do new directions
come from?" — it operates entirely within the SAE's own behavioural
distribution (no auxiliary models, no human-supplied seeds), it's
deterministic given a fixed RNG seed, and it's one numpy call away
once residuals are cached. The §4.1 finding (PR #18: Polygram-
predicted overlap tracks decoder cosine at Spearman 0.94 on the
real SAE) suggests that decoder directions live in a structured
sub-manifold; cluster centroids of *failure-mode residuals* are a
principled way to populate previously-empty parts of that manifold.

Other strategies are named in the enum because they are reasonable
alternatives a future change might ship:

- `high_decoder_norm_random`: pick random directions, scaled to
  the surviving features' decoder-norm distribution. Maximally
  cheap, no activation pass needed; useful as a baseline against
  which `residual_kmeans` proves it has signal.
- `orthogonal_noise_scaled`: project a random direction onto the
  orthogonal complement of the surviving decoder rows, then
  rescale. Useful when residual k-means produces redundant
  centroids (collapses to the same cluster across runs).

Shipping just `residual_kmeans` keeps the spec compact and avoids
locking in default behaviour for two strategies before we have
data on which is preferable.

### Why provenance is structural, not free-form

`CompressionReport.to_json()` carries source + output sha256s plus
the upstream `ValidationReport`'s `dictionary_name`. The natural
chain is: validator → compressor → regrower, with each stage's
report linking back to the previous one's identifying hash. The
`Regrower`'s `RegrowReport.provenance` field is therefore a
typed dict — `{"compression_report_sha256": "...",
"compression_report_dictionary_name": "..."}` — populated by
`from_compression_report` and empty for the direct constructor.
That makes "which compression run produced this regrown
checkpoint?" answerable purely from on-disk artifacts, without
needing a session log.

### Why both construction modes

The chained `from_compression_report` is the natural shape inside
a Polygram-only workflow (compress → regrow). The direct
`Regrower(zeroed=...)` constructor is the natural shape for the
orca-lang workflow (every node consumes a previous node's output;
the workflow language tracks the provenance chain at the FSM
level, not inside Polygram). Both constructors emit the same
`RegrowResult`; the only difference is whether `RegrowReport.provenance`
is populated.

### Why determinism matters

A workflow language consuming Regrower's output benefits from
"same input → same output" semantics: the workflow's transition
guards can verify a regrown checkpoint by recomputing it. The
spec therefore commits to **bit-identical determinism** for
`residual_kmeans` given fixed inputs (source checkpoint sha256,
zeroed set, residual array, strategy parameters, seed). This is
achievable for the strategies named (k-means on a fixed numpy
array with a fixed seed is deterministic; no GPU-nondeterminism
applies to the strategies as designed). The spec asserts this as
a requirement so future strategy additions inherit it.
