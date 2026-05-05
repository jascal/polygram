## Context

`add-compression-action` zeros encoder columns, encoder biases, and
decoder rows for every non-representative member of a confirmed
redundancy cluster. Those slots become pristine zeros in an SAE
that otherwise retains its full row count. The slots are *capacity*
— a fresh decoder direction populated into a zeroed slot extends
the dictionary's effective coverage of the activation manifold
without changing any other feature's tensors and without changing
the SAE's overall shape.

The minimal operation that turns capacity into a populated
dictionary is *regrow*: pick a direction for each zeroed slot,
write it. Regrow is not training. It does not optimize a loss, it
does not take gradients, it does not update existing features.
It's a one-shot transform — and that simplicity is the property
that lets it ship as a small spec change with clean handoff
semantics to whatever consumes the primed checkpoint.

The hard questions the design has to answer:

1. Where do directions come from? (Strategy semantics.)
2. How do strategies translate cluster centroids / random samples
   into the four tensors that need populating? (Encode-decode
   coupling, sign convention, norm matching, bias init.)
3. How do we keep the operation deterministic for workflow
   verification? (Strategy contracts on RNG, ordering, dependence
   on input hashes.)
4. How does the chained-from-CompressionReport mode differ from
   the direct mode in observable behaviour? (Provenance fields,
   nothing else.)
5. What are the failure modes the strategy can encounter, and how
   are they reported? (Empty residual streams, degenerate k-means
   convergence, slot count larger than residual cluster count.)

Each decision below commits to a concrete answer.

## Goals / Non-Goals

**Goals:**

- A `Regrower` with two-stage `plan() / apply()` API matching the
  `Compressor` discipline.
- One concrete strategy (`residual_kmeans`) with fully-specified
  semantics from residual extraction through tensor population.
- Two construction modes (direct, chained) producing the same
  output bytes for matching inputs; the only difference is
  `RegrowReport.provenance` population.
- Determinism: bit-identical output checkpoints for
  bit-identical inputs (including RNG seed).
- Atomic checkpoint write: temp file + `os.replace`, rejecting
  source-equals-output collisions before any I/O.
- A `RegrowReport` carrying source + output sha256s, the strategy
  name and parameters, per-slot population diagnostics, and
  optional provenance back to a `CompressionReport`.
- A `polygram regrow` CLI subcommand with file-based inputs.
- Test ladder: unit tests for residual extraction, k-means
  semantics, tensor population invariants, postinit rejections,
  JSON round-trip, CLI argument paths.
- Worked example mirroring `examples/compress_validated.py`.

**Non-Goals:**

- Training, fine-tuning, gradient descent.
- A `from_epoch_report` constructor (deferred to follow-up; the
  `EpochReport` type doesn't yet exist in implementation).
- `high_decoder_norm_random` or `orthogonal_noise_scaled`
  implementations (named in the dispatcher; bodies raise
  `NotImplementedError`).
- Auto-strategy selection.
- Iteration or composition with `Compressor`/`EpochCompressor`.
- A hard "this many slots must be repopulated" success criterion.
  The strategy reports what it did; the consumer judges.

## Decisions

### Decision 1 — Strategy enum with one implemented body

```python
class RegrowStrategy(StrEnum):
    RESIDUAL_KMEANS = "residual_kmeans"
    HIGH_DECODER_NORM_RANDOM = "high_decoder_norm_random"
    ORTHOGONAL_NOISE_SCALED = "orthogonal_noise_scaled"
```

`Regrower.strategy: str` is required (no default). The dispatcher
inside `apply()` raises `NotImplementedError` with a clear "ship a
follow-up change" hint for the two unimplemented strategies. The
implemented strategy is `residual_kmeans`.

The `__post_init__` validation rejects any `strategy` value not in
the enum.

### Decision 2 — `residual_kmeans` semantics

**Step 1 — Residual extraction.** The Regrower runs the SAE's
forward pass on the cached or freshly-captured residuals and
computes the per-token residual:

```
sae_pre = (residuals - b_dec) @ W_enc + b_enc      # (n_tokens, n_features)
sae_act = np.maximum(sae_pre, 0.0)                 # ReLU; SAEs are positive-activation by convention
sae_recon = sae_act @ W_dec + b_dec                # (n_tokens, d_model)
residual_stream = residuals - sae_recon            # (n_tokens, d_model)
```

The residual stream is the "what the SAE failed to reconstruct"
signal at every token. Strategy variants that don't need residuals
(future `high_decoder_norm_random`) skip this step.

**Step 2 — K-means clustering.** Run `sklearn.cluster.KMeans(
n_clusters=K, n_init=n_init, random_state=seed,
algorithm='lloyd')` on `residual_stream`, where:

- `K = len(zeroed)`, the number of slots to repopulate. (One
  centroid per slot. Slots are populated independently; if the
  user wants fewer-clusters-than-slots, they should run with a
  smaller `zeroed` set.)
- `n_init = 4`. Default in sklearn ≥ 1.4 is `'auto'` which picks
  10; 4 is faster and gives the determinism guarantee a stable
  seed. Configurable via `Regrower.n_init`.
- `random_state = self.seed`. Default `seed = 0`. The seed is
  surfaced as a `Regrower` field and recorded in `RegrowReport`
  for reproducibility.
- `algorithm = 'lloyd'`. The default `'auto'` switches to
  `'elkan'` on dense data which is faster but produces different
  numeric tiebreaks. Pinning to `'lloyd'` keeps determinism
  across sklearn versions.

If `K = 0` (no zeroed slots), the strategy is a no-op: the regrown
checkpoint equals the source byte-for-byte (modulo the safetensors
metadata's potentially-different mtime; the spec asserts the
*tensors* are equal, not the file bytes).

**Step 3 — Centroid-to-tensor mapping.** For each zeroed slot
`fid` and its assigned centroid `c[k]`:

- **Decoder row** `W_dec[fid, :] = c[k] / max(‖c[k]‖, eps)`. The
  centroid is normalized to unit L2 norm and assigned as the
  decoder direction. Surviving features in jbloom-style SAEs
  have decoder rows with varying L2 norms (typically 0.5–2.5 on
  GPT-2 small layer 10); the regrown row is unit-norm (a
  conservative choice — the slot becomes "alive but quiet" until
  any downstream fine-tune scales it).
- **Encoder column** `W_enc[:, fid] = W_dec[fid, :].T`. SAE
  trainers typically tie `W_enc = W_dec.T` at initialization;
  regrow inherits that convention. The decoder is the directional
  truth; the encoder is its reciprocal.
- **Encoder bias** `b_enc[fid] = 0`. Zero bias means the slot's
  pre-activation starts at `(residual - b_dec) @ W_enc[:, fid]`
  with no offset; downstream training can shift it.
- **Decoder bias** `b_dec` is untouched (global; not feature-
  specific).

The unit-norm + zero-bias choice produces a "primed but quiet"
slot: it has a direction and will fire on activations aligned
with that direction, but it has no learned amplitude or offset.
This is the most conservative initialization — any downstream
fine-tune can re-discover whatever amplitude the slot wants
without having to first overcome a bad random init.

**Step 4 — Slot-to-centroid assignment.** Sort `zeroed` ascending
by feature id; assign centroid `k` to the k-th slot in that
order. Determinism: this means re-running with the same `zeroed`
set produces the same slot→centroid pairing across runs, even
though k-means cluster ids are arbitrary internally. K-means
itself returns clusters in label-order matching the centroid
array's row order, which is deterministic under fixed
`random_state` + fixed `algorithm` + fixed input array.

**Step 5 — Per-slot diagnostics.** For each slot, populate a
`SlotPopulation` record:
- `feature_id`: the zeroed feature id this slot occupies.
- `cluster_size`: count of residual tokens whose nearest centroid
  was this one.
- `decoder_norm`: 1.0 by construction (post-normalization).
- `encoder_norm`: norm of the encoder column post-population
  (equal to decoder_norm in this strategy, kept as a separate
  field so future strategies that decouple encoder from decoder
  can populate it differently).

A slot whose `cluster_size` is 0 (sklearn's k-means *can* produce
empty clusters under degenerate input) is **left zero, not
populated**. The strategy reports this via
`RegrowReport.n_slots_left_zero`. Rationale: a centroid with no
support is a noise direction, not a missing-feature signal.
Better to leave the slot empty than populate it with random
geometry.

### Decision 3 — Two construction modes, one observable difference

```python
@dataclass
class Regrower:
    sae_checkpoint: Path
    strategy: str
    zeroed: set[int]
    seed: int = 0
    n_init: int = 4

    # One of these two is required:
    prompts: Sequence[str] | None = None
    cached_residuals: np.ndarray | None = None

    # Only when prompts is supplied:
    model_name: str = "gpt2"
    layer: int = 10
    device: str | None = None

    @classmethod
    def from_compression_report(
        cls,
        report: CompressionReport,
        sae_checkpoint: Path,
        *,
        strategy: str,
        prompts: Sequence[str] | None = None,
        cached_residuals: np.ndarray | None = None,
        seed: int = 0,
        n_init: int = 4,
        model_name: str = "gpt2",
        layer: int = 10,
        device: str | None = None,
    ) -> "Regrower":
        zeroed = {
            fid
            for cluster in report.plan.clusters
            for fid in cluster.zeroed
        }
        instance = cls(
            sae_checkpoint=sae_checkpoint,
            strategy=strategy,
            zeroed=zeroed,
            seed=seed,
            n_init=n_init,
            prompts=prompts,
            cached_residuals=cached_residuals,
            model_name=model_name,
            layer=layer,
            device=device,
        )
        # Stash provenance for the eventual RegrowReport.
        object.__setattr__(
            instance,
            "_provenance",
            {
                "compression_report_source_sha256":
                    report.source_checkpoint_sha256,
                "compression_report_output_sha256":
                    report.output_checkpoint_sha256,
                "compression_report_dictionary_name":
                    report.validation_report_dictionary_name,
            },
        )
        return instance
```

The two constructors produce instances with bit-identical fields
*except* for an optional internal `_provenance` dict. The dict is
empty for the direct constructor, populated for
`from_compression_report`. `RegrowReport.provenance` reflects this
field: the JSON serialization carries an empty `{}` for the direct
case, a populated map for the chained case.

`__post_init__` enforces:

- exactly one of `prompts` or `cached_residuals` is supplied
  (XOR);
- `sae_checkpoint` exists on disk;
- `strategy` is in the enum;
- `seed >= 0`;
- `n_init >= 1`;
- if `prompts` is supplied: `prompts` non-empty, `layer >= 0`
  (the `allow_layer_zero` discussion is irrelevant here — regrow
  doesn't run ablations, just one forward pass for residual
  capture);
- `zeroed` is a set of non-negative integers, all of which are
  valid feature ids in the SAE checkpoint (i.e., `< W_dec.shape[0]`).
  This last check requires reading the checkpoint header (cheap,
  one safetensors metadata read).

### Decision 4 — Atomic write, source-equals-output rejection

`apply(plan, output_checkpoint)`:

1. Resolve `output_checkpoint`. Reject if equal to
   `self.sae_checkpoint` (`Path.resolve()` comparison; same
   contract `Compressor` ships).
2. Create the parent directory if missing.
3. Open a `NamedTemporaryFile(dir=parent, prefix=".regrow.",
   suffix=".tmp", delete=False)`; close it immediately so we have
   a path.
4. Call `safetensors.numpy.save_file` on the rewritten state-dict
   to the temp path.
5. `os.replace(tmp_path, output_checkpoint)` — atomic on POSIX
   when source and dest are on the same filesystem, which is the
   case here by construction.
6. On any exception in steps 4–5, unlink the temp file (best
   effort) and re-raise.
7. Compute output sha256 over the final file's bytes; populate
   `RegrowReport.output_checkpoint_sha256`.

### Decision 5 — Determinism contract

For the `residual_kmeans` strategy, given:

- the same `sae_checkpoint` (same bytes — sha256-checkable),
- the same `zeroed` set,
- the same `cached_residuals` array (or the same `prompts` +
  `model_name` + `layer` + `device` — but the `prompts` path
  inherits whatever non-determinism torch + transformers carries,
  see caveat below),
- the same `seed` and `n_init`,

the resulting checkpoint MUST be byte-identical when written by
the same Polygram + sklearn + numpy versions.

The contract holds for the `cached_residuals` path
unconditionally. The `prompts` path additionally requires that the
torch forward pass be deterministic — for CPU and MPS this is
generally true given torch's default determinism settings; CUDA
introduces non-determinism via cuDNN algorithm selection. The spec
notes this caveat but does not require a `torch.use_deterministic_algorithms(True)`
call — that flag has performance implications and is the
consumer's call.

A scenario in `specs/compression/spec.md` asserts byte-identical
determinism on the `cached_residuals` path.

### Decision 6 — Failure mode: degenerate k-means

Failure modes the strategy can encounter:

1. **`zeroed` is empty.** No-op: write a checkpoint identical in
   tensors to the source (modulo metadata). `n_slots_repopulated
   = 0`, `n_slots_left_zero = 0`.
2. **`residual_stream` is all-zero or near-zero.** Indicates the
   SAE is a perfect reconstructor on this prompt set —
   exceedingly unlikely on real SAEs but possible on a synthetic
   fixture where the SAE has been hand-tuned to reconstruct the
   prompts exactly. K-means on all-zero input produces all-zero
   centroids. The strategy detects this (`residual_stream.std()
   < 1e-9`) and raises `RuntimeError("residual stream has no
   signal — try a more diverse prompt set")` rather than
   silently populating slots with zeros.
3. **`n_residual_tokens < K`.** Fewer tokens than slots. K-means
   degenerates (sklearn raises). The strategy catches this
   pre-call and raises a clearer error: "n_residual_tokens=N is
   less than the K=M zeroed slots; use more prompts or fewer
   slots."
4. **K-means produces empty clusters.** Sklearn's k-means can
   produce a cluster with zero assigned points if the
   initialization is unlucky on a small dataset. Per Decision 2
   step 5, slots assigned to empty clusters are left zero and
   recorded in `RegrowReport.n_slots_left_zero`.

Failure modes (1) and (4) are recoverable (the strategy reports
and continues); (2) and (3) are hard errors raised at
`plan()`-time so the caller never gets to `apply()` with a
broken plan.

### Decision 7 — Lazy torch import

`Regrower` itself is torch-free. The `prompts` construction path
triggers torch import inside `_capture_residuals(self)`, called
during `plan()` only when `cached_residuals is None`. The
`cached_residuals` path is fully torch-free. The
`from_compression_report` constructor doesn't change this — it
inherits whichever path the user supplies via `prompts` or
`cached_residuals`.

This keeps the orca-lang demo path (which will likely use
`cached_residuals` from a prior workflow node) on a torch-free
import surface.

### Decision 8 — `RegrowReport` JSON layout

```json
{
  "schema_version": 1,
  "source_checkpoint": "...",
  "source_checkpoint_sha256": "...",
  "output_checkpoint": "...",
  "output_checkpoint_sha256": "...",
  "strategy": "residual_kmeans",
  "n_slots_repopulated": 5,
  "n_slots_left_zero": 0,
  "feature_ids": [0, 1, 2, ..., 24575],
  "plan": {
    "strategy": "residual_kmeans",
    "n_residual_tokens": 654,
    "zeroed_input": [42, 100, 256, 512, 1024],
    "feature_ids": [0, 1, 2, ..., 24575],
    "slots": [
      {"feature_id": 42, "cluster_size": 87,
       "decoder_norm": 1.000000, "encoder_norm": 1.000000},
      {"feature_id": 100, "cluster_size": 134, ...},
      ...
    ]
  },
  "strategy_params": {
    "seed": 0,
    "n_init": 4
  },
  "provenance": {}
}
```

`RegrowReport.from_json(report.to_json()) == report` holds.
Floats use the `format(v, ".6g")` discipline matching
`CompressionReport`. The `provenance` field is `{}` for the
direct constructor and a populated map for
`from_compression_report`.
