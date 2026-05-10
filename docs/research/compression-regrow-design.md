# Compression regrow — design notes

> Engineering note pointing at the
> [`add-compression-regrow`](../../openspec/changes/add-compression-regrow/)
> spec. Captures one observation from the §4.1 / §4.4 evidence base
> that justifies the default `residual_kmeans` strategy choice.

## Pointer

The full spec for the regrow primitive lives at
[`openspec/changes/add-compression-regrow/`](../../openspec/changes/add-compression-regrow/).
That directory carries `proposal.md`, `design.md` (the eight-decision
table), `tasks.md` (the implementation checklist), and the
capability-spec deltas under `specs/`.

## Why `residual_kmeans` is the default strategy

The post-compression residual stream — `activation -
SAE_reconstruct(activation)` — is exactly the signal the SAE was
*supposed* to reconstruct but didn't. Components that consistently
appear in the residual across many tokens are the directions the
current dictionary fails to represent. K-means on those residuals
finds clustered failure modes; cluster centroids are reasonable
candidate decoder directions for "what's missing."

PR #18 (`docs/research/decoder-gram-validity.md`) measured Spearman
0.94 between Polygram-predicted overlap and the real-decoder
squared-cosine Gram on the §4.4-class SAE — i.e., decoder directions
live in a structured sub-manifold that's well-described by their
pairwise cosine geometry. Cluster centroids of *failure-mode
residuals* are a principled way to populate previously-empty parts
of that manifold: the centroids are themselves directions in
d_model-space, and they're chosen to span exactly the regions the
surviving features don't already cover.

## Why the strategy hands a "primed but quiet" slot

Each populated slot has `‖W_dec[fid, :]‖ = 1` (unit-normed centroid),
`W_enc[:, fid] = W_dec[fid, :]` (decoder-encoder tie at init, the
SAE-trainer convention), and `b_enc[fid] = 0` (no offset). This is
the most conservative initialization — the slot has a direction and
will fire on activations aligned with that direction, but it has no
learned amplitude or bias offset. Whatever downstream consumer
takes the primed checkpoint (an orca-lang fine-tune workflow node,
a research notebook measuring redundancy regrowth without
fine-tuning, etc.) can re-discover whatever amplitude the slot
wants without having to first overcome a bad random init.

The "primed but quiet" framing is also why `b_dec` is left
untouched — the decoder bias is global and absorbing-into-it
doesn't help the slot's reconstruction. Only the slot's specific
encoder and decoder rows change.

## Why determinism is asserted on the cached_residuals path only

The `Regrower(cached_residuals=...)` construction path is fully
deterministic given fixed (`sae_checkpoint`, `zeroed`, `residuals`,
`seed`, `n_init`) inputs: numpy + sklearn KMeans with
`algorithm='lloyd'` and a fixed `random_state` produces byte-
identical centroids across runs. The orca-lang workflow case feeds
this path — residuals come from a prior workflow node, the regrower
is a deterministic transformation, the workflow language can verify
the transition by recomputing.

The `Regrower(prompts=...)` construction path is only as
deterministic as torch + transformers' forward pass. CPU and MPS
are generally deterministic; CUDA is not without explicit
`use_deterministic_algorithms(True)` flags, which carry performance
implications. The spec calls this out as a caveat rather than
requiring users to make the call.

## Capping regrowth with `top_k`

`RegrowConfig.top_k` (and the matching `Regrower.from_compression_report(top_k=...)`
kwarg) caps the number of zeroed slots regrown per call. Selection is
plan order — the first `top_k` slots in `RegrowPlan.zeroed_input` order
(ascending by feature id). Remaining zeroed slots stay zero in the
output checkpoint.

When to use it: an adaptive caller (e.g. sae-forge's `adaptive-regrow`
controller) wants to drive a per-cycle growth count from external
signals — current basis size vs target, damping schedule, etc. The
controller computes "+N slots" and passes `top_k=N`; polygram
executes the count without needing to know why.

Guarantees:

- `top_k=None` (the default) is byte-identical to the pre-change
  behavior — every zeroed slot regrown. This is the load-bearing
  acceptance gate; the test in `tests/compression/test_regrow_top_k.py`
  pins it empirically.
- `top_k=0` is a valid no-op: the output checkpoint equals the
  source checkpoint, byte-for-byte.
- `top_k >= len(zeroed)` is a no-op cap — equivalent to `top_k=None`.
- `top_k < 0` is rejected at config / constructor `__post_init__`
  with a `ValueError` naming the field and value.

Selection strategies beyond plan order (by cluster size, by feature
id, caller-supplied list) are deferred to a follow-up
(`regrow-selection-strategies`) once a downstream consumer needs them.

## See also

- [`compression-action-design.md`](compression-action-design.md) —
  the upstream half (component-first compression).
- [`add-compression-regrow/proposal.md`](../../openspec/changes/add-compression-regrow/proposal.md) —
  what changes and why.
- [`add-compression-regrow/design.md`](../../openspec/changes/add-compression-regrow/design.md) —
  the eight-decision table.
- [`decoder-gram-validity.md`](decoder-gram-validity.md) — the §4.1
  evidence (Spearman 0.94) that decoder directions live in a
  structured sub-manifold.
