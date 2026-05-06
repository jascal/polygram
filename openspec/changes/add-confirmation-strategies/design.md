## Context

`Compressor` consumes a `ValidationReport` with a `confirmed` field — a tuple of `(i, j)` feature-id pairs deemed redundant. Currently the only path to a populated `confirmed` is `BehaviouralValidator.validate()`, which lazy-imports torch + transformers and hooks `model.transformer.h[layer]` — a GPT-2-specific forward-pass hook. Any SAE from a different architecture (Gemma Scope, Llama, Mistral) requires bespoke scripting to produce a synthetic `ValidationReport`.

`BehaviouralValidator` already has a `predict()` stage (torch-free, computes decoder cosine²) but it always emits `gate_pass=False`, leaving `confirmed` empty. The geometry is already computed; what's missing is a path that promotes geometry into confirmed pairs without requiring a forward pass.

`convert_gemma_scope_to_safetensors.py` was written for polygon geometry work only and discards `W_enc`, `b_enc`, `b_dec`, `threshold`. The compressor's `apply_zero` strategy calls `_load_sae_full()`, which requires all four keys. The script's scope was narrow-by-design but now creates a silent footgun for anyone trying to compress Gemma Scope SAEs.

## Goals / Non-Goals

**Goals:**
- A `Confirmer` protocol (`run() -> ValidationReport`) that all strategies satisfy
- `DecoderGeometryConfirmer`: confirms pairs where decoder cosine² ≥ threshold; numpy-only
- `ClusterConfirmer`: confirms all within-cluster pairs from a `SelectionReport`; numpy-only
- `BehaviouralValidator` documented as the behavioural strategy; no API change to it
- `convert_gemma_scope_to_safetensors.py` preserves all SAE keys by default
- Both new strategies exported from `polygram.__init__`

**Non-Goals:**
- Changing `Compressor`, `ValidationReport`, or `BehaviouralValidator` APIs
- A `merge` compression strategy (deferred, existing note in compressor.py)
- Geometry-based strategy for architectures other than residual-stream SAEs
- Automatic strategy selection

## Decisions

### Decision 1 — Protocol, not ABC

Use `typing.Protocol` (structural subtyping) rather than an `ABC`. `BehaviouralValidator` already has `.run() -> ValidationReport`; making it satisfy the protocol retroactively requires zero changes to its code. An ABC would require it to inherit from a new base class — a breaking change to its MRO and a reason to touch tested, stable code.

**Alternative considered**: Abstract base class with `register()`. Rejected: requires modifying `BehaviouralValidator`.

### Decision 2 — `DecoderGeometryConfirmer` threshold on decoder cosine², not Polygram overlap

The decoder cosine² (`decoder_overlap` in `CandidatePair`) is the raw geometry signal, independent of any encoding or φ parameter. Polygram overlap folds in the encoding's φ-space structure, which the user may not have configured yet when they want to confirm. Using decoder cosine² makes the confirmer usable before or without a `Dictionary`.

**Default threshold: 0.8** — consistent with the existing `BehaviouralValidator.polygram_overlap_threshold` of 0.7 (slightly stricter on the raw geometry side, where the signal is noisier without behavioural gating).

**Alternative considered**: threshold on Polygram overlap (requires a `Dictionary`). Rejected: adds a dependency on encoding choice, complicates the API.

### Decision 3 — `ClusterConfirmer` takes a `SelectionReport`, not raw cluster assignments

`SelectionReport.cluster_assignments` (a `dict[str, str]` mapping feature name to cluster name) is already computed by `from_sae_lens`. Accepting `SelectionReport` directly avoids re-clustering and keeps the two calls naturally chained:

```python
dictionary, report = from_sae_lens(records, ids, n_clusters=6)
val_report = ClusterConfirmer(report, sae_checkpoint).run()
compressor = Compressor(val_report, sae_checkpoint)
```

**Alternative considered**: accept raw `dict[int, str]` cluster map. Rejected: requires caller to re-derive what `from_sae_lens` already computed; doesn't compose as naturally.

### Decision 4 — New module `polygram/confirmation/`

The two strategies live in `polygram/confirmation/__init__.py` (or `confirmer.py` + per-strategy files). Putting them in `polygram/behavioural/` would mislead — geometry strategies have no behavioural component. A new `confirmation/` module parallels the existing `behavioural/` and `compression/` structure.

### Decision 5 — `convert_gemma_scope_to_safetensors.py` preserves all keys by default

The script's docstring describes it as a converter for "projection-geometry work" and explicitly converts only `W_dec`. Since the compressor silently fails when other keys are missing, always preserving all keys is safer. The script gains a `--dec-only` flag for callers that genuinely want the projection-only file (e.g., the polygon geometry pipeline).

**Alternative considered**: add a `--full` flag, keep dec-only as default. Rejected: the default that causes a silent footgun should be the safe one. Callers who want dec-only can pass `--dec-only` explicitly.

## Risks / Trade-offs

- **Geometry-only confirmation is weaker than behavioural** — high decoder cosine² is necessary but not sufficient for behavioural redundancy. `DecoderGeometryConfirmer` may confirm pairs that co-fire differently in practice. Mitigation: document clearly; the strategy name makes the tradeoff explicit.

- **`ClusterConfirmer` threshold sensitivity** — cluster membership depends on `n_clusters` and the k-means random seed. Two features in the same cluster may have modest cosine similarity. Mitigation: surface the within-cluster cosine² distribution in the report; let the caller inspect before compressing.

- **`ValidationReport` fields not meaningful for geometry strategies** — `model_name`, `n_prompts`, `n_tokens`, jaccard fields are NaN or zero. Mitigation: set `model_name="geometry"` as a sentinel; document which fields are populated vs NaN for each strategy.

## Open Questions

- Should `ClusterConfirmer` expose a `min_cosine` guard (refuse to confirm within-cluster pairs below some floor)? Left open — can be added in a follow-up without breaking the interface.
- Should the `Confirmer` protocol be exported publicly or kept internal? Current decision: export it so users can implement custom strategies.
