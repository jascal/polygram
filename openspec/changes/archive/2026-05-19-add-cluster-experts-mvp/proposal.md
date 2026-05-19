## Why

`ClusteredDictionary` is already a list-of-Dictionary-blocks plus a
sparse cross-block adjacency. Downstream consumers (sae-forge, bio-SAE
work) want a routing primitive on top of that: given a per-feature
activation vector, return the top-k expert blocks responsible. This
is the missing "expert formation" verb in polygram's vocabulary —
the analytic-side mirror of MoE routing.

The MVP slice is intentionally tight: a thin wrapper that *names* the
existing clustered-block structure as `ExpertDictionary` and adds the
one method downstream actually needs — `route(activations, top_k)`.
No new clustering strategies (cosine is already implemented; the
existing co_firing placeholder still raises NotImplementedError, so
the MVP exposes `method="cosine"` only and reserves `method=` for the
co_firing wiring whenever that lands). No router training. No bio
metrics. No new runtime dependencies.

If users hit the wall on this surface, the follow-ups (Louvain,
HDBSCAN, MLP router, GO-term enrichment) get scoped against real
usage data instead of speculation.

## What Changes

- New module `polygram/experts.py` exposing:
  - `ExpertDictionary` — frozen dataclass holding `experts: tuple[Dictionary, ...]`
    plus a precomputed feature→expert map. Constructed from an existing
    flat `Dictionary` via `cluster_experts(...)`.
  - `cluster_experts(dictionary, decoder_vectors, *, method="cosine",
    coherence_threshold=0.3, max_features_per_expert=None,
    activations=None)` — factory that delegates block formation to the
    existing `build_clustered_dictionary` path and wraps the result.
- `ExpertDictionary.route(activations, top_k)` — top-k expert indices
  by summed per-expert activation. Numpy-only; no torch dependency.
- Re-export both symbols from `polygram/__init__.py`.
- New tests under `tests/test_experts.py` covering planted-cluster
  recovery, routing correctness, and round-trip with the source
  Dictionary's primitives.

## Impact

- **Affected specs**: new capability `experts`.
- **Affected code**: new `polygram/experts.py`; `polygram/__init__.py`
  re-exports.
- **No new runtime deps.** Reuses `clustered_dictionary` machinery.
- **Risk**: low. Strictly additive — no existing API touched.

## Explicitly out of scope

- `method="coactivation"` — blocked on the existing `co_firing`
  `NotImplementedError` in `clustered_dictionary._form_blocks_co_firing`.
  Will be lit up automatically when that stub is implemented.
- `method="louvain"` / `method="hdbscan"` — new heavyweight deps;
  premature without usage data on cosine.
- Trained router (MLP) — belongs in sae-forge, where torch already lives.
- Bio-specific scoring (GO enrichment, motif overlap) — belongs in a
  downstream package or `polygram[bio]` extra.
- `min_cluster_size` / `max_experts` post-processing — straightforward
  add-on once the base API has consumers; deferred.
