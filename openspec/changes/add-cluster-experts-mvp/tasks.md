## 1. New `experts` module

- [x] 1.1 Create `polygram/experts.py`. Implement `ExpertDictionary` as a frozen dataclass with fields `experts: tuple[Dictionary, ...]`, `source: Dictionary`, and `_feature_to_expert: tuple[int, ...]` (precomputed map for O(N) routing).
- [x] 1.2 Construction-time invariants: every `source.features` name appears in exactly one expert; `_feature_to_expert` length equals `len(source.features)`; experts are non-empty.
- [x] 1.3 `ExpertDictionary.n_experts` and `.n_features` properties.
- [x] 1.4 `ExpertDictionary.route(activations, top_k)` returning `list[int]` of expert indices ordered by descending summed activation. Validates `activations.shape == (n_features,)` and `1 <= top_k <= n_experts`.

## 2. `cluster_experts` factory

- [x] 2.1 Add `cluster_experts(dictionary, decoder_vectors, *, method, coherence_threshold, max_features_per_expert, activations)` in the same module.
- [x] 2.2 `method="cosine"` (the only valid value in MVP) dispatches to the existing `build_clustered_dictionary(strategy="cosine", ...)` path with `cosine_threshold=coherence_threshold` and `block_size_max=max_features_per_expert`. The encoding is read off the source dictionary.
- [x] 2.3 `method="coactivation"` raises `NotImplementedError` matching `_form_blocks_co_firing`'s message; the kwarg is reserved for the inevitable co_firing landing.
- [x] 2.4 Any other `method` value raises `ValueError`.
- [x] 2.5 Wrap the returned `ClusteredDictionary.blocks` as the `experts` tuple, build the feature→expert map, and return `ExpertDictionary`.

## 3. Re-exports

- [x] 3.1 Re-export `ExpertDictionary` and `cluster_experts` from `polygram/__init__.py` alongside `ClusteredDictionary`.

## 4. Tests

- [x] 4.1 `tests/test_experts.py::test_cluster_experts_recovers_planted_clusters` — build a synthetic flat Dictionary whose decoder vectors fall into two well-separated antipodal groups, call `cluster_experts(..., method="cosine", coherence_threshold=0.5)`, assert the resulting `ExpertDictionary` partitions the features into exactly those two groups.
- [x] 4.2 `tests/test_experts.py::test_route_top_k_returns_dominant_expert` — given activations whose mass is concentrated in expert 0's features, `route(..., top_k=1)` returns `[0]`; with `top_k=2` it returns `[0, <next-highest>]`.
- [x] 4.3 `tests/test_experts.py::test_route_validates_input_shape` — wrong-shape activations or out-of-range `top_k` raises `ValueError`.
- [x] 4.4 `tests/test_experts.py::test_method_coactivation_raises_not_implemented` — pins the reserved kwarg's NotImplementedError so the contract is explicit.
- [x] 4.5 `tests/test_experts.py::test_expert_blocks_keep_source_dictionary_primitives` — each expert is itself a `Dictionary` and exposes `.features`, `.gram()`, etc., so downstream callers can keep using the existing primitives per-expert without indirection.

## 5. Validation

- [x] 5.1 `openspec validate add-cluster-experts-mvp --strict`.
- [x] 5.2 Full `pytest` non-sklearn pass.
- [x] 5.3 Commit + PR.
