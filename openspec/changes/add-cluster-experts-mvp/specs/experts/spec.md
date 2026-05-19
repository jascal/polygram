## ADDED Requirements

### Requirement: `ExpertDictionary` groups a flat Dictionary's features into routable experts

`polygram.experts.ExpertDictionary` SHALL be a frozen dataclass holding a tuple of expert blocks (`experts: tuple[Dictionary, ...]`), the source flat Dictionary (`source: Dictionary`), and a precomputed feature→expert index map enabling O(N) routing.

Every feature name in `source.features` SHALL appear in exactly one expert; every expert SHALL be non-empty.

#### Scenario: feature partition invariant

- **WHEN** an `ExpertDictionary` is constructed
- **THEN** the disjoint union of expert feature names equals
  `{f.name for f in source.features}`

#### Scenario: expert is itself a Dictionary

- **WHEN** a caller indexes into `expert_dict.experts[i]`
- **THEN** the returned object is a `Dictionary` exposing
  `.features`, `.gram()`, and every other primitive on the parent
  type — no shim layer

### Requirement: `cluster_experts` factory reuses BlockFormation cosine path

`polygram.experts.cluster_experts(dictionary, decoder_vectors, *, method, coherence_threshold, max_features_per_expert, activations)` SHALL be the entry point for building an `ExpertDictionary` from a flat `Dictionary`.

For `method="cosine"`, the factory SHALL delegate block formation to the existing `build_clustered_dictionary(strategy="cosine", ...)` path, with `cosine_threshold=coherence_threshold` and `block_size_max=max_features_per_expert`. The encoding SHALL be read off the source dictionary.

#### Scenario: cosine recovers planted antipodal clusters

- **WHEN** a flat `Dictionary` is built with decoder vectors falling
  into two well-separated antipodal groups and `cluster_experts(...,
  method="cosine", coherence_threshold=0.5)` is called
- **THEN** the returned `ExpertDictionary` has exactly two experts,
  one per planted group

### Requirement: `cluster_experts` rejects unimplemented methods

`cluster_experts(..., method="coactivation")` SHALL raise `NotImplementedError`. Any `method` value outside `{"cosine", "coactivation"}` SHALL raise `ValueError`.

The `"coactivation"` token is reserved for the implementation that will land alongside `clustered_dictionary._form_blocks_co_firing`.

#### Scenario: coactivation raises NotImplementedError

- **WHEN** `cluster_experts(..., method="coactivation")` is called
- **THEN** `NotImplementedError` is raised with a message naming the
  blocking dependency (the co_firing block-formation stub)

#### Scenario: unknown method raises ValueError

- **WHEN** `cluster_experts(..., method="louvain")` is called
- **THEN** `ValueError` is raised naming the supported method set

### Requirement: `ExpertDictionary.route` returns top-k experts by summed activation

`ExpertDictionary.route(activations, top_k)` SHALL return a `list[int]` of expert indices ordered by descending summed per-expert activation. `activations` SHALL be a numpy array of shape `(n_features,)` indexed in the order of `source.features`. `top_k` SHALL satisfy `1 <= top_k <= n_experts`.

Routing SHALL NOT require any decoder-vector or torch dependency at call time — the precomputed feature→expert map makes it a `np.add.at` reduction.

#### Scenario: dominant expert wins top_k=1

- **WHEN** activations are concentrated in the features of expert 0
  and `route(activations, top_k=1)` is called
- **THEN** the return value is `[0]`

#### Scenario: route validates input

- **WHEN** `route` is called with `activations.shape != (n_features,)`
  or `top_k <= 0` or `top_k > n_experts`
- **THEN** `ValueError` is raised
