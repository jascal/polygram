## ADDED Requirements

### Requirement: ClusteredDictionary holds N features as ≤K-feature blocks plus sparse cross-block adjacency

`polygram.clustered.ClusteredDictionary` SHALL be a frozen dataclass exposing the following fields:

- `blocks: list[Dictionary]` — each block holds at most `encoding.max_features` features.
- `cross_block_pairs: Mapping[tuple[int, int, int, int], float]` — sparse adjacency keyed by `(block_i, feat_i, block_j, feat_j)` with value equal to the decoder-vector cosine similarity.
- `block_topology: Graph | None` — optional cluster-of-clusters hierarchy.
- `block_formation: BlockFormation` — the strategy + parameters used to construct this clustering.

Every block SHALL share the same `encoding`. Every feature SHALL appear in exactly one block (hard partition invariant in v1).

#### Scenario: construction with mixed encodings raises

- **WHEN** `ClusteredDictionary` is constructed with two blocks whose `encoding` types differ
- **THEN** a `ValueError` is raised whose message names both encoding types

#### Scenario: feature appears in two blocks raises

- **WHEN** `ClusteredDictionary` is constructed with two blocks containing a feature with the same name
- **THEN** a `ValueError` is raised whose message names the duplicated feature

#### Scenario: block size exceeds encoding cap raises

- **WHEN** `ClusteredDictionary` is constructed with a 12-feature block on `MPSRung1` encoding (cap 8)
- **THEN** a `ValueError` is raised whose message names the block index, the encoding, and the cap

### Requirement: BlockFormation declares the block-construction strategy

`polygram.clustered.BlockFormation` SHALL be a config dataclass with fields:

- `strategy: Literal["cosine", "co_firing", "user_declared"]` (no default — caller must pick).
- `cosine_threshold: float = 0.3` — threshold for admitting cross-block edges.
- `block_size_max: int | None = None` — defaults to `encoding.max_features` at construction time.
- `firing_corpus: Sequence[str] | None = None` — required when `strategy="co_firing"`; `None` otherwise.

Validation SHALL raise `ValueError` if `strategy="co_firing"` and `firing_corpus is None`.

#### Scenario: co_firing without corpus raises

- **WHEN** `BlockFormation(strategy="co_firing", firing_corpus=None)` is constructed
- **THEN** a `ValueError` is raised naming the required `firing_corpus` parameter

#### Scenario: cosine default threshold

- **WHEN** `BlockFormation(strategy="cosine")` is constructed without an explicit threshold
- **THEN** `cosine_threshold == 0.3`

### Requirement: ClusteredDictionary.gram returns a BlockSparseGram

`ClusteredDictionary.gram() -> BlockSparseGram` SHALL compute and return:

- A list of per-block dense complex grams (one `K_i × K_i` matrix per block, computed via the existing `Dictionary.gram()` path on each block).
- A dict of cross-block Gram entries — one complex value per cross-block edge in `cross_block_pairs`, computed as the direct inner product of the two features' decoder vectors (no quantum encoding round-trip).

The return value SHALL satisfy `result.shape == (n_features, n_features)` where `n_features` is the sum of block sizes.

#### Scenario: per-block gram matches single-Dictionary gram

- **WHEN** `clustered.gram()` is invoked on a `ClusteredDictionary` with two blocks
- **THEN** for each block, the per-block dense gram equals the gram of a single `Dictionary` constructed with the same features and encoding (to 1e-12 absolute tolerance)

#### Scenario: cross-block edge present when cosine exceeds threshold

- **WHEN** a `ClusteredDictionary` has a cross-block feature pair with decoder-vector cosine 0.5 and `BlockFormation.cosine_threshold = 0.3`
- **THEN** that pair appears in `result.cross_block_entries()` with a complex Gram value matching the direct decoder-vector inner product

### Requirement: cross_block_redundant_pairs surfaces high-cosine cross-block pairs

`ClusteredDictionary.cross_block_redundant_pairs(threshold: float = 0.7) -> CrossBlockRedundancyReport` SHALL return a report listing every cross-block feature pair whose decoder-vector cosine equals or exceeds the supplied threshold. Pairs SHALL be ordered by cosine descending.

The report SHALL include metadata: the threshold used, the total number of cross-block edges examined, the count of pairs above threshold, and a per-block-pair coverage summary.

#### Scenario: planted duplicate is caught first

- **WHEN** two features in different blocks share an identical decoder vector (cosine 1.0) and the threshold is 0.7
- **THEN** that pair appears in the result's pair list, ranked first by cosine score

#### Scenario: threshold filtering is monotone

- **WHEN** `cross_block_redundant_pairs(threshold=0.5)` returns N pairs and `cross_block_redundant_pairs(threshold=0.9)` returns M pairs on the same clustered dictionary
- **THEN** `M <= N` and every pair in the 0.9 result is present in the 0.5 result

### Requirement: emit_qorca writes one .q.orca.md per block plus a manifest

`ClusteredDictionary.emit_qorca(output_dir: Path) -> dict[str, Path]` SHALL write:

- One `.q.orca.md` file per block (file naming: `<block_id>.q.orca.md`).
- A `manifest.json` listing block IDs, per-block feature names, cross-block edges, and the `BlockFormation` config used.

Each per-block `.q.orca.md` SHALL be independently round-trippable through Q-OrCA's verifier (`q-orca verify <path>`).

#### Scenario: per-block machines verify independently

- **WHEN** `emit_qorca` writes machines for a 3-block clustered dictionary
- **THEN** each of the 3 emitted `.q.orca.md` files passes Q-OrCA's 5-stage verification pipeline

#### Scenario: manifest captures cross-block adjacency

- **WHEN** `emit_qorca` is called on a clustered dictionary with N cross-block edges above threshold
- **THEN** the resulting `manifest.json` contains N entries in `cross_block_edges`, each with `from`, `to`, and `cosine` fields
