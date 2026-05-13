## MODIFIED Requirements

### Requirement: from_sae_lens supports a clustered import path

`polygram.sae_import.from_sae_lens` SHALL accept two new optional keyword arguments:

- `clustered: bool = False` — when `True`, the loader returns a `ClusteredDictionary` instead of a single `Dictionary`.
- `block_formation: BlockFormation | None = None` — when supplied, configures the block-formation strategy. When `clustered=True` and `block_formation is None`, defaults to `BlockFormation(strategy="cosine", cosine_threshold=0.3, block_size_max=encoding.max_features)`.

When `clustered=False` (the default), the loader's behaviour SHALL be byte-identical to the pre-change implementation. The new arguments are additive; no existing caller breaks.

When `clustered=True`, the returned object is a `ClusteredDictionary`. The `SelectionReport` returned alongside SHALL include per-clustering stats: `n_blocks`, `mean_block_size`, `n_cross_block_edges`.

#### Scenario: clustered=False default preserves the existing path

- **WHEN** `from_sae_lens(... )` is called without the `clustered` or `block_formation` kwargs
- **THEN** the return value is a `Dictionary` (not a `ClusteredDictionary`) and the result is byte-identical to the pre-change implementation on the same inputs

#### Scenario: clustered=True returns a ClusteredDictionary

- **WHEN** `from_sae_lens(... , clustered=True)` is called with 64 feature ids against an `MPSRung1` encoding (cap 8)
- **THEN** the return value is a `ClusteredDictionary` with at least 8 blocks (64 features / 8 per block) and every block has at most 8 features

#### Scenario: cosine default threshold when block_formation omitted

- **WHEN** `from_sae_lens(... , clustered=True)` is called without `block_formation`
- **THEN** the loader constructs a default `BlockFormation(strategy="cosine", cosine_threshold=0.3)` with `block_size_max` set to `encoding.max_features`

### Requirement: SelectionReport reports per-clustering stats when clustered import is used

`polygram.sae_import.SelectionReport` SHALL gain three optional fields populated when the clustered loader path runs:

- `n_blocks: int | None` — count of blocks in the resulting `ClusteredDictionary`.
- `mean_block_size: float | None` — average block size.
- `n_cross_block_edges: int | None` — count of cross-block adjacency edges above threshold.

When the non-clustered path runs, all three fields SHALL be `None`. Existing fields SHALL be unchanged.

#### Scenario: non-clustered SelectionReport has None for clustering stats

- **WHEN** `from_sae_lens(... )` is called with `clustered=False`
- **THEN** the returned `SelectionReport` has `n_blocks is None`, `mean_block_size is None`, `n_cross_block_edges is None`

#### Scenario: clustered SelectionReport reports populated stats

- **WHEN** `from_sae_lens(... , clustered=True)` is called and produces a 6-block clustered dictionary with 4 cross-block edges
- **THEN** the returned `SelectionReport` has `n_blocks == 6`, `mean_block_size` matching the average block size, and `n_cross_block_edges == 4`
