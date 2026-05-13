## MODIFIED Requirements

### Requirement: `from_compression_panels` is the production conversion point for `EpochCompressor`

`polygram.clustered.ClusteredDictionary.from_compression_panels` SHALL serve as the production conversion from `_select_panels`'s output (`list[Panel]`) to a `ClusteredDictionary` view, both for downstream callers and for `EpochCompressor.run`'s internal pipeline.

The method's surface SHALL remain as shipped in `clustered-dictionary-analysis` (PR #44): `(panels, state_dict, encoding, *, name, cosine_threshold, feature_records)` accepting an iterable of `Panel`-shaped objects and producing a `ClusteredDictionary` whose `blocks` align element-wise with the input panels.

#### Scenario: block ordering aligns with panel ordering

- **WHEN** `ClusteredDictionary.from_compression_panels(panels=..., ...)` is called on an ordered list of panels
- **THEN** the returned `ClusteredDictionary.blocks[k]` is constructed from `panels[k]`'s `feature_ids` for every `k`

#### Scenario: feature names round-trip through panel members

- **WHEN** a panel's `feature_ids` are `(0, 1, 2)` and no `feature_records` are supplied
- **THEN** the corresponding block contains features named `f0`, `f1`, `f2` in that order

#### Scenario: cross-block adjacency populated from W_dec

- **WHEN** two panels share no member but their members' decoder vectors include a cosine pair above the `cosine_threshold`
- **THEN** that pair appears in `clustered.cross_block_pairs` with the canonical `(bi, fi, bj, fj)` ordering and the correct cosine value
