## ADDED Requirements

### Requirement: merge strategy preserves cluster norm mass
When `strategy="merge"`, `Compressor.apply()` SHALL rescale the representative's W_dec row so that its L2 norm equals `merged_norm` (the weighted average of all cluster members' original norms), while preserving the representative's original direction.

Formula for `merge_mode="freq_weighted"`:
```
merged_norm = Σ(norm_f * fires_f) / Σ(fires_f)   for f in cluster
```
where `fires_f` is the total summed `n_fires` for feature `f` across all confirmed pairs in the cluster.

Formula for `merge_mode="simple_mean"`:
```
merged_norm = mean(norm_f for f in cluster)
```

After computing `merged_norm`, the representative's row SHALL be rescaled:
```
W_dec[rep, :] = W_dec[rep, :] * merged_norm / (||W_dec[rep, :]|| + ε)
```
where `ε = 1e-8` prevents division by zero.

Non-representative members' W_dec rows SHALL still be zeroed. Encoder columns and biases for non-representatives SHALL be zeroed (same as the zero strategy).

#### Scenario: freq_weighted merge rescales representative row
- **WHEN** `strategy="merge"` and `merge_mode="freq_weighted"` and a two-member cluster has representative with norm 1.0 and fires 10, and redundant with norm 2.0 and fires 10
- **THEN** the output W_dec row for the representative has norm 1.5 and the same direction as the original representative row

#### Scenario: simple_mean merge averages norms
- **WHEN** `strategy="merge"` and `merge_mode="simple_mean"` and a three-member cluster has norms [1.0, 2.0, 3.0]
- **THEN** the representative's output row has norm 2.0

#### Scenario: merge zeros non-representative rows
- **WHEN** `strategy="merge"` is applied to a cluster with three members
- **THEN** the output W_dec rows for the two non-representatives have norm 0.0

#### Scenario: merge falls back safely on zero-norm representative
- **WHEN** the representative's original W_dec row has norm 0
- **THEN** the rescaling is skipped (no division by zero) and the row remains zero-norm

### Requirement: merge_mode parameter validation
`Compressor.__post_init__` SHALL raise `ValueError` if `merge_mode` is not one of `{"freq_weighted", "simple_mean"}`.

#### Scenario: invalid merge_mode raises at construction time
- **WHEN** `Compressor` is constructed with `merge_mode="weighted_kl"`
- **THEN** a `ValueError` is raised containing the invalid value and the supported set

### Requirement: merge_mode ignored for zero strategy
`merge_mode` SHALL have no effect when `strategy="zero"`.

#### Scenario: merge_mode does not affect zero strategy output
- **WHEN** `strategy="zero"` and `merge_mode="simple_mean"`
- **THEN** output is identical to `strategy="zero"` with the default `merge_mode`
