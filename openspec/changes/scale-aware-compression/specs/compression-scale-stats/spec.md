## ADDED Requirements

### Requirement: ClusterPlan carries source norm statistics
`ClusterPlan` SHALL expose `cluster_norm_mean: float`, `cluster_norm_std: float`, and `merged_norm: float | None` fields. These are populated during `apply()` and are `None` before apply runs (i.e., after `plan()` alone).

For the `zero` strategy, `merged_norm` SHALL be `None`. For the `merge` strategy, `merged_norm` SHALL equal the value used to rescale the representative's W_dec row.

`cluster_norm_mean` and `cluster_norm_std` SHALL be the mean and standard deviation of the L2 norms of all cluster members' original (pre-compression) W_dec rows.

#### Scenario: zero strategy leaves merged_norm as None
- **WHEN** `strategy="zero"` and `apply()` completes
- **THEN** every `ClusterPlan.merged_norm` in the returned result is `None`

#### Scenario: merge strategy populates merged_norm
- **WHEN** `strategy="merge"` and `apply()` completes
- **THEN** every `ClusterPlan.merged_norm` in the returned result is a positive float

#### Scenario: cluster_norm_mean and std are correct
- **WHEN** a cluster has members with W_dec norms [1.0, 3.0]
- **THEN** `cluster_norm_mean == 2.0` and `cluster_norm_std == 1.0`

### Requirement: CompressionReport carries scale_compression_ratio
`CompressionReport` SHALL include `scale_compression_ratio: float` — the ratio of total preserved norm mass to total source norm mass across all cluster members.

Per cluster, "preserved mass" is:
- `strategy="zero"`: `||W_dec[rep]||` before compression (only the rep survives).
- `strategy="merge"`: `merged_norm × cluster_size` — the rescaled rep's row stands in for every member it absorbed.

```
scale_compression_ratio = Σ(preserved_mass_c) / Σ(||W_dec[f]||_before for all f in all clusters)
```

Under `merge_mode="simple_mean"` the merge case is mathematically `Σ(mean_c × |c|) = Σ(sum_c)`, so ratio = 1.0. Under `freq_weighted` it depends on how fires correlate with norms.

#### Scenario: zero strategy scale_compression_ratio < 1
- **WHEN** `strategy="zero"` and clusters are non-trivial (each has at least 2 members)
- **THEN** `scale_compression_ratio < 1.0`

#### Scenario: merge strategy scale_compression_ratio approaches 1
- **WHEN** `strategy="merge"` and `merge_mode="simple_mean"`
- **THEN** `scale_compression_ratio` is approximately 1.0 (within floating-point rounding)
