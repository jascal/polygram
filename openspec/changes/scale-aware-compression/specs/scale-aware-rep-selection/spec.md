## ADDED Requirements

### Requirement: scale_aware representative selection
When `rep_selection="scale_aware"`, `Compressor` SHALL score each candidate in a cluster using a weighted combination of three signals and return the highest-scoring candidate as the representative (tiebreak: lowest feature id).

The score for candidate `f` in a cluster `C` of size `n` SHALL be:

```
score(f) = 0.4 * norm_proximity(f, C)
         + 0.4 * ablation_importance(f, C)
         + 0.2 * log_freq(f, C)
```

where:
- `norm_proximity(f, C)` = `1 - |norm_f - median_norm_C| / (median_norm_C + ε)`, clipped to [0, 1]; `ε = 1e-8`.
- `ablation_importance(f, C)` = sum of `kl_ablate` for all confirmed pairs involving `f` within `C`, normalised to [0, 1] across the cluster (min-max; no-op if all equal).
- `log_freq(f, C)` = `log(total_n_fires_f + 1e-8)`, normalised to [0, 1] across the cluster.

Norms SHALL be computed as `np.linalg.norm` of each feature's W_dec row loaded from the SAE checkpoint.

#### Scenario: scale_aware picks norm-central high-ablation candidate
- **WHEN** a cluster contains three members where one has norm close to the median and the highest KL-ablation score
- **THEN** `Compressor.plan()` with `rep_selection="scale_aware"` returns that member as the representative

#### Scenario: NaN fallback for geometry-only confirmer
- **WHEN** all `kl_ablate` values for pairs in a cluster are NaN (geometry-only confirmer path)
- **THEN** the ablation term is zeroed and selection falls back to `n_fires`-only scoring, and a `UserWarning` is emitted once per `plan()` call

#### Scenario: n_fires default is unchanged
- **WHEN** `rep_selection="n_fires"` (the default)
- **THEN** `Compressor.plan()` behaviour is identical to the pre-change implementation

### Requirement: rep_selection parameter validation
`Compressor.__post_init__` SHALL raise `ValueError` if `rep_selection` is not one of `{"n_fires", "scale_aware"}`.

#### Scenario: invalid rep_selection raises at construction time
- **WHEN** `Compressor` is constructed with `rep_selection="unknown"`
- **THEN** a `ValueError` is raised containing the invalid value and the supported set
