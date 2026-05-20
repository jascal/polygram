# design — emit-cluster-metadata-from-epoch-compressor

## Context

The two compression entry points differ in what they persist alongside
the compressed safetensors:

| entry point | safetensors | sidecar JSON | report has n_clusters | report has cluster_assignments |
|---|---|---|---|---|
| `Compressor.apply(out)` | ✓ | ✓ `<out>_compression_report.json` | ✓ | derivable from `plan.clusters[*].members`, not surfaced as a single field |
| `EpochCompressor.run(out)` | ✓ | ❌ NONE | ❌ (field absent from EpochReport) | ❌ |

This asymmetry is invisible until a downstream consumer expects the
cluster signal. sae-forge's `add-concept-anchored-finetune`'s polygram
backend tried to read `basis.metadata['n_clusters']` and
`basis.metadata['cluster_assignments']` and got `0` and `None`
respectively, because `FeatureBasis.from_polygram_checkpoint` looked
for the sidecar (which exists for `Compressor.apply`) and found
nothing for `EpochCompressor` outputs.

`EpochCompressor` internally **does** know the cluster structure:
the `cluster_fingerprints` list tracked across iterations holds the
per-iteration cluster sets, and the final cluster structure
determines which features ended up zeroed / merged. We just don't
persist it.

## Goals / Non-Goals

**Goals:**

- Surface `n_clusters` and `cluster_assignments` on the EpochReport,
  the EpochCompressor sidecar, and the downstream FeatureBasis.
- Preserve schema/JSON round-trip semantics (back-compat for v3
  payloads, schema bump to v4 for the new fields).
- Keep the sidecar-write byte-equivalent in spirit to
  `Compressor.apply()`'s pattern — same suffix, same content shape.
- Zero behavioural change for callers that ignore the new metadata.

**Non-Goals:**

- Per-iteration cluster history surfaces. Final iteration only.
- safetensors-header metadata embedding. Sidecar JSON is sufficient.
- Changing the cluster-discovery algorithm itself.
- Mirroring the new field onto `CompressionReport` (the
  `Compressor.apply` path); cluster_assignments are already
  derivable from `plan.clusters[*].members` there. Symmetric mirror
  is a separate follow-up if a consumer asks.

## Decisions

### Decision 1 — Field semantics: cluster_assignments[i] = cluster_id or -1

`cluster_assignments` is a tuple of length `n_features_input` where
element `i` is:

- An integer in `[0, n_clusters)` giving feature `i`'s cluster id
  in the FINAL iteration's compression. Features in the same cluster
  share the same id.
- `-1` if feature `i` was fully zeroed across all iterations (no
  cluster).

`n_clusters` is the count of distinct non-negative ids in the
assignments. A feature that ended up as a singleton (its own
representative; no merger) gets a unique cluster id; it's a
"cluster of one." `n_clusters` therefore counts both multi-feature
clusters AND singleton clusters that survived compression.

**Alternative considered**: report only multi-feature clusters in
`n_clusters` (matching the historical reading where "cluster" meant
"merged group"). Rejected — the supervised-concept backend needs to
know about singletons too; they're legitimate per-feature concepts.

### Decision 2 — Sidecar JSON suffix matches Compressor.apply's

Use `<out>_compression_report.json`. Existing sae-forge
`FeatureBasis.from_polygram_checkpoint` already checks all three
suffixes (`_compression_report.json`, `.compression_report.json`,
`_report.json`) — no downstream code changes needed.

### Decision 3 — Schema bump v3 → v4 with back-compat loader

`EpochReport.SCHEMA_VERSION 3 -> 4`. `from_json` reads v3 payloads
without error, defaulting `n_clusters` to `0` and `cluster_assignments`
to `None`. Equality / hash include the new fields (NaN-aware on
`n_clusters` is unnecessary because it's int; `cluster_assignments`
None compares by None or tuple equality).

### Decision 4 — Compute cluster_assignments in run() and pass to EpochReport

`EpochCompressor.run` already iterates `cluster_fingerprints` per
iteration. At the end of `run()` (before returning the
`EpochResult`), materialise the final iteration's frozenset clusters
into a length-`n_features_input` tuple:

```python
def _final_cluster_assignments(
    cluster_fingerprints: list[frozenset],
    n_features_input: int,
    final_state_dict: dict[str, np.ndarray],
) -> tuple[int, ...]:
    """Return per-feature cluster id; -1 for fully-zeroed features."""
    last = cluster_fingerprints[-1] if cluster_fingerprints else ()
    assignments = [-1] * n_features_input
    for cluster_id, members in enumerate(sorted(last, key=lambda c: min(c))):
        for fid in members:
            assignments[fid] = cluster_id
    return tuple(assignments)
```

Singleton fids that survived but never appeared in any
`cluster_fingerprints` entry default to `-1`. The implementation
SHOULD additionally scan the final state's W_dec to bump those
surviving singletons into their own cluster_id (matches Decision 1's
semantics — surviving singletons get a real id, not `-1`).

### Decision 5 — Sidecar write is in the same atomic-write block as the safetensors

`EpochCompressor.run` already writes the safetensors via
tempfile + `os.replace`. Bundle the JSON sidecar write into the same
block: write `<out>.tmp` for safetensors, write
`<out>_compression_report.json.tmp` for the report, atomically
replace both. A failed sidecar write should NOT leave a stale
safetensors lying around without its companion; if either replace
fails, both temps get cleaned up.

**Alternative considered**: write the sidecar after the safetensors
replace, accepting that a crash between the two leaves a sidecar-less
safetensors. Rejected — back-compat with existing `Compressor.apply`
sidecar emission is the goal, and even a half-written sidecar is
worse than no sidecar (downstream loaders see partial data).

## Risks / Trade-offs

- **Frozen-fixture tests will need refresh.** `test_epoch_clustered_consume`
  has `epoch_result_reference{,_multi_iter}.json` fixtures. The schema
  bump + new fields require regenerating these (or updating with
  `n_clusters` / `cluster_assignments` and schema_version). Same
  pattern as the v0.11.0 release's fixture updates.

- **`cluster_fingerprints` may not exist for max_iterations=0** runs
  (degenerate but legal). Defensive default: `n_clusters=0`,
  `cluster_assignments=()` of length `n_features_input` filled with
  -1.

- **Sidecar discovery convention** is already established by
  `Compressor.apply` (`<out>_compression_report.json`). No new
  convention added; the asymmetry is closed.

- **No behavioural change** for callers that ignore the new fields.
  The sae-forge side reads them when present and falls back to the
  documented round-robin partition when absent (with a clear
  UserWarning) — that behaviour stays.

## Migration

- No migration required for existing callers; the new fields are
  additive and the sidecar is opt-in.
- Existing `_compression_report.json` files from `Compressor.apply`
  stay valid; their schema version is `CompressionReport`'s, not
  `EpochReport`'s — different report types.
