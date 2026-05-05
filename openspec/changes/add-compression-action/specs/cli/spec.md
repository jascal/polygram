## ADDED Requirements

### Requirement: polygram compress wraps Compressor.run() with file-based inputs

`polygram compress` SHALL be a new CLI subcommand that wraps `Compressor.run()` with file-based inputs. The flag set:

- `--validation-report PATH` — required. Path to a JSON file
  produced by `ValidationReport.to_json()` (or by
  `polygram validate --output`).
- `--sae-checkpoint PATH` — required. Source SAE
  `.safetensors` file. Must exist on disk.
- `--output-checkpoint PATH` — required. Where the rewritten
  `.safetensors` is written. Must differ from
  `--sae-checkpoint`.
- `--strategy STR` — required. Must be `"zero"` (the only
  currently-implemented strategy). Other values exit with code
  2 and a message naming the supported set.
- `--output PATH` — required. Where the
  `CompressionReport` JSON is written.
- `--representatives SPEC` — optional. Comma-separated
  `cluster_id=fid` pairs (e.g.,
  `0=12999,1=4192,2=8371`). Cluster ids referenced must exist
  in the plan; fids must be members of their named cluster.

### Requirement: polygram compress prints honest stage progress

`polygram compress` SHALL print one line per major stage to stderr:

```
polygram compress: loading validation report ...
polygram compress: loading source SAE checkpoint ...
polygram compress: building compression plan ... (3 clusters, 5 features to zero)
polygram compress: rewriting checkpoint ...
polygram compress: writing compression report ...
polygram compress: done. Source SHA256 abcd... → Output SHA256 wxyz...
```

The final line SHALL include both checkpoint hashes (truncated to 12 hex chars for legibility) so the operation's identity is visible in shell logs.

### Requirement: polygram compress exits non-zero on missing inputs or invalid arguments

`polygram compress` SHALL exit with code `2` and a message on stderr when:

- `--validation-report` does not exist or is not a file.
- `--sae-checkpoint` does not exist or is not a file.
- `--output-checkpoint` resolves to the same path as
  `--sae-checkpoint`.
- `--strategy` is not one of the supported values.
- `--representatives` references a cluster id not in the plan.
- `--representatives` references a fid not in its named cluster.

The message SHALL name the offending argument and the offending value.
