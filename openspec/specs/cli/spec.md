# cli Specification

## Purpose

The `polygram` console-script entry point and its subcommands. Covers
how Polygram artifacts are produced from the command line: `run`
loads and executes example modules; `analyze` triages SAE feature
subsets via the analytic Gram. New subcommands SHALL extend this
spec rather than introduce parallel entry points.
## Requirements
### Requirement: Console-script entry point

Polygram SHALL register a `polygram` console script. Invocation forms:

- `polygram --version` — print `polygram <__version__>` and exit 0
- `polygram run <target> [--output-dir DIR] [--n-points N]` — load
  the target module and invoke its `main(output_dir=...)` callable

`<target>` SHALL accept either a filesystem path to a `.py` file or
a `pkg.module:callable` reference. When the path form is used,
Polygram loads the module via `importlib.util` and looks up
`main`. The CLI SHALL exit 2 with a clear error if the target does
not expose `main(output_dir=...)`.

#### Scenario: filesystem-path target runs and writes to output dir

- **WHEN** the CLI is invoked as `polygram run /tmp/myexample.py
  --output-dir /tmp/out` and `myexample.py` defines
  `def main(output_dir): Path(output_dir).joinpath("hello").write_text("hi")`
- **THEN** the process exits 0 and `/tmp/out/hello` contains `"hi"`

#### Scenario: missing main raises clear error

- **WHEN** the CLI is invoked on a module with no `main` function
- **THEN** the process exits non-zero and stderr names the missing
  `main(output_dir=...)` callable

### Requirement: analyze subcommand triages an SAE feature subset

The CLI SHALL register an `analyze` subcommand that takes a positional
SAE-path argument, a required `--features` option (comma-separated
feature ids), and an optional `--output` option (default
`analysis_report.md`). The handler SHALL load the SAE JSON file
(schema matching `tests/fixtures/toy_sae.json`), select the requested
feature ids, build a rung-1 Polygram Dictionary via `from_sae_lens`,
run `polygram.analysis.predict_cancellation_depth`, write the
rendered markdown report to the output path, and print the
suitability score to stdout.

The subcommand SHALL accept four additional optional flags:

- `--sharing-graph <path.json>` — when supplied, the handler SHALL
  invoke `polygram.analysis.build_sharing_graph` on the prediction
  and write the result of `FeatureGraph.to_json()` to the named
  path. Default behavior unchanged when omitted.
- `--sharing-threshold <float>` — threshold forwarded to
  `build_sharing_graph(threshold=...)`. Defaults to `0.5`. Ignored
  unless `--sharing-graph` is also supplied.
- `--separation-graph <path.json>` — when supplied, the handler
  SHALL invoke `polygram.analysis.build_separation_graph` on the
  prediction and write the result of `FeatureGraph.to_json()` to
  the named path. Default behavior unchanged when omitted.
- `--separation-threshold <float>` — threshold forwarded to
  `build_separation_graph(threshold=...)`. Defaults to `0.2`.
  Ignored unless `--separation-graph` is also supplied.

The two graph flags are independent — the user MAY supply neither,
either, or both. When both are supplied, the handler SHALL emit two
JSON files.

The subcommand SHALL exit 0 on success and non-zero with a clear
error message on any of: missing/invalid `<sae_path>`, malformed
`--features` argument, malformed `--sharing-threshold` or
`--separation-threshold` value, or any error raised by
`predict_cancellation_depth`, `build_sharing_graph`, or
`build_separation_graph`.

#### Scenario: analyze writes a report and prints a score

- **WHEN** the CLI is invoked as `polygram analyze
  tests/fixtures/toy_sae.json --features 0,1,4,5 --output /tmp/r.md`
- **THEN** the process exits 0
- **AND** `/tmp/r.md` exists and contains a markdown report with the
  expected section headings
- **AND** stdout contains a line naming the suitability score

#### Scenario: malformed features argument is rejected

- **WHEN** the CLI is invoked with `--features not-a-number`
- **THEN** the process exits non-zero with an error mentioning the
  malformed feature id

#### Scenario: sharing-graph flag emits a parseable JSON artifact

- **WHEN** the CLI is invoked as `polygram analyze
  tests/fixtures/toy_sae.json --features 0,1,4,5 --output /tmp/r.md
  --sharing-graph /tmp/g.json --sharing-threshold 0.4`
- **THEN** the process exits 0
- **AND** `/tmp/g.json` exists and parses cleanly via `json.loads`
- **AND** the parsed dict has the keys `"kind"`, `"nodes"`,
  `"edges"`, `"clusters"`, and `"metadata"`, with
  `"kind" == "sharing"`

#### Scenario: separation-graph flag emits a parseable JSON artifact

- **WHEN** the CLI is invoked as `polygram analyze
  tests/fixtures/toy_sae.json --features 0,1,4,5 --output /tmp/r.md
  --separation-graph /tmp/s.json --separation-threshold 0.15`
- **THEN** the process exits 0
- **AND** `/tmp/s.json` exists and parses cleanly via `json.loads`
- **AND** the parsed dict's `"kind"` field equals `"separation"`

#### Scenario: both graph flags supplied emit both artifacts

- **WHEN** the CLI is invoked with both `--sharing-graph` and
  `--separation-graph` paths
- **THEN** both JSON files exist and parse cleanly, with the
  expected `"kind"` values

### Requirement: batch subcommand consumes a FeatureGraph and a Dictionary

The `polygram` CLI SHALL register a `batch` subcommand that consumes a serialized `FeatureGraph`, a `Dictionary` reference, and runs `BatchExperiment.run()` on the graph's top-K edges, writing a `batch_results.json` artifact.

Argument set:

- `--feature-graph FILE.json` — required. Path to a JSON document
  produced by `FeatureGraph.to_json()` (output of
  `polygram.analysis.build_sharing_graph` or
  `build_separation_graph`). The CLI parses it via the
  `FeatureGraph.from_json` round-trip helper.
- `--dictionary REF` — required. Either a `.q.orca.md` file path
  (parsed via the existing q-orca round-trip) or a `module:callable`
  reference whose callable returns a `Dictionary` (e.g.
  `examples.animals_hea:build_dictionary`).
- `--top-k N` (default `8`) — forwarded to `BatchExperiment.top_k`.
  Hard cap of 16 is enforced at the dataclass layer; values outside
  `[1, 16]` SHALL produce an argparse-level error before the
  `BatchExperiment` is constructed.
- `--knobs cluster_shared|per_feature` (default `cluster_shared`) —
  forwarded to `BatchExperiment.knobs`.
- `--output-dir DIR` (default: a fresh temp directory) — forwarded
  to `BatchExperiment.output_dir`. The resolved path is printed to
  stdout regardless of whether the user provided it.

The handler SHALL:

1. Parse `--feature-graph` via `FeatureGraph.from_json`. On parse
   failure, exit non-zero with stderr naming the offending path and
   the parse error.
2. Resolve `--dictionary` via the existing q-orca file loader or
   `module:callable` import. On any resolution error, exit non-zero
   with stderr.
3. Construct a `BatchExperiment` with the parsed arguments. Any
   `ValueError` raised by `__post_init__` (e.g. graph node missing
   from dictionary, `top_k` out of range) SHALL be caught and
   reported on stderr with exit code non-zero.
4. Call `BatchExperiment.run()`. Write `batch_results.json` at the
   top of `--output-dir`.
5. Print the resolved `batch_results.json` path on stdout. Exit 0.

#### Scenario: end-to-end run on a separation graph

- **GIVEN** a `FeatureGraph` produced by
  `build_separation_graph(predict_cancellation_depth(toy_sae,
  [0,1,4,5]))` and serialized to `/tmp/sep.json`, AND
  `examples/animals_hea.py` exposes
  `build_dictionary()` returning the matching dictionary
- **WHEN** the CLI is invoked as `polygram batch --feature-graph
  /tmp/sep.json --dictionary examples.animals_hea:build_dictionary
  --top-k 2 --output-dir /tmp/out`
- **THEN** the process exits 0, `/tmp/out/batch_results.json`
  exists and parses as a JSON document with exactly 2 `runs`
  entries, and the printed stdout line names that path

#### Scenario: feature-graph with node not in dictionary rejected

- **GIVEN** a `FeatureGraph` whose `nodes` includes a name not
  declared by the resolved dictionary
- **WHEN** the CLI is invoked
- **THEN** the process exits non-zero with stderr naming the
  missing feature(s) and pointing at the dictionary reference

#### Scenario: malformed feature-graph JSON rejected

- **WHEN** the CLI is invoked with `--feature-graph` pointing at a
  file that is not valid `FeatureGraph` JSON
- **THEN** the process exits non-zero with stderr naming the path
  and the parse error

#### Scenario: top-k above 16 rejected at argparse layer

- **WHEN** the CLI is invoked with `--top-k 17`
- **THEN** the process exits non-zero with stderr naming the value
  `17` and the 16-pair cap, before any `BatchExperiment` is
  constructed

#### Scenario: top-k below 1 rejected

- **WHEN** the CLI is invoked with `--top-k 0`
- **THEN** the process exits non-zero with stderr naming the value
  and the `[1, 16]` valid range

#### Scenario: unknown knobs value rejected

- **WHEN** the CLI is invoked with `--knobs bogus`
- **THEN** the process exits non-zero with stderr listing the
  supported choices (`cluster_shared`, `per_feature`)

#### Scenario: defaults produce a runnable invocation

- **GIVEN** a valid `--feature-graph` and `--dictionary`
- **WHEN** the CLI is invoked WITHOUT `--top-k`, `--knobs`, or
  `--output-dir`
- **THEN** the process exits 0, `--top-k` defaults to `8`,
  `--knobs` defaults to `cluster_shared`, `--output-dir` defaults
  to a freshly-created temp directory, and the resolved temp path
  is printed on stdout

### Requirement: sae-import subcommand converts a safetensors file to the toy-SAE JSON schema

The `polygram` CLI SHALL register a `sae-import` subcommand that loads a `.safetensors` file via `polygram.load_sae_safetensors`, optionally selects a subset, and writes the result as JSON in the same schema as `tests/fixtures/toy_sae.json`. The emitted JSON SHALL be directly readable by `polygram.load_toy_sae` and consumable by every downstream surface that already accepts that schema (notably `polygram analyze`).

Argument set:

- `<path>` (positional, required) — path to the source `.safetensors` file.
- `--features 0,12,1042` — optional comma-separated list of feature ids to keep. When omitted, every loaded feature is emitted. Out-of-range or non-integer entries SHALL exit non-zero with stderr naming the offending entry.
- `--names path.json` — optional path to a JSON file mapping either `{feature_id_int: name_str}` or `{name_str: feature_id_int}`. The handler SHALL detect the mapping direction by inspecting the first value's type: string-valued maps are interpreted as `id → name`; int-valued maps as `name → id` and inverted before being passed to `load_sae_safetensors`. Mixed-type values SHALL exit non-zero.
- `--output path.json` — optional output path. When omitted, the JSON document SHALL be written to stdout.

The handler SHALL:

1. Resolve `<path>`. Missing or unreadable files SHALL exit non-zero with stderr naming the path.
2. Parse `--names` if supplied; surface JSON parse errors and ambiguous-shape errors to stderr.
3. Call `load_sae_safetensors(path, names=resolved_names_or_None, feature_ids=parsed_features_or_None)`. When `--features` is supplied, the parsed id list SHALL be passed through to the loader's lazy-slice mode rather than loading the full decoder tensor and filtering after — for GB-class SAEs this is the difference between OOM and a sub-MB read.
4. If `--features` is supplied and any requested id is outside the loader's valid range, the loader's `ValueError` SHALL surface as a non-zero exit with a stderr message naming the offending id and the valid range.
5. Emit a JSON document with shape `{"schema_version": 1, "description": "<short auto-generated note>", "features": [...]}` where each feature object carries the same fields as `tests/fixtures/toy_sae.json` entries (`feature_id`, `name`, `projection`, optional `label`, optional `activation_mean`, optional `activation_std`). Records whose optional fields are `None` SHALL be omitted from the JSON object (matching `load_toy_sae` semantics).
6. Print nothing to stdout when `--output` is supplied; print the resolved output path on stderr regardless. When `--output` is omitted, print the JSON document on stdout.

#### Scenario: end-to-end safetensors → JSON → analyze

- **GIVEN** a `.safetensors` file at `/tmp/sae.safetensors` with `W_dec` shape `(16, 8)`
- **WHEN** the CLI is invoked as `polygram sae-import /tmp/sae.safetensors --features 0,1,4,5 --output /tmp/picked.json`
- **THEN** the process exits 0
- **AND** `/tmp/picked.json` exists and is readable by `polygram.load_toy_sae`
- **AND** the loaded dict has exactly 4 entries keyed by `0`, `1`, `4`, `5`
- **AND** invoking `polygram analyze /tmp/picked.json --features 0,1,4,5` succeeds (exit 0)

#### Scenario: --features with unknown id rejected

- **GIVEN** a `.safetensors` file with 4 features (ids `0..3`)
- **WHEN** the CLI is invoked with `--features 0,5`
- **THEN** the process exits non-zero with stderr naming `5` and the valid id range

#### Scenario: --names accepts {id: name}

- **GIVEN** a `.safetensors` file at path `/tmp/sae.safetensors` and a labels file `/tmp/labels.json` with content `{"0": "dog_poodle", "2": "bird_hawk"}`
- **WHEN** the CLI is invoked as `polygram sae-import /tmp/sae.safetensors --names /tmp/labels.json --output /tmp/picked.json`
- **THEN** the process exits 0
- **AND** the emitted JSON's feature with `feature_id=0` has `name="dog_poodle"` and the feature with `feature_id=2` has `name="bird_hawk"`

#### Scenario: --names accepts {name: id} and inverts

- **GIVEN** a labels file `/tmp/labels.json` with content `{"dog_poodle": 0, "bird_hawk": 2}`
- **WHEN** the CLI is invoked with `--names /tmp/labels.json`
- **THEN** the inversion produces the same effect as the `{id: name}` form
- **AND** the emitted feature with `feature_id=0` has `name="dog_poodle"`

#### Scenario: --names with mixed value types rejected

- **GIVEN** a labels file containing `{"0": "dog_poodle", "1": 5}` (mixed string and int values)
- **WHEN** the CLI is invoked with that file as `--names`
- **THEN** the process exits non-zero with stderr naming the file and the ambiguous-shape error

#### Scenario: --output omitted writes to stdout

- **WHEN** the CLI is invoked without `--output`
- **THEN** the JSON document is written to stdout
- **AND** stderr does NOT echo the document

#### Scenario: missing source file rejected

- **WHEN** the CLI is invoked with `<path>` pointing at a file that does not exist
- **THEN** the process exits non-zero with stderr naming the missing path

### Requirement: analyze subcommand exposes assign_gamma and n_clusters

The `polygram analyze` subcommand SHALL accept two additional optional flags forwarded directly to `polygram.from_sae_lens` via `predict_cancellation_depth`:

- `--assign-gamma` (boolean flag, default false) — when set, forwards `assign_gamma=True` so each feature's γ knob is derived from per-cluster PCA on the centered projection vectors. Without this flag every feature's γ stays at 0, which collapses within-cluster overlaps to 1.0 on diverse-projection inputs (the default `from_sae_lens` behaviour). Real-SAE workloads almost always need it.
- `--n-clusters N` (integer, default `None` → `from_sae_lens` defaults to 2) — forwarded as the `n_clusters` argument when `from_sae_lens` falls back to k-means. Out-of-range values (`< 1`) SHALL be rejected at argparse layer.

These flags affect the `Dictionary` produced by `from_sae_lens` and therefore every downstream artifact: the markdown report's pair predictions, the optional `--sharing-graph` and `--separation-graph` JSON outputs, and the suitability score. They do not interact with each other beyond what `from_sae_lens` documents.

#### Scenario: --assign-gamma forwards to from_sae_lens

- **GIVEN** a toy-SAE JSON file with at least one cluster of size ≥ 2 whose features have diverse projection vectors
- **WHEN** the CLI is invoked with `--assign-gamma`
- **THEN** the report's per-pair `current` overlaps for within-cluster pairs are NOT all `1.0000` (γ-PCA differentiates siblings)
- **AND WHEN** the same file is analyzed without `--assign-gamma`
- **THEN** within-cluster `current` overlaps DO collapse to `1.0000`

#### Scenario: --n-clusters forwards to from_sae_lens

- **GIVEN** a toy-SAE JSON whose features lack `cluster/name` labels (so k-means is the cluster path)
- **WHEN** the CLI is invoked with `--n-clusters 3 --features 0,1,2,3`
- **THEN** the report's `Cluster method` line names `kmeans`
- **AND** the per-pair table reflects 3-cluster grouping (not the default 2-cluster)

#### Scenario: invalid --n-clusters rejected

- **WHEN** the CLI is invoked with `--n-clusters 0`
- **THEN** the process exits non-zero with stderr naming the value and the valid range

