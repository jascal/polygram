## ADDED Requirements

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
