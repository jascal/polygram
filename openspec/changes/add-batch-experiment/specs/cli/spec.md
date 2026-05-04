## ADDED Requirements

### Requirement: batch subcommand runs BatchExperiment from the CLI

The `polygram` CLI SHALL register a `batch` subcommand that builds a
`Dictionary` from one of two input forms, runs a
`BatchExperiment`, and writes a `sharing_graph.json` artifact.

Argument set:

- `--sae PATH` — load an SAE JSON file (schema matching
  `tests/fixtures/toy_sae.json`) via the existing `from_sae_lens`
  helper. Required `--features id1,id2,...` selects which feature
  ids to include.
- `--dictionary REF` — load a `Dictionary` either from a
  `.q.orca.md` file (parsed via the existing q-orca round-trip) or
  from a `module:callable` reference exposing a
  `build_dictionary()` function. When this form is used,
  `--features` is ignored.
- `--experiments KINDS` (default `"sweep,cancellation"`) — comma-
  separated list of experiment kinds.
- `--pairs SEL` (default `"all"`) — pair selection forwarded to
  `BatchExperiment.pairs`. The CLI accepts only the string forms
  (`"all"`, `"cross_cluster"`, `"within_cluster"`); explicit pair
  lists are not exposed at the CLI level (use the Python API).
- `--output-dir DIR` (default: a fresh temp directory) — directory
  to write per-pair sub-artifacts and `sharing_graph.json` into.
- `--force` — flag forwarded to `BatchExperiment.force` to
  override the ≤50-pair safety rail.

Exactly one of `--sae` or `--dictionary` SHALL be required;
specifying neither or both SHALL raise an argparse usage error and
exit non-zero.

The handler SHALL:

1. Build a `Dictionary` per the chosen input form.
2. Construct a `BatchExperiment` with the parsed arguments.
3. Call `BatchExperiment.run()`, materializing per-pair
   sub-artifacts under `--output-dir` and writing
   `sharing_graph.json` at the top level.
4. Print the `sharing_graph.json` path to stdout.
5. Exit 0 on success.

The subcommand SHALL exit non-zero with a clear stderr message on
any of: missing/invalid input file, malformed `--features`,
unknown experiment kind, or any error raised by
`BatchExperiment.run()`.

#### Scenario: --sae invocation writes a SharingGraph JSON

- **WHEN** the CLI is invoked as `polygram batch --sae
  tests/fixtures/toy_sae.json --features 0,1,4,5
  --experiments cancellation --output-dir /tmp/out`
- **THEN** the process exits 0 and `/tmp/out/sharing_graph.json`
  exists and parses as a JSON document with at least 6 edges

#### Scenario: --dictionary REF on a build_dictionary callable

- **WHEN** the CLI is invoked as `polygram batch --dictionary
  examples.animals_hea:build_dictionary --pairs cross_cluster
  --output-dir /tmp/out` and `examples/animals_hea.py` exposes
  `build_dictionary()` returning a 4-feature 2-cluster Dictionary
- **THEN** the process exits 0 and the produced
  `sharing_graph.json` has exactly 4 edges, all with
  cross-cluster `(a, b)` endpoints

#### Scenario: --sae and --dictionary mutually exclusive

- **WHEN** the CLI is invoked with both `--sae PATH` and
  `--dictionary REF`
- **THEN** the process exits non-zero with stderr explaining the
  mutual exclusion

#### Scenario: oversized batch rejected without --force

- **WHEN** the CLI is invoked on a Dictionary whose pair count
  exceeds 50 without `--force`
- **THEN** the process exits non-zero with stderr naming the
  pair count and recommending `--force` or a narrower `--pairs`

#### Scenario: unknown experiment kind rejected

- **WHEN** the CLI is invoked as `polygram batch ...
  --experiments bogus,sweep`
- **THEN** the process exits non-zero with stderr listing the
  supported kinds (`sweep`, `cancellation`)
