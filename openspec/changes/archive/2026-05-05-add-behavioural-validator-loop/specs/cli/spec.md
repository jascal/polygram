## ADDED Requirements

### Requirement: validate subcommand runs the BehaviouralValidator end-to-end

The `polygram` CLI SHALL register a `validate` subcommand that wraps `polygram.behavioural.BehaviouralValidator.run()` with file-based inputs and writes a structured `ValidationReport` to disk.

Argument set:

- `--dictionary <path>` (required) — path to a JSON file in the
  `tests/fixtures/toy_sae.json` schema, loadable by
  `polygram.load_toy_sae`. The CLI SHALL build the `Dictionary` via
  `from_sae_lens(records, feature_ids, assign_gamma=True)` using the
  ids supplied by `--feature-ids`.
- `--sae-checkpoint <path>` (required) — path to a `.safetensors`
  file with `W_enc / b_enc / W_dec / b_dec`. Missing files SHALL
  exit non-zero with stderr naming the path.
- `--feature-ids 12999,19398,...` (required) — comma-separated SAE
  feature indices, in the same order as the dictionary's features.
  Length mismatch SHALL exit non-zero with stderr naming both
  counts.
- `--prompts <path>` (required) — text file, one prompt per
  non-empty line. Lines starting with `#` SHALL be treated as
  comments and skipped. Empty file SHALL exit non-zero with a
  message naming the file.
- `--layer <int>` (required) — passed through to
  `BehaviouralValidator.layer`. The CLI does not enforce an upper
  bound; the model determines the valid range.
- `--model <name>` (default `"gpt2"`) — passed through to
  `BehaviouralValidator.model_name`. Any value other than `"gpt2"`
  SHALL emit a stderr warning that the validator's empirical
  threshold defaults are calibrated on GPT-2 small only.
- `--polygram-threshold <float>` (default `0.7`) — passed through to
  `BehaviouralValidator.polygram_overlap_threshold`.
- `--jaccard-threshold <float>` (default `0.30`) — passed through.
- `--min-firing-rate <float>` (default `0.01`) — passed through.
- `--min-both-fire <int>` (default `5`) — passed through.
- `--allow-layer-zero` (flag, default off) — passed through.
- `--output <path>` (required) — JSON output path; the CLI SHALL
  write the full `ValidationReport.to_json(...)` document there.
- `--csv <path>` (optional) — when set, also write
  `ValidationReport.to_csv(<path>)`.

The handler SHALL:

1. Resolve all paths. Missing required files SHALL exit non-zero
   with stderr naming the path.
2. Load the dictionary JSON and build the `Dictionary` per
   §4.2 above.
3. Build the `BehaviouralValidator` with the resolved arguments.
   Any `__post_init__` `ValueError` SHALL surface as a non-zero
   exit, with the validator's message printed verbatim to stderr.
4. Print one stderr progress line per major stage:
   `validate: predict ...`, `validate: load model ...`,
   `validate: forward N prompts ...`, `validate: SAE encode ...`,
   `validate: ablation k/N feat_<id> ...`, `validate: aggregate ...`.
5. Write the JSON report. When `--csv` is set, also write the CSV.
6. Print the resolved output path(s) to stderr; print nothing to
   stdout.

#### Scenario: end-to-end happy path

- **GIVEN** a `.safetensors` file at `/tmp/sae.safetensors` with
  `W_enc / b_enc / W_dec / b_dec` and 24576 features
- **AND** a dictionary JSON at `/tmp/dict.json` with 8 features
  whose names match a `from_sae_lens` build over ids
  `12999,19398,4192,23625,8371,2287,68,13737`
- **AND** a prompts file at `/tmp/prompts.txt` with 12 paragraphs
- **WHEN** the CLI is invoked as `polygram validate
  --dictionary /tmp/dict.json --sae-checkpoint /tmp/sae.safetensors
  --feature-ids 12999,19398,4192,23625,8371,2287,68,13737
  --prompts /tmp/prompts.txt --layer 10 --output /tmp/report.json
  --csv /tmp/pairs.csv`
- **THEN** the process exits 0
- **AND** `/tmp/report.json` is readable by
  `ValidationReport.from_json` and round-trips
- **AND** `/tmp/pairs.csv` has 28 data rows with the column order
  named in the `behavioural` capability's "ValidationReport
  supports CSV emission" requirement

#### Scenario: layer 0 rejected without override

- **WHEN** the CLI is invoked with `--layer 0` (no
  `--allow-layer-zero`)
- **THEN** the process exits non-zero with stderr naming layer 0
  and pointing at `docs/research/deeper-layer-ablation-probe.md`

#### Scenario: layer 0 accepted with override + warning

- **WHEN** the CLI is invoked with `--layer 0 --allow-layer-zero`
- **THEN** the process emits a `RuntimeWarning` to stderr (the
  validator's `__post_init__` warning surfaces) but continues to
  run

#### Scenario: feature-ids length mismatch rejected

- **GIVEN** a dictionary JSON with 8 features
- **WHEN** the CLI is invoked with `--feature-ids 1,2,3` (3 ids)
- **THEN** the process exits non-zero with stderr naming both `3`
  (supplied) and `8` (expected)

#### Scenario: missing torch + missing CSV flag still emits JSON-only report

- **GIVEN** torch + transformers are not installed
- **WHEN** the CLI is invoked end-to-end (no `--csv`)
- **THEN** the process exits non-zero with stderr naming
  `pip install polygram[behavioural]` as the resolution

#### Scenario: non-default --model emits warning but proceeds

- **WHEN** the CLI is invoked with `--model EleutherAI/pythia-1b`
- **THEN** stderr contains a warning that the threshold defaults
  are calibrated on GPT-2 small only
- **AND** the process proceeds with the supplied model
