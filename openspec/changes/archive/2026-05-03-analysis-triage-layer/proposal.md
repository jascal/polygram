## Why

> Retroactive: this proposal documents work already shipped in commit
> `a4a906a`. The implementation bypassed the OpenSpec workflow that
> `AGENTS.md` mandates; this change closes the process gap so the
> capabilities are reflected in `openspec/specs/`.

`Cancellation` and `InterferenceSweep` answer phase-tuning questions
on a *given* small Dictionary. But a real SAE has 16k+ features and
the upstream question — "which 4–8 features should I encode?" — has
no primitive yet. Building a Dictionary, encoding, and simulating
every plausible subset is not tractable.

The rung-1 closed-form Gram makes a cheaper triage possible: for any
selected subset, the per-pair `(M, V, structural_floor,
cancellation_gap)` decomposition (the same one
`cancellation-floor-diagnostic` exposed at the per-pair scalar level)
can be computed for *all* pairs at once from `n_features + 1` Gram
evaluations — one all-zero baseline plus one φ_i = π flip per
feature. No quantum simulation, no encoding step, no q-orca call.

This change introduces a `polygram.analysis` package and a
`polygram analyze` CLI subcommand so researchers can score and
compare candidate subsets before committing to encoding.

## What Changes

- **NEW** `analysis` capability:
  - `polygram.analysis.predict_cancellation_depth(sae_dict,
    feature_ids, **from_sae_lens_kwargs) -> TriagePrediction`
    — entry point. Builds the rung-1 Dictionary via `from_sae_lens`,
    evaluates `n_features + 1` Gram configurations, and returns
    per-pair `(current_overlap, m_pi, M, V, structural_floor,
    cancellation_gap)`, per-feature sensitivity, and an aggregate
    `encoding_suitability_score`.
  - `PairPrediction` dataclass — frozen, per-pair record with
    `is_cross_cluster` derived property.
  - `TriagePrediction` dataclass — bundles dictionary, selection
    report, pairs list, sensitivity dict, scalar score, and the
    `SUITABILITY_FORMULA` documentation string.
  - `feature_sensitivity(...) -> dict[str, float]` —
    convenience wrapper returning mean `|V_ij|` per feature.
  - `encoding_suitability_score(...) -> float` — convenience
    wrapper returning the aggregate score (formula in
    `SUITABILITY_FORMULA`):
    `mean_cancellation_gap × min_pairwise_separation`, where
    `min_pairwise_separation = 1 − max_pair_current_overlap`.
    Both factors live in `[0, 1]`; higher is better.
  - `render_report(prediction) -> str` — markdown report with
    summary table, per-pair table sorted by gap, per-feature
    sensitivity, and an embedded formula explanation.
  - The triage layer is **encoding-free**: it never builds a quantum
    state and never calls q-orca. Cost is `O(n_features)` Gram
    evaluations, each `O(n_features²)` — fine for triaging up to
    ~16 features at a time.

- **MODIFIED** `cli` capability:
  - New `polygram analyze <sae_path> --features <ids> --output
    <md>` subcommand. Loads a toy-SAE JSON (schema matches
    `tests/fixtures/toy_sae.json`), runs
    `predict_cancellation_depth`, writes the rendered report to
    `--output`, and prints the suitability score to stdout.

## Capabilities

### New Capabilities

- `analysis` — pre-encoding feature triage on rung-1 Dictionaries
  using only the closed-form Gram.

### Modified Capabilities

- `cli` — gains the `analyze` subcommand.

## Impact

- `polygram/analysis/__init__.py` — public re-exports
- `polygram/analysis/triage.py` — 312 LOC implementation
- `polygram/cli.py` — `analyze` subparser + `_cmd_analyze` handler
- `tests/test_analysis.py` — 142 LOC test module
- `README.md` — `polygram analyze` quickstart snippet

No breaking changes. No new runtime dependencies (numpy was already
required). No q-orca version bump — this layer is upstream of
encoding entirely.
