# analysis-triage-layer — tasks

> **Retroactive change** — work shipped in commit `a4a906a` on main.
> All boxes ticked at proposal time; this file documents the shipped
> state for the OpenSpec record. The work was not gated on these
> tasks; the audit trail just reflects what already landed.

## 1. Analysis package

- [x] 1.1 `polygram/analysis/__init__.py` re-exports
      `PairPrediction`, `TriagePrediction`,
      `SUITABILITY_FORMULA`, `predict_cancellation_depth`,
      `feature_sensitivity`, `encoding_suitability_score`,
      `render_report`
- [x] 1.2 `polygram/analysis/triage.py` —
      `predict_cancellation_depth` builds rung-1 Dictionary via
      `from_sae_lens`, evaluates `n + 1` Gram configurations
      (one all-zero baseline, one φ_i = π flip per feature), and
      returns `TriagePrediction`
- [x] 1.3 `PairPrediction` dataclass with
      `feature_a`, `feature_b`, `cluster_a`, `cluster_b`,
      `current_overlap`, `m_pi`, `M`, `V`, `structural_floor`,
      `cancellation_gap`, plus `is_cross_cluster` derived property
- [x] 1.4 `TriagePrediction` dataclass bundles `dictionary`,
      `selection_report`, `pairs`, `feature_sensitivity`,
      `encoding_suitability_score`, `suitability_formula`
- [x] 1.5 `feature_sensitivity()` convenience wrapper —
      `dict[str, float]` of mean `|V_ij|` per feature
- [x] 1.6 `encoding_suitability_score()` convenience wrapper —
      scalar in `[0, 1]`,
      `mean_cancellation_gap × min_pairwise_separation`
- [x] 1.7 `SUITABILITY_FORMULA` module constant documents the
      score formula and intuition
- [x] 1.8 `render_report()` emits a deterministic markdown report
      with summary line, per-pair table, per-feature sensitivity,
      and quoted formula footer
- [x] 1.9 No q-orca / quantum-simulation calls in this module

## 2. CLI

- [x] 2.1 `polygram analyze <sae_path> --features <ids>
      [--output <path>]` subcommand registered in `polygram/cli.py`
- [x] 2.2 `_cmd_analyze` loads the SAE JSON, parses
      `--features`, runs `predict_cancellation_depth`, writes the
      rendered report, prints the suitability score
- [x] 2.3 `_parse_feature_ids` rejects malformed inputs with a
      clear error

## 3. Tests

- [x] 3.1 `tests/test_analysis.py` — `predict_cancellation_depth`
      shape + closed-form-Gram agreement on toy SAE fixture
- [x] 3.2 `feature_sensitivity` keys match selected features,
      values non-negative
- [x] 3.3 `encoding_suitability_score` lies in `[0, 1]`
- [x] 3.4 `render_report` contains all required sections and
      every feature name
- [x] 3.5 `polygram analyze` CLI integration test — exit 0,
      report file written, score printed to stdout

## 4. README + docs

- [x] 4.1 README — `polygram analyze` quickstart snippet, link to
      `polygram.analysis` programmatic API

## 5. Validate + commit

- [x] 5.1 `openspec validate analysis-triage-layer --strict` ✓
      (this commit; validates the retroactive proposal)
- [x] 5.2 All tests pass; ruff clean
- [x] 5.3 Commit + push (work landed as `a4a906a`)
