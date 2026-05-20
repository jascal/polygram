## Why

Two diagnostic gaps in the current toolchain make existing outputs
hard to interpret without insider knowledge:

1. **Cancellation at the structural floor is invisible.** When the
   target pair's `before_overlap` already sits on the encoding's
   structural floor, `Cancellation.run()` returns
   `cancellation_efficiency=None`. That's the same value reported
   when the floor is `NaN` ("undefined for this encoding/knob
   configuration") — so a user can't tell whether the optimiser
   succeeded *because nothing was left to gain* or whether the floor
   was simply unmeasurable. Either way they leave the call unsure
   whether to re-run with a richer knob set. The previous
   `cancellation-floor-diagnostic` change (archived 2026-05-03)
   introduced the floor and the efficiency ratio but stopped short
   of distinguishing these two cases.

2. **Compression reports don't say whether the kept features still
   span the activations.** `CompressionReport`, `EpochReport`, and
   `RegrowReport` record what was kept and what was zeroed, but
   nothing about how well the kept basis covers the input subspace.
   Downstream forge-MSE numbers (computed in the host repo, not in
   polygram) get interpreted in a vacuum: a low MSE could mean the
   strategy preserved the subspace OR that the test prompts didn't
   exercise it. Two cheap rank / projection diagnostics computed at
   `Compressor.apply()` time close that gap.

## What Changes

### `experiment` capability — modified

- **MODIFIED** `CancellationResult` gains
  `at_structural_floor: bool` (default `False`). `True` when
  `before_overlap`, `after_overlap`, and `structural_floor` agree
  within an absolute tolerance of `1e-6`.
- **MODIFIED** `Cancellation.run()` and the rung3/rung4/MPSRung1
  variants populate the new field on every returned result and emit
  a `UserWarning` (stacklevel 2) when it is `True`, naming the
  three values and the conclusion "operating at the structural
  floor".
- **MODIFIED** When `at_structural_floor` is `True` and the
  existing efficiency path would return `None`,
  `cancellation_efficiency` is coerced to `0.0` instead. This
  preserves the rule "`None` means floor is undefined" — a
  legibly-at-floor result is *numerically zero gap consumed*, not
  unmeasurable.
- README "Cancellation" section gains a small "Which tool should
  I use?" pivot table mapping a user's question
  (orthogonalise pair / coherent in SAE / encoding's structural
  floor / compress while preserving structure) to the right tool
  and the required encoding rung.

### `pareto-compression` capability — modified

- **MODIFIED** `CompressionReport` adds four optional fields:
  - `rank_ratio: float | None` —
    `basis_rank(W_dec_kept) / d_model`, where `W_dec_kept` is the
    cluster-representative rows of the rewritten decoder.
  - `post_A: float | None` —
    `1 - var(h - h_proj) / var(h)`, computed as a weighted ratio
    over cluster representatives using the *source* `W_dec` and a
    projector `P = pinv(W_kept) @ W_kept`. Reads "fraction of
    representative-direction variance preserved by the kept-feature
    subspace".
  - `forge_mse: float | None` — caller-provided MSE from the host
    repo's forge pipeline. Polygram does not compute this; it
    stores it for traceability.
  - `informative_metric: Literal["post_A", "both", "forge_mse"] | None` —
    derived from `rank_ratio`: `"post_A"` when `<0.95`, `"both"`
    when `0.95 ≤ x ≤ 1.05`, `"forge_mse"` when `>1.05`. `None` when
    `rank_ratio` is `None`.
- **MODIFIED** `EpochReport` and `RegrowReport` carry the same four
  fields, with the same defaults and the same back-compat load
  behaviour.
- **MODIFIED** `Compressor.apply()` computes `rank_ratio` and
  `post_A` from the rewritten / source decoders respectively and
  stores them on the produced `CompressionReport`. `forge_mse`
  stays `None` (the host repo sets it post-hoc).
- **MODIFIED** Schema versions bump:
  `CompressionReport.SCHEMA_VERSION 1 → 2`,
  `EpochReport.SCHEMA_VERSION 2 → 3`,
  `RegrowReport.SCHEMA_VERSION 1 → 2`. All three loaders accept
  prior-version payloads; missing diagnostic fields default to
  `None`. Equality and hash include the new fields (NaN-aware).
- Shared helpers (`json_finite`, `floats_eq`,
  `informative_metric`, `compute_rank_ratio`) move to
  `polygram/compression/_helpers.py` so all three report modules
  and the `Compressor` import one canonical implementation.

## Capabilities

### New Capabilities

*(none)*

### Modified Capabilities

- `experiment` — `Cancellation` results report an
  `at_structural_floor` flag and emit a `UserWarning` when the
  optimiser is operating at the floor.
- `pareto-compression` — `CompressionReport`, `EpochReport`, and
  `RegrowReport` gain `rank_ratio`, `post_A`, `forge_mse`,
  `informative_metric` fields; `Compressor.apply()` computes the
  first two; schema versions bump with back-compat loaders.

## Impact

- `polygram/cancellation.py` — ~40 LOC delta (new field, helper,
  warning + efficiency coercion in four `run`-style paths).
- `polygram/compression/report.py`,
  `polygram/compression/epoch_report.py`,
  `polygram/compression/regrow_report.py` — schema bump + four new
  fields each, NaN-aware equality, back-compat loaders.
- `polygram/compression/_helpers.py` — new file (~60 LOC) with
  `json_finite`, `floats_eq`, `informative_metric`,
  `compute_rank_ratio`.
- `polygram/compression/compressor.py` — ~50 LOC for the two
  metric computations + storing them on the produced report.
- `tests/test_cancellation.py` — +2 tests
  (`test_at_floor_flag_and_warning`,
  `test_efficiency_zero_when_at_floor`).
- `tests/test_compression_report.py`,
  `tests/test_epoch_report.py`,
  `tests/test_regrow_report.py` — +1 round-trip test each, +1
  back-compat-load test each.
- `tests/test_compressor.py` — +2 tests
  (`test_apply_populates_rank_ratio_and_post_A`,
  `test_informative_metric_thresholds`).
- `README.md` — small extension to existing tour section.

No new runtime dependencies. The two new metrics use only
`numpy.linalg.matrix_rank` and `numpy.linalg.pinv` on a
`(n_kept, d_model)` matrix — fast for the SAE scales currently in
use. Existing report consumers ignoring the new fields continue to
work; v1 / v2 JSON payloads still load.
