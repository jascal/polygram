## Why

`from_sae_lens(records, feature_ids, ...)` raises a `ValueError` when
`len(feature_ids)` exceeds the target encoding's `max_features`
(e.g., 9 features against `MPSRung1`'s cap of 8). The error message
points the caller at `clustered=True`, but the caller has to know to
type it. For SAEs at real scale (Gemma-Scope, Llama-Scope, Qwen-Scope
ship 16k–1M features per layer), every loader call hits the cap and
the only correct path is the clustered one — yet `clustered=True`
remains opt-in.

This change makes `clustered=True` the **implicit default** when N
exceeds the encoding's cap. The auto-promoted call returns a
`ClusteredDictionary` and surfaces a `SelectionReport` warning so
the promotion is observable. Callers who explicitly want the strict
error continue to pass `clustered=False`.

The change also removes the `_LEGACY_MAX_FEATURES = 8` fallback in
`polygram/clustered_dictionary.py`. That fallback was added defensively
while the `per-encoding-feature-cap` change was in flight; it has been
archived (2026-05-16) and every encoding now exposes
`max_features` natively. The defensive `getattr` is dead code.

## What Changes

- `polygram.sae_import.from_sae_lens`: `clustered: bool = False` →
  `clustered: bool | None = None`. Three modes:
    - `None` (new default) — auto: clustered when N > encoding cap,
      flat otherwise. Auto-promotion appends a warning to
      `SelectionReport.warnings`.
    - `True` — always clustered (unchanged).
    - `False` — strict; raise the existing `ValueError` if N exceeds
      the cap (unchanged from previous default behaviour).
- `polygram/clustered_dictionary.py`: drop `_LEGACY_MAX_FEATURES` and
  the `_encoding_max_features` `getattr` fallback. Every encoding
  exposes `max_features` natively after the archived
  `per-encoding-feature-cap` change.
- Tests that asserted the old strict-raise default are updated to pass
  `clustered=False` explicitly. A new test asserts the auto-promote
  path returns a `ClusteredDictionary` and emits a warning.

No new public API surface. Backwards-compatible for every caller that
either passed `clustered=` explicitly or stayed within the cap.

## Impact

- **Affected specs**: `sae` (capacity-limit-enforced scenario amended).
- **Affected code**: `polygram/sae_import.py`,
  `polygram/clustered_dictionary.py`.
- **Affected tests**: `tests/test_sae_import.py` (5 cap-related
  assertions switched to explicit `clustered=False`; one new
  auto-promote test).
- **Risk**: low. The auto-promote path replaces an error with a
  working result; explicit `clustered=False` preserves the error
  exactly. The dead-code removal touches a path no longer reached
  after the archived `per-encoding-feature-cap` change.
