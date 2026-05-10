## Why

`Regrower.from_compression_report(...).run(output_path)` regrows
**every** zeroed slot in the supplied `CompressionReport`. There is
no public knob to limit the number of regrown slots per call. The
count is fully determined by the just-completed compression's
`n_features_zeroed`.

This works for polygram's primary use case (one-shot regrowth
back to the original SAE size), but it blocks downstream
*adaptive* consumers that want to drive growth signals into
polygram. Concretely:

- **sae-forge `adaptive-regrow`** ([proposal](https://github.com/jascal/sae-forge/pull/16))
  defines a controller that picks a per-cycle growth count based
  on the current basis size vs a configured target. The
  controller's output (`effective_regrow_count`) is meaningful
  *only* if the regrower can be told to regrow exactly that many
  slots. Without a polygram-side knob, the controller is
  effectively informational — the basis size after regrow is
  fixed by the prior compression's zero-count, regardless of
  what the controller computes.

The discovered limitation surfaced during sae-forge's
`adaptive-regrow` implementation work — the implementation
stopped at the orchestration layer (controller class +
`ForgePipeline` knobs landed; per-cycle wiring blocked) and
parked pending a polygram-side knob.

This change adds an opt-in `top_k` selector to `RegrowConfig`.
When set, the regrower regrows only the first `top_k` zeroed
slots in plan order; remaining slots stay zero in the output
checkpoint. The default behavior (regrow every zeroed slot) is
preserved byte-identically when `top_k is None`.

## What Changes

### Scope

A single optional field on `RegrowConfig` plus a few-line gating
in `Regrower.run` to honor it. No change to the regrower's
strategy implementations, residual capture, or output schema.

### New artifacts

- **Tests** in `tests/test_regrow_top_k.py`:
  - `top_k=None` (default) is byte-identical to the pre-change
    behavior — every zeroed slot regrown.
  - `top_k=N` where `N < n_features_zeroed` regrows exactly `N`
    slots, in plan order; remaining slots stay zero.
  - `top_k=N` where `N >= n_features_zeroed` is equivalent to
    `top_k=None` — every slot regrown, no cap effect.
  - `top_k=0` is a valid no-op — same as skipping the regrower
    entirely (the resulting checkpoint equals the input
    `CompressionReport.output_checkpoint`).
  - Determinism: two runs with the same seed and same `top_k`
    produce byte-identical output checkpoints.

### Modified artifacts

- **`polygram/regrowth/config.py`** (or wherever `RegrowConfig`
  is defined) — add `top_k: int | None = None` field. Document
  the semantic and the byte-equivalence guarantee under `None`.
- **`polygram/regrowth/regrower.py`** — `Regrower.run(...)` reads
  `self.top_k` after planning; if non-None and less than the
  plan's slot count, slices the regrow plan to the first `top_k`
  slots in plan order before executing.
- **`polygram/regrowth/regrower.py: from_compression_report`** —
  accepts an optional `top_k` kwarg that overrides
  `config.top_k` when both are provided (precedence matches the
  existing per-field-kwarg-vs-config rule).
- **`CHANGELOG.md`** — `## [Unreleased]` entry documenting the
  new field and its byte-equivalence guarantee under default.
- **`docs/`** (whichever doc covers regrowth) — short subsection
  on the new `top_k` knob.

### CLI surface

The polygram CLI does not expose regrowth directly today; this
change does not add a CLI flag. Downstream consumers
(sae-forge) wire `top_k` via `RegrowConfig.from_dict(ctx[...])`.

### Out of scope (deferred)

- **Selection strategy beyond plan order.** v1 picks the first
  `top_k` slots in `RegrowPlan.populations` order. Other
  selection strategies (by cluster size, by feature ID, by
  caller-supplied list) are deferred to a follow-up
  (`regrow-selection-strategies`) once a downstream consumer
  needs them.
- **Per-strategy `top_k` overrides.** v1 applies `top_k`
  uniformly across all regrow strategies (`residual_kmeans`,
  any future ones). Strategy-specific behavior is deferred.
- **Auto-tuning `top_k` from internal signals.** v1 is purely
  caller-driven. Polygram does not introspect to pick a value.
  The signal-driven controller lives downstream (sae-forge's
  `adaptive-regrow`).

## Capabilities

### Modified Capabilities

- **`tuning-config`** — extends `RegrowConfig` with a `top_k`
  field and adds two new requirements documenting its
  semantics: (1) `None` default preserves byte-equivalence with
  pre-change behavior; (2) integer values cap the per-call
  regrowth count in plan order. The existing
  `Regrower.from_compression_report requires model_name and layer`
  requirement is unchanged.

## Impact

- **No public API breakage.** `top_k` defaults to `None`. Every
  existing caller continues to regrow every zeroed slot. The
  byte-equivalence guarantee under `None` is the load-bearing
  acceptance check.
- **Minimal runtime cost.** When `top_k is None` the regrower
  takes the same code path as today. When set, an O(plan_size)
  list slice runs once before execution.
- **Unblocks sae-forge `adaptive-regrow`.** Once this lands,
  sae-forge can resume implementation: the parked controller
  scaffolding wires through to `ctx["regrow"]["top_k"] =
  effective_regrow_count`, and `perform_regrowth` actually
  consumes the controller's output.
- **Test surface.** ~6 new tests in
  `tests/test_regrow_top_k.py`. No existing tests modified.

## Sequencing

- **Independent of polygram's queued changes.** No conflict
  with `add-confirmation-strategies`, `normalise-sae-loader`,
  or the active `tech-debt-backlog` items.
- **Single PR.** Small surface area; byte-equiv under default
  is the gate.
- **Downstream resume.** sae-forge's `adaptive-regrow`
  (currently parked at PR #16) can resume implementation as
  soon as a polygram release containing `top_k` is published.
