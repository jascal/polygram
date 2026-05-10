# Design: regrow-top-k

## Why a single integer field, not a list of feature IDs

Two API shapes were considered:

### Option A (rejected): caller passes feature IDs

```python
RegrowConfig(top_k_feature_ids=[42, 71, 88])
```

Pro: maximally expressive — caller picks exactly which slots.
Con: pushes the selection problem onto callers. The
`CompressionReport`'s plan exposes `populations` ordered by
clustering output; a caller would have to inspect the plan to
build the list, which means polygram leaks plan-order details
into the caller's surface. That's a bigger API contract than
the use case justifies.

### Option B (picked): caller passes a count

```python
RegrowConfig(top_k=3)  # regrow first 3 zeroed slots in plan order
```

Pro: tiny API surface (one int). Selection deterministic
(plan order). Sufficient for the queued use case
(sae-forge's adaptive controller wants "+N slots", doesn't care
which N).
Con: callers can't pick *which* slots. Mitigated by the
deferred-follow-up (`regrow-selection-strategies`) if/when a
caller needs that.

**Picked: Option B.** The counted form is the smallest API that
unblocks the queued downstream consumer. If a future caller
needs feature-ID selection, it's an additive extension
(`top_k_feature_ids` as a sibling field, or a new
`selection: list[int] | None` field) — not a redesign.

## Selection in plan order

`RegrowPlan.populations` is the ordered list of slots the
regrower will populate. Today the regrower iterates this list
and runs the configured strategy (e.g. `residual_kmeans`) for
each. With `top_k=N` the regrower SHALL slice
`populations[:N]` before iterating.

The order is determined by the polygram clustering output,
which is itself deterministic given the seed and the input SAE.
Two callers passing the same `top_k=N` against the same
`CompressionReport` SHALL therefore see the same N slots
regrown — the determinism guarantee.

## Implementation site

`Regrower.run(self, output_checkpoint)` already builds the
plan via `self.plan()`. The change is one conditional just
after `self.plan()` returns:

```python
plan = self.plan()
if self.top_k is not None and self.top_k < len(plan.populations):
    plan = dataclasses.replace(
        plan, populations=plan.populations[:self.top_k]
    )
# ... existing iteration over plan.populations ...
```

The remaining zeroed slots are NOT touched by the regrower —
they stay zero in the output checkpoint. The output
`RegrowReport.populations` list reflects only the regrown
subset.

## Edge cases pinned

- **`top_k = 0`.** Valid. Equivalent to skipping the regrower
  entirely. Output checkpoint equals the input
  `CompressionReport.output_checkpoint`. The output
  `RegrowReport.populations` is empty.
- **`top_k >= len(populations)`.** Valid. Equivalent to
  `top_k = None` (no cap effect). Every slot regrown.
- **`top_k < 0`.** Rejected at `RegrowConfig.__post_init__`
  with `ValueError`. Guards against accidental negative values
  from arithmetic.
- **`top_k` and pre-change callers.** Pre-change callers do
  not pass `top_k`; the field defaults to `None`; behavior is
  byte-identical. The CHANGELOG entry calls out this guarantee
  explicitly.

## RegrowConfig.from_dict round-trip

`RegrowConfig` already supports `from_dict` (used by sae-forge
to thread the config through ctx). The new field SHALL
round-trip cleanly:

- `RegrowConfig(...).to_dict()` returns a dict that includes
  `"top_k": <value or None>`.
- `RegrowConfig.from_dict(d)` reads `"top_k"` and sets the
  field; missing key defaults to `None`.

This matches the existing pattern for `prompts`, `seed`,
`n_init`, `device`. No special handling required.

## Per-field kwarg precedence on from_compression_report

`Regrower.from_compression_report` already follows a precedence
rule: per-field kwarg (when non-None) > `config.<field>` >
required-field error. The new `top_k` kwarg follows the same
pattern. Adding the kwarg is a one-line change in the
constructor's `if config is not None: ...` block.

```python
if top_k is None:
    top_k = config.top_k if config is not None else None
```

## Determinism guarantee

The regrower's strategies (today: `residual_kmeans`) are
deterministic given the seed. The plan-order slicing introduced
here is deterministic given the plan. The plan itself is
deterministic given the input SAE and the seed. Therefore: two
calls with identical inputs produce byte-identical output
checkpoints.

This is the existing polygram regrower invariant, preserved
under the new field.

## Test plan

`tests/test_regrow_top_k.py`:

| Test | Assertion |
|---|---|
| `test_top_k_none_is_byte_identical_to_pre_change` | `top_k=None` produces a checkpoint with the same SHA as a (mocked) pre-change run on the same seed |
| `test_top_k_caps_population_count` | `top_k=N` where `N < n_features_zeroed` regrows exactly `N` slots; the remaining `n_features_zeroed - N` slots stay zero |
| `test_top_k_above_zeroed_count_is_no_op_cap` | `top_k=999` against a 5-slot plan regrows 5; no error |
| `test_top_k_zero_is_no_regrow` | `top_k=0` produces a checkpoint equal to the compression's output (no rows changed) |
| `test_top_k_deterministic` | Two runs with same seed + same `top_k=N` produce byte-identical checkpoints |
| `test_top_k_negative_raises` | `RegrowConfig(top_k=-1)` raises `ValueError` at construction |

The first test is the load-bearing acceptance gate. The byte-
identity guarantee under `None` is what unblocks sae-forge's
adaptive-regrow without breaking polygram's existing callers.

## Why this is a small change

Surface area: one new field, one new conditional in `run()`,
one new kwarg on `from_compression_report`, ~6 tests. The
spec delta is two new requirements + one MODIFIED to mention
the new field. No existing test modified.

The decoupling story is also clean: polygram exposes the lever,
sae-forge pulls it. Polygram does NOT need to know about
sae-forge's controller, target sizes, or adaptive logic —
those live downstream.

## Risk: silently changing semantics for callers using strategy="residual_kmeans"

Mitigated by the byte-equivalence test under `top_k=None`. If
the slicing code is reached only when `top_k is not None`, the
default code path is unchanged. The byte-equivalence test
verifies this empirically.

## Risk: polygram release coordination with sae-forge

The downstream consumer (sae-forge `adaptive-regrow`) is parked
pending this change's release. Once a polygram version with
`top_k` is published and pinned in sae-forge's
`pyproject.toml`, the parked impl branch can resume.
Coordination cost: one polygram release + one sae-forge
dep-bump + the resumed adaptive-regrow impl PR.
