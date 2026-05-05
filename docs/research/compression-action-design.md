# Compression action — design notes

> Engineering note pointing at the
> [`add-compression-action`](../../openspec/changes/add-compression-action/)
> spec. Captures one empirical observation from the §5.1 live validator
> run that forced a specific shape on the action.

## Pointer

The full spec for the compression action lives at
[`openspec/changes/add-compression-action/`](../../openspec/changes/add-compression-action/).
That directory carries `proposal.md`, `design.md` (the eight-decision
table), `tasks.md` (the implementation checklist), and the
capability-spec deltas under `specs/`.

## Why component-first compression beats pair-first

The §5.1 validator's live run on the §4.4 selection (12999, near-cluster
{19398, 4192, 23625}, far-cluster {8371, 2287, 68, 13737}) emitted 8
confirmed candidate pairs. Two were the obvious near-cluster
redundancies. The other six were *every pair* drawn from the
far-cluster — i.e. all four far-cluster features fire on the same 12
tokens.

Pair-by-pair compression of that far-cluster would zero each member
three times (once per pair it appears in), and the final state of any
given feature would depend on which pair the strategy processed last.
That ordering dependence is a property of the algorithm, not of the
underlying redundancy.

Component-first compression — union-find on the confirmed pair list,
collapse to connected components, pick one representative per
component, zero the others *exactly once* — is order-independent,
idempotent under re-application, and matches the natural notion of
"this is one redundancy clique." The spec encodes this; the
implementation lands as `Compressor.plan()` running union-find before
`apply()` ever touches a tensor.

## Why `zero` first

`zero` writes nothing synthetic to the SAE. Redundant features become
fully inert (encoder column zeroed, encoder bias zeroed, decoder row
zeroed), the representative survives unchanged, and `b_dec` (which is
global, not feature-specific) is left alone. That gives the loop a
clean baseline against which a future `merge` strategy — decoder-row
centroid, magnitude-weighted mean across cluster members — can be
benchmarked.

Shipping `merge` first would force a choice over several plausible
centroid definitions and lock in one before there's empirical evidence
about which to prefer. Deferred to its own change.

## See also

- [`behavioural-scaleup-probe.md`](behavioural-scaleup-probe.md) — the
  §4.4 calibration that produced the per-pair Polygram → Jaccard
  Spearman = +0.637 result.
- [`behavioural-validator-design.md`](behavioural-validator-design.md) —
  the upstream half of the loop. The compression action consumes its
  output directly.
- [`add-compression-action/proposal.md`](../../openspec/changes/add-compression-action/proposal.md) —
  what changes and why.
- [`add-compression-action/design.md`](../../openspec/changes/add-compression-action/design.md) —
  the eight-decision table.
