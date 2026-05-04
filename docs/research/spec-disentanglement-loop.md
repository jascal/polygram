# Quantum-informed disentanglement loop — research note

**Status:** research-track. Not an OpenSpec proposal. Captured on
branch `add-sharing-graph-triage` (2026-05-04) when the
"must-separate map" half of a user-supplied uncompress sketch was
folded into `add-sharing-graph-triage` and the active-disentanglement
half was deferred here.

## The pitch

Polygram's triage layer flags pairs whose squared overlap stays high
across all phase configurations (`structural_floor` close to
`current_overlap`). The phase-only `Cancellation` primitive cannot
disambiguate them — `cancellation-phase-floor.md` documents this as
the "structural floor at M − |V|" finding.

The user-supplied sketch proposed an active *disentanglement
primitive*: take a flagged dangerous pair, run interference
experiments to compute a "quantum-informed gradient," and update the
SAE decoder weights to push the pair into more orthogonal
directions. Output: a modified SAE with cleaner feature directions
and a before/after disentanglement report.

## Why it's not a proposal yet

Three concrete blockers, in order of severity.

### 1. The gradient path isn't differentiable

The proposed loss flows
`SAE_decoder_weights → from_sae_lens → Dictionary → Gram → loss`.
Inside `from_sae_lens` are KMeans (cluster assignment) and PCA
(per-cluster β/γ extraction). KMeans is non-differentiable; PCA is
differentiable in principle but the current code uses
`numpy.linalg.svd` and discrete cluster routing, with no gradient
plumbing. There is no `torch` or `jax` pathway that would let an
optimizer back-propagate through the import.

A v0 disentanglement primitive would either need:

- (a) a *differentiable surrogate* `from_sae_lens` (replace KMeans
  with a soft-assignment relaxation; replace SVD with a
  differentiable eigendecomposition), or
- (b) a *direct β/α/γ search* that bypasses the SAE-weight layer
  entirely — operate on the Polygram `Dictionary` parameters, then
  project the change back through a pseudo-inverse to recover an
  SAE-weight delta. (b) is closer to how `Cancellation` works
  today and probably the smaller leap.

Either path is a meaningful infrastructure investment. Neither has
been prototyped.

### 2. No training-loop infrastructure

`Cancellation` runs an Optuna search over a 4-axis grid. A
disentanglement loop is structurally different: stochastic gradient
descent (or its surrogate) over a much larger weight space, with
optimizer state, learning rate scheduling, and a *capability-
preservation* metric — i.e. "does the modified SAE still decode the
features it was supposed to?" Polygram has none of these. q-orca is
designed around discrete machine emission and verification, not
classical training. Adding a training loop is a major scope
expansion.

### 3. No fixture demonstrates the gradient signal exists

Every Polygram primitive that has shipped — `Cancellation`,
`InterferenceSweep`, the triage layer, cluster-shared knobs — was
preceded by a fixture experiment that showed the underlying signal
was real (the M+V·cos(δ) decomposition; the per-pair
`structural_floor`; the cluster-shattering hazard of per-feature
θ knobs). The disentanglement pitch has no analogue. Before
proposing it, we'd want to demonstrate on a toy fixture (e.g. a
synthetic 2-cluster SAE with a hand-injected polysemantic feature)
that:

- the closed-form Gram identifies the dangerous pair,
- a small β/α/γ perturbation in a specific direction reduces the
  `structural_floor` measurably,
- that direction is computable from the triage data without an
  expensive search (i.e. there's an actual gradient signal, not a
  needle-in-a-haystack search).

If (3) holds, (2) becomes worth investing in. If (3) fails, the
whole pitch collapses to "search β/α/γ exhaustively for each
flagged pair" — which is `cancellation-phase-floor.md` Implication
#1, and that route was deferred there for the same reason it's
deferred here: it's a re-engineering of the encoding, not a steering
primitive.

## What the separation graph (in `add-sharing-graph-triage`) buys us

The `build_separation_graph` requirement in
`add-sharing-graph-triage` covers the *flagging* half of the
uncompress pitch. It tells the user which cross-cluster pairs have
high `structural_floor` — i.e. which pairs a future disentanglement
primitive would target. That's a useful artifact independent of
whether the disentanglement loop ever ships: a researcher inspecting
an SAE can see "these N cross-cluster pairs have irreducible 35%+
overlap; phase tuning won't help them; if you care, the path is
encoding-level intervention, not steering."

So the user-visible "must-separate map" deliverable from the
original sketch ships now via the separation graph. The active-
repair half waits for (1)–(3).

## When to revisit

Promote this note to an OpenSpec proposal when at least the
following are in hand:

- A toy fixture (≤8 features, ≥2 clusters, one hand-injected
  high-floor cross-cluster pair) demonstrating the gradient signal.
- A scratch implementation showing that perturbing β/α/γ along a
  specific direction reduces the `structural_floor` measurably,
  with the direction computable from the triage data alone.
- A defensible answer to "how do we know the modified Dictionary
  still represents the same concepts?" — i.e. the capability-
  preservation metric. Without this, the loop can drive overlaps
  to zero by collapsing the entire dictionary.

If a quarter passes with no progress on (1) or (3), this note can
be retitled and the direction marked as "tried, didn't pan out, see
[follow-up]" — the cost of writing it down was low; the cost of
proposing it without these is vaporware code.

## Out of scope for this note

- "Quantum-informed gradient" framing. Once (1)–(3) are answered
  the gradient is just the chain rule through a differentiable
  pipeline, no quantum mystique required. The "quantum-informed"
  label is marketing.
- Comparison to existing disentanglement losses in the SAE
  literature (orthogonality penalties, sparse-autoencoder-of-
  autoencoders constructions). Those exist; the question for
  Polygram is whether the closed-form Gram + interference data adds
  any signal beyond what an orthogonality penalty already provides.
  Worth surveying when (3) is being pursued, not before.

## References

- `docs/research/cancellation-phase-floor.md` — phase-only search has a
  structural floor; β/α/γ search and richer encodings are the listed
  paths past it.
- `openspec/changes/archive/2026-05-04-add-cluster-shared-knobs/` —
  the cluster-shattering hazard that motivates not running per-pair
  Cancellation as part of triage.
- `openspec/changes/add-sharing-graph-triage/` — the proposal that
  ships the *flagging* half of the uncompress pitch via
  `build_separation_graph`.
