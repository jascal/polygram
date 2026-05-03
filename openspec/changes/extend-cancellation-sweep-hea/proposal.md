## Why

`add-hea-encoding-emission` shipped HEA dictionary emission and the
`concept_gram_tier_separation` invariant, but it left
`Cancellation` and `InterferenceSweep` operating on a hard-coded
`<feature>.phi` axis only. That's adequate for rung-1 MPS (where the
sole non-amplitude knob is the `Rz(qs[1], φ)` phase), but on HEA the
per-feature θ tensor of shape `(|rotations|, depth, n_qubits)` exposes
many more knobs. Researchers running interference experiments on HEA
dictionaries today must either (a) drop down to writing custom
`scipy` calls against `Dictionary.gram()`, or (b) restrict every HEA
experiment to the single `.phi` axis the `_default_hea_theta` helper
synthesizes — both of which forfeit the primitive's value.

This change generalizes the Cancellation/Sweep search space to a
named-knob list, lets `InterferenceSweep` carry a per-sweep-point
tier-separation measure (and an optional invariant assertion), adds a
combined before/after Gram-matrix visualization, and updates the
Animals HEA example to demonstrate both primitives end-to-end.

`Cancellation.structural_floor()` is **deliberately scoped narrowly**:
the analytic `M − |V|` bound (`|<A|B>|²(δ) = M + V·cos(δ)`) is exact
because Pauli-Rz has eigenvalues ±1 and the squared overlap reduces
to a single sinusoid in `δ = φ_A − φ_B`. The default 2-φ rung-1 search
satisfies that exactly. Multi-knob HEA configurations do **not**
inherit a closed-form floor — even a single `.theta[r,d,q]` knob in a
layered ansatz is not in general bilinear in the difference of two
features' values. Rather than ship a "best-found-so-far" number
labeled as `structural_floor`, this change makes the method raise
`NotImplementedError` on every configuration outside the canonical
rung-1 2-φ shape.

A side-experiment on the Animals HEA dictionary surfaced an additional
reason the deferral is load-bearing, beyond analytic tractability. The
default 2-φ knob set on HEA cannot reduce
`(dog_poodle, bird_hawk)` below `0.7686` (the φ-overlap landscape on
HEA is essentially flat — only 4 unique overlap values across 50
feasible points of a 12×12 grid). Switching to a 4-θ knob set on the
Ry rotations *does* drive the target pair to `≈ 0` — but it
**shatters the cluster**: sibling overlaps collapse from `0.9999` to
`~0.58`, and `tier_separation` flips from `+0.2226` to `−0.1957`
(siblings are now less similar than cross-cluster pairs). The
per-feature θ rotation surgically targets the named pair while
ignoring that they live in clusters. A useful HEA floor therefore has
to be defined with respect to a **cluster-respecting knob set** (e.g.
shared θ across siblings, or constrained θ-deltas) — not just over the
raw `(R, D, Q)` slot space. A defensible HEA floor (e.g. a Lipschitz
upper bound on `|∂overlap/∂θ|` giving a non-tight lower bound, defined
over a cluster-respecting knob set) is **deliberately deferred** to a
research-track proposal once a concrete SAE workload needs it.

## What Changes

### `Cancellation` — multi-knob joint optimization

- Add a `knobs: list[str] | None = None` field. Each entry is a knob
  path of the form:
  - `<feature>.phi` — works on both `MPSRung1` and `HEA_Rung2` (for
    HEA, sets the `phi` scalar consumed by `_default_hea_theta`).
  - `<feature>.theta[r,d,q]` — HEA-only; sets the named slot of the
    feature's θ tensor (overriding the default if `Feature.theta` is
    `None`, or lifting then patching if `Feature.theta` is set).
- When `knobs is None`, the default is
  `[f"{a}.phi", f"{b}.phi"]` — preserves today's 2-φ behavior bit-for-
  bit.
- Extend `Cancellation._dictionary_at(*values)` to a variadic helper
  that walks `knobs` and applies each value via a new
  `Dictionary.with_knob(path, value)` (added to the dictionary
  capability — see below).
- Search bounds: `[0.0, 2π]` for `.phi`; `[-π, π]` for
  `.theta[r,d,q]`. (Empirical: `_default_hea_theta` produces values
  well within ±π for the Animals example, so this range covers the
  practical design space; users with a tighter operational window can
  pass a fixed θ then sweep over a `.phi` shim, which is exactly the
  pattern proposed in `add-hea-encoding-emission`'s out-of-scope
  list.)
- Grid backend: `max_steps` now means "points per axis"; total
  evaluations = `max_steps ** len(knobs)`. Researcher-friendly safety
  rail: `len(knobs) > 4` raises `ValueError` recommending
  `method="scipy"`. Scipy backend: bounds list lifted from the knob
  list; `differential_evolution` is dimension-agnostic.
- `trajectory` widens from `(M, 3)` to `(M, len(knobs) + 1)` —
  one column per knob plus the overlap. Existing 2-φ default still
  produces `(M, 3)`. `feasible_mask` shape `(M,)` unchanged.
- `optimized_phis` field renamed to `optimized_knobs:
  dict[str, float]` (keyed by knob path, not just feature name).
  Old field name removed; downstream callers update at the same time.

### `Cancellation.structural_floor()` — encoding-aware contract

- Defined exactly when:
  1. `dictionary.encoding` is `MPSRung1`, AND
  2. `knobs` is the default 2-φ pair (`["{a}.phi", "{b}.phi"]`).
- Outside that shape, raises `NotImplementedError` with a message
  naming the configuration and pointing at this proposal's deferral
  paragraph. `cancellation_efficiency` is `None` whenever
  `structural_floor()` is undefined (in addition to the existing
  `before − floor < 1e-9` case).

### `InterferenceSweep` — sweep keys + tier-separation surfacing

- Sweep keys generalize: same `<feature>.phi` and
  `<feature>.theta[r,d,q]` syntax as Cancellation knobs. MPS
  dictionaries continue to accept `.phi` only; HEA accepts both.
- `ExperimentResult` gains:
  - `tier_separation: np.ndarray | None` — same shape as `overlaps`
    (`*sweep_dims`); `None` when the dictionary geometry has every
    cluster a singleton (matching `Dictionary.tier_separation()`'s
    contract). Populated from
    `q_orca.compiler.concept_gram_hea.compute_tier_separation` per
    sweep point — runs on both encodings (the metric is encoding-
    agnostic; the Gram itself dispatches).
  - A new optional assertion
    `concept_gram_tier_separation_bound_holds`, supported only for
    HEA dictionaries with a non-`None`
    `encoding.tier_separation_bound`. Asserts
    `tier_separation[idx] >= encoding.tier_separation_bound` per
    sweep point; raises `ValueError` at construction time if the
    user requests it on a dictionary that doesn't carry a bound.
- `Experiment.materialize` is unchanged in signature — its emitted
  `.q.orca.md` already carries the `## invariants` section by virtue
  of the encoding-driven dispatch shipped in
  `add-hea-encoding-emission`. The new pieces are the per-sweep-point
  measure and the optional assertion.

### `CancellationResult` — combined before/after visualization

- `CancellationResult.plot(path, kind="before_after")` produces a
  three-panel figure:
  1. Before Gram heatmap (`|gram|²`).
  2. After Gram heatmap, same colorbar scale.
  3. Bar chart with `before_overlap`, `after_overlap`, and (if
     defined) `structural_floor` for the target pair, plus a tier-
     separation indicator if the dictionary carries clusters with
     ≥ 2 features.
- Existing `kind="grid"` and `kind="scipy"` line plots stay the
  default for backwards compatibility — `plot(path)` without `kind`
  picks the per-method default.

### `examples/animals_hea.py` — demonstrates both primitives

- Extends the existing example to:
  1. Run the current emit + verify path (unchanged).
  2. Run an `InterferenceSweep` over a single
     `dog_poodle.phi` axis on the HEA dictionary, materializing the
     result and asserting `concept_gram_tier_separation_bound_holds`
     across the sweep.
  3. Run a `Cancellation` on `(dog_poodle, bird_hawk)` with the
     default 2-φ knobs and `method="grid"`, then materialize and
     emit the before/after viz.
  4. Print a small tier-separation rollup so the researcher can see
     the headline numbers without opening the artifact files.

## Capabilities

### Modified Capabilities

- `experiment` — Cancellation knobs surface; encoding-aware
  `structural_floor()` contract; InterferenceSweep tier-separation
  measure + optional assertion; before/after plot kind.
- `dictionary` — `Dictionary.with_knob(path, value)` helper for
  single-slot knob mutations on both encodings.

### New Capabilities

*(none — this change is purely additive on existing capabilities)*

## Out of Scope

The following items appeared in scoping discussions and are
explicitly **not** part of this change:

- **Defensible HEA structural-floor.** A Lipschitz/spectral-norm
  upper bound on `|∂overlap/∂θ|` (or any other principled non-tight
  lower bound on the achievable overlap on multi-knob HEA) is a
  research question. Defer to a follow-up proposal once an SAE
  workload tells us what shape of bound is useful.
- **Auto-generation of HEA θ tensors from a sweep range.** Same
  reasoning as `add-hea-encoding-emission`'s out-of-scope item —
  sweep semantics over an `(R, D, Q)` parameter space is non-
  trivial. Users sweep over named single slots via the new
  `.theta[r,d,q]` syntax instead.
- **Cancellation under amplitude variation (β/α/γ).** Pure-phase
  search bounded below by the rung-1 `M − |V|` floor is a feature,
  not a bug — driving overlap below requires amplitude variation,
  which is a separate research-track question.
- **Cluster-respecting HEA knob sets.** The current θ-knob surface
  is per-feature, so a researcher cancelling
  `(dog_poodle, bird_hawk)` via θ-rotations can drive their overlap
  to `≈ 0` while collapsing sibling overlaps (`dog_poodle`,
  `dog_beagle`) and inverting the tier-separation invariant — see
  the deferral paragraph above. A future proposal can introduce
  cluster-shared knobs (one `.theta[r,d,q]` slot patched
  identically across all siblings) or invariant-preserving
  optimization (Cancellation runs with `tier_separation_bound` as a
  hard constraint), but the surface for that is a research-track
  question. For now, researchers using θ knobs on HEA should
  hand-check the `tier_separation` measure on the materialized
  `_at_optimum.q.orca.md` rather than trust the cancellation
  result alone.
- **`QFT_Rung3` and other algebraic encoding families.** Speculative
  until SAE evidence justifies them.

## Impact

- `polygram/cancellation.py` — `knobs` field, generalized
  `_dictionary_at`, encoding-aware `structural_floor()`, before/after
  plot kind.
- `polygram/experiment.py` — sweep-key generalization,
  `ExperimentResult.tier_separation`, new
  `concept_gram_tier_separation_bound_holds` assertion.
- `polygram/dictionary.py` — `with_knob(path, value)` helper.
- `polygram/_assertions.py` — new
  `concept_gram_tier_separation_bound_holds` checker.
- `tests/test_cancellation.py`, `tests/test_experiment.py`,
  `tests/test_dictionary.py`, `tests/test_examples.py` — extended
  for the new surfaces.
- `examples/animals_hea.py` — extended to demonstrate both
  primitives end-to-end.
- No q-orca dependency change — relies entirely on v0.9.0's already-
  shipped surface (`compute_tier_separation`, the HEA Gram helper,
  the invariant emission machinery).
