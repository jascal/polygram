## Why

`extend-cancellation-sweep-hea` (archived 2026-05-03) shipped multi-
knob `Cancellation` with named `<feature>.phi` and
`<feature>.theta[r,d,q]` paths. An empirical experiment captured in
that change's deferral paragraph showed a sharp hazard: per-feature
θ knobs surgically target the named pair while ignoring that they
live in clusters. On the Animals HEA dictionary, a 4-θ Ry knob set
drove `(dog_poodle, bird_hawk)` overlap from `0.7686` to `≈ 0` —
*while shattering siblings* (`dog_poodle × dog_beagle`:
`0.9999 → 0.5735`) and inverting `tier_separation`
(`+0.2226 → −0.1957`).

The same change documented this as Out-of-Scope ("Cluster-respecting
HEA knob sets") and `tech-debt-backlog` §2.1 captured the
research-track follow-up. This change promotes that bullet into a
real proposal, because the principled mechanic falls out of unitarity
algebra rather than a heuristic.

**The mechanic.** A *cluster-shared* knob is a knob path that, when
applied, sets the same value across every feature in a named cluster.
For features `a, b` both in cluster `C`, the same unitary `U_C` ends
up on both branches. Then:

```
<U_C a | U_C b> = <a | U_C† U_C | b> = <a|b>
```

— so within-cluster Gram entries are *exactly* unchanged. Bit-for-bit,
not approximately. Cross-cluster Gram entries (where the two sides
get different unitaries `U_X`, `U_Y`) are the cancellation lever; the
optimizer pushes those down without touching siblings. The cluster
invariant — every sibling is more similar than every cross-cluster
pair — is preserved by construction.

This is genuinely the *honest* version of "safe knobs." Per-feature
knobs trade safety for expressivity; cluster-shared knobs trade
expressivity (fewer degrees of freedom — one θ per cluster instead of
one per feature) for an algebraic invariant.

## What Changes

### `Dictionary.with_knob` — cluster-shared path syntax

- Extend the path grammar to accept `<cluster>.phi` and
  `<cluster>.theta[r,d,q]`. Resolution order: feature-name first, then
  cluster-name. Existing `<feature>.<...>` paths are unchanged.
- Implementation: `with_knob("dogs.theta[0,0,0]", v)` walks
  `hierarchy["dogs"]` and applies the per-feature mutation to each
  member, returning a single `Dictionary`.
- Disambiguation guard: `Dictionary.__post_init__` rejects
  construction when a feature name and a cluster name collide, with a
  `ValueError` naming both. This is new validation; existing fixtures
  in the repo do not collide (verified) so the guard ships clean.

### `Cancellation` — cluster-shared knobs in the knob list

- The `knobs: list[str]` field accepts cluster-shared paths in any
  position, mixed freely with per-feature paths. Validation reuses the
  extended `Dictionary.with_knob` grammar; bounds are unchanged
  (`(0, 2π)` for `.phi`, `(-π, π)` for `.theta[r,d,q]`).
- Grid backend continues to enforce `len(knobs) ≤ 4`. Cluster-shared
  knobs count as a single axis (one cluster, one value per grid
  point), so a 2-cluster dictionary can fit a comfortable 2-cluster
  sweep within the limit.
- `_dictionary_at(*values)` mutation chain works without changes —
  `with_knob` now does the per-feature fanout internally.
- `optimized_knobs` keys remain the knob path as written (e.g.
  `"dogs.theta[0,0,0]"`), so the trajectory CSV header faithfully
  records what was searched.

### Cancellation summary — cluster-knob mode caveat replaced

When the knob list contains *only* cluster-shared paths, the
materialized summary's `## Caveat` section is replaced with:

> **Note:** cluster-shared knob set. Within-cluster Gram entries are
> preserved exactly (algebraic invariant from unitarity). Verify the
> after-optimum `concept_gram_tier_separation` if you want a numeric
> floor on the cluster ordering, but the qualitative invariant
> (siblings > cross-cluster) holds by construction.

When the list mixes per-feature and cluster-shared paths, the existing
multi-knob caveat continues to fire — mixed lists do *not* inherit the
within-cluster invariant.

### `examples/animals_hea.py` — demonstrate cluster-shared knobs

Extend the existing example to add a third Cancellation run alongside
the existing 2-φ default:

- A cluster-shared run with `knobs=["dogs.theta[0,0,0]",
  "birds.theta[0,0,0]"]` (Ry layer-0, qubit 0; the shape that
  succeeded in the per-feature experiment, now applied
  cluster-respectfully).
- Materialize a second before/after figure and print a comparison row
  showing: target overlap (before/after), worst sibling overlap
  (before/after), tier-separation (before/after). The cluster-shared
  row's "worst sibling" column should be **bit-for-bit identical**
  before vs after.
- Existing 2-φ run stays first, unchanged.

### Tests

- `tests/test_dictionary.py::TestClusterKnob` — `<cluster>.phi`
  fans out across siblings; `<cluster>.theta[r,d,q]` fans out;
  unknown cluster name rejected; feature/cluster collision rejected
  at construction.
- `tests/test_cancellation.py::TestClusterSharedKnobs` —
  cluster-shared knobs accepted; trajectory shape correct;
  within-cluster Gram preserved bit-for-bit at the optimum (compute
  pairwise sibling overlaps before vs at-optimum, assert equal to
  numeric tolerance); summary caveat text reflects cluster-shared
  mode.
- `tests/test_cancellation.py::TestMixedKnobs` — mixed per-feature +
  cluster-shared list accepted; summary caveat falls back to the
  multi-knob warning (mixed lists do *not* preserve within-cluster
  Gram).
- `tests/test_examples.py::test_animals_hea_example_runs` —
  extended to assert the second before/after figure exists and that
  the cluster-shared run preserves sibling overlaps.

## Capabilities

### Modified Capabilities

- `dictionary` — `Dictionary.with_knob` accepts `<cluster>.<slot>`
  paths; `Dictionary.__post_init__` validates feature/cluster name
  uniqueness.
- `experiment` — `Cancellation.knobs` accepts cluster-shared paths;
  materialized summary distinguishes cluster-shared, mixed, and
  per-feature configurations.

### New Capabilities

*(none — additive on existing capabilities)*

## Out of Scope

- **Knob-binding across non-cluster groupings.** Arbitrary equality
  constraints (e.g. "tie `dog_poodle.theta[0,0,0]` to
  `dog_beagle.theta[0,0,0]` only") are not supported. Cluster-shared
  is the only binding mode introduced — that's enough to give
  invariant preservation, and arbitrary bindings invite surface
  inflation without empirical justification.
- **`tier_separation_bound` as a hard optimization constraint.** A
  separate, complementary path (run optimizer with the bound as a
  feasibility filter, reject points that fall through it). Mentioned
  in `tech-debt-backlog` §2.1 — staying deferred. Cluster-shared
  knobs prevent the violation from happening at all; the constraint
  approach handles per-feature knobs that *might* violate. They're
  orthogonal.
- **`suggest_safe_knobs()` heuristic helper.** An external task spec
  proposed a helper that returns "safe" path lists. Empirically
  there is no defensible per-feature safe choice on HEA (Rz layer-0
  has zero leverage on `|0⟩` initial states; Ry knobs are exactly the
  cluster-shatterer). This proposal supersedes that idea: the
  principled answer is a binding mechanic, not a curated path list.
- **Per-knob and joint structural-floor diagnostics.** Same deferral
  as `tech-debt-backlog` §2.1 — best-found-so-far values labeled as
  "structural floor" mislead. Cluster-shared knobs make the
  numerical bound *more* meaningful (the optimizer is exploring a
  smaller, principled subspace), but the analytic-bound question
  remains research-track.

## Impact

- `polygram/dictionary.py` — extended path grammar in
  `_parse_knob_path`; `with_knob` cluster-fanout; `__post_init__`
  collision check.
- `polygram/cancellation.py` — knob-list validation accepts cluster
  paths; summary renderer distinguishes pure-cluster, mixed, and
  per-feature shapes.
- `examples/animals_hea.py` — third Cancellation run + comparison
  printout.
- `tests/test_dictionary.py`, `tests/test_cancellation.py`,
  `tests/test_examples.py` — extended for the new surfaces.
- No q-orca dependency change — the binding semantics live entirely
  on the polygram side; `with_knob` produces a regular `Dictionary`
  whose emit/verify path is unchanged.
- `tech-debt-backlog` §2.1 — mark superseded by this change once the
  impl PR ships.
