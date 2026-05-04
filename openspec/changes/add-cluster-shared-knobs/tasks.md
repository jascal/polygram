# add-cluster-shared-knobs — tasks

## 1. Dictionary collision guard

- [ ] 1.1 `polygram/dictionary.py` — `Dictionary.__post_init__`
      rejects construction when any cluster key collides with any
      feature name. `ValueError` names the colliding identifier.
- [ ] 1.2 Verify existing fixtures (`tests/`, `examples/`) do not
      collide. (Spot-checked in proposal; confirm in impl.)

## 2. with_knob cluster-shared path syntax

- [ ] 2.1 Extend `_parse_knob_path` to return a `(name, kind, slot)`
      tuple unchanged; resolution moves to `with_knob` which checks
      feature names first, then cluster names. New return shape:
      None — keep parser pure, dispatch in `with_knob`.
- [ ] 2.2 `Dictionary.with_knob(path, value)` — when the leading
      identifier matches a cluster (and not a feature), iterate
      `self.hierarchy[name]`, apply the per-feature mutation to each
      member, return a single `Dictionary`. Unknown name (neither
      feature nor cluster) raises `ValueError` mentioning both
      candidates.
- [ ] 2.3 Cluster-shared `<cluster>.theta[r,d,q]` rejected on
      `MPSRung1` with the same message as the per-feature case.
- [ ] 2.4 Out-of-range slot on cluster-shared theta path: validation
      runs once against `encoding.theta_shape`, error names the
      cluster (not an arbitrary member feature).

## 3. Cancellation knob-list acceptance

- [ ] 3.1 `polygram/cancellation.py` — `__post_init__` validation
      accepts cluster-shared paths via the same `with_knob` grammar.
      Bounds: `(0, 2π)` for `.phi`, `(-π, π)` for `.theta[r,d,q]`,
      independent of cluster-shared vs per-feature.
- [ ] 3.2 `_dictionary_at(*values)` requires no changes — `with_knob`
      handles the fanout. Verify by extension test.
- [ ] 3.3 Grid backend `len(knobs) ≤ 4` cap continues to count
      cluster-shared paths as a single axis.

## 4. Summary caveat — cluster-shared mode

- [ ] 4.1 `polygram/cancellation.py` — `_render_summary` detects
      knob-list shape: pure-cluster (every path's leading identifier
      is a cluster), mixed (some cluster, some feature), per-feature
      (existing behavior).
- [ ] 4.2 Pure-cluster: replace the multi-knob caveat with the
      cluster-invariant note (text in proposal). Mixed: emit both
      caveats — the existing multi-knob warning *and* an explicit
      "this list mixes per-feature and cluster-shared paths;
      within-cluster invariant does NOT hold." Per-feature:
      unchanged from `extend-cancellation-sweep-hea`.

## 5. Example update

- [ ] 5.1 `examples/animals_hea.py` — add a third Cancellation run
      with `knobs=["dogs.theta[0,0,0]", "birds.theta[0,0,0]"]`,
      materialize a second before/after figure, print a comparison
      row including target overlap, worst sibling overlap, and tier-
      separation (before/after). Existing 2-φ run stays first.
- [ ] 5.2 Module docstring updated with the new output layout
      (`cluster_shared/` subdirectory under `cancellation/`).

## 6. Tests

- [ ] 6.1 `tests/test_dictionary.py::TestClusterKnob` —
      `<cluster>.phi` fans out, `<cluster>.theta[r,d,q]` fans out
      across HEA siblings, unknown cluster rejected, feature/cluster
      collision rejected at construction.
- [ ] 6.2 `tests/test_cancellation.py::TestClusterSharedKnobs` —
      cluster-shared knobs accepted, trajectory shape correct,
      sibling overlaps preserved bit-for-bit (within numeric
      tolerance) at the optimum, summary text reflects cluster-shared
      mode.
- [ ] 6.3 `tests/test_cancellation.py::TestMixedKnobs` — mixed
      per-feature + cluster-shared list accepted; summary contains
      both caveats; sibling overlaps are NOT guaranteed preserved.
- [ ] 6.4 `tests/test_examples.py::test_animals_hea_example_runs` —
      asserts the cluster_shared before/after figure exists and that
      the cluster-shared run preserves sibling overlaps.

## 7. Validate + commit

- [ ] 7.1 `openspec validate add-cluster-shared-knobs --strict` ✓
- [ ] 7.2 Full pytest suite green; ruff clean
- [ ] 7.3 Commit + push, open PR, merge after review

## 8. Backlog cleanup

- [ ] 8.1 Mark `tech-debt-backlog` §2.1 with a `[~]` (superseded)
      note pointing at this change's archive entry.

## 9. Archive

- [ ] 9.1 `openspec archive add-cluster-shared-knobs` after merge —
      propagate the new requirements into
      `openspec/specs/{dictionary,experiment}/spec.md`.
