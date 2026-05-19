## Why

The [sm-sae benchmark fixture](https://jascal.github.io/sm-sae/) publishes
a "Recommended defaults" table that calls out two polygram defaults
as having **measured** evidence for change:

| field | upstream default | sm-sae recommendation | rationale |
|---|---|---|---|
| `SAEImportConfig.assign_amp_knobs` | `False` | `True` | Without amp knobs, cancellation is stuck at the structural floor. For any benchmark that measures structured-feature recovery, this should be flipped. |
| `SAEImportConfig.assign_phase_knobs` | `False` | `True` | Required to populate the `.phi` entries the 4-knob Rung3 search expects. |

Both are tagged `measured` in the table (directly supported by sweep
results, not just first-principles guess). The article's general
framing — *"For any benchmark that measures structured-feature
recovery, this should be flipped"* — generalises the recommendation
beyond sm-sae to any ground-truth-rich fixture.

The corresponding `Cancellation(encoding=...)` row in the same table
recommends switching from `None` → `"rung3"`. **Investigation shows
this is already in effect by inference**: `Cancellation.encoding=None`
calls `_infer_encoding_string(dictionary.encoding)` which returns
`"rung3"` for `Rung3` dictionaries, picking up the 4-knob
`[a.phi, b.phi, b.theta_amp, b.psi_aux]` list automatically.
Verified by exercising `Cancellation(dictionary=<Rung3 dict>, ...)`
with no `encoding=` kwarg — see proposal.md notes. No code change
needed.

The encoding-side recommendation `Rung3(bond_dim=2)` in the same
table is tagged `provisional` (pending sweep), and switching
`from_sae_lens`'s default encoding from `MPSRung1` is a wider
behavioural change with downstream implications. Deferred to a
separate change.

The `Rung5(n_amp_qubits=2)` over `Rung4` recommendation in the
article is a performance issue tracked upstream as polygram#86,
not a default-value question; out of scope here.

## What Changes

- `SAEImportConfig.assign_amp_knobs`: `False` → `True`. Comment
  describing the previous "preserves byte-identical behaviour" is
  updated to reference the sm-sae measured recommendation.
- `SAEImportConfig.assign_phase_knobs`: `False` → `True`. Same comment
  update.
- Tests that asserted the previous default-False behaviour are updated
  to pass the kwarg explicitly where the assertion's intent is
  "knob assignment is off."

## Impact

- **Affected specs**: `sae`.
- **Affected code**: `polygram/config.py` (two field defaults).
- **Affected tests**: any test that exercised the `False` default
  implicitly. To be enumerated during implementation; expected
  ≤ 5 sites.
- **Risk**: medium. Public API behaviour change — callers using
  `SAEImportConfig()` defaults now get phase and amp knobs populated.
  Cancellation runs that previously hit the structural floor will
  now actually search the amp branch. This is the *intended* effect,
  per the sm-sae writeup.

## Verified

- `Cancellation(dictionary=<Rung3 dict>, target_pair=('a','b'))` with
  no `encoding=` kwarg produces `encoding='rung3'` and `knobs=['a.phi',
  'b.phi', 'b.theta_amp', 'b.psi_aux']`. The sm-sae recommendation is
  already the live default by inference. (Verified locally.)
