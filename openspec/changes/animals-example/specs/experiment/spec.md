## MODIFIED Requirements

### Requirement: Q-Orca file emission with provenance

Polygram SHALL expose `polygram.emit.write_qorca(dictionary, path)` that
writes a `.q.orca.md` file readable by `q_orca.parser.parse_q_orca_markdown`
and verifiable by `q_orca.verifier.verify`. The emitted file SHALL begin
with a comment block naming the source `Dictionary`, the generation
timestamp, and the git revision (or "unversioned" outside a repo).

The shipped Animals example SHALL exercise this path end-to-end as part
of the test suite — closing the v0 milestone.

#### Scenario: emitted file parses and verifies clean

- **WHEN** `write_qorca` is called for a 4-feature, 2-cluster dictionary
  and the result is round-tripped through `parse_q_orca_markdown` +
  `verifier.verify`
- **THEN** `verifier.verify` reports `valid == True`

#### Scenario: emitter never produces inverse-form when phi nonzero

- **WHEN** any feature has `phi != 0` and `write_qorca` is called
- **THEN** the emitted transitions table uses preparation-form call
  sites (`prepare_*` events into distinct `prepared_*` states), never
  inverse-form rollback transitions

#### Scenario: animals example produces a valid q-orca artifact

- **WHEN** `tests/test_examples.py::test_animals_interference_runs`
  executes a coarsened version of `examples/animals_interference.py`
- **THEN** the emitted `.q.orca.md` parses and `verify(...).valid` is
  `True`, the destructive-endpoint assertion passes at `phi = π`, and
  the hierarchical-ordering assertion holds at every sweep point
