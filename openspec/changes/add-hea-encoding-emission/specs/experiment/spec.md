## MODIFIED Requirements

### Requirement: Q-Orca file emission with provenance

Polygram SHALL expose `polygram.emit.write_qorca(dictionary, path)` that
writes a `.q.orca.md` file readable by `q_orca.parser.parse_q_orca_markdown`
and verifiable by `q_orca.verifier.verify`. The emitted file SHALL begin
with a comment block naming the source `Dictionary`, the generation
timestamp, and the git revision (or "unversioned" outside a repo).

The renderer SHALL dispatch on `dictionary.encoding`. For `MPSRung1`
the emitted body SHALL be the existing rung-1 staircase layout
unchanged. For `HEA_Rung2` the emitted body SHALL include three
extra sections in order:

1. A `## encoding` table with `kind: hea`, `depth`, `entangler`,
   and `rotations` matching `dictionary.encoding`.
2. A `## theta` table with three columns
   `| concept | tensor | cluster |`. The `concept` column carries
   the feature slug, `tensor` carries the literal-eval-able Python
   list form of each feature's θ tensor (using the encoding's
   default-tensor generator when `feature.theta is None`), and
   `cluster` carries the feature's `cluster` field verbatim.
3. A `## invariants` section declaring
   `- concept_gram_tier_separation >= <bound>` whenever
   `encoding.tier_separation_bound is not None`. When the field
   is `None`, the section SHALL be omitted from the HEA branch.

The shipped Animals example SHALL exercise this path end-to-end as part
of the example test (rung-1). A new `examples/animals_hea.py` SHALL
exercise the HEA branch, producing a file that
`q_orca.verifier.verify` accepts under default options (Stage 4b
including the tier-separation invariant).

#### Scenario: HEA dictionary emits encoding/theta/invariants

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` with three
  features grouped two-and-one across clusters `s1` and `s2`
- **WHEN** `polygram.emit.write_qorca(dictionary, path)` runs
- **THEN** the written file contains a `## encoding` section with
  `kind: hea`, a 3-column `## theta` table whose `cluster` column
  reads `s1, s1, s2`, and a `## invariants` section listing
  `concept_gram_tier_separation >= 0.025`

#### Scenario: HEA dictionary with bound=None omits invariants

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2,
  tier_separation_bound=None))`
- **WHEN** the emitter runs
- **THEN** the produced markdown does not contain a
  `## invariants` section

#### Scenario: HEA emission verifies clean

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` with
  features whose θ tensors satisfy the declared
  `tier_separation_bound`
- **WHEN** the emitted file is parsed and passed to
  `q_orca.verifier.verify`
- **THEN** `result.valid` is `True` and no error of code
  `HEA_GRAM_INVALID`, `HEA_TIER_INVARIANT_VIOLATED`, or
  `HEA_TIER_UNDEFINED` is reported
