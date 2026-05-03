# add-hea-encoding-emission — tasks

## 1. Dependency + version

- [ ] 1.1 `pyproject.toml` — bump `q-orca>=0.7.1` to `q-orca>=0.9.0`
- [ ] 1.2 Confirm `pip install -e ".[dev]"` resolves against the
      published v0.9.0 wheel
- [ ] 1.3 `AGENTS.md` — update the Q-Orca dependency-contract
      paragraph to mention HEA + tier invariants are now part of
      the stable surface

## 2. Encoding marker

- [ ] 2.1 `polygram/encoding.py` — `HEA_Rung2` frozen dataclass with
      fields `depth: int`, `entangler: str = "ring"`,
      `rotations: tuple[str, ...] = ("Ry", "Rz")`,
      `tier_separation_bound: float | None = 0.025`,
      `n_qubits: int = 3`
- [ ] 2.2 `__post_init__` validates `depth >= 1`,
      `entangler in {"ring", "chain"}`, every rotation in
      `{"Rx", "Ry", "Rz"}`, and (if not None)
      `0.0 <= tier_separation_bound <= 1.0`
- [ ] 2.3 Public re-export from `polygram.__init__` alongside
      `MPSRung1`

## 3. Feature θ tensor

- [ ] 3.1 `polygram/dictionary.py` — `Feature` gains optional
      `theta: np.ndarray | None = None`. Default `None` means
      "let the emitter generate a tensor from `(α, β, γ, φ)`."
- [ ] 3.2 New helper `_default_hea_theta(feature, encoding)` builds
      a `(|rotations|, depth, n_qubits)` tensor by spreading the
      four scalar knobs across the first layer's qubits and zeroing
      the rest. Cohesion-preserving: small `(α, β, γ)` keeps overlaps
      high; the outsider tier (large `(α, β, γ)`) gets a magnitude
      shift.
- [ ] 3.3 `Feature` validates that an explicitly-passed `theta` has
      the encoding's expected shape; mismatch raises `ValueError`
      naming the offending feature

## 4. Dictionary dispatch

- [ ] 4.1 `Dictionary.encoding` field type widens from `MPSRung1` to
      `MPSRung1 | HEA_Rung2`; default unchanged (`MPSRung1()`)
- [ ] 4.2 `Dictionary.gram()` dispatches: `MPSRung1` →
      `compute_concept_gram_mps`; `HEA_Rung2` →
      `compute_concept_gram_hea`
- [ ] 4.3 New `Dictionary.tier_separation()` returns
      `compute_tier_separation(self.gram(), [f.cluster for f in
      self.features])` — public surface for users who want the
      metric without going through the verifier

## 5. Q-Orca emit (HEA branch)

- [ ] 5.1 `_qorca_emit.render_machine_markdown` dispatches on
      `dictionary.encoding`. Existing rung-1 path stays bit-for-bit
      identical.
- [ ] 5.2 HEA branch emits a `## encoding` table with `kind: hea`,
      `depth`, `entangler`, `rotations` — matches the q-orca-lang
      `examples/larql-hea-minimal.q.orca.md` shape.
- [ ] 5.3 HEA branch emits a 3-column `## theta` table
      `| concept | tensor | cluster |`. Each row's `concept` is the
      feature slug; `tensor` is the literal-eval-able Python list
      form of the θ tensor; `cluster` is the feature's `cluster`
      field verbatim.
- [ ] 5.4 HEA branch emits `## invariants` with
      `- concept_gram_tier_separation >= <bound>` when
      `encoding.tier_separation_bound is not None`. Suppressed when
      the user passes `tier_separation_bound=None`.
- [ ] 5.5 The HEA branch emits a single `query_concept` action and
      one transition per feature, mirroring the rung-1 layout's
      "one prep_<slug> event per feature" style — verified by
      `q_orca.parser.parse_q_orca_markdown` returning no errors.

## 6. Tests

### Encoding

- [ ] 6.1 `tests/test_encoding.py::TestHEARung2` — depth/entangler/
      rotations/bound validation, defaults, frozen-dataclass
      equality.

### Dictionary

- [ ] 6.2 `tests/test_dictionary.py::TestHEADictionary` — accept
      `HEA_Rung2`, validate Feature θ shape, dispatch
      `Dictionary.gram()` to the HEA helper, `tier_separation()`
      returns a positive float for a clearly-tiered fixture.

### Emit

- [ ] 6.3 `tests/test_qorca_emit.py::TestHEAEmit` — emitted markdown
      contains the three sections in the right order, parses cleanly
      under `parse_q_orca_markdown`, the parsed `QMachineDef` has
      `encoding.kind == "hea"`, the parsed θ rows carry the declared
      `cluster` strings, and the parsed machine declares one
      `concept_gram_tier_separation` invariant.
- [ ] 6.4 `tests/test_qorca_emit.py::TestHEAEmit::test_no_invariant`
      — passing `tier_separation_bound=None` produces no
      `## invariants` section.
- [ ] 6.5 `tests/test_emit.py` — `write_qorca` round-trips an
      HEA dictionary: file written, file parses, `verify(machine)`
      returns `result.valid == True` (Stage 4b green including the
      tier-separation invariant).

## 7. Example

- [ ] 7.1 `examples/animals_hea.py` — same Animals dictionary as
      `examples/animals.py` but constructed with
      `encoding=HEA_Rung2(depth=2)` and per-feature θ tensors that
      the existing example proves separable. Asserts
      `verify(machine).valid` and prints the tier separation.
- [ ] 7.2 The original `examples/animals.py` is untouched —
      additive change.

## 8. Validate + commit

- [ ] 8.1 `openspec validate add-hea-encoding-emission --strict` ✓
- [ ] 8.2 Full pytest suite green; ruff clean
- [ ] 8.3 Commit + push, open PR, merge after review

## 9. Archive

- [ ] 9.1 `openspec archive add-hea-encoding-emission` after merge —
      populate the new requirements into
      `openspec/specs/{dictionary,experiment}/spec.md`
