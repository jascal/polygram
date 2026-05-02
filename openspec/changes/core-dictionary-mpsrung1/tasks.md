# core-dictionary-mpsrung1 — tasks

## 1. Data classes

- [ ] 1.1 `Feature` dataclass — name, cluster, α, β, γ, φ; α/γ default
      to 0.0, β required, φ defaults to 0.0
- [ ] 1.2 `Dictionary` dataclass — name, features, hierarchy; `__post_init__`
      validates each feature belongs to exactly one cluster in `hierarchy`
- [ ] 1.3 `MPSRung1` dataclass — bond_dim (default 2), phase_knobs (default
      True); raises if bond_dim != 2 (rung-1 only for v0)

## 2. Default-angle helper

- [ ] 2.1 Function `assign_default_betas(clusters: list[str]) -> dict[str, float]`
      spreads β evenly in `[-0.5, 0.5]` (e.g. K=2 → {-0.5, +0.5}, K=3 →
      {-0.5, 0.0, +0.5})
- [ ] 2.2 `Dictionary.with_default_angles()` classmethod-style helper:
      build a Dictionary from `{cluster: [feature_names]}` and apply the
      default β per cluster, leaving α=γ=φ=0

## 3. In-memory Q-Orca emit + Gram

- [ ] 3.1 `_qorca_emit.build_machine_def(dictionary)` returns a
      `QMachineDef` matching the larql-animals-interference style
      (states `idle` / `prepared_<feature>` / `done`, single
      `prepare_concept` action with the Rz-bearing staircase signature)
- [ ] 3.2 `Dictionary.gram()` calls
      `q_orca.compiler.concept_gram_mps.compute_concept_gram_mps`
      with `form="preparation"` (avoiding the inverse-form Rz symmetry
      break documented in q-orca-lang's
      `larql-animals-interference.q.orca.md`)
- [ ] 3.3 Returned Gram is a 2D NumPy array indexed by feature order; add
      `Dictionary.feature_index(name) -> int` for label lookups

## 4. Tests

- [ ] 4.1 `test_dictionary.py` — feature/cluster validation, default-β spread
- [ ] 4.2 `test_encoding.py` — rejects bond_dim != 2; phase_knobs flag is
      surfaced on the dataclass
- [ ] 4.3 `test_gram.py` — toy 2-cluster × 2-feature dictionary reproduces
      the larql-animals-interference Gram (1.000 / 0.8851 / 0.6816 /
      0.5931 tiers) within 1e-4
