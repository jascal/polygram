# core-dictionary-mpsrung1 — tasks

## 1. Data classes

- [x] 1.1 `Feature` dataclass — name, cluster, α, β, γ, φ; α/γ default
      to 0.0, β required, φ defaults to 0.0
- [x] 1.2 `Dictionary` dataclass — name, features, hierarchy; `__post_init__`
      validates each feature belongs to exactly one cluster in `hierarchy`
- [x] 1.3 `MPSRung1` dataclass — bond_dim (default 2), phase_knobs (default
      True); raises if bond_dim != 2 (rung-1 only for v0)

## 2. Default-angle helper

- [x] 2.1 Function `_default_betas(clusters)` spreads β evenly in
      `[-0.5, 0.5]` (K=2 → {-0.5, +0.5}, K=3 → {-0.5, 0.0, +0.5})
- [x] 2.2 `Dictionary.with_default_angles()` classmethod-style helper:
      build a Dictionary from `{cluster: [feature_names]}` and apply the
      default β per cluster, leaving α=γ=φ=0

## 3. In-memory Q-Orca emit + Gram

- [x] 3.1 `_qorca_emit.build_machine(dictionary)` returns a `QMachineDef`
      matching the larql-animals-interference style (states `idle` /
      `prepared_<feature>` / `done`, single `prepare_concept` action with
      the Rz-bearing staircase signature). Implemented as render-markdown
      + parse round-trip, so the same renderer used by `Dictionary.gram()`
      is what the public `polygram.emit.write_qorca` will reuse in change
      `experiment-interference-sweep`.
- [x] 3.2 `Dictionary.gram()` calls
      `q_orca.compiler.concept_gram_mps.compute_concept_gram_mps`. The
      preparation form is implicit — Polygram only emits prep-form action
      call sites, and the safe-`Rz` matcher rejects inverse-form `Rz` at
      runtime, so this can't drift.
- [x] 3.3 Returned Gram is a 2D NumPy array indexed by feature order;
      `Dictionary.feature_index(name) -> int` exposed for label lookups
      (plus convenience `Dictionary.feature(name)`)

## 4. Tests

- [x] 4.1 `test_dictionary.py` — 11 tests covering feature/cluster
      validation (5 negative cases), `feature_index`, default-β spread,
      `with_default_angles` and `with_phi` helpers
- [x] 4.2 `test_encoding.py` — rejects bond_dim != 2; phase_knobs flag
      surfaced on the dataclass and toggleable
- [x] 4.3 `test_gram.py` — 4 tests: shape + diagonal == 1, three published
      Gram tiers (0.8851 / 0.6816 / 0.5931 within 1e-3), strict ordering,
      and the φ=0 collapse case (same-cluster overlaps go to 1.0,
      cross-cluster all hit the rung-1 product-state cosine 0.5931)
