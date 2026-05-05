## ADDED Requirements

### Requirement: Rung3 is a 5-qubit encoding parallel to MPSRung1 and HEA_Rung2

`polygram.encoding.Rung3` SHALL be a frozen dataclass parallel to `MPSRung1` and `HEA_Rung2`, exposing the following per-feature fields:

- `alpha: float` — same as MPSRung1.
- `beta: float` — same as MPSRung1.
- `gamma: float` — same as MPSRung1.
- `phi: float` — same as MPSRung1's primary phase knob.
- `theta_amp: float = math.pi / 4` — amplitude knob on the
  auxiliary qubit pair (qubits 3–4). Default chosen so the
  amplitude branch reduces to identity for the joint state
  (see Requirement: Rung3 defaults reduce to MPSRung1
  behaviour).
- `psi_aux: float = 0.0` — auxiliary phase knob.

The class SHALL expose `compute_concept_gram(other: "Rung3") -> float` returning the analytic squared overlap `|⟨ψ_self | ψ_other⟩|²`. The implementation MUST be analytic (closed-form) — no simulator round-trip is permitted in the gram path. Implementations MAY reuse `MPSRung1.compute_concept_gram` on the (α, β, γ, φ) subset for qubits 0–2 and add the amplitude branch contribution in closed form.

### Requirement: Rung3 defaults reduce to MPSRung1 behaviour

When `theta_amp == math.pi / 4` and `psi_aux == 0.0` for both `Rung3` instances, `Rung3.compute_concept_gram(other)` SHALL return a value equal (within float64 tolerance, 1e-12 absolute) to `MPSRung1(alpha, beta, gamma, phi).compute_concept_gram(MPSRung1(alpha_b, beta_b, gamma_b, phi_b))` evaluated on the same (α, β, γ, φ) subset.

This property guarantees that a baseline Rung3 dictionary with default knobs behaves identically to its MPSRung1 counterpart for cancellation_gap ranking, dictionary materialization, and downstream validation.

### Requirement: Dictionary dispatches on encoding="rung3"

`polygram.Dictionary` SHALL accept `encoding="rung3"` as a value parallel to the existing `"mps"` and `"hea"`. Dispatch MUST construct `Rung3` per-feature objects from the dictionary's (α, β, γ, φ) plus default `theta_amp` and `psi_aux`. Existing `"mps"` and `"hea"` paths SHALL be unchanged.

### Requirement: Rung3 is constructed from MPSRung1 deterministically

`Rung3.from_mps(mps: MPSRung1, theta_amp: float = math.pi / 4, psi_aux: float = 0.0) -> Rung3` SHALL construct a Rung3 instance from an MPSRung1 with the same (α, β, γ, φ) and the supplied amplitude / aux knobs. The default-knob construction `Rung3.from_mps(mps)` SHALL produce a Rung3 whose `compute_concept_gram` matches `mps.compute_concept_gram` to 1e-12 (per Requirement: Rung3 defaults reduce to MPSRung1 behaviour).

### Requirement: Rung3 surface is torch-free and analytic

`polygram.encoding.Rung3` SHALL NOT import torch or transformers. The `compute_concept_gram` path SHALL be implementable in numpy + math only (the existing `polygram` baseline stack). No new optional extra is introduced.
