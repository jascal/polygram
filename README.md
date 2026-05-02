# Polygram

**Quantum Interference Laboratory for Polysemantic Feature Dictionaries**

Polygram is a researcher-friendly Python frontend that emits verifiable
[Q-Orca](https://github.com/jascal/q-orca-lang) `.q.orca.md` machines for
mechanistic-interpretability experiments on hierarchical polysemantic feature
dictionaries.

It builds on Q-Orca's rung-1 MPS encoding with safe `Rz` phase knobs (q-orca
PR #51) to enable phase-interference sweeps, destructive cancellation studies,
entanglement probes, and hybrid measurement-feedback steering on
SAE-style dictionaries.

## Status

Pre-alpha. v0 scope is being staged through OpenSpec changes — see
`openspec/changes/`.

## Layout

```
polygram/         — Python package
openspec/         — spec-driven change proposals + capability specs
tests/            — pytest suite
examples/         — Python + notebook examples (added in later stages)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Relationship to Q-Orca

Polygram does **not** define a new file format. It generates standard
Q-Orca `.q.orca.md` files (matching the style of
`examples/larql-animals-interference.q.orca.md` from q-orca-lang) and uses
Q-Orca for verification, simulation, and the analytic Gram helper
`compute_concept_gram_mps`.

## License

Apache-2.0.
