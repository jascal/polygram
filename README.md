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

Pre-alpha (v0 milestone). The 4-stage v0 scope (bootstrap → core dictionary →
experiment/sweep → animals example) is staged through OpenSpec changes — see
`openspec/changes/`.

## Layout

```
polygram/         — Python package
openspec/         — spec-driven change proposals + capability specs
tests/            — pytest suite
examples/         — Python script + notebook walking tour
```

## Quickstart

```python
import numpy as np

from polygram import Dictionary, Experiment, Feature, MPSRung1

dictionary = Dictionary(
    name="AnimalsInterference",
    features=[
        Feature("dog_poodle",   "dogs",  beta=-0.5),
        Feature("dog_beagle",   "dogs",  beta=-0.5),
        Feature("bird_hawk",    "birds", beta= 0.5),
        Feature("bird_sparrow", "birds", beta= 0.5),
    ],
    hierarchy={"dogs": ["dog_poodle", "dog_beagle"],
               "birds": ["bird_hawk", "bird_sparrow"]},
    encoding=MPSRung1(bond_dim=2, phase_knobs=True),
)

experiment = Experiment(
    name=dictionary.name,
    dictionary=dictionary,
    target_pair=("dog_poodle", "bird_hawk"),
    sweep={"bird_hawk.phi": np.linspace(0.0, np.pi, 40)},
    measures=["overlap", "gram_matrix", "schmidt_rank"],
    assertions=["hierarchical_ordering_preserved"],
)

experiment.materialize("examples/output/")   # emits a verifiable .q.orca.md
result = experiment.run()                    # analytic Gram per sweep point
result.to_csv("examples/output/result.csv")
```

See `examples/animals_interference.py` and the matching
`examples/animals_interference.ipynb` notebook for the full walking tour.

### Plots

`result.plot(path)` saves a default figure: 1D sweep → line plot of
target-pair overlap with sibling and cross-cluster tier baselines; 2D
sweep → heatmap. Requires the `[plot]` extra (`pip install polygram[plot]`).

**1D sweep** — `bird_hawk.phi` from 0 to π. Single-φ steering on this
geometry leaves the cross-cluster overlap above the matched-φ baseline
of `cos(0.5)⁴ ≈ 0.5931` and below the sibling tier; it never destroys.

![1D sweep](docs/img/animals_overlap_1d.png)

**2D sweep** — `(dog_poodle.phi, bird_hawk.phi)` 24×24 grid. The
antidiagonal channel (one feature's φ near 0, the other's near π) is
where the cross-cluster overlap drops; the diagonal channel keeps
overlap high. This is the asymmetric-φ direction the future
`Cancellation` primitive will exploit.

![2D heatmap](docs/img/animals_overlap_2d.png)

### CLI

The `polygram` console script runs an example or experiment module
that exposes `main(output_dir=...)`:

```bash
polygram run examples/animals_interference.py --output-dir results/
polygram --version
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
