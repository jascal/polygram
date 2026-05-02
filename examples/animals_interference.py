"""Animals interference example — Polygram v0 walking tour.

Defines a 4-feature, 2-cluster Animals dictionary (dog_poodle,
dog_beagle, bird_hawk, bird_sparrow) and runs a phase sweep over
`bird_hawk.phi` while watching the (dog_poodle, bird_hawk)
cross-cluster overlap.

The hierarchy ordering should hold throughout: in-cluster overlaps
(0.8851 between dog siblings, 0.8851 between bird siblings) stay above
the cross-cluster target overlap. Single-φ sweep on this geometry does
*not* drive destructive interference at the endpoint — overlap actually
rises slightly with φ_bird, peaking near π. Driving the cross-cluster
overlap to zero needs an antisymmetric φ on both sides; that is the
`Cancellation` primitive's territory (roadmap), not v0.

Outputs land in `examples/output/`:
- AnimalsInterference.q.orca.md       (verifiable Q-Orca artifact)
- run_AnimalsInterference.py          (self-contained runner)
- AnimalsInterference_result.npz      (raw arrays)
- AnimalsInterference_result.csv      (flat overlaps + assertions)
"""

from pathlib import Path

import numpy as np

from polygram import Dictionary, Experiment, Feature, MPSRung1


def build_dictionary() -> Dictionary:
    return Dictionary(
        name="AnimalsInterference",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.5),
            Feature("dog_beagle", "dogs", beta=-0.5),
            Feature("bird_hawk", "birds", beta=0.5),
            Feature("bird_sparrow", "birds", beta=0.5),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=MPSRung1(bond_dim=2, phase_knobs=True),
    )


def build_experiment(dictionary: Dictionary, n_points: int = 40) -> Experiment:
    return Experiment(
        name=dictionary.name,
        dictionary=dictionary,
        target_pair=("dog_poodle", "bird_hawk"),
        sweep={"bird_hawk.phi": np.linspace(0.0, np.pi, n_points)},
        measures=["overlap", "gram_matrix", "schmidt_rank"],
        assertions=["hierarchical_ordering_preserved"],
    )


def main(output_dir: str | Path = "examples/output") -> None:
    out = Path(output_dir)
    experiment = build_experiment(build_dictionary())
    experiment.materialize(out)
    result = experiment.run()
    result.save(out / f"{experiment.name}_result.npz")
    result.to_csv(out / f"{experiment.name}_result.csv")

    phis = result.sweep_axes["bird_hawk.phi"]
    overlaps = result.overlaps
    print(f"swept {len(phis)} points in [{phis[0]:.3f}, {phis[-1]:.3f}]")
    print(f"overlap range: [{overlaps.min():.4f}, {overlaps.max():.4f}]")
    print(f"endpoint overlap: {overlaps[-1]:.4f}")
    for name, ok in result.assertion_pass.items():
        print(f"  {name}: {'PASS' if bool(ok[-1]) else 'FAIL'}")


if __name__ == "__main__":
    main()
