"""Animals interference example — Polygram v0 walking tour.

Defines a 4-feature, 2-cluster Animals dictionary (dog_poodle,
dog_beagle, bird_hawk, bird_sparrow) and runs a phase sweep over
`bird_hawk.phi` (single-axis mode, default) — or a 2D grid over
`(dog_poodle.phi, bird_hawk.phi)` (`mode="two_axis"`) — while watching
the (dog_poodle, bird_hawk) cross-cluster overlap.

The hierarchy ordering should hold throughout: in-cluster overlaps
(0.8851 between dog siblings, 0.8851 between bird siblings) stay above
the cross-cluster target overlap. Single-φ sweep on this geometry does
*not* drive destructive interference at the endpoint — overlap actually
rises slightly with φ_bird, peaking near π. Driving the cross-cluster
overlap to zero needs antisymmetric φ on both sides; the 2D heatmap
makes the asymmetry visible. (Full destructive cancellation is the
`Cancellation` primitive's territory, on the v0+ roadmap.)

Outputs land in `<output_dir>/animals_interference/`:
- AnimalsInterference.q.orca.md       (verifiable Q-Orca artifact)
- AnimalsInterference_summary.md      (config + tier rollup + asserts)
- run_AnimalsInterference.py          (self-contained runner)
- AnimalsInterference_result.npz      (raw arrays)
- AnimalsInterference_result.csv      (flat overlaps + tiers + asserts)
- AnimalsInterference_overlap.png     (line plot or heatmap)
"""

from pathlib import Path
from typing import Literal

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


def build_experiment(
    dictionary: Dictionary,
    n_points: int = 40,
    mode: Literal["single", "two_axis"] = "single",
) -> Experiment:
    if mode == "single":
        sweep = {"bird_hawk.phi": np.linspace(0.0, np.pi, n_points)}
    elif mode == "two_axis":
        sweep = {
            "dog_poodle.phi": np.linspace(0.0, np.pi, n_points),
            "bird_hawk.phi": np.linspace(0.0, np.pi, n_points),
        }
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'single' or 'two_axis'")
    return Experiment(
        name=dictionary.name,
        dictionary=dictionary,
        target_pair=("dog_poodle", "bird_hawk"),
        sweep=sweep,
        measures=["overlap", "gram_matrix", "schmidt_rank"],
        assertions=["hierarchical_ordering_preserved"],
    )


def main(
    output_dir: str | Path = "examples/output",
    n_points: int | None = None,
    mode: Literal["single", "two_axis"] = "single",
) -> None:
    out = Path(output_dir) / "animals_interference"
    out.mkdir(parents=True, exist_ok=True)

    pts = n_points if n_points is not None else (12 if mode == "two_axis" else 40)
    experiment = build_experiment(build_dictionary(), n_points=pts, mode=mode)
    experiment.materialize(out)

    result = experiment.run()
    result.save(out / f"{experiment.name}_result.npz")
    result.to_csv(out / f"{experiment.name}_result.csv")
    result.write_summary(out / f"{experiment.name}_summary.md")

    try:
        result.plot(out / f"{experiment.name}_overlap.png")
        plot_msg = f"plot:    {experiment.name}_overlap.png"
    except ImportError as exc:
        plot_msg = f"plot skipped: {exc}"

    overlaps = result.overlaps
    print(f"mode:    {mode}")
    print(f"sweep:   {dict((k, len(v)) for k, v in result.sweep_axes.items())}")
    print(f"overlap range: [{overlaps.min():.4f}, {overlaps.max():.4f}]")
    print(f"sibling tier mean: {float(np.nanmean(result.tier_stats['sibling'])):.4f}")
    print(f"cross   tier mean: {float(np.nanmean(result.tier_stats['cross_cluster'])):.4f}")
    for name, ok in result.assertion_pass.items():
        print(f"  {name}: pass-rate {ok.mean():.0%}")
    print(plot_msg)


if __name__ == "__main__":
    main()
