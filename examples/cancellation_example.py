"""Cancellation example — combined SAE → InterferenceSweep → Cancellation walk.

Loads the bundled 4-cluster toy SAE, picks 4 features (mammals/dog_*,
birds/hawk_*), then runs both Polygram experiment primitives:

1. `InterferenceSweep` — landscape exploration over `hawk_red.phi`
2. `Cancellation` — goal-directed φ search over the
   `(dog_poodle, hawk_red)` target pair, preserving cluster tiers

All artifacts land in `<output_dir>/cancellation_example/`:

- `<dictionary>.q.orca.md`           — Polygram-rendered Q-Orca machine
- `<dictionary>_summary.md`           — InterferenceSweep config + tier rollup
- `<dictionary>_result.npz` / `.csv`  — sweep arrays + flat overlap table
- `<dictionary>_overlap.png`          — sweep overlap line plot
- `<optimum>.q.orca.md`               — optimized Dictionary at φ_optimum
- `<optimum>_summary.md`              — Cancellation summary (before/after)
- `<optimum>_trajectory.csv`          — every (φ_a, φ_b, overlap, feasible)
- `<optimum>_grid.png`                — Cancellation grid heatmap
"""

from pathlib import Path

import numpy as np

from polygram import Cancellation, Experiment, from_sae_lens, load_toy_sae

FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "toy_sae.json"
)


def build_dictionary_and_report():
    records = load_toy_sae(FIXTURE)
    return from_sae_lens(records, [0, 1, 4, 5], name="ToySAEAnimals4")


def main(
    output_dir: str | Path = "examples/output",
    n_points: int | None = None,
) -> None:
    out = Path(output_dir) / "cancellation_example"
    out.mkdir(parents=True, exist_ok=True)

    dictionary, report = build_dictionary_and_report()
    sweep_pts = n_points if n_points is not None else 30
    grid_res = max(8, sweep_pts // 2)

    # 1) InterferenceSweep over hawk_red.phi
    experiment = Experiment(
        name=dictionary.name,
        dictionary=dictionary,
        target_pair=("dog_poodle", "hawk_red"),
        sweep={"hawk_red.phi": np.linspace(0.0, np.pi, sweep_pts)},
        measures=["overlap", "gram_matrix", "schmidt_rank"],
        assertions=["hierarchical_ordering_preserved"],
    )
    experiment.materialize(out)
    sweep_result = experiment.run()
    sweep_result.save(out / f"{experiment.name}_result.npz")
    sweep_result.to_csv(out / f"{experiment.name}_result.csv")
    sweep_result.write_summary(out / f"{experiment.name}_summary.md")
    try:
        sweep_result.plot(out / f"{experiment.name}_overlap.png")
        plot_msg = f"sweep plot: {experiment.name}_overlap.png"
    except ImportError as exc:
        plot_msg = f"sweep plot skipped: {exc}"

    # 2) Cancellation — goal-directed search over the target pair
    cancellation = Cancellation(
        dictionary=dictionary,
        target_pair=("dog_poodle", "hawk_red"),
        tolerance=0.05,
        preserve_tiers=True,
        optimize={"method": "grid", "max_steps": grid_res},
    )
    canc_result = cancellation.run()
    canc_result.materialize(out)
    try:
        canc_result.plot(
            out / f"{canc_result.dictionary_at_optimum.name}_grid.png"
        )
        canc_msg = "cancellation plot written"
    except ImportError as exc:
        canc_msg = f"cancellation plot skipped: {exc}"

    print(f"selection: {[f.name for f in dictionary.features]}")
    print(f"hierarchy: {dictionary.hierarchy}")
    print(f"cluster method: {report.cluster_method}")
    print(
        f"β-variance explained: {report.beta_variance_explained:.4f} | "
        f"tier_preservation: {report.tier_preservation}"
    )
    print(f"{plot_msg}")
    print(
        f"sweep overlap range: [{sweep_result.overlaps.min():.4f}, "
        f"{sweep_result.overlaps.max():.4f}]"
    )
    print(
        f"cancellation: before={canc_result.before_overlap:.4f} "
        f"→ after={canc_result.after_overlap:.4f} "
        f"(tolerance_met={canc_result.tolerance_met}, "
        f"feasible={canc_result.feasible_count}/"
        f"{canc_result.trajectory.shape[0]})"
    )
    print(
        f"optimized φ: {dict((k, round(v, 4)) for k, v in canc_result.optimized_phis.items())}"
    )
    print(canc_msg)


if __name__ == "__main__":
    main()
