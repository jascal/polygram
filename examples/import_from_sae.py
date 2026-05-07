"""Import an SAE dictionary into Polygram and run an InterferenceSweep.

This example uses the bundled `tests/fixtures/toy_sae.json` — a 16-feature,
4-cluster, 8-dim deterministic toy. It illustrates the selection-first
import flow: pick a small subset of features (≤8), get a `Dictionary`
back, run a phase sweep, save artifacts.

## Swapping in a real SAE

Real SAEs (Anthropic's Claude SAEs, SAE-Lens .pt files, HF safetensors)
ship 16k–1M features. Polygram caps a Dictionary at 8 features per
the rung-1 MPS encoding, so you must select a small subset by feature
id. A future `polygram.sae_import.load_sae_lens(...)` loader will read
SAE-Lens / safetensors directly; for now, hand-roll a `dict[int,
SAEFeatureRecord]` from your own loader. Pseudocode:

    # NOT shipped in v0 — sketch only:
    # import safetensors.torch as sft
    # sd = sft.load_file("path/to/sae.safetensors")
    # decoder = sd["W_dec"].numpy()  # shape (n_features, d_model)
    # records = {
    #     i: SAEFeatureRecord(
    #         feature_id=i,
    #         name=metadata.get(i, {}).get("label", f"feat_{i}"),
    #         projection=decoder[i],
    #         label=metadata.get(i, {}).get("label"),
    #     )
    #     for i in range(decoder.shape[0])
    # }
    # d, report = from_sae_lens(records, [42, 117, 308, 421])

When the projection vectors are high-dimensional (e.g. 4096), the
clustering still runs in pure numpy without further deps.
"""

from pathlib import Path

import numpy as np

from polygram import (
    Experiment,
    from_sae_lens,
    load_toy_sae,
)

FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "toy_sae.json"
)


def build_dictionary_and_report():
    records = load_toy_sae(FIXTURE)
    selection = [0, 1, 4, 5]  # mammals/dog_{poodle, beagle} + birds/hawk_{red, cooper}
    # The toy fixture's projections are tuned for the all-γ=0 path;
    # ``assign_gamma=True`` (polygram's new default for real SAEs) gives
    # this toy dictionary nonzero γs that violate the
    # hierarchical_ordering_preserved invariant on the demo sweep. Pin
    # the legacy behaviour explicitly here so the example's downstream
    # assertions remain stable.
    return from_sae_lens(
        records, selection, name="ToySAEAnimals", assign_gamma=False
    )


def main(
    output_dir: str | Path = "examples/output",
    n_points: int | None = None,
) -> None:
    out = Path(output_dir) / "import_from_sae"
    out.mkdir(parents=True, exist_ok=True)

    dictionary, report = build_dictionary_and_report()
    pts = n_points if n_points is not None else 30
    experiment = Experiment(
        name=dictionary.name,
        dictionary=dictionary,
        target_pair=("dog_poodle", "hawk_red"),
        sweep={"hawk_red.phi": np.linspace(0.0, np.pi, pts)},
        measures=["overlap", "gram_matrix", "schmidt_rank"],
        assertions=["hierarchical_ordering_preserved"],
    )
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

    print(f"selection: {[f.name for f in dictionary.features]}")
    print(f"hierarchy: {dictionary.hierarchy}")
    print(f"cluster method: {report.cluster_method}")
    print(f"β-variance explained by clustering: "
          f"{report.beta_variance_explained:.4f}")
    if report.warnings:
        print(f"warnings: {report.warnings}")
    print(f"sweep: {dict((k, len(v)) for k, v in result.sweep_axes.items())}")
    print(f"overlap range: [{result.overlaps.min():.4f}, "
          f"{result.overlaps.max():.4f}]")
    for name, ok in result.assertion_pass.items():
        print(f"  {name}: pass-rate {ok.mean():.0%}")
    print(plot_msg)


if __name__ == "__main__":
    main()
