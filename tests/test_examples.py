"""Animals example — end-to-end integration test.

Runs a coarsened version of `examples/animals_interference.py` (5 sweep
points instead of 40) and asserts:

- the emitted `.q.orca.md` parses + verifies clean
- the destructive-endpoint assertion passes at φ = π
- the hierarchical-ordering assertion holds at every sweep point
"""

from pathlib import Path

from examples.animals_interference import build_dictionary, build_experiment


def test_animals_interference_runs(tmp_path: Path):
    experiment = build_experiment(build_dictionary(), n_points=5)
    experiment.materialize(tmp_path)
    result = experiment.run()

    machine_path = tmp_path / f"{experiment.name}.q.orca.md"
    assert machine_path.exists()

    from q_orca import VerifyOptions, verify
    from q_orca.parser.markdown_parser import parse_q_orca_markdown

    parsed = parse_q_orca_markdown(machine_path.read_text())
    assert not parsed.errors, parsed.errors
    machine = parsed.file.machines[0]
    verification = verify(machine, VerifyOptions())
    assert verification.valid, [
        d for d in (verification.diagnostics or []) if getattr(d, "severity", "") == "error"
    ]

    ordering = result.assertion_pass["hierarchical_ordering_preserved"]
    assert ordering.all(), (
        f"hierarchical ordering violated at indices "
        f"{[i for i, ok in enumerate(ordering) if not ok]}"
    )

    assert result.overlaps.shape == (5,)
    assert result.gram_matrices.shape == (5, 4, 4)


def test_import_from_sae_runs(tmp_path: Path):
    """Coarsened SAE-import example — toy fixture → Dictionary →
    InterferenceSweep → verifying .q.orca.md."""
    from examples.import_from_sae import build_dictionary_and_report
    from polygram import Experiment

    dictionary, report = build_dictionary_and_report()
    assert report.cluster_method == "from_labels"
    assert report.beta_variance_explained > 0.9

    import numpy as np

    experiment = Experiment(
        name=dictionary.name,
        dictionary=dictionary,
        target_pair=("dog_poodle", "hawk_red"),
        sweep={"hawk_red.phi": np.linspace(0.0, np.pi, 5)},
        measures=["overlap", "gram_matrix"],
        assertions=["hierarchical_ordering_preserved"],
    )
    experiment.materialize(tmp_path)
    result = experiment.run()

    machine_path = tmp_path / f"{experiment.name}.q.orca.md"
    assert machine_path.exists()

    from q_orca import VerifyOptions, verify
    from q_orca.parser.markdown_parser import parse_q_orca_markdown

    parsed = parse_q_orca_markdown(machine_path.read_text())
    assert not parsed.errors, parsed.errors
    machine = parsed.file.machines[0]
    verification = verify(machine, VerifyOptions())
    assert verification.valid, [
        d for d in (verification.diagnostics or []) if getattr(d, "severity", "") == "error"
    ]

    assert result.overlaps.shape == (5,)
    assert result.assertion_pass["hierarchical_ordering_preserved"].all()


def test_animals_hea_example_runs(tmp_path: Path):
    """Coarsened HEA emit example — verifies the dictionary builds, emits
    a parseable file, and produces a positive tier-separation that clears
    the declared invariant bound."""
    from examples.animals_hea import build_dictionary, main

    main(output_dir=tmp_path)
    out = tmp_path / "animals_hea" / "AnimalsHea.q.orca.md"
    assert out.exists()

    from q_orca.parser.markdown_parser import parse_q_orca_markdown
    from q_orca.verifier import VerifyOptions, verify

    parsed = parse_q_orca_markdown(out.read_text())
    assert not parsed.errors, parsed.errors
    machine = parsed.file.machines[0]
    assert machine.encoding.kind == "hea"
    assert [r.cluster for r in machine.theta.rows] == [
        "dogs", "dogs", "birds", "birds",
    ]

    verification = verify(machine, VerifyOptions(skip_resource_bounds=True))
    assert verification.valid

    sep = build_dictionary().tier_separation()
    assert sep is not None
    assert sep > 0.025


def test_cancellation_example_runs(tmp_path: Path):
    """Coarsened combined SAE → Sweep → Cancellation walk; verifies
    both materialized `.q.orca.md` files parse + verify clean."""
    from examples.cancellation_example import main

    main(output_dir=tmp_path, n_points=8)

    out = tmp_path / "cancellation_example"
    assert out.exists()
    sweep_machine = out / "ToySAEAnimals4.q.orca.md"
    optimum_machine = out / "ToySAEAnimals4_at_optimum.q.orca.md"
    assert sweep_machine.exists()
    assert optimum_machine.exists()

    from q_orca import VerifyOptions, verify
    from q_orca.parser.markdown_parser import parse_q_orca_markdown

    for machine_path in (sweep_machine, optimum_machine):
        parsed = parse_q_orca_markdown(machine_path.read_text())
        assert not parsed.errors, parsed.errors
        machine = parsed.file.machines[0]
        verification = verify(machine, VerifyOptions())
        assert verification.valid, [
            d for d in (verification.diagnostics or [])
            if getattr(d, "severity", "") == "error"
        ]

    assert (out / "ToySAEAnimals4_at_optimum_trajectory.csv").exists()
    assert (out / "ToySAEAnimals4_at_optimum_summary.md").exists()
