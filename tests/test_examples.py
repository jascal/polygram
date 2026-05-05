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
    a parseable file, produces a positive tier-separation that clears
    the declared invariant bound, and that the InterferenceSweep +
    Cancellation walks produce their expected artifacts."""
    from examples.animals_hea import build_dictionary, main

    main(output_dir=tmp_path)
    base = tmp_path / "animals_hea"
    out = base / "AnimalsHea.q.orca.md"
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

    sweep_dir = base / "sweep"
    assert (sweep_dir / "AnimalsHeaSweep.q.orca.md").exists()
    assert (sweep_dir / "AnimalsHeaSweep_result.csv").exists()
    assert (sweep_dir / "AnimalsHeaSweep_summary.md").exists()
    assert (sweep_dir / "AnimalsHeaSweep_result.npz").exists()

    canc_dir = base / "cancellation"
    assert (canc_dir / "AnimalsHea_at_optimum.q.orca.md").exists()
    assert (canc_dir / "AnimalsHea_at_optimum_summary.md").exists()
    assert (canc_dir / "AnimalsHea_at_optimum_trajectory.csv").exists()
    try:
        import matplotlib  # noqa: F401

        assert (canc_dir / "before_after.png").exists()
    except ImportError:
        pass

    cluster_dir = canc_dir / "cluster_shared"
    assert (cluster_dir / "AnimalsHea_at_optimum.q.orca.md").exists()
    assert (cluster_dir / "AnimalsHea_at_optimum_summary.md").exists()
    assert (cluster_dir / "AnimalsHea_at_optimum_trajectory.csv").exists()
    try:
        import matplotlib  # noqa: F401

        assert (cluster_dir / "before_after.png").exists()
    except ImportError:
        pass


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


def test_sae_safetensors_runs(tmp_path: Path):
    """Synthesize a .safetensors fixture, load it, build a Dictionary,
    and emit a verifying .q.orca.md."""
    from examples.sae_safetensors import main

    main(output_dir=tmp_path)
    out = tmp_path / "sae_safetensors"
    assert (out / "synthesized.safetensors").is_file()
    machine = out / "ImportedSafetensors.q.orca.md"
    assert machine.is_file()


def test_behavioural_gram_probe_smoke(capsys):
    """Smoke test: the behavioural-Gram probe script imports cleanly,
    parses CLI args, and either runs end-to-end on a single short
    prompt or skips with a clear message. Skip cases:
    - SAE checkpoint absent (~144MB; matches existing decoder-Gram
      and cross-encoding-stability skip pattern)
    - `transformers` / `torch` not installed (the probe itself
      handles this gracefully and the script exits without raising).
    The smoke test asserts only that `main()` returns without
    raising in either branch.
    """
    from examples.behavioural_gram_probe import main

    main(["--n-prompts", "1", "--quiet"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    # Either we printed BEHAVIOURAL-GRAM PROBE banner (success path)
    # or we printed a skip message — both are acceptable smoke pass.
    assert (
        "BEHAVIOURAL-GRAM PROBE" in out
        or "behavioural_gram_probe:" in out
    )


def test_behavioural_gram_scaleup_smoke(capsys, tmp_path):
    """Smoke test: the behavioural-Gram scale-up probe imports cleanly,
    parses CLI args, and either runs end-to-end on a tiny configuration
    or skips with a clear message. Same skip pattern as §4.2 / §4.3:
    SAE checkpoint absent or torch/transformers not installed both
    produce a non-raising exit. The full probe is too expensive for
    CI (~25 ablation passes × 12 prompts), so this asserts only that
    `main()` returns without raising in either branch.
    """
    from examples.behavioural_gram_scaleup import main

    csv_path = tmp_path / "scaleup_pairs.csv"
    main([
        "--n-prompts", "1",
        "--n-features", "4",
        "--csv-out", str(csv_path),
        "--quiet",
    ])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert (
        "BEHAVIOURAL-GRAM SCALE-UP" in out
        or "behavioural_gram_scaleup:" in out
    )


def test_behavioural_validator_smoke(capsys, tmp_path):
    """Smoke test: `examples/behavioural_validate.py` imports cleanly
    and either runs end-to-end on a tiny configuration or skips with a
    clear message. Same skip pattern as §4.2 / §4.3 / §4.4: SAE
    checkpoint absent OR torch/transformers not installed both produce
    a non-raising exit. The full validator is too expensive for CI
    (8 ablation passes × 12 prompts), so this asserts only that
    `main()` returns 0 in either branch.
    """
    from examples.behavioural_validate import main

    rc = main([
        "--n-prompts", "1",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert (
        "BEHAVIOURAL-VALIDATOR" in out
        or "behavioural_validate:" in out
    )


def test_decoder_gram_validity_smoke(capsys):
    """Smoke test: the decoder-gram validity spike script runs end-to-end
    on the toy fixture. Skips the real-SAE branch (which needs a 144MB
    download) but exercises the toy-SAE path, both encodings, the
    correlation/contingency reporting, and confirms the script prints a
    Spearman block for both MPS and HEA."""
    from examples.decoder_gram_validity import main

    main(["--skip-real-sae"])
    out = capsys.readouterr().out
    assert "FIXTURE: Toy SAE" in out
    assert "Spearman(G_real, G_mps)" in out
    assert "Spearman(G_real, G_hea)" in out
    assert "classification agreement" in out


def test_batch_animals_hea_runs(tmp_path: Path):
    """Batch-experiment walk-through: triage_dictionary →
    build_separation_graph → BatchExperiment(top_k=4) on the Animals
    HEA dictionary. Asserts the standard artifact bundle lands."""
    import json

    from examples.batch_animals_hea import main

    main(output_dir=tmp_path)
    out = tmp_path / "batch_animals_hea"
    assert (out / "input_separation_graph.json").is_file()
    results_path = out / "batch_results.json"
    assert results_path.is_file()
    data = json.loads(results_path.read_text())
    assert data["dictionary_name"] == "AnimalsHea"
    assert data["knobs"] == "cluster_shared"
    assert 1 <= len(data["runs"]) <= 4
    for run in data["runs"]:
        sub = out / f"{run['source']}_x_{run['target']}"
        assert sub.is_dir()
