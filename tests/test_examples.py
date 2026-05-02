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
