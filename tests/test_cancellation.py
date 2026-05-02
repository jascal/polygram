"""Cancellation primitive — grid + scipy backends, materialize, plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polygram import Cancellation, Dictionary, Feature, MPSRung1


def _animals(
    dog_phi: float = 0.0, hawk_phi: float = 0.0
) -> Dictionary:
    return Dictionary(
        name="AnimalsCanc",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.5, phi=dog_phi),
            Feature("dog_beagle", "dogs", beta=-0.5),
            Feature("bird_hawk", "birds", beta=0.5, phi=hawk_phi),
            Feature("bird_sparrow", "birds", beta=0.5),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=MPSRung1(bond_dim=2, phase_knobs=True),
    )


def test_grid_finds_best_feasible_point():
    canc = Cancellation(
        dictionary=_animals(dog_phi=np.pi / 2, hawk_phi=0.0),
        target_pair=("dog_poodle", "bird_hawk"),
        tolerance=0.05,
        preserve_tiers=True,
        optimize={"method": "grid", "max_steps": 30},
    )
    result = canc.run()
    assert result.after_overlap <= result.before_overlap + 1e-9
    assert result.feasible_count > 0
    a = result.dictionary_at_optimum.feature_index("dog_poodle")
    b = result.dictionary_at_optimum.feature_index("bird_hawk")
    g = result.dictionary_at_optimum.gram()
    assert abs(float(np.abs(g[a, b]) ** 2) - result.after_overlap) < 1e-6


def test_grid_preserve_tiers_false_at_least_as_good():
    """Without the feasibility filter, the optimum can only equal or
    beat the constrained run."""
    common = dict(
        dictionary=_animals(dog_phi=np.pi / 3, hawk_phi=np.pi / 5),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 20},
    )
    constrained = Cancellation(preserve_tiers=True, **common).run()
    unconstrained = Cancellation(preserve_tiers=False, **common).run()
    assert unconstrained.after_overlap <= constrained.after_overlap + 1e-9


def test_optimize_all_not_yet_implemented():
    with pytest.raises(NotImplementedError, match="optimize_all"):
        Cancellation(
            dictionary=_animals(),
            target_pair=("dog_poodle", "bird_hawk"),
            optimize_all=True,
        )


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="bogus"):
        Cancellation(
            dictionary=_animals(),
            target_pair=("dog_poodle", "bird_hawk"),
            optimize={"method": "bogus"},
        )


def test_target_pair_must_reference_features():
    with pytest.raises(ValueError, match="ghost"):
        Cancellation(
            dictionary=_animals(),
            target_pair=("dog_poodle", "ghost"),
        )


def test_trajectory_shape_matches_grid_resolution():
    canc = Cancellation(
        dictionary=_animals(),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 20},
    )
    result = canc.run()
    assert result.trajectory.shape == (400, 3)
    assert result.feasible_mask.shape == (400,)


def test_materialize_writes_optimized_qorca(tmp_path: Path):
    from q_orca import verify
    from q_orca.parser.markdown_parser import parse_q_orca_markdown

    canc = Cancellation(
        dictionary=_animals(dog_phi=np.pi / 4),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 12},
    )
    result = canc.run()
    artifacts = result.materialize(tmp_path)

    assert artifacts["machine"].exists()
    assert artifacts["summary"].exists()
    assert artifacts["trajectory"].exists()

    parsed = parse_q_orca_markdown(artifacts["machine"].read_text())
    assert not parsed.errors, parsed.errors
    machine = parsed.file.machines[0]
    report = verify(machine)
    assert report.valid, report

    csv_text = artifacts["trajectory"].read_text().splitlines()
    assert csv_text[0] == "phi_a,phi_b,overlap,feasible"
    assert len(csv_text) == 1 + 12 * 12


def test_plot_grid_writes_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    canc = Cancellation(
        dictionary=_animals(),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 10},
    )
    result = canc.run()
    out = result.plot(tmp_path / "p.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_scipy_backend_or_skip():
    pytest.importorskip("scipy")
    canc = Cancellation(
        dictionary=_animals(dog_phi=np.pi / 4),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "scipy", "max_steps": 5},
        preserve_tiers=False,
    )
    result = canc.run()
    assert result.method == "scipy"
    assert result.trajectory.shape[1] == 3
    assert result.trajectory.shape[0] >= 1
    assert result.after_overlap <= result.before_overlap + 1e-6


def test_summary_contains_optimum(tmp_path: Path):
    canc = Cancellation(
        dictionary=_animals(dog_phi=0.5),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 10},
    )
    result = canc.run()
    artifacts = result.materialize(tmp_path)
    text = artifacts["summary"].read_text()
    assert "Optimum" in text
    assert "before:" in text
    assert "after:" in text
    assert "dog_poodle" in text
    assert "bird_hawk" in text
