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
    assert csv_text[0] == "dog_poodle.phi,bird_hawk.phi,overlap,feasible"
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


def test_structural_floor_matches_grid_minimum():
    canc = Cancellation(
        dictionary=_animals(),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 50},
        preserve_tiers=False,
    )
    floor = canc.structural_floor()
    result = canc.run()
    assert abs(floor - float(result.trajectory[:, 2].min())) < 1e-9
    assert abs(floor - result.structural_floor) < 1e-12


def test_efficiency_one_when_floor_reached():
    canc = Cancellation(
        dictionary=_animals(dog_phi=np.pi / 2),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 30},
        preserve_tiers=False,
    )
    result = canc.run()
    assert result.cancellation_efficiency is not None
    assert abs(result.cancellation_efficiency - 1.0) < 1e-9
    assert abs(result.after_overlap - result.structural_floor) < 1e-9


def test_efficiency_none_when_already_at_floor():
    canc = Cancellation(
        dictionary=_animals(),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 10},
    )
    result = canc.run()
    assert result.cancellation_efficiency is None
    assert abs(result.before_overlap - result.structural_floor) < 1e-9


def test_summary_includes_floor_section(tmp_path: Path):
    canc = Cancellation(
        dictionary=_animals(dog_phi=0.5),
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 10},
    )
    result = canc.run()
    artifacts = result.materialize(tmp_path)
    text = artifacts["summary"].read_text()
    assert "Structural floor" in text
    assert "structural_floor:" in text
    assert "cancellation_efficiency:" in text
    assert "interpretation:" in text


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


def _hea_pair() -> Dictionary:
    """Two-cluster HEA fixture for knob-list tests."""
    from polygram import HEA_Rung2

    return Dictionary(
        name="HeaPair",
        features=[
            Feature("a", "s1", beta=0.10, alpha=0.05, gamma=0.02),
            Feature("b", "s1", beta=0.11, alpha=0.04, gamma=0.03),
            Feature("c", "s2", beta=1.20, alpha=1.10, gamma=1.00),
        ],
        hierarchy={"s1": ["a", "b"], "s2": ["c"]},
        encoding=HEA_Rung2(depth=2),
    )


class TestKnobsList:
    def test_default_knobs_resolve_to_two_phi(self):
        canc = Cancellation(
            dictionary=_animals(),
            target_pair=("dog_poodle", "bird_hawk"),
            optimize={"method": "grid", "max_steps": 5},
        )
        assert canc.knobs == ["dog_poodle.phi", "bird_hawk.phi"]
        result = canc.run()
        # 2 knobs → trajectory has 2 + 1 = 3 cols and `5*5 = 25` rows.
        assert result.trajectory.shape == (25, 3)
        assert set(result.optimized_knobs.keys()) == {
            "dog_poodle.phi", "bird_hawk.phi"
        }

    def test_explicit_hea_theta_knobs_run(self):
        canc = Cancellation(
            dictionary=_hea_pair(),
            target_pair=("a", "c"),
            knobs=["a.theta[0,0,1]", "c.theta[1,0,1]"],
            optimize={"method": "grid", "max_steps": 4},
        )
        result = canc.run()
        assert result.trajectory.shape == (16, 3)
        # Theta knob bounds are [-π, π]; verify values stay in range.
        assert float(result.trajectory[:, 0].min()) >= -np.pi - 1e-9
        assert float(result.trajectory[:, 0].max()) <= np.pi + 1e-9

    def test_grid_rejects_more_than_four_knobs(self):
        d = _hea_pair()
        with pytest.raises(ValueError, match="at most 4 knobs"):
            Cancellation(
                dictionary=d,
                target_pair=("a", "c"),
                knobs=[
                    "a.theta[0,0,0]", "a.theta[0,0,1]", "a.theta[0,0,2]",
                    "a.theta[1,0,0]", "a.theta[1,0,1]",
                ],
                optimize={"method": "grid", "max_steps": 5},
            )

    def test_theta_knob_rejected_on_mps(self):
        with pytest.raises(ValueError, match="HEA-only"):
            Cancellation(
                dictionary=_animals(),
                target_pair=("dog_poodle", "bird_hawk"),
                knobs=["dog_poodle.theta[0,0,1]"],
            )

    def test_unknown_knob_feature_rejected(self):
        with pytest.raises(ValueError, match="not declared"):
            Cancellation(
                dictionary=_animals(),
                target_pair=("dog_poodle", "bird_hawk"),
                knobs=["nope.phi", "bird_hawk.phi"],
            )


class TestStructuralFloorContract:
    def test_canonical_mps_floor_is_float(self):
        canc = Cancellation(
            dictionary=_animals(),
            target_pair=("dog_poodle", "bird_hawk"),
        )
        floor = canc.structural_floor()
        assert isinstance(floor, float)

    def test_hea_default_2phi_raises(self):
        canc = Cancellation(
            dictionary=_hea_pair(),
            target_pair=("a", "c"),
            optimize={"method": "grid", "max_steps": 4},
        )
        assert canc.knobs == ["a.phi", "c.phi"]
        with pytest.raises(NotImplementedError, match="HEA"):
            canc.structural_floor()

    def test_mps_with_non_canonical_knobs_raises(self):
        canc = Cancellation(
            dictionary=_animals(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dog_poodle.phi"],
        )
        with pytest.raises(NotImplementedError, match="canonical 2-φ"):
            canc.structural_floor()

    def test_run_on_hea_carries_nan_floor_and_none_efficiency(self):
        import math

        canc = Cancellation(
            dictionary=_hea_pair(),
            target_pair=("a", "c"),
            optimize={"method": "grid", "max_steps": 4},
        )
        result = canc.run()
        assert math.isnan(result.structural_floor)
        assert result.cancellation_efficiency is None

    def test_summary_marks_undefined_floor_on_hea(self, tmp_path: Path):
        canc = Cancellation(
            dictionary=_hea_pair(),
            target_pair=("a", "c"),
            optimize={"method": "grid", "max_steps": 4},
        )
        result = canc.run()
        artifacts = result.materialize(tmp_path)
        text = artifacts["summary"].read_text()
        assert "undefined for this configuration" in text
        assert "## Caveat" in text
        assert "best value found" in text
        # 2-φ HEA has no θ knob → tier-invariant addendum should not fire.
        assert "concept_gram_tier_separation" not in text

    def test_summary_caveat_flags_theta_cluster_hazard(
        self, tmp_path: Path
    ):
        canc = Cancellation(
            dictionary=_hea_pair(),
            target_pair=("a", "c"),
            knobs=["a.theta[0,0,0]", "c.theta[0,0,0]"],
            optimize={"method": "grid", "max_steps": 4},
        )
        result = canc.run()
        artifacts = result.materialize(tmp_path)
        text = artifacts["summary"].read_text()
        assert "## Caveat" in text
        assert "concept_gram_tier_separation" in text

    def test_summary_omits_caveat_on_canonical_mps(self, tmp_path: Path):
        canc = Cancellation(
            dictionary=_animals(dog_phi=np.pi / 3),
            target_pair=("dog_poodle", "bird_hawk"),
            optimize={"method": "grid", "max_steps": 6},
        )
        result = canc.run()
        artifacts = result.materialize(tmp_path)
        text = artifacts["summary"].read_text()
        assert "## Caveat" not in text


class TestBeforeAfterPlot:
    def test_before_after_writes_png(self, tmp_path: Path):
        pytest.importorskip("matplotlib")
        canc = Cancellation(
            dictionary=_animals(dog_phi=np.pi / 3),
            target_pair=("dog_poodle", "bird_hawk"),
            optimize={"method": "grid", "max_steps": 6},
        )
        result = canc.run()
        out = result.plot(tmp_path / "ba.png", kind="before_after")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_grid_plot_refused_on_three_knobs(self, tmp_path: Path):
        pytest.importorskip("matplotlib")
        canc = Cancellation(
            dictionary=_hea_pair(),
            target_pair=("a", "c"),
            knobs=[
                "a.theta[0,0,0]", "a.theta[0,0,1]", "c.theta[0,0,2]",
            ],
            optimize={"method": "grid", "max_steps": 3},
        )
        result = canc.run()
        with pytest.raises(NotImplementedError, match="len\\(knobs\\) == 2"):
            result.plot(tmp_path / "p.png", kind="grid")
