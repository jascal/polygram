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


def _hea_two_clusters() -> Dictionary:
    """HEA fixture with two size-2 clusters for cluster-shared tests."""
    from polygram import HEA_Rung2

    return Dictionary(
        name="HeaTwoClusters",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.50, alpha=0.05, gamma=0.02),
            Feature("dog_beagle", "dogs", beta=-0.48, alpha=0.04, gamma=0.03),
            Feature("bird_hawk", "birds", beta=0.50, alpha=-0.04, gamma=0.02),
            Feature("bird_sparrow", "birds", beta=0.52, alpha=-0.03, gamma=0.01),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=HEA_Rung2(depth=2),
    )


class TestClusterSharedKnobs:
    def test_cluster_shared_theta_knobs_run(self):
        canc = Cancellation(
            dictionary=_hea_two_clusters(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dogs.theta[0,0,0]", "birds.theta[0,0,0]"],
            optimize={"method": "grid", "max_steps": 8},
            preserve_tiers=False,
        )
        result = canc.run()
        assert result.trajectory.shape == (64, 3)
        assert set(result.optimized_knobs.keys()) == {
            "dogs.theta[0,0,0]", "birds.theta[0,0,0]",
        }

    def test_cluster_shared_phi_accepted(self):
        canc = Cancellation(
            dictionary=_hea_two_clusters(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dogs.phi", "birds.phi"],
            optimize={"method": "grid", "max_steps": 6},
            preserve_tiers=False,
        )
        result = canc.run()
        assert result.trajectory.shape == (36, 3)

    def test_mps_cluster_shared_phi_preserves_sibling_overlaps(self):
        # Bit-for-bit case: MPS rung-1 + cluster-shared phi.
        d = _animals()  # all phi default to 0 in this fixture path
        canc = Cancellation(
            dictionary=d,
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dogs.phi", "birds.phi"],
            optimize={"method": "grid", "max_steps": 6},
            preserve_tiers=False,
        )
        result = canc.run()
        for cluster in ("dogs", "birds"):
            i, j = (d.feature_index(m) for m in d.hierarchy[cluster])
            assert abs(
                result.before_gram[i, j] - result.after_gram[i, j]
            ) < 1e-9

    def test_hea_cluster_shared_theta_does_not_preserve_siblings(self):
        # On diverse-sibling HEA fixtures the bit-for-bit invariant does
        # NOT hold. Run completes; sibling Gram MAY drift. Assert only
        # the trajectory shape (search-space-reduction guarantee).
        canc = Cancellation(
            dictionary=_hea_two_clusters(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dogs.theta[0,0,0]", "birds.theta[0,0,0]"],
            optimize={"method": "grid", "max_steps": 6},
            preserve_tiers=False,
        )
        result = canc.run()
        # 2 cluster-shared axes at resolution 6 → 36 evaluations.
        assert result.trajectory.shape == (36, 3)

    def test_summary_caveat_mps_phi_names_factorization(
        self, tmp_path: Path
    ):
        # Pure cluster-shared phi on MPS — bit-for-bit caveat.
        # Use HEA-encoded version since structural_floor only fires NaN
        # outside the canonical 2-φ shape; cluster-shared phi on MPS is
        # not the canonical 2-phi shape so floor will be NaN.
        canc = Cancellation(
            dictionary=_animals(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dogs.phi", "birds.phi"],
            optimize={"method": "grid", "max_steps": 4},
        )
        result = canc.run()
        artifacts = result.materialize(tmp_path)
        text = artifacts["summary"].read_text()
        assert "## Caveat" in text
        assert "final-Rz factorization" in text
        assert "bit-for-bit" in text
        assert "best value found" not in text

    def test_summary_caveat_hea_pure_cluster_names_search_space(
        self, tmp_path: Path
    ):
        canc = Cancellation(
            dictionary=_hea_two_clusters(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dogs.theta[0,0,0]", "birds.theta[0,0,0]"],
            optimize={"method": "grid", "max_steps": 4},
        )
        result = canc.run()
        artifacts = result.materialize(tmp_path)
        text = artifacts["summary"].read_text()
        assert "## Caveat" in text
        assert "search-space dimensionality reduction" in text
        assert "MAY drift" in text
        assert "concept_gram_tier_separation" in text
        assert "best value found" not in text

    def test_grid_4_axis_cap_counts_cluster_shared_as_one(self):
        # Five paths total, but len(knobs) is what the cap watches.
        # Confirm a 4-axis cluster-shared list is accepted; a 5th entry
        # (whether feature or cluster) trips the cap.
        from polygram import HEA_Rung2

        d = Dictionary(
            name="HeaFour",
            features=[
                Feature("a", "g1", beta=0.10, alpha=0.05, gamma=0.02),
                Feature("b", "g1", beta=0.11, alpha=0.04, gamma=0.03),
                Feature("c", "g2", beta=0.20, alpha=0.05, gamma=0.02),
                Feature("d", "g2", beta=0.22, alpha=0.04, gamma=0.03),
                Feature("e", "g3", beta=0.30, alpha=0.05, gamma=0.02),
                Feature("f", "g3", beta=0.32, alpha=0.04, gamma=0.03),
                Feature("g", "g4", beta=0.40, alpha=0.05, gamma=0.02),
                Feature("h", "g4", beta=0.42, alpha=0.04, gamma=0.03),
                Feature("i", "g5", beta=0.50, alpha=0.05, gamma=0.02),
                Feature("j", "g5", beta=0.52, alpha=0.04, gamma=0.03),
            ],
            hierarchy={
                "g1": ["a", "b"], "g2": ["c", "d"], "g3": ["e", "f"],
                "g4": ["g", "h"], "g5": ["i", "j"],
            },
            encoding=HEA_Rung2(depth=2),
        )
        Cancellation(
            dictionary=d,
            target_pair=("a", "c"),
            knobs=[
                "g1.theta[0,0,0]", "g2.theta[0,0,0]",
                "g3.theta[0,0,0]", "g4.theta[0,0,0]",
            ],
            optimize={"method": "grid", "max_steps": 3},
        )
        with pytest.raises(ValueError, match="at most 4 knobs"):
            Cancellation(
                dictionary=d,
                target_pair=("a", "c"),
                knobs=[
                    "g1.theta[0,0,0]", "g2.theta[0,0,0]",
                    "g3.theta[0,0,0]", "g4.theta[0,0,0]",
                    "g5.theta[0,0,0]",
                ],
                optimize={"method": "grid", "max_steps": 3},
            )

    def test_unknown_cluster_rejected(self):
        with pytest.raises(ValueError, match="not declared"):
            Cancellation(
                dictionary=_hea_two_clusters(),
                target_pair=("dog_poodle", "bird_hawk"),
                knobs=["cats.phi", "birds.phi"],
            )


class TestMixedKnobs:
    def test_mixed_knob_list_accepted(self):
        canc = Cancellation(
            dictionary=_hea_two_clusters(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dog_poodle.theta[0,0,0]", "birds.theta[0,0,0]"],
            optimize={"method": "grid", "max_steps": 4},
            preserve_tiers=False,
        )
        result = canc.run()
        assert result.trajectory.shape == (16, 3)

    def test_summary_caveat_names_both_warnings_for_mixed(
        self, tmp_path: Path
    ):
        canc = Cancellation(
            dictionary=_hea_two_clusters(),
            target_pair=("dog_poodle", "bird_hawk"),
            knobs=["dog_poodle.theta[0,0,0]", "birds.theta[0,0,0]"],
            optimize={"method": "grid", "max_steps": 4},
        )
        result = canc.run()
        artifacts = result.materialize(tmp_path)
        text = artifacts["summary"].read_text()
        assert "## Caveat" in text
        assert "best value found" in text
        assert "mixes per-feature and cluster-shared" in text


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


class TestRung3Cancellation:
    @staticmethod
    def _rung3_pair(
        beta_a: float = -0.5, beta_b: float = 0.5, alpha_b: float = 0.0
    ) -> Dictionary:
        from polygram import Rung3

        return Dictionary(
            name="Rung3Pair",
            features=[
                Feature("a", "ca", beta=beta_a, phi=0.3),
                Feature("b", "cb", beta=beta_b, alpha=alpha_b, phi=0.7),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=Rung3(),
        )

    def test_cancellation_rung3_smoke(self):
        """Synthesize a tiny rung-3 dictionary, run the joint optimizer,
        confirm the new fields are populated and the structural_floor
        matches the MPS-equivalent floor of the same (α, β, γ)."""
        pytest.importorskip("scipy")
        from polygram import MPSRung1

        d = self._rung3_pair()
        canc = Cancellation(
            dictionary=d,
            target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(3, 3),
            optimize={"method": "grid", "max_steps": 12},
        )
        assert canc.encoding == "rung3"
        result = canc.run()

        # New fields are populated (not NaN).
        import math
        assert not math.isnan(result.theta_amp_optimum)
        assert not math.isnan(result.psi_aux_optimum)

        # Equivalent MPS floor matches (α, β, γ) tuple.
        from dataclasses import replace
        mps_dict = replace(d, encoding=MPSRung1())
        mps_canc = Cancellation(
            dictionary=mps_dict,
            target_pair=("a", "b"),
            preserve_tiers=False,
        )
        mps_floor = mps_canc.structural_floor()
        assert result.structural_floor == pytest.approx(mps_floor, abs=1e-9)

        # Method label and trajectory shape.
        assert result.method == "rung3_joint"
        assert result.trajectory.shape[1] == 5  # 4 knobs + overlap
        assert result.trajectory.shape[0] >= 3 * 3  # at least the outer grid

        # Optimized knobs dict has all four entries.
        for path in [
            "a.phi", "b.phi", "b.theta_amp", "b.psi_aux",
        ]:
            assert path in result.optimized_knobs

    def test_cancellation_mps_result_has_nan_amp_aux_fields(self):
        from polygram import MPSRung1

        d = Dictionary(
            name="MpsTwo",
            features=[
                Feature("a", "ca", beta=-0.5, phi=0.3),
                Feature("b", "cb", beta=0.5, phi=0.7),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=MPSRung1(),
        )
        result = Cancellation(
            dictionary=d, target_pair=("a", "b"),
            preserve_tiers=False,
            optimize={"method": "grid", "max_steps": 8},
        ).run()
        import math
        assert math.isnan(result.theta_amp_optimum)
        assert math.isnan(result.psi_aux_optimum)

    def test_cancellation_rung3_breaks_floor_synthetic(self):
        """Hand-crafted pair with non-trivial (θ_b, ψ_b) optimum that
        demonstrably reaches below the MPS phase-only floor M − |V|."""
        pytest.importorskip("scipy")
        # Choose alpha_b so the MPS pair has high pre-overlap and a
        # non-zero V (so M − |V| > 0). The amp branch can multiply
        # cos(ψ_b)·sin(2θ_b) into the overlap; ψ_b ≈ π drives the
        # amp factor toward 0, so the rung-3 overlap can dip below
        # the MPS floor.
        d = self._rung3_pair(beta_a=0.05, beta_b=0.05, alpha_b=0.05)

        canc = Cancellation(
            dictionary=d,
            target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(5, 5),
            optimize={"method": "grid", "max_steps": 12},
        )
        result = canc.run()
        floor = result.structural_floor
        assert floor > 0.05, (
            f"need a non-trivial floor to demonstrate breakage; got {floor}"
        )
        assert result.after_overlap < floor - 1e-3, (
            f"rung-3 optimizer failed to break the MPS floor "
            f"(after={result.after_overlap}, floor={floor})"
        )

    def test_rung3_requires_rung3_dictionary(self):
        """Cancellation(encoding="rung3") on an MPS dictionary must
        raise — the rung-3 path needs the per-feature amp knobs."""
        from polygram import MPSRung1

        d = Dictionary(
            name="MpsTwo",
            features=[
                Feature("a", "ca", beta=-0.5, phi=0.3),
                Feature("b", "cb", beta=0.5, phi=0.7),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=MPSRung1(),
        )
        with pytest.raises(ValueError, match="requires a Rung3 dictionary"):
            Cancellation(
                dictionary=d, target_pair=("a", "b"),
                encoding="rung3",
            )

    def test_rung3_custom_knob_list_rejected(self):
        d = self._rung3_pair()
        with pytest.raises(ValueError, match="canonical 4-knob list"):
            Cancellation(
                dictionary=d, target_pair=("a", "b"),
                knobs=["a.phi", "b.phi"],
            )

    def test_rung3_efficiency_can_exceed_zero(self):
        """When rung-3 reaches below the MPS floor, the conventional
        ``cancellation_efficiency`` clamps at 1.0 because the formula's
        denominator (before − floor) is non-negative."""
        pytest.importorskip("scipy")
        d = self._rung3_pair(beta_a=0.05, beta_b=0.05, alpha_b=0.05)
        result = Cancellation(
            dictionary=d, target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(3, 3),
            optimize={"method": "grid", "max_steps": 8},
        ).run()
        if result.cancellation_efficiency is not None:
            assert 0.0 <= result.cancellation_efficiency <= 1.0

    def test_min_amp_overlap_blocks_trivial_amp_zeroing(self):
        """Without the constraint the rung-3 optimizer converges to the
        trivial amp-zeroing solution (θ_b≈π/4, ψ_b≈π) and post-overlap
        is essentially zero. With min_amp_overlap=0.5 the optimizer is
        forbidden from landing on configurations whose amp factor falls
        below 0.5, so post-overlap must be at least
        ``min_amp_overlap × mps_floor``."""
        pytest.importorskip("scipy")
        d = self._rung3_pair(beta_a=0.05, beta_b=0.05, alpha_b=0.05)

        unconstrained = Cancellation(
            dictionary=d, target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(5, 5),
            optimize={"method": "grid", "max_steps": 12},
        ).run()
        constrained = Cancellation(
            dictionary=d, target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(5, 5),
            optimize={"method": "grid", "max_steps": 12},
            min_amp_overlap=0.5,
        ).run()

        floor = unconstrained.structural_floor
        assert constrained.structural_floor == pytest.approx(floor, abs=1e-9)

        # Unconstrained drops far below the lower bound the constraint
        # would impose.
        lower_bound = 0.5 * floor
        assert unconstrained.after_overlap < lower_bound
        # Constrained respects the bound (small tolerance for numerical
        # slop in the gram).
        assert constrained.after_overlap >= lower_bound - 1e-6

    def test_min_amp_overlap_invalid_range_rejected(self):
        d = self._rung3_pair()
        with pytest.raises(ValueError, match="min_amp_overlap"):
            Cancellation(
                dictionary=d, target_pair=("a", "b"),
                min_amp_overlap=1.5,
            )
        with pytest.raises(ValueError, match="min_amp_overlap"):
            Cancellation(
                dictionary=d, target_pair=("a", "b"),
                min_amp_overlap=-0.1,
            )

    def test_min_amp_overlap_rejected_on_non_rung3(self):
        from polygram import MPSRung1

        d = Dictionary(
            name="MpsTwo",
            features=[
                Feature("a", "ca", beta=-0.5, phi=0.3),
                Feature("b", "cb", beta=0.5, phi=0.7),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=MPSRung1(),
        )
        # post-Rung4: min_amp_overlap is meaningful for rung3 OR rung4
        # encodings; rejected on MPS / HEA.
        with pytest.raises(ValueError, match="only meaningful for "):
            Cancellation(
                dictionary=d, target_pair=("a", "b"),
                min_amp_overlap=0.5,
            )


# ---------------------------------------------------------------------------
# Tasks §3.3 — Cancellation accepts CancellationConfig with override
# precedence: per-field kwarg > config > dataclass-default. See
# polygram.config for the documented rule.
# ---------------------------------------------------------------------------


class TestCancellationConfigPassthrough:
    def _animals_local(self):
        return _animals()

    def test_no_config_preserves_legacy_defaults(self):
        from polygram import Cancellation

        canc = Cancellation(
            dictionary=self._animals_local(), target_pair=("dog_poodle", "bird_hawk")
        )
        # Pre-config legacy defaults round-tripped through CancellationConfig.
        assert canc.tolerance == 0.05
        assert canc.preserve_tiers is True
        assert canc.optimize == {"method": "grid", "max_steps": 50}
        assert canc.grid_outer == (5, 5)
        assert canc.min_amp_overlap == 0.0

    def test_config_supplies_unset_fields(self):
        from polygram import Cancellation, CancellationConfig

        cfg = CancellationConfig(tolerance=0.01, preserve_tiers=False)
        canc = Cancellation(
            dictionary=self._animals_local(),
            target_pair=("dog_poodle", "bird_hawk"),
            config=cfg,
        )
        # Config fills in for the unsupplied fields...
        assert canc.tolerance == 0.01
        assert canc.preserve_tiers is False
        # ...and other fields keep CancellationConfig's own defaults.
        assert canc.optimize == {"method": "grid", "max_steps": 50}
        assert canc.grid_outer == (5, 5)

    def test_per_field_kwarg_overrides_config(self):
        from polygram import Cancellation, CancellationConfig

        cfg = CancellationConfig(tolerance=0.01)
        canc = Cancellation(
            dictionary=self._animals_local(),
            target_pair=("dog_poodle", "bird_hawk"),
            config=cfg,
            tolerance=0.001,
        )
        # kwarg wins over config.
        assert canc.tolerance == 0.001

    def test_config_optimize_dict_propagates(self):
        from polygram import Cancellation, CancellationConfig

        cfg = CancellationConfig(optimize={"method": "grid", "max_steps": 7})
        canc = Cancellation(
            dictionary=self._animals_local(),
            target_pair=("dog_poodle", "bird_hawk"),
            config=cfg,
        )
        assert canc.optimize == {"method": "grid", "max_steps": 7}


# ---------------------------------------------------------------------------
# Rung4 cancellation (add-rung4-encoding-mvp §5)
# ---------------------------------------------------------------------------


class TestRung4Cancellation:
    @staticmethod
    def _rung4_pair(
        beta_a: float = -0.5, beta_b: float = 0.5, alpha_b: float = 0.0
    ) -> Dictionary:
        from polygram.encoding import Rung4

        return Dictionary(
            name="Rung4Pair",
            features=[
                Feature("a", "ca", beta=beta_a, phi=0.3),
                Feature("b", "cb", beta=beta_b, alpha=alpha_b, phi=0.7),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=Rung4(),
        )

    def test_canonical_six_knob_list(self):
        # Rung4 cancellation auto-builds the 6-knob canonical list.
        d = self._rung4_pair()
        canc = Cancellation(
            dictionary=d,
            target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(2, 2),
        )
        assert canc.encoding == "rung4"
        assert canc.knobs == [
            "a.phi", "b.phi",
            "b.theta_amp", "b.psi_aux",
            "b.theta_amp_b", "b.psi_amp_b",
        ]

    def test_custom_knob_list_rejected(self):
        d = self._rung4_pair()
        with pytest.raises(ValueError, match="canonical 6-knob list"):
            Cancellation(
                dictionary=d,
                target_pair=("a", "b"),
                encoding="rung4",
                knobs=["a.phi", "b.phi"],  # too few
            )

    def test_rung4_dispatch_requires_rung4_dict(self):
        from polygram import Rung3

        d_r3 = Dictionary(
            name="r3",
            features=[
                Feature("a", "ca", beta=-0.5),
                Feature("b", "cb", beta=0.5),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=Rung3(),
        )
        with pytest.raises(ValueError, match="encoding='rung4'.*requires"):
            Cancellation(
                dictionary=d_r3,
                target_pair=("a", "b"),
                encoding="rung4",
            )

    def test_cancellation_rung4_smoke(self):
        """Synthesize a tiny Rung4 dictionary, run the joint optimizer,
        confirm fields populate and structural_floor matches MPS-equivalent.
        """
        pytest.importorskip("scipy")
        from polygram import MPSRung1

        d = self._rung4_pair()
        canc = Cancellation(
            dictionary=d,
            target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(2, 2),  # 2^4 = 16 outer cells — fast
            optimize={"method": "grid", "max_steps": 8},
        )
        assert canc.encoding == "rung4"
        result = canc.run()

        # result is a CancellationResult with the new fields populated.
        assert result.method == "rung4_joint"
        assert "b.theta_amp_b" in result.optimized_knobs
        assert "b.psi_amp_b" in result.optimized_knobs
        assert result.cancellation_efficiency is None or (
            0.0 <= result.cancellation_efficiency <= 1.0
        )

        # Structural floor matches the MPS-equivalent on (α, β, γ).
        from dataclasses import replace
        mps_dict = replace(d, encoding=MPSRung1())
        mps_canc = Cancellation(
            dictionary=mps_dict,
            target_pair=("a", "b"),
            preserve_tiers=False,
        )
        assert result.structural_floor == pytest.approx(
            mps_canc.structural_floor(), abs=1e-9
        )

    def test_cancellation_rung4_lowers_overlap(self):
        pytest.importorskip("scipy")
        d = self._rung4_pair()
        canc = Cancellation(
            dictionary=d,
            target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(2, 2),
            optimize={"method": "grid", "max_steps": 8},
        )
        result = canc.run()
        # The joint optimizer should not INCREASE the overlap. With a
        # 4D outer grid + scipy refine, we expect at-or-below the
        # before value.
        assert result.after_overlap <= result.before_overlap + 1e-9

    def test_min_amp_overlap_constraint_active(self):
        """With min_amp_overlap > 0, the optimizer must respect the
        constraint — at the chosen optimum the amp factor satisfies
        the bound (or the feasible set was empty and we fell back to
        unconstrained)."""
        pytest.importorskip("scipy")
        from polygram.encoding import rung4_amp_overlap_squared

        d = self._rung4_pair()
        canc = Cancellation(
            dictionary=d,
            target_pair=("a", "b"),
            preserve_tiers=False,
            grid_outer=(2, 2),
            optimize={"method": "grid", "max_steps": 4},
            min_amp_overlap=0.3,
        )
        result = canc.run()
        # Compute the amp factor at the chosen optimum.
        a_feat = d.features[0]
        amp_sq = rung4_amp_overlap_squared(
            a_feat.theta_amp, a_feat.psi_aux,
            a_feat.theta_amp_b, a_feat.psi_amp_b,
            result.optimized_knobs["b.theta_amp"],
            result.optimized_knobs["b.psi_aux"],
            result.optimized_knobs["b.theta_amp_b"],
            result.optimized_knobs["b.psi_amp_b"],
        )
        # Either constraint respected, or we fell back to unconstrained.
        feasible_pool_existed = result.feasible_count > 0
        if feasible_pool_existed:
            assert amp_sq >= 0.3 - 1e-9, f"amp_sq={amp_sq} < 0.3"
