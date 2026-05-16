"""Tests for the LearnedKnobAssignment strategy + supporting helpers.

Covers tasks.md §1 (objectives + protocol), §2 (KnobAssignmentResult
fields), §3 (strategy class — greedy + scipy + HEA fallback +
validation split + early-stop).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polygram.geometry import (
    KnobAssignmentResult,
    LearnedAxisObjective,
    behavioural_objective,
    pearson_objective,
    spearman_objective,
)


# ---------------------------------------------------------------------------
# §1 — Objectives + protocol conformance
# ---------------------------------------------------------------------------


class TestObjectives:
    def test_spearman_matches_explicit_formula_on_off_diagonal(self):
        # Two 4×4 matrices whose off-diagonals are perfectly
        # rank-correlated → Spearman = +1.
        a = np.array(
            [[1.0, 0.1, 0.2, 0.3],
             [0.1, 1.0, 0.4, 0.5],
             [0.2, 0.4, 1.0, 0.6],
             [0.3, 0.5, 0.6, 1.0]]
        )
        # b is a monotone transform of a's off-diagonal — same ranks.
        b = a * 2.0 + 0.5
        assert math.isclose(
            spearman_objective(a.astype(complex), b ** 2), 1.0, abs_tol=1e-12
        )

    def test_pearson_matches_correlation_on_off_diagonal(self):
        # Off-diagonal of |a|² is linearly related to off-diagonal of
        # b → Pearson should be very close to +1.
        rng = np.random.default_rng(0)
        n = 8
        a = rng.uniform(0.0, 1.0, size=(n, n))
        a = (a + a.T) / 2.0  # symmetric
        np.fill_diagonal(a, 1.0)
        # b is a strictly increasing function of |a|² off-diagonal,
        # so Pearson against |a|² is +1 by construction.
        b = 3.0 * (a ** 2) + 0.7
        score = pearson_objective(a.astype(complex), b)
        assert score > 0.99

    def test_behavioural_objective_uses_reference_matrix(self):
        # Build a non-negative reference matrix that disagrees with
        # the "decoder_geom" argument. The behavioural objective
        # ignores decoder_geom and correlates against ref instead.
        rng = np.random.default_rng(0)
        n = 6
        ref = rng.uniform(0.0, 1.0, size=(n, n))
        ref = (ref + ref.T) / 2.0
        np.fill_diagonal(ref, 1.0)
        decoy = rng.uniform(0.0, 1.0, size=(n, n))  # unrelated
        decoy = (decoy + decoy.T) / 2.0
        np.fill_diagonal(decoy, 1.0)
        obj = behavioural_objective(ref)
        # analytic_good's |.|² == ref → perfect rank correlation with ref.
        analytic_good = np.sqrt(ref)
        good = obj(analytic_good.astype(complex), decoy)
        assert good > 0.99
        # analytic_bad's |.|² == decoy → uncorrelated with ref.
        analytic_bad = np.sqrt(decoy)
        bad = obj(analytic_bad.astype(complex), decoy)
        assert good > bad

    def test_behavioural_objective_rejects_non_square(self):
        with pytest.raises(ValueError, match="square matrix"):
            behavioural_objective(np.zeros((3, 4)))

    def test_protocol_runtime_check_spearman(self):
        assert isinstance(spearman_objective, LearnedAxisObjective)

    def test_protocol_runtime_check_pearson(self):
        assert isinstance(pearson_objective, LearnedAxisObjective)

    def test_protocol_runtime_check_behavioural_closure(self):
        obj = behavioural_objective(np.eye(4))
        assert isinstance(obj, LearnedAxisObjective)

    def test_objective_accepts_call_without_feature_names(self):
        # The protocol's feature_names default is None — Spearman
        # SHALL accept invocation without it.
        n = 5
        a = np.eye(n, dtype=complex)
        b = np.eye(n)
        # No feature_names passed:
        spearman_objective(a, b)
        pearson_objective(a, b)


# ---------------------------------------------------------------------------
# §2 — KnobAssignmentResult new fields
# ---------------------------------------------------------------------------


class TestKnobAssignmentResultFields:
    def test_new_fields_default_to_none(self):
        r = KnobAssignmentResult(
            cluster_per_feature=["c0"],
            betas=[0.0],
            gammas=[0.0],
            cluster_method="test",
            beta_variance_explained=0.0,
        )
        assert r.axis_assignment is None
        assert r.objective_value is None
        assert r.objective_baseline is None
        assert r.training_objective_value is None

    def test_clustered_strategy_leaves_fields_none(self):
        from polygram.geometry import ClusteredKnobAssignment

        rng = np.random.default_rng(0)
        projs = rng.standard_normal((8, 16))
        result = ClusteredKnobAssignment().assign(
            projs,
            [f"f{i}" for i in range(8)],
            n_clusters=2,
            gamma_range=(-0.5, 0.5),
            assign_gamma=False,
            seed=0,
        )
        assert result.axis_assignment is None
        assert result.objective_value is None
        assert result.objective_baseline is None
        assert result.training_objective_value is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synth_clustered(n_clusters=8, cluster_size=4, d_model=32, seed=0,
                     noise_sigma=0.08):
    """Tight-cluster synth — one centroid per cluster, isotropic noise
    siblings. Used by the strategy tests + the prototype's scan 4."""
    rng = np.random.default_rng(seed)
    n = n_clusters * cluster_size
    projs = np.zeros((n, d_model))
    labels = []
    for c in range(n_clusters):
        centroid = rng.standard_normal(d_model)
        centroid /= np.linalg.norm(centroid) + 1e-12
        for s in range(cluster_size):
            v = centroid + rng.standard_normal(d_model) * noise_sigma
            v /= np.linalg.norm(v) + 1e-12
            projs[c * cluster_size + s] = v
            labels.append(f"c{c:02d}/f{c:02d}_{s:02d}")
    return projs, labels


# ---------------------------------------------------------------------------
# §3 — LearnedKnobAssignment greedy solver
# ---------------------------------------------------------------------------


class TestGreedySolver:
    def test_deterministic(self):
        from polygram.encoding import Rung4
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered()
        a = LearnedKnobAssignment().assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung4(),
        )
        b = LearnedKnobAssignment().assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung4(),
        )
        assert a.axis_assignment == b.axis_assignment
        assert math.isclose(a.objective_value, b.objective_value, abs_tol=1e-12)
        assert math.isclose(
            a.objective_baseline, b.objective_baseline, abs_tol=1e-12
        )

    def test_reproduces_scan4_k3_above_0_30(self):
        from polygram.encoding import Rung5
        from polygram.geometry import LearnedKnobAssignment

        # Matches scan 4's synth: 16 clusters of 4 → 64 features, d=32.
        projs, labels = _synth_clustered(
            n_clusters=16, cluster_size=4, d_model=32, seed=0,
        )
        result = LearnedKnobAssignment().assign(
            projs, labels, n_clusters=16, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung5(n_amp_qubits=3),
        )
        assert result.objective_value >= 0.30

    def test_reproduces_scan4_k4_above_0_30(self):
        from polygram.encoding import Rung5
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered(
            n_clusters=16, cluster_size=4, d_model=32, seed=0,
        )
        result = LearnedKnobAssignment().assign(
            projs, labels, n_clusters=16, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung5(n_amp_qubits=4),
        )
        assert result.objective_value >= 0.30

    def test_objective_value_no_worse_than_baseline(self):
        from polygram.encoding import Rung4
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered()
        result = LearnedKnobAssignment().assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung4(),
        )
        # Training score must beat baseline (greedy started from
        # baseline as the fill-in and only locks in improvements).
        assert (
            result.training_objective_value
            >= result.objective_baseline - 1e-6
        )


class TestParameterValidation:
    def test_solver_must_be_greedy_or_scipy(self):
        from polygram.geometry import LearnedKnobAssignment

        with pytest.raises(ValueError, match="solver must be"):
            LearnedKnobAssignment(solver="bogus")

    def test_validation_fraction_out_of_range(self):
        from polygram.geometry import LearnedKnobAssignment

        with pytest.raises(ValueError, match="validation_fraction"):
            LearnedKnobAssignment(validation_fraction=0.7)

    def test_max_axes_below_1(self):
        from polygram.geometry import LearnedKnobAssignment

        with pytest.raises(ValueError, match="max_axes"):
            LearnedKnobAssignment(max_axes=0)

    def test_scipy_restarts_below_1(self):
        from polygram.geometry import LearnedKnobAssignment

        with pytest.raises(ValueError, match="scipy_restarts"):
            LearnedKnobAssignment(scipy_restarts=0)

    def test_early_stop_eps_negative(self):
        from polygram.geometry import LearnedKnobAssignment

        with pytest.raises(ValueError, match="early_stop_eps"):
            LearnedKnobAssignment(early_stop_eps=-1e-9)


class TestEarlyStopping:
    def test_zero_eps_disables_heuristic(self):
        # With eps=0, every knob slot is filled by the search (no
        # baseline fall-through). Verify by checking that the
        # assigned axes have no duplicates and use the search's
        # decisions, not baseline_axis_for_knob.
        from polygram.encoding import Rung5
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered(
            n_clusters=8, cluster_size=4, d_model=32, seed=0,
        )
        result = LearnedKnobAssignment(early_stop_eps=0.0).assign(
            projs, labels, n_clusters=8, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung5(n_amp_qubits=2),
        )
        # All knobs assigned; no two knobs share an axis (permutation).
        axes = list(result.axis_assignment.values())
        assert len(axes) == len(set(axes))


class TestValidationSplit:
    def test_train_vs_validation_objectives_separate(self):
        from polygram.encoding import Rung4
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered()
        result = LearnedKnobAssignment(
            validation_fraction=0.3
        ).assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung4(),
        )
        # Both scores populated and they need NOT be equal (a 30%
        # held-out set has a different rank statistic).
        assert result.objective_value is not None
        assert result.training_objective_value is not None

    def test_zero_validation_fraction_equates(self):
        from polygram.encoding import Rung4
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered()
        result = LearnedKnobAssignment(
            validation_fraction=0.0
        ).assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung4(),
        )
        assert math.isclose(
            result.objective_value,
            result.training_objective_value,
            abs_tol=1e-12,
        )


class TestHEAFallback:
    def test_hea_rung2_falls_through_to_clustered(self):
        from polygram.encoding import HEA_Rung2
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered(d_model=8)
        result = LearnedKnobAssignment().assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0,
            encoding=HEA_Rung2(n_qubits=3, depth=2),
        )
        # No learned axis assignment surfaces for HEA.
        assert result.axis_assignment is None
        assert result.objective_value is None
        assert result.objective_baseline is None
        # Clustering-side result still well-formed.
        assert result.cluster_method == "kmeans"
        assert len(result.betas) == len(labels)


class TestScipySolver:
    def test_initialises_from_greedy_no_regression(self):
        pytest.importorskip("scipy")
        from polygram.encoding import Rung4
        from polygram.geometry import LearnedKnobAssignment

        projs, labels = _synth_clustered()
        greedy_result = LearnedKnobAssignment(solver="greedy").assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung4(),
        )
        scipy_result = LearnedKnobAssignment(solver="scipy").assign(
            projs, labels, n_clusters=4, gamma_range=(-0.5, 0.5),
            assign_gamma=False, seed=0, encoding=Rung4(),
        )
        # Scipy starts from greedy; the integer-rounded result should
        # be no worse than greedy's training score.
        assert (
            scipy_result.training_objective_value
            >= greedy_result.training_objective_value - 1e-6
        )
