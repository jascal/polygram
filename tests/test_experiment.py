"""Experiment + InterferenceSweep — declarative phase sweeps."""

import math

import numpy as np
import pytest

from polygram.dictionary import Dictionary, Feature
from polygram.experiment import Experiment


def _animals() -> Dictionary:
    return Dictionary(
        name="AnimalsExp",
        features=[
            Feature("dog_a", "dogs", beta=-0.5),
            Feature("dog_b", "dogs", beta=-0.5),
            Feature("bird_a", "birds", beta=0.5),
            Feature("bird_b", "birds", beta=0.5),
        ],
        hierarchy={"dogs": ["dog_a", "dog_b"], "birds": ["bird_a", "bird_b"]},
    )


def test_unknown_target_pair_feature_rejected():
    with pytest.raises(ValueError, match="not declared"):
        Experiment(
            name="Bad",
            dictionary=_animals(),
            target_pair=("dog_a", "ghost"),
            sweep={"dog_a.phi": np.linspace(0, 1, 3)},
        )


def test_unknown_measure_rejected():
    with pytest.raises(ValueError, match="unknown measure"):
        Experiment(
            name="Bad",
            dictionary=_animals(),
            target_pair=("dog_a", "bird_a"),
            sweep={"dog_a.phi": np.linspace(0, 1, 3)},
            measures=["bogus"],
        )


def test_unknown_assertion_rejected():
    with pytest.raises(ValueError, match="unknown assertion"):
        Experiment(
            name="Bad",
            dictionary=_animals(),
            target_pair=("dog_a", "bird_a"),
            sweep={"dog_a.phi": np.linspace(0, 1, 3)},
            assertions=["bogus"],
        )


def test_sweep_key_must_reference_feature():
    with pytest.raises(ValueError, match="unknown feature"):
        Experiment(
            name="Bad",
            dictionary=_animals(),
            target_pair=("dog_a", "bird_a"),
            sweep={"ghost.phi": np.linspace(0, 1, 3)},
        )


def test_sweep_key_must_match_grammar():
    with pytest.raises(ValueError, match="grammar"):
        Experiment(
            name="Bad",
            dictionary=_animals(),
            target_pair=("dog_a", "bird_a"),
            sweep={"dog_a.beta": np.linspace(0, 1, 3)},
        )


def test_qutip_backend_not_implemented():
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"dog_a.phi": np.linspace(0, 1, 3)},
    )
    with pytest.raises(NotImplementedError, match="not supported"):
        exp.run(backend="qutip")


def test_run_produces_correctly_shaped_result():
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"dog_b.phi": np.linspace(0, math.pi / 2, 5)},
    )
    res = exp.run()
    assert res.gram_matrices.shape == (5, 4, 4)
    assert res.overlaps.shape == (5,)
    assert res.schmidt_ranks.shape == (5, 4)


def test_overlap_at_phi_zero_matches_baseline():
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"dog_a.phi": np.array([0.0, math.pi / 4, math.pi / 2])},
    )
    res = exp.run()
    expected_zero = np.cos(0.5) ** 4
    np.testing.assert_allclose(res.overlaps[0], expected_zero, atol=1e-3)


def test_destructive_assertion_at_endpoint():
    """Push the target pair toward destruction by sweeping bird_a's phi
    enough that |<dog_a|bird_a>|² drops; the assertion should still
    return a bool array of correct shape."""
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"bird_a.phi": np.linspace(0, math.pi, 5)},
        assertions=["target_pair_destructive_at_endpoint"],
    )
    res = exp.run()
    assert "target_pair_destructive_at_endpoint" in res.assertion_pass
    assert res.assertion_pass["target_pair_destructive_at_endpoint"].shape == (5,)


def test_hierarchical_ordering_assertion_array():
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"dog_a.phi": np.linspace(0, math.pi / 4, 4)},
        assertions=["hierarchical_ordering_preserved"],
    )
    res = exp.run()
    assert res.assertion_pass["hierarchical_ordering_preserved"].shape == (4,)
    assert res.assertion_pass["hierarchical_ordering_preserved"].all()


def test_materialize_writes_artifacts(tmp_path):
    exp = Experiment(
        name="PoodleHawk",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"bird_a.phi": np.linspace(0, math.pi, 3)},
    )
    arts = exp.materialize(tmp_path)
    assert (tmp_path / "PoodleHawk.q.orca.md").exists()
    assert (tmp_path / "run_PoodleHawk.py").exists()
    assert arts["machine"].name == "PoodleHawk.q.orca.md"
    assert arts["runner"].name == "run_PoodleHawk.py"


def test_runner_script_is_syntactically_valid(tmp_path):
    """The emitted runner must at least compile cleanly."""
    import py_compile

    exp = Experiment(
        name="PoodleHawk",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"bird_a.phi": np.linspace(0, math.pi, 3)},
    )
    exp.materialize(tmp_path)
    py_compile.compile(
        str(tmp_path / "run_PoodleHawk.py"), doraise=True
    )


def test_result_save_and_load_roundtrip(tmp_path):
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"dog_a.phi": np.linspace(0, 1, 3)},
        assertions=["hierarchical_ordering_preserved"],
    )
    res = exp.run()
    p = res.save(tmp_path / "res.npz")
    assert p.exists()
    loaded = np.load(p)
    assert loaded["gram_matrices"].shape == (3, 4, 4)
    assert loaded["assert_hierarchical_ordering_preserved"].shape == (3,)
    assert loaded["tier_sibling"].shape == (3,)
    assert loaded["tier_cross_cluster"].shape == (3,)


def test_two_axis_sweep_shape():
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={
            "dog_a.phi": np.linspace(0, math.pi, 3),
            "bird_a.phi": np.linspace(0, math.pi, 5),
        },
    )
    res = exp.run()
    assert res.gram_matrices.shape == (3, 5, 4, 4)
    assert res.overlaps.shape == (3, 5)
    assert res.schmidt_ranks.shape == (3, 5, 4)
    assert res.tier_stats["sibling"].shape == (3, 5)
    assert res.tier_stats["cross_cluster"].shape == (3, 5)


def test_tier_stats_match_baselines_at_phi_zero():
    """At phi=0 for all features in the default Animals dictionary
    (α=γ=0, β=±0.5 by cluster), siblings within a cluster are identical
    states (overlap 1.0); cross-cluster pairs have |<A|B>|² = cos(0.5)⁴.
    """
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"dog_a.phi": np.array([0.0])},
    )
    res = exp.run()
    np.testing.assert_allclose(res.tier_stats["self"][0], 1.0, atol=1e-6)
    np.testing.assert_allclose(res.tier_stats["sibling"][0], 1.0, atol=1e-4)
    np.testing.assert_allclose(
        res.tier_stats["cross_cluster"][0], np.cos(0.5) ** 4, atol=1e-4
    )


def test_summary_md_written_by_materialize(tmp_path):
    exp = Experiment(
        name="PoodleHawk",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"bird_a.phi": np.linspace(0, math.pi, 3)},
        assertions=["hierarchical_ordering_preserved"],
    )
    exp.materialize(tmp_path)
    summary = (tmp_path / "PoodleHawk_summary.md").read_text()
    assert "PoodleHawk" in summary
    assert "AnimalsExp" in summary
    assert "bird_a.phi" in summary
    assert "dog_a" in summary and "bird_a" in summary


def test_write_summary_appends_results(tmp_path):
    exp = Experiment(
        name="PoodleHawk",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"bird_a.phi": np.linspace(0, math.pi, 4)},
        assertions=["hierarchical_ordering_preserved"],
    )
    exp.materialize(tmp_path)
    res = exp.run()
    res.write_summary(tmp_path / "PoodleHawk_summary.md")
    body = (tmp_path / "PoodleHawk_summary.md").read_text()
    assert "Tier rollup" in body
    assert "sibling" in body
    assert "cross_cluster" in body
    assert "Assertion pass-rate" in body
    assert "hierarchical_ordering_preserved" in body


def test_plot_1d_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"bird_a.phi": np.linspace(0, math.pi, 5)},
    )
    res = exp.run()
    p = res.plot(tmp_path / "p.png")
    assert p.exists() and p.stat().st_size > 0


def test_plot_2d_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={
            "dog_a.phi": np.linspace(0, math.pi, 4),
            "bird_a.phi": np.linspace(0, math.pi, 5),
        },
    )
    res = exp.run()
    p = res.plot(tmp_path / "p.png")
    assert p.exists() and p.stat().st_size > 0


def test_plot_3d_raises(tmp_path):
    pytest.importorskip("matplotlib")
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={
            "dog_a.phi": np.linspace(0, math.pi, 2),
            "dog_b.phi": np.linspace(0, math.pi, 2),
            "bird_a.phi": np.linspace(0, math.pi, 2),
        },
    )
    res = exp.run()
    with pytest.raises(NotImplementedError, match="1D and 2D"):
        res.plot(tmp_path / "p.png")


def test_to_csv_includes_tier_columns(tmp_path):
    exp = Experiment(
        name="X",
        dictionary=_animals(),
        target_pair=("dog_a", "bird_a"),
        sweep={"bird_a.phi": np.linspace(0, math.pi, 3)},
    )
    res = exp.run()
    p = res.to_csv(tmp_path / "r.csv")
    header = p.read_text().splitlines()[0]
    assert "tier_sibling" in header
    assert "tier_cross_cluster" in header
    assert "overlap" in header


def _hea_tiered(encoding=None):
    from polygram import HEA_Rung2

    encoding = encoding or HEA_Rung2(depth=2)
    return Dictionary(
        name="HeaSweep",
        features=[
            Feature("a", "s1", beta=0.10, alpha=0.05, gamma=0.02),
            Feature("b", "s1", beta=0.11, alpha=0.04, gamma=0.03),
            Feature("c", "s2", beta=1.20, alpha=1.10, gamma=1.00),
        ],
        hierarchy={"s1": ["a", "b"], "s2": ["c"]},
        encoding=encoding,
    )


class TestSweepKnobs:
    def test_phi_axis_on_hea(self):
        exp = Experiment(
            name="X",
            dictionary=_hea_tiered(),
            target_pair=("a", "c"),
            sweep={"a.phi": np.linspace(0.0, math.pi, 5)},
        )
        result = exp.run()
        assert result.overlaps.shape == (5,)

    def test_theta_axis_on_hea(self):
        exp = Experiment(
            name="X",
            dictionary=_hea_tiered(),
            target_pair=("a", "c"),
            sweep={"a.theta[1,0,1]": np.linspace(-math.pi, math.pi, 5)},
        )
        result = exp.run()
        assert result.overlaps.shape == (5,)

    def test_theta_axis_rejected_on_mps(self):
        with pytest.raises(ValueError, match="HEA-only"):
            Experiment(
                name="Bad",
                dictionary=_animals(),
                target_pair=("dog_a", "bird_a"),
                sweep={"dog_a.theta[0,0,1]": np.linspace(0, 1, 3)},
            )

    def test_theta_slot_out_of_range_rejected(self):
        from polygram import HEA_Rung2

        d = _hea_tiered(HEA_Rung2(depth=2))
        with pytest.raises(ValueError, match=r"theta_shape="):
            Experiment(
                name="Bad",
                dictionary=d,
                target_pair=("a", "c"),
                sweep={"a.theta[5,0,0]": np.linspace(0, 1, 3)},
            )


class TestTierSeparationMeasure:
    def test_tiered_dictionary_carries_array(self):
        exp = Experiment(
            name="X",
            dictionary=_hea_tiered(),
            target_pair=("a", "c"),
            sweep={"a.phi": np.linspace(0, math.pi, 4)},
        )
        result = exp.run()
        assert result.tier_separation is not None
        assert result.tier_separation.shape == (4,)
        assert (result.tier_separation > 0).all()

    def test_all_singleton_dictionary_carries_none(self):
        from polygram import HEA_Rung2

        d = Dictionary(
            name="Singletons",
            features=[
                Feature("a", "s1", beta=0.1),
                Feature("b", "s2", beta=0.2),
                Feature("c", "s3", beta=0.3),
            ],
            hierarchy={"s1": ["a"], "s2": ["b"], "s3": ["c"]},
            encoding=HEA_Rung2(depth=2),
        )
        exp = Experiment(
            name="X",
            dictionary=d,
            target_pair=("a", "c"),
            sweep={"a.phi": np.linspace(0, math.pi, 3)},
        )
        result = exp.run()
        assert result.tier_separation is None

    def test_csv_and_npz_round_trip(self, tmp_path):
        exp = Experiment(
            name="X",
            dictionary=_hea_tiered(),
            target_pair=("a", "c"),
            sweep={"a.phi": np.linspace(0, math.pi, 3)},
        )
        result = exp.run()
        csv_path = result.to_csv(tmp_path / "r.csv")
        assert "tier_separation" in csv_path.read_text().splitlines()[0]

        npz_path = result.save(tmp_path / "r.npz")
        loaded = np.load(npz_path)
        assert "tier_separation" in loaded.files
        assert loaded["tier_separation"].shape == (3,)


class TestTierBoundAssertion:
    def test_passes_on_clearly_tiered_hea(self):
        exp = Experiment(
            name="X",
            dictionary=_hea_tiered(),
            target_pair=("a", "c"),
            sweep={"a.phi": np.linspace(0, math.pi, 4)},
            assertions=["concept_gram_tier_separation_bound_holds"],
        )
        result = exp.run()
        passes = result.assertion_pass[
            "concept_gram_tier_separation_bound_holds"
        ]
        assert passes.shape == (4,)
        assert passes.all()

    def test_rejected_on_mps_dictionary(self):
        with pytest.raises(ValueError, match="tier_separation_bound"):
            Experiment(
                name="Bad",
                dictionary=_animals(),
                target_pair=("dog_a", "bird_a"),
                sweep={"dog_a.phi": np.linspace(0, math.pi, 3)},
                assertions=["concept_gram_tier_separation_bound_holds"],
            )

    def test_rejected_on_hea_with_no_bound(self):
        from polygram import HEA_Rung2

        d = _hea_tiered(HEA_Rung2(depth=2, tier_separation_bound=None))
        with pytest.raises(ValueError, match="tier_separation_bound"):
            Experiment(
                name="Bad",
                dictionary=d,
                target_pair=("a", "c"),
                sweep={"a.phi": np.linspace(0, math.pi, 3)},
                assertions=["concept_gram_tier_separation_bound_holds"],
            )
