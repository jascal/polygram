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


def test_sweep_key_must_use_phi():
    with pytest.raises(ValueError, match="only the `.phi` knob"):
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
