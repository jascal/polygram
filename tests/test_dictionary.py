import pytest

from polygram.dictionary import Dictionary, Feature, _default_betas


def _animals():
    return Dictionary(
        name="Animals",
        features=[
            Feature("dog_at_rest", "dogs", beta=-0.5),
            Feature("dog_in_motion", "dogs", beta=-0.5, phi=1.5707963),
            Feature("bird_at_rest", "birds", beta=0.5),
            Feature("bird_in_motion", "birds", beta=0.5, phi=1.5707963),
        ],
        hierarchy={
            "dogs": ["dog_at_rest", "dog_in_motion"],
            "birds": ["bird_at_rest", "bird_in_motion"],
        },
    )


def test_well_formed_dictionary_constructs():
    d = _animals()
    assert len(d.features) == 4
    assert d.feature_index("bird_at_rest") == 2


def test_feature_in_unknown_cluster_raises():
    with pytest.raises(ValueError, match="not a key in hierarchy"):
        Dictionary(
            name="Bad",
            features=[Feature("x", "missing", beta=0.0)],
            hierarchy={"present": ["x"]},
        )


def test_feature_listed_in_two_clusters_raises():
    with pytest.raises(ValueError, match="listed in two clusters"):
        Dictionary(
            name="Bad",
            features=[Feature("x", "a", beta=0.0)],
            hierarchy={"a": ["x"], "b": ["x"]},
        )


def test_duplicate_feature_name_raises():
    with pytest.raises(ValueError, match="duplicate feature"):
        Dictionary(
            name="Bad",
            features=[Feature("x", "a", beta=0.0), Feature("x", "a", beta=0.5)],
            hierarchy={"a": ["x"]},
        )


def test_unknown_name_in_hierarchy_raises():
    with pytest.raises(ValueError, match="unknown feature"):
        Dictionary(
            name="Bad",
            features=[Feature("x", "a", beta=0.0)],
            hierarchy={"a": ["x", "ghost"]},
        )


def test_feature_cluster_inconsistent_with_hierarchy_raises():
    with pytest.raises(ValueError, match="not listed under hierarchy"):
        Dictionary(
            name="Bad",
            features=[Feature("x", "a", beta=0.0)],
            hierarchy={"a": [], "b": ["x"]},
        )


def test_invalid_machine_name_raises():
    with pytest.raises(ValueError, match="must match"):
        Dictionary(name="bad name", features=[], hierarchy={})


def test_default_betas_two_clusters():
    out = _default_betas(["a", "b"])
    assert out == {"a": -0.5, "b": 0.5}


def test_default_betas_three_clusters():
    out = _default_betas(["a", "b", "c"])
    assert out["a"] == pytest.approx(-0.5)
    assert out["b"] == pytest.approx(0.0)
    assert out["c"] == pytest.approx(0.5)


def test_with_default_angles_helper():
    d = Dictionary.with_default_angles(
        name="Animals",
        hierarchy={"dogs": ["d1", "d2"], "birds": ["b1", "b2"]},
    )
    assert all(f.alpha == 0.0 and f.gamma == 0.0 and f.phi == 0.0 for f in d.features)
    assert d.feature("d1").beta == pytest.approx(-0.5)
    assert d.feature("b2").beta == pytest.approx(0.5)


def test_with_phi_returns_modified_copy():
    d = _animals()
    d2 = d.with_phi("dog_at_rest", 0.7)
    assert d.feature("dog_at_rest").phi == 0.0
    assert d2.feature("dog_at_rest").phi == pytest.approx(0.7)
    assert d2.feature("bird_at_rest").phi == d.feature("bird_at_rest").phi
