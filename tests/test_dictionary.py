import numpy as np
import pytest

from polygram.dictionary import Dictionary, Feature, _default_betas, _default_hea_theta
from polygram.encoding import HEA_Rung2


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


def _hea_animals(encoding=None):
    encoding = encoding or HEA_Rung2(depth=2)
    return Dictionary(
        name="HeaTiered",
        features=[
            Feature("a", "s1", beta=0.10, alpha=0.05, gamma=0.02),
            Feature("b", "s1", beta=0.11, alpha=0.04, gamma=0.03),
            Feature("c", "s2", beta=1.20, alpha=1.10, gamma=1.00),
        ],
        hierarchy={"s1": ["a", "b"], "s2": ["c"]},
        encoding=encoding,
    )


class TestHEADictionary:
    def test_feature_default_theta_is_none(self):
        f = Feature("a", "s1", beta=0.1)
        assert f.theta is None

    def test_dictionary_accepts_hea_encoding(self):
        d = _hea_animals()
        assert isinstance(d.encoding, HEA_Rung2)
        assert d.encoding.depth == 2

    def test_feature_with_well_shaped_theta_is_accepted(self):
        encoding = HEA_Rung2(depth=2)
        theta = np.zeros(encoding.theta_shape)
        Dictionary(
            name="Ok",
            features=[
                Feature("a", "s1", beta=0.0, theta=theta),
                Feature("b", "s2", beta=0.0),
            ],
            hierarchy={"s1": ["a"], "s2": ["b"]},
            encoding=encoding,
        )

    def test_feature_with_wrong_shape_theta_raises(self):
        encoding = HEA_Rung2(depth=3, rotations=("Ry", "Rz"))
        bad = np.zeros((2, 2, 3))
        with pytest.raises(ValueError, match=r"a.*\(2, 2, 3\).*\(2, 3, 3\)"):
            Dictionary(
                name="Bad",
                features=[Feature("a", "s1", beta=0.0, theta=bad)],
                hierarchy={"s1": ["a"]},
                encoding=encoding,
            )

    def test_default_hea_theta_lays_knobs_on_first_layer(self):
        encoding = HEA_Rung2(depth=2, rotations=("Ry", "Rz"))
        f = Feature("a", "s1", beta=0.5, alpha=0.1, gamma=0.2, phi=0.7)
        theta = _default_hea_theta(f, encoding)
        assert theta.shape == (2, 2, 3)
        assert theta[0, 0, 0] == pytest.approx(0.1)
        assert theta[0, 0, 1] == pytest.approx(0.5)
        assert theta[0, 0, 2] == pytest.approx(0.2)
        assert theta[1, 0, 1] == pytest.approx(0.7)
        assert theta[0, 1, :].sum() == 0.0
        assert theta[1, 1, :].sum() == 0.0

    def test_gram_dispatches_to_hea_helper(self):
        d = _hea_animals()
        gram = d.gram()
        assert gram.shape == (3, 3)
        assert np.iscomplexobj(gram)
        for i in range(3):
            assert abs(gram[i, i]) == pytest.approx(1.0, abs=1e-9)

    def test_tier_separation_positive_for_clearly_tiered_fixture(self):
        d = _hea_animals()
        sep = d.tier_separation()
        assert sep is not None
        assert sep > 0.5

    def test_tier_separation_returns_none_for_all_singletons(self):
        encoding = HEA_Rung2(depth=2)
        d = Dictionary(
            name="Singletons",
            features=[
                Feature("a", "s1", beta=0.1),
                Feature("b", "s2", beta=0.2),
                Feature("c", "s3", beta=0.3),
            ],
            hierarchy={"s1": ["a"], "s2": ["b"], "s3": ["c"]},
            encoding=encoding,
        )
        assert d.tier_separation() is None


class TestWithKnob:
    def test_phi_path_works_on_mps(self):
        d = _animals()
        d2 = d.with_knob("dog_at_rest.phi", 0.7)
        assert d.feature("dog_at_rest").phi == 0.0
        assert d2.feature("dog_at_rest").phi == pytest.approx(0.7)
        assert d2.feature("bird_at_rest").phi == d.feature("bird_at_rest").phi

    def test_phi_path_works_on_hea(self):
        d = _hea_animals()
        d2 = d.with_knob("a.phi", 0.5)
        assert d.feature("a").phi == 0.0
        assert d2.feature("a").phi == pytest.approx(0.5)
        for other in ("b", "c"):
            assert d2.feature(other).phi == d.feature(other).phi

    def test_theta_path_rejected_on_mps(self):
        d = _animals()
        with pytest.raises(ValueError, match=r"HEA-only"):
            d.with_knob("dog_at_rest.theta[0,0,1]", 0.3)

    def test_theta_path_writes_single_slot_on_hea(self):
        d = _hea_animals()
        original_default = _default_hea_theta(d.feature("a"), d.encoding)
        d2 = d.with_knob("a.theta[1,0,1]", 0.5)
        theta = d2.feature("a").theta
        assert theta is not None
        assert theta.shape == (2, 2, 3)
        assert theta[1, 0, 1] == pytest.approx(0.5)
        for r in range(theta.shape[0]):
            for q in range(theta.shape[2]):
                if (r, 0, q) == (1, 0, 1):
                    continue
                assert theta[r, 0, q] == pytest.approx(original_default[r, 0, q])
        assert d.feature("a").theta is None

    def test_theta_path_lifts_existing_tensor(self):
        encoding = HEA_Rung2(depth=2)
        baseline = np.full(encoding.theta_shape, 0.25)
        d = Dictionary(
            name="Lift",
            features=[
                Feature("a", "s1", beta=0.0, theta=baseline.copy()),
                Feature("b", "s2", beta=0.0),
            ],
            hierarchy={"s1": ["a"], "s2": ["b"]},
            encoding=encoding,
        )
        d2 = d.with_knob("a.theta[0,1,2]", 1.0)
        new_theta = d2.feature("a").theta
        assert new_theta[0, 1, 2] == pytest.approx(1.0)
        assert new_theta is not d.feature("a").theta
        assert d.feature("a").theta[0, 1, 2] == pytest.approx(0.25)

    def test_out_of_range_slot_raises(self):
        d = _hea_animals()
        with pytest.raises(ValueError, match=r"\(2, 0, 0\).*theta_shape=\(2, 2, 3\)"):
            d.with_knob("a.theta[2,0,0]", 0.0)

    def test_malformed_path_raises(self):
        d = _hea_animals()
        with pytest.raises(ValueError, match=r"grammar"):
            d.with_knob("a.theta", 0.0)
        with pytest.raises(ValueError, match=r"grammar"):
            d.with_knob("a", 0.0)

    def test_unknown_feature_raises(self):
        d = _hea_animals()
        with pytest.raises(ValueError, match="nope"):
            d.with_knob("nope.phi", 0.0)


class TestClusterKnob:
    def test_cluster_phi_fans_out_across_siblings(self):
        d = _hea_animals()
        d2 = d.with_knob("s1.phi", 0.7)
        assert d2.feature("a").phi == pytest.approx(0.7)
        assert d2.feature("b").phi == pytest.approx(0.7)
        assert d2.feature("c").phi == d.feature("c").phi

    def test_cluster_theta_fans_out_on_hea(self):
        d = _hea_animals()
        d2 = d.with_knob("s1.theta[0,0,0]", 1.5)
        for member in ("a", "b"):
            theta = d2.feature(member).theta
            assert theta is not None
            assert theta[0, 0, 0] == pytest.approx(1.5)
            original = _default_hea_theta(d.feature(member), d.encoding)
            for r in range(theta.shape[0]):
                for layer in range(theta.shape[1]):
                    for q in range(theta.shape[2]):
                        if (r, layer, q) == (0, 0, 0):
                            continue
                        assert theta[r, layer, q] == pytest.approx(
                            original[r, layer, q]
                        )
        assert d2.feature("c").theta is None

    def test_cluster_theta_rejected_on_mps(self):
        d = _animals()
        with pytest.raises(ValueError, match="HEA-only"):
            d.with_knob("dogs.theta[0,0,1]", 0.3)

    def test_unknown_identifier_rejected(self):
        d = _hea_animals()
        with pytest.raises(ValueError, match="cats"):
            d.with_knob("cats.phi", 0.0)

    def test_feature_cluster_collision_rejected_at_construction(self):
        with pytest.raises(ValueError, match="name collision"):
            Dictionary(
                name="Bad",
                features=[
                    Feature("dogs", "dogs", beta=-0.5),
                    Feature("bird_hawk", "birds", beta=0.5),
                ],
                hierarchy={"dogs": ["dogs"], "birds": ["bird_hawk"]},
            )

    def test_cluster_theta_out_of_range_names_cluster(self):
        d = _hea_animals()
        with pytest.raises(ValueError, match=r"s1\.theta\[2,0,0\]"):
            d.with_knob("s1.theta[2,0,0]", 0.0)

    def test_mps_cluster_shared_phi_preserves_within_cluster_gram(self):
        # Bit-for-bit invariant: MPSRung1 + cluster-shared phi + sibling
        # pre-mutation phi agreement. Final-Rz factorization makes the
        # cluster-shared rotation cancel in `<a|b>`.
        d = Dictionary(
            name="MpsClean",
            features=[
                Feature("dog_a", "dogs", beta=-0.5),
                Feature("dog_b", "dogs", beta=-0.5),
                Feature("bird_a", "birds", beta=0.5),
                Feature("bird_b", "birds", beta=0.5),
            ],
            hierarchy={"dogs": ["dog_a", "dog_b"], "birds": ["bird_a", "bird_b"]},
        )
        before = d.gram()
        d2 = d.with_knob("dogs.phi", 0.4)
        after = d2.gram()
        for cluster in ("dogs", "birds"):
            i, j = (d.feature_index(m) for m in d.hierarchy[cluster])
            assert abs(after[i, j] - before[i, j]) < 1e-9

    def test_hea_cluster_shared_theta_may_drift(self):
        # Bit-for-bit preservation on HEA requires fully identical sibling
        # baselines. _hea_animals has α=0.05/γ=0.02 vs α=0.04/γ=0.03 —
        # cluster-shared θ on slot (0,0,0) shifts the within-cluster Gram.
        d = _hea_animals()
        before = d.gram()
        d2 = d.with_knob("s1.theta[0,0,0]", 1.0)
        after = d2.gram()
        i_a = d.feature_index("a")
        i_b = d.feature_index("b")
        # With diverse sibling baselines, the unitarity argument doesn't
        # apply — drift is allowed. Just confirm the call runs.
        assert d2.feature("a").theta is not None
        assert d2.feature("b").theta is not None
        # Drift may be small but nonzero; do NOT assert equality.
        _ = abs(after[i_a, i_b] - before[i_a, i_b])
