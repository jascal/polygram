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


# ---------------------------------------------------------------------------
# Rung4 gram dispatch (add-rung4-encoding-mvp §3)
# ---------------------------------------------------------------------------


class TestRung4GramDispatch:
    def _two_feature_dict(self, *, encoding, alpha=(-0.5, 0.5),
                          theta_amp=(0.0, 0.0), psi_aux=(0.0, 0.0),
                          theta_amp_b=(0.0, 0.0), psi_amp_b=(0.0, 0.0)):
        from polygram import Dictionary, Feature

        feats = [
            Feature(
                name=f"f{i}",
                cluster="g",
                beta=alpha[i],
                theta_amp=theta_amp[i],
                psi_aux=psi_aux[i],
                theta_amp_b=theta_amp_b[i],
                psi_amp_b=psi_amp_b[i],
            )
            for i in (0, 1)
        ]
        return Dictionary(
            name="dict",
            features=feats,
            hierarchy={"g": [f.name for f in feats]},
            encoding=encoding,
        )

    def test_default_knobs_match_mpsrung1_gram(self):
        # Rung4 with all amp knobs at 0 produces a gram identical to
        # MPSRung1 on the same (α, β, γ, φ). This is the load-bearing
        # "default reduces to MPS" invariant.
        import numpy as np

        from polygram.encoding import MPSRung1, Rung4

        d_mps = self._two_feature_dict(encoding=MPSRung1())
        d_r4 = self._two_feature_dict(encoding=Rung4())
        g_mps = d_mps.gram()
        g_r4 = d_r4.gram()
        np.testing.assert_allclose(g_r4, g_mps, atol=1e-12)

    def test_nondefault_q3_amp_knobs_change_gram(self):
        import numpy as np

        from polygram.encoding import MPSRung1, Rung4

        d_mps = self._two_feature_dict(encoding=MPSRung1())
        d_r4 = self._two_feature_dict(
            encoding=Rung4(),
            theta_amp=(0.3, 0.7),
            psi_aux=(0.1, 0.4),
        )
        g_mps = d_mps.gram()
        g_r4 = d_r4.gram()
        # On-diagonal still 1 (state normalised).
        np.testing.assert_allclose(np.abs(g_r4.diagonal()), 1.0, atol=1e-12)
        # Off-diagonal differs from MPS path.
        assert not np.allclose(g_r4, g_mps, atol=1e-9)

    def test_nondefault_q4_amp_knobs_change_gram(self):
        import numpy as np

        from polygram.encoding import MPSRung1, Rung4

        d_mps = self._two_feature_dict(encoding=MPSRung1())
        # Only q4 amp knobs vary; q3 amp stays at default.
        d_r4 = self._two_feature_dict(
            encoding=Rung4(),
            theta_amp_b=(0.3, 0.7),
            psi_amp_b=(0.1, 0.4),
        )
        g_mps = d_mps.gram()
        g_r4 = d_r4.gram()
        np.testing.assert_allclose(np.abs(g_r4.diagonal()), 1.0, atol=1e-12)
        assert not np.allclose(g_r4, g_mps, atol=1e-9)

    def test_q3_q4_factorise(self):
        # The Rung4 amp factor for a pair (i, j) equals
        # _single_qubit_overlap(q3_i, q3_j) * _single_qubit_overlap(q4_i, q4_j).
        # We verify this end-to-end: the gram[i,j] / mps_gram[i,j]
        # should equal the product of the two single-qubit overlaps.
        import numpy as np

        from polygram.encoding import (
            _single_qubit_overlap,
            MPSRung1,
            Rung4,
        )

        # Pick non-default knobs on BOTH q3 and q4 to exercise both
        # factors.
        d_mps = self._two_feature_dict(encoding=MPSRung1())
        d_r4 = self._two_feature_dict(
            encoding=Rung4(),
            theta_amp=(0.3, 0.5),
            psi_aux=(0.1, 0.2),
            theta_amp_b=(0.4, 0.6),
            psi_amp_b=(0.3, 0.7),
        )
        g_mps = d_mps.gram()
        g_r4 = d_r4.gram()
        # Off-diagonal entry (0, 1).
        expected_factor = (
            _single_qubit_overlap(0.3, 0.1, 0.5, 0.2)
            * _single_qubit_overlap(0.4, 0.3, 0.6, 0.7)
        )
        # gram_r4[i,j] = gram_mps[i,j] * amp_factor[i,j]
        observed_factor = g_r4[0, 1] / g_mps[0, 1]
        np.testing.assert_allclose(
            observed_factor, expected_factor, atol=1e-12
        )


class TestRung5FeatureAmpKnobs:
    def test_feature_default_amp_knobs_is_empty_tuple(self):
        from polygram import Feature

        f = Feature(name="f", cluster="c", beta=0.1)
        assert f.amp_knobs == ()

    def test_feature_explicit_amp_knobs(self):
        from polygram import Feature

        amp = ((0.1, 0.2), (0.3, 0.4))
        f = Feature(name="f", cluster="c", beta=0.1, amp_knobs=amp)
        assert f.amp_knobs == amp

    def test_with_default_amp_knobs_pads_for_rung5(self):
        from polygram import Feature
        from polygram.encoding import Rung5

        f = Feature(name="f", cluster="c", beta=0.1)
        padded = f.with_default_amp_knobs(Rung5(n_amp_qubits=4))
        assert padded.amp_knobs == ((0.0, 0.0),) * 4
        # Other fields unchanged.
        assert padded.name == f.name
        assert padded.cluster == f.cluster
        assert padded.beta == f.beta

    def test_with_default_amp_knobs_preserves_populated_amp_knobs(self):
        from polygram import Feature
        from polygram.encoding import Rung5

        amp = ((0.1, 0.2), (0.3, 0.4))
        f = Feature(name="f", cluster="c", beta=0.1, amp_knobs=amp)
        result = f.with_default_amp_knobs(Rung5(n_amp_qubits=2))
        assert result.amp_knobs == amp

    def test_with_default_amp_knobs_is_noop_for_non_rung5(self):
        from polygram import Feature
        from polygram.encoding import HEA_Rung2, MPSRung1, Rung3, Rung4

        f = Feature(name="f", cluster="c", beta=0.1)
        for encoding in (
            MPSRung1(),
            Rung3(),
            Rung4(),
            HEA_Rung2(n_qubits=3, depth=1),
        ):
            assert f.with_default_amp_knobs(encoding) is f


class TestRung5DictionaryValidation:
    def _make_dict(
        self,
        *,
        encoding,
        amp_knobs=(),
        amp_knobs_per_feature=None,
    ):
        from polygram import Dictionary, Feature

        if amp_knobs_per_feature is None:
            amp_knobs_per_feature = [amp_knobs, amp_knobs]
        feats = [
            Feature(
                name=f"f{i}",
                cluster="g",
                beta=0.1 * (i - 0.5),
                amp_knobs=amp_knobs_per_feature[i],
            )
            for i in (0, 1)
        ]
        return Dictionary(
            name="d",
            features=feats,
            hierarchy={"g": [f.name for f in feats]},
            encoding=encoding,
        )

    def test_correctly_sized_amp_knobs_accepted(self):
        from polygram.encoding import Rung5

        d = self._make_dict(
            encoding=Rung5(n_amp_qubits=3),
            amp_knobs=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        )
        assert len(d.features) == 2

    def test_mismatched_amp_knobs_length_rejected(self):
        from polygram.encoding import Rung5

        with pytest.raises(ValueError, match="amp_knobs has length"):
            self._make_dict(
                encoding=Rung5(n_amp_qubits=3),
                amp_knobs=((0.0, 0.0), (0.0, 0.0)),
            )

    def test_empty_amp_knobs_on_rung5_rejected(self):
        from polygram.encoding import Rung5

        with pytest.raises(ValueError, match="amp_knobs has length"):
            self._make_dict(encoding=Rung5(n_amp_qubits=2), amp_knobs=())

    def test_non_tuple_entry_in_amp_knobs_rejected(self):
        from polygram.encoding import Rung5

        with pytest.raises(ValueError, match="must be a 2-tuple"):
            self._make_dict(
                encoding=Rung5(n_amp_qubits=2),
                amp_knobs=((0.0, 0.0), [0.0, 0.0]),  # list, not tuple
            )

    def test_non_rung5_dictionary_with_amp_knobs_accepted_and_ignored(self):
        # Non-Rung5 dicts ignore amp_knobs; Dictionary construction
        # SHALL succeed regardless of amp_knobs content.
        from polygram.encoding import MPSRung1

        d = self._make_dict(
            encoding=MPSRung1(),
            amp_knobs=((0.1, 0.2),),  # populated, "wrong length" for any k
        )
        assert len(d.features) == 2

    def test_existing_non_rung5_dictionary_round_trip(self):
        # Existing MPSRung1/Rung3/Rung4 fixtures don't populate
        # amp_knobs; the default `()` SHALL round-trip cleanly.
        from polygram import Dictionary, Feature
        from polygram.encoding import MPSRung1, Rung3, Rung4

        for encoding in (MPSRung1(), Rung3(), Rung4()):
            feats = [
                Feature(name="f0", cluster="g", beta=-0.1),
                Feature(name="f1", cluster="g", beta=0.1),
            ]
            d = Dictionary(
                name="d",
                features=feats,
                hierarchy={"g": ["f0", "f1"]},
                encoding=encoding,
            )
            for f in d.features:
                assert f.amp_knobs == ()


class TestRung5GramDispatch:
    def _two_feature_dict(
        self,
        *,
        encoding,
        alpha=(-0.3, 0.3),
        beta=(-0.2, 0.2),
        gamma=(0.05, -0.05),
        phi=(0.2, -0.2),
        amp_knobs_per_feature=None,
    ):
        from polygram import Dictionary, Feature

        feats = []
        for i in (0, 1):
            kwargs = {
                "name": f"f{i}",
                "cluster": "g",
                "beta": beta[i],
                "alpha": alpha[i],
                "gamma": gamma[i],
                "phi": phi[i],
            }
            if amp_knobs_per_feature is not None:
                kwargs["amp_knobs"] = amp_knobs_per_feature[i]
            feats.append(Feature(**kwargs))
        return Dictionary(
            name="d",
            features=feats,
            hierarchy={"g": [f.name for f in feats]},
            encoding=encoding,
        )

    def test_default_amp_knobs_match_mpsrung1_gram(self):
        # Rung5 with every (θ_i, ψ_i) = (0, 0) produces a gram equal
        # to MPSRung1 on the same (α, β, γ, φ) — the load-bearing
        # "default reduces to MPS" invariant generalised across k.
        import numpy as np

        from polygram.encoding import MPSRung1, Rung5

        for k in (1, 2, 3, 4):
            default_amp = ((0.0, 0.0),) * k
            d_mps = self._two_feature_dict(encoding=MPSRung1())
            d_r5 = self._two_feature_dict(
                encoding=Rung5(n_amp_qubits=k),
                amp_knobs_per_feature=[default_amp, default_amp],
            )
            g_mps = d_mps.gram()
            g_r5 = d_r5.gram()
            np.testing.assert_allclose(
                g_r5, g_mps.astype(complex), atol=1e-12,
                err_msg=f"k={k}",
            )

    def test_nondefault_amp_knobs_differ_from_mpsrung1(self):
        import numpy as np

        from polygram.encoding import MPSRung1, Rung5

        d_mps = self._two_feature_dict(encoding=MPSRung1())
        d_r5 = self._two_feature_dict(
            encoding=Rung5(n_amp_qubits=3),
            amp_knobs_per_feature=[
                ((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)),
                ((0.15, 0.25), (0.35, 0.45), (0.55, 0.65)),
            ],
        )
        g_mps = d_mps.gram()
        g_r5 = d_r5.gram()
        np.testing.assert_allclose(
            np.abs(g_r5.diagonal()), 1.0, atol=1e-12
        )
        assert not np.allclose(g_r5, g_mps.astype(complex), atol=1e-9)

    def test_gram_factorises_through_single_qubit_overlaps(self):
        # Off-diagonal entry of the Rung5 gram equals the MPSRung1
        # gram entry times the k-fold product of single-qubit
        # overlaps — same factorisation pattern as Rung4 generalised
        # over k qubits.
        from polygram.encoding import (
            MPSRung1,
            Rung5,
            _single_qubit_overlap,
        )

        amp = [
            ((0.3, 0.1), (0.5, 0.2), (0.7, 0.4)),
            ((0.4, 0.0), (0.6, 0.5), (0.8, 0.3)),
        ]
        d_mps = self._two_feature_dict(encoding=MPSRung1())
        d_r5 = self._two_feature_dict(
            encoding=Rung5(n_amp_qubits=3),
            amp_knobs_per_feature=amp,
        )
        g_mps = d_mps.gram()
        g_r5 = d_r5.gram()
        expected_factor = complex(1.0, 0.0)
        for (ta, pa), (tb, pb) in zip(amp[0], amp[1]):
            expected_factor *= _single_qubit_overlap(ta, pa, tb, pb)
        observed_factor = g_r5[0, 1] / g_mps[0, 1]
        assert abs(observed_factor - expected_factor) < 1e-12

    def test_gram_is_psd_and_symmetric(self):
        import numpy as np

        from polygram.encoding import Rung5

        d = self._two_feature_dict(
            encoding=Rung5(n_amp_qubits=2),
            amp_knobs_per_feature=[
                ((0.3, 0.1), (0.5, 0.2)),
                ((0.4, 0.0), (0.6, 0.5)),
            ],
        )
        g = d.gram()
        # Symmetric (Hermitian for complex)
        np.testing.assert_allclose(g, g.T.conj(), atol=1e-12)
        # PSD: real eigenvalues, all non-negative (within FP noise)
        eigs = np.linalg.eigvalsh((g + g.conj().T) / 2)
        assert (eigs > -1e-10).all(), eigs


class TestRung5WithKnob:
    def _rung5_dict(self, k=2, n_features=2):
        from polygram import Dictionary, Feature
        from polygram.encoding import Rung5

        default_amp = ((0.0, 0.0),) * k
        feats = [
            Feature(
                name=f"f{i}", cluster="g", beta=0.1 * (i - 0.5),
                amp_knobs=default_amp,
            )
            for i in range(n_features)
        ]
        return Dictionary(
            name="d",
            features=feats,
            hierarchy={"g": [f.name for f in feats]},
            encoding=Rung5(n_amp_qubits=k),
        )

    def test_set_amp_knobs_theta(self):
        d = self._rung5_dict(k=3)
        d2 = d.with_knob("f0.amp_knobs[1].theta", 0.7)
        assert d2.features[0].amp_knobs == (
            (0.0, 0.0), (0.7, 0.0), (0.0, 0.0)
        )
        # Other feature unchanged
        assert d2.features[1].amp_knobs == ((0.0, 0.0),) * 3

    def test_set_amp_knobs_psi(self):
        d = self._rung5_dict(k=2)
        d2 = d.with_knob("f1.amp_knobs[0].psi", 1.5)
        assert d2.features[1].amp_knobs == ((0.0, 1.5), (0.0, 0.0))

    def test_cluster_shared_amp_knobs(self):
        d = self._rung5_dict(k=2, n_features=3)
        d2 = d.with_knob("g.amp_knobs[1].theta", 0.5)
        for f in d2.features:
            assert f.amp_knobs == ((0.0, 0.0), (0.5, 0.0))

    def test_index_out_of_range_rejected(self):
        d = self._rung5_dict(k=3)
        with pytest.raises(ValueError, match="amp index 5 is outside"):
            d.with_knob("f0.amp_knobs[5].theta", 0.5)

    def test_amp_knobs_path_on_non_rung5_rejected(self):
        from polygram import Dictionary, Feature
        from polygram.encoding import MPSRung1, Rung4

        for encoding in (
            MPSRung1(),
            Rung4(),
        ):
            d = Dictionary(
                name="d",
                features=[Feature(name="f0", cluster="g", beta=0.1)],
                hierarchy={"g": ["f0"]},
                encoding=encoding,
            )
            with pytest.raises(ValueError, match="require"):
                d.with_knob("f0.amp_knobs[0].theta", 0.5)

    def test_with_knob_chains(self):
        d = self._rung5_dict(k=2)
        d2 = (
            d.with_knob("f0.amp_knobs[0].theta", 0.3)
            .with_knob("f0.amp_knobs[0].psi", 0.6)
            .with_knob("f0.amp_knobs[1].theta", 0.9)
        )
        assert d2.features[0].amp_knobs == (
            (0.3, 0.6), (0.9, 0.0)
        )


class TestRung5MatchesRung4AtK2:
    """Internal consistency check: `Rung5(n_amp_qubits=2)` and Rung4
    are different encoding *classes* with different dispatch paths,
    but their amp branches are mathematically the same (both are
    products of two `_single_qubit_overlap` evaluations). The two
    grams MUST agree numerically on the same knobs.

    Documented in `docs/research/rung5-encoding.md`; not exposed as a
    public API equivalence."""

    def _build_pair_at_knobs(
        self,
        encoding,
        theta0_a, psi0_a, theta1_a, psi1_a,
        theta0_b, psi0_b, theta1_b, psi1_b,
    ):
        from polygram import Dictionary, Feature
        from polygram.encoding import Rung4, Rung5

        if isinstance(encoding, Rung4):
            feats = [
                Feature(
                    name="a", cluster="g", beta=-0.3, alpha=0.1,
                    gamma=0.05, phi=0.2,
                    theta_amp=theta0_a, psi_aux=psi0_a,
                    theta_amp_b=theta1_a, psi_amp_b=psi1_a,
                ),
                Feature(
                    name="b", cluster="g", beta=0.3, alpha=-0.1,
                    gamma=-0.05, phi=-0.2,
                    theta_amp=theta0_b, psi_aux=psi0_b,
                    theta_amp_b=theta1_b, psi_amp_b=psi1_b,
                ),
            ]
        else:
            assert isinstance(encoding, Rung5)
            feats = [
                Feature(
                    name="a", cluster="g", beta=-0.3, alpha=0.1,
                    gamma=0.05, phi=0.2,
                    amp_knobs=((theta0_a, psi0_a), (theta1_a, psi1_a)),
                ),
                Feature(
                    name="b", cluster="g", beta=0.3, alpha=-0.1,
                    gamma=-0.05, phi=-0.2,
                    amp_knobs=((theta0_b, psi0_b), (theta1_b, psi1_b)),
                ),
            ]
        return Dictionary(
            name="d",
            features=feats,
            hierarchy={"g": ["a", "b"]},
            encoding=encoding,
        )

    def test_grams_match_at_k2(self):
        import numpy as np

        from polygram.encoding import Rung4, Rung5

        knobs = (0.3, 0.1, 0.5, 0.2, 0.4, 0.0, 0.6, 0.5)
        g_r4 = self._build_pair_at_knobs(Rung4(), *knobs).gram()
        g_r5 = self._build_pair_at_knobs(
            Rung5(n_amp_qubits=2), *knobs
        ).gram()
        np.testing.assert_allclose(g_r4, g_r5, atol=1e-12)
