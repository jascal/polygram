"""Tests for `polygram.experts` — cluster_experts + ExpertDictionary."""

from __future__ import annotations

import numpy as np
import pytest

from polygram.dictionary import Dictionary, Feature
from polygram.encoding import MPSRung1
from polygram.experts import ExpertDictionary, cluster_experts


def _planted_two_cluster_inputs() -> tuple[Dictionary, np.ndarray]:
    """Build a flat Dictionary + decoder matrix with two clearly
    antipodal clusters of 4 features each. Cosine clustering on the
    decoder vectors should split them along the planted boundary."""
    rng = np.random.default_rng(42)
    d_model = 6

    base_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    base_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    rows = []
    feats = []
    hierarchy: dict[str, list[str]] = {"a": [], "b": []}
    for i in range(4):
        v = base_a + rng.normal(scale=0.02, size=d_model)
        v /= np.linalg.norm(v)
        rows.append(v)
        name = f"a{i}"
        feats.append(Feature(name=name, cluster="a", beta=0.5))
        hierarchy["a"].append(name)
    for i in range(4):
        v = base_b + rng.normal(scale=0.02, size=d_model)
        v /= np.linalg.norm(v)
        rows.append(v)
        name = f"b{i}"
        feats.append(Feature(name=name, cluster="b", beta=-0.5))
        hierarchy["b"].append(name)

    decoder = np.stack(rows)
    dictionary = Dictionary(
        name="planted",
        features=feats,
        hierarchy=hierarchy,
        encoding=MPSRung1(),
    )
    return dictionary, decoder


def test_cluster_experts_recovers_planted_clusters():
    dictionary, decoder = _planted_two_cluster_inputs()

    experts = cluster_experts(
        dictionary,
        decoder,
        method="cosine",
        coherence_threshold=0.5,
        max_features_per_expert=4,
    )

    assert isinstance(experts, ExpertDictionary)
    assert experts.n_features == 8
    expert_groups = [{f.name for f in e.features} for e in experts.experts]
    assert {"a0", "a1", "a2", "a3"} in expert_groups
    assert {"b0", "b1", "b2", "b3"} in expert_groups


def test_route_top_k_returns_dominant_expert():
    dictionary, decoder = _planted_two_cluster_inputs()
    experts = cluster_experts(
        dictionary, decoder, coherence_threshold=0.5,
        max_features_per_expert=4,
    )

    a_expert_idx = next(
        i for i, e in enumerate(experts.experts)
        if {f.name for f in e.features} == {"a0", "a1", "a2", "a3"}
    )
    b_expert_idx = 1 - a_expert_idx

    activations = np.zeros(8)
    a_feature_idxs = [
        i for i, f in enumerate(dictionary.features) if f.cluster == "a"
    ]
    activations[a_feature_idxs] = 1.0

    assert experts.route(activations, top_k=1) == [a_expert_idx]
    top2 = experts.route(activations, top_k=2)
    assert top2[0] == a_expert_idx
    assert top2[1] == b_expert_idx


def test_route_validates_input_shape_and_top_k():
    dictionary, decoder = _planted_two_cluster_inputs()
    experts = cluster_experts(
        dictionary, decoder, coherence_threshold=0.5,
        max_features_per_expert=4,
    )

    with pytest.raises(ValueError, match="activations.shape"):
        experts.route(np.zeros(7), top_k=1)

    with pytest.raises(ValueError, match="top_k"):
        experts.route(np.zeros(8), top_k=0)

    with pytest.raises(ValueError, match="top_k"):
        experts.route(np.zeros(8), top_k=experts.n_experts + 1)

    with pytest.raises(ValueError, match="ndarray"):
        experts.route([0.0] * 8, top_k=1)  # type: ignore[arg-type]


def test_method_coactivation_raises_not_implemented():
    dictionary, decoder = _planted_two_cluster_inputs()
    with pytest.raises(NotImplementedError, match="co_firing"):
        cluster_experts(dictionary, decoder, method="coactivation")


def test_unknown_method_raises_value_error():
    dictionary, decoder = _planted_two_cluster_inputs()
    with pytest.raises(ValueError, match="unknown method"):
        cluster_experts(dictionary, decoder, method="louvain")  # type: ignore[arg-type]


def test_expert_blocks_keep_source_dictionary_primitives():
    dictionary, decoder = _planted_two_cluster_inputs()
    experts = cluster_experts(
        dictionary, decoder, coherence_threshold=0.5,
        max_features_per_expert=4,
    )

    for expert in experts.experts:
        assert isinstance(expert, Dictionary)
        gram = expert.gram()
        n = len(expert.features)
        assert gram.shape == (n, n)
        diag = np.diag(gram)
        np.testing.assert_allclose(diag, 1.0, atol=1e-9)


def test_decoder_axis_mismatch_raises():
    dictionary, _ = _planted_two_cluster_inputs()
    with pytest.raises(ValueError, match="decoder_vectors first axis"):
        cluster_experts(dictionary, np.zeros((7, 6)))


def test_expert_dictionary_partition_invariant_enforced():
    dictionary, _ = _planted_two_cluster_inputs()

    expert_a = Dictionary(
        name="a", features=list(dictionary.features[:4]),
        hierarchy={"a": [f.name for f in dictionary.features[:4]]},
        encoding=MPSRung1(),
    )
    expert_b = Dictionary(
        name="b", features=list(dictionary.features[4:]),
        hierarchy={"b": [f.name for f in dictionary.features[4:]]},
        encoding=MPSRung1(),
    )

    with pytest.raises(ValueError, match="length"):
        ExpertDictionary(
            experts=(expert_a, expert_b),
            source=dictionary,
            _feature_to_expert=(0, 0, 0, 0, 1, 1, 1),
        )

    # Duplicate feature across experts.
    dup_b = Dictionary(
        name="dup", features=[dictionary.features[0], *dictionary.features[5:]],
        hierarchy={
            "a": [dictionary.features[0].name],
            "b": [f.name for f in dictionary.features[5:]],
        },
        encoding=MPSRung1(),
    )
    with pytest.raises(ValueError, match="appears in more than one"):
        ExpertDictionary(
            experts=(expert_a, dup_b),
            source=dictionary,
            _feature_to_expert=(0, 0, 0, 0, 1, 1, 1, 1),
        )
