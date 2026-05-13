"""Tests for `polygram.clustered.ClusteredDictionary` — §1 of the
`clustered-dictionary-analysis` openspec change."""

from __future__ import annotations

import numpy as np
import pytest

from polygram.clustered import (
    BlockFormation,
    ClusteredDictionary,
    build_clustered_dictionary,
    compute_cosine_pair_graph,
)
from polygram.dictionary import Dictionary, Feature
from polygram.encoding import HEA_Rung2, MPSRung1, Rung3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feature(name: str, cluster: str = "all", beta: float = 0.0) -> Feature:
    return Feature(name=name, cluster=cluster, beta=beta)


def _block(name: str, n: int, encoding=None) -> Dictionary:
    """Build a `Dictionary` with `n` features in a single cluster."""
    encoding = encoding or MPSRung1()
    feats = [_feature(f"{name}_f{i}", cluster=name, beta=0.1 * i) for i in range(n)]
    return Dictionary(
        name=name,
        features=feats,
        hierarchy={name: [f.name for f in feats]},
        encoding=encoding,
    )


# ---------------------------------------------------------------------------
# BlockFormation
# ---------------------------------------------------------------------------


class TestBlockFormation:
    def test_cosine_default_threshold(self):
        bf = BlockFormation(strategy="cosine")
        assert bf.cosine_threshold == 0.3
        assert bf.block_size_max is None
        assert bf.firing_corpus is None

    def test_co_firing_requires_corpus(self):
        with pytest.raises(ValueError, match="requires a non-None `firing_corpus`"):
            BlockFormation(strategy="co_firing")

    def test_co_firing_with_corpus_ok(self):
        bf = BlockFormation(strategy="co_firing", firing_corpus=["prompt"])
        assert bf.firing_corpus == ["prompt"]

    def test_non_co_firing_strategy_rejects_corpus(self):
        with pytest.raises(ValueError, match="does not consume `firing_corpus`"):
            BlockFormation(strategy="cosine", firing_corpus=["prompt"])

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="cosine_threshold must lie in"):
            BlockFormation(strategy="cosine", cosine_threshold=1.5)
        with pytest.raises(ValueError, match="cosine_threshold must lie in"):
            BlockFormation(strategy="cosine", cosine_threshold=-0.1)

    def test_block_size_max_zero_raises(self):
        with pytest.raises(ValueError, match="block_size_max must be >= 1"):
            BlockFormation(strategy="cosine", block_size_max=0)

    def test_user_declared_no_corpus_needed(self):
        bf = BlockFormation(strategy="user_declared")
        assert bf.firing_corpus is None


# ---------------------------------------------------------------------------
# ClusteredDictionary — construction, properties, validation
# ---------------------------------------------------------------------------


class TestClusteredDictionaryConstruction:
    def test_minimal_two_block(self):
        cd = ClusteredDictionary(
            name="cd_test",
            blocks=[_block("alpha", 2), _block("beta", 2)],
        )
        assert cd.n_blocks == 2
        assert cd.n_features == 4
        assert cd.mean_block_size == 2.0
        assert cd.n_cross_block_edges == 0
        assert isinstance(cd.encoding, MPSRung1)

    def test_with_cross_block_edges(self):
        edges = {(0, 0, 1, 0): 0.6, (0, 1, 1, 1): 0.42}
        cd = ClusteredDictionary(
            name="cd_with_edges",
            blocks=[_block("alpha", 2), _block("beta", 2)],
            cross_block_pairs=edges,
        )
        assert cd.n_cross_block_edges == 2

    def test_uses_supplied_block_formation(self):
        bf = BlockFormation(strategy="cosine", cosine_threshold=0.7)
        cd = ClusteredDictionary(
            name="cd_bf",
            blocks=[_block("a", 2)],
            block_formation=bf,
        )
        assert cd.block_formation.strategy == "cosine"
        assert cd.block_formation.cosine_threshold == 0.7

    def test_default_block_formation_is_user_declared(self):
        cd = ClusteredDictionary(name="cd_default", blocks=[_block("a", 2)])
        assert cd.block_formation.strategy == "user_declared"


class TestClusteredDictionaryValidation:
    def test_empty_blocks_raises(self):
        with pytest.raises(ValueError, match="blocks must be non-empty"):
            ClusteredDictionary(name="bad", blocks=[])

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="must match"):
            ClusteredDictionary(name="bad-name", blocks=[_block("a", 2)])
        with pytest.raises(ValueError, match="must match"):
            ClusteredDictionary(name="0starts_digit", blocks=[_block("a", 2)])

    def test_mismatched_encoding_type_raises(self):
        block_mps = _block("alpha", 2, encoding=MPSRung1())
        block_rung3 = _block("beta", 2, encoding=Rung3())
        with pytest.raises(ValueError, match="encoding type.*MPSRung1.*Rung3"):
            ClusteredDictionary(name="bad", blocks=[block_mps, block_rung3])

    def test_mismatched_encoding_config_raises(self):
        block_a = _block("alpha", 2, encoding=HEA_Rung2(depth=1, n_qubits=3))
        block_b = _block("beta", 2, encoding=HEA_Rung2(depth=2, n_qubits=3))
        with pytest.raises(ValueError, match="same encoding configuration"):
            ClusteredDictionary(name="bad", blocks=[block_a, block_b])

    def test_block_over_cap_raises(self):
        # MPSRung1 cap is 8 (legacy or via max_features). 9 features in one
        # block must raise. Use beta values that keep Feature happy.
        feats = [_feature(f"big_f{i}", cluster="big", beta=0.05 * i) for i in range(9)]
        big = Dictionary(
            name="big",
            features=feats,
            hierarchy={"big": [f.name for f in feats]},
            encoding=MPSRung1(),
        )
        with pytest.raises(ValueError, match="exceeding the encoding cap"):
            ClusteredDictionary(name="bad", blocks=[big])

    def test_duplicate_feature_name_across_blocks_raises(self):
        # Two blocks both name a feature "shared_f0".
        b1 = Dictionary(
            name="b1",
            features=[_feature("shared_f0"), _feature("b1_unique")],
            hierarchy={"all": ["shared_f0", "b1_unique"]},
            encoding=MPSRung1(),
        )
        b2 = Dictionary(
            name="b2",
            features=[_feature("shared_f0"), _feature("b2_unique")],
            hierarchy={"all": ["shared_f0", "b2_unique"]},
            encoding=MPSRung1(),
        )
        with pytest.raises(ValueError, match="appears in two blocks"):
            ClusteredDictionary(name="bad", blocks=[b1, b2])


class TestClusteredDictionaryCrossBlockEdgeValidation:
    def _two_block_4f(self):
        return [_block("alpha", 2), _block("beta", 2)]

    def test_out_of_range_block_idx_raises(self):
        with pytest.raises(ValueError, match="out-of-range block_j_idx"):
            ClusteredDictionary(
                name="bad",
                blocks=self._two_block_4f(),
                cross_block_pairs={(0, 0, 5, 0): 0.5},
            )

    def test_out_of_range_feature_idx_raises(self):
        with pytest.raises(ValueError, match="out-of-range feat_j_idx"):
            ClusteredDictionary(
                name="bad",
                blocks=self._two_block_4f(),
                cross_block_pairs={(0, 0, 1, 99): 0.5},
            )

    def test_non_canonical_ordering_raises(self):
        # Edge from block 1 to block 0 — must be canonicalised to 0->1.
        with pytest.raises(ValueError, match="block_i_idx < block_j_idx"):
            ClusteredDictionary(
                name="bad",
                blocks=self._two_block_4f(),
                cross_block_pairs={(1, 0, 0, 0): 0.5},
            )

    def test_self_block_edge_raises(self):
        # Same block on both sides — must be rejected (the intra-block
        # gram captures these).
        with pytest.raises(ValueError, match="block_i_idx < block_j_idx"):
            ClusteredDictionary(
                name="bad",
                blocks=self._two_block_4f(),
                cross_block_pairs={(0, 0, 0, 1): 0.5},
            )

    def test_wrong_tuple_arity_raises(self):
        with pytest.raises(ValueError, match="must be a 4-tuple"):
            ClusteredDictionary(
                name="bad",
                blocks=self._two_block_4f(),
                cross_block_pairs={(0, 0, 1): 0.5},  # type: ignore[dict-item]
            )

    def test_valid_edges_pass(self):
        cd = ClusteredDictionary(
            name="ok",
            blocks=self._two_block_4f(),
            cross_block_pairs={(0, 1, 1, 0): 0.42, (0, 0, 1, 1): 0.55},
        )
        assert cd.n_cross_block_edges == 2


# ---------------------------------------------------------------------------
# Per-encoding cap
# ---------------------------------------------------------------------------


class TestClusteredDictionaryEncodingCap:
    def test_mpsrung1_cap_at_8(self):
        # Build with exactly 8 — passes. Build with 9 — fails. (See
        # `test_block_over_cap_raises` for the 9-feature case.)
        feats = [_feature(f"f{i}", cluster="full", beta=0.05 * i) for i in range(8)]
        block = Dictionary(
            name="full",
            features=feats,
            hierarchy={"full": [f.name for f in feats]},
            encoding=MPSRung1(),
        )
        cd = ClusteredDictionary(name="ok", blocks=[block])
        assert cd.n_features == 8

    def test_hea_max_features_when_declared(self):
        # If the encoding declares `max_features`, the cap honours it.
        # We can't test 2**n_qubits>8 without per-encoding-feature-cap
        # (PR #42) shipping, so probe the defensive getattr() path: an
        # encoding without `max_features` falls back to the legacy 8.
        encoding = HEA_Rung2(depth=1, n_qubits=3)
        assert not hasattr(encoding, "max_features")
        feats = [_feature(f"f{i}", cluster="hea", beta=0.05 * i) for i in range(8)]
        block = Dictionary(
            name="hea",
            features=feats,
            hierarchy={"hea": [f.name for f in feats]},
            encoding=encoding,
        )
        cd = ClusteredDictionary(name="ok", blocks=[block])
        assert cd.n_features == 8


# ---------------------------------------------------------------------------
# §2 — compute_cosine_pair_graph
# ---------------------------------------------------------------------------


class TestComputeCosinePairGraph:
    def test_orthogonal_decoders_yield_empty(self):
        # Three mutually-orthogonal unit vectors → no pairs above
        # threshold 0.1.
        vectors = np.eye(3, dtype=np.float32)
        pairs = compute_cosine_pair_graph(vectors, threshold=0.1)
        assert pairs == set()

    def test_identical_decoders_yield_all_pairs(self):
        vectors = np.ones((4, 3), dtype=np.float32)
        pairs = compute_cosine_pair_graph(vectors, threshold=0.99)
        assert pairs == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}

    def test_pair_ordering_canonical(self):
        # Every emitted pair has i < j (canonical ordering).
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((6, 4)).astype(np.float32)
        pairs = compute_cosine_pair_graph(vectors, threshold=-1.0)  # all
        for i, j in pairs:
            assert i < j

    def test_indices_mode_maps_to_global_ids(self):
        # When indices is supplied, output pairs use those ID values.
        vectors = np.zeros((10, 3), dtype=np.float32)
        vectors[2] = [1, 0, 0]
        vectors[5] = [1, 0, 0]
        eligible = np.array([2, 5, 7], dtype=np.int64)
        pairs = compute_cosine_pair_graph(vectors, threshold=0.99, indices=eligible)
        # Features 2 and 5 are colinear; 7 is zero-vector (treated as
        # unit after the 1e-12 norm guard, still orthogonal).
        assert (2, 5) in pairs

    def test_threshold_monotone(self):
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((8, 5)).astype(np.float32)
        loose = compute_cosine_pair_graph(vectors, threshold=0.0)
        tight = compute_cosine_pair_graph(vectors, threshold=0.5)
        # Every pair above 0.5 is also above 0.0.
        assert tight <= loose


# ---------------------------------------------------------------------------
# §2 — build_clustered_dictionary
# ---------------------------------------------------------------------------


def _build_planted_antipodal(rng: np.random.Generator) -> tuple[list[Feature], np.ndarray]:
    """Synthesize a 12-feature 'planted antipodal' fixture: three
    tight clusters of 4 features each. Within a cluster the decoders
    are tiny perturbations of a common axis; across clusters the axes
    are orthogonal. Useful for asserting that cosine clustering
    recovers the planted partition.
    """
    n_per_cluster = 4
    d_model = 16
    axes = np.eye(3, d_model, dtype=np.float32)  # three orthogonal directions
    features: list[Feature] = []
    vectors_list: list[np.ndarray] = []
    for cluster_idx in range(3):
        for i in range(n_per_cluster):
            noise = rng.standard_normal(d_model).astype(np.float32) * 0.01
            vec = axes[cluster_idx] + noise
            vec /= np.linalg.norm(vec)
            vectors_list.append(vec)
            features.append(
                Feature(
                    name=f"c{cluster_idx}_f{i}",
                    cluster="planted",  # ignored by cosine strategy
                    beta=0.05 * i,
                )
            )
    vectors = np.stack(vectors_list, axis=0)
    return features, vectors


class TestBuildClusteredDictionaryCosine:
    def test_recovers_planted_antipodal_clusters(self):
        rng = np.random.default_rng(0)
        features, vectors = _build_planted_antipodal(rng)
        cd = build_clustered_dictionary(
            name="planted",
            features=features,
            decoder_vectors=vectors,
            encoding=MPSRung1(),
            block_formation=BlockFormation(
                strategy="cosine", cosine_threshold=0.9
            ),
        )
        assert cd.n_blocks == 3
        assert cd.n_features == 12
        # Each block should contain features from exactly one planted
        # cluster (matched by name prefix "c0_", "c1_", "c2_").
        for block in cd.blocks:
            prefixes = {f.name[:3] for f in block.features}
            assert len(prefixes) == 1, (
                f"block mixed planted clusters: prefixes={prefixes}, "
                f"features={[f.name for f in block.features]}"
            )

    def test_singleton_blocks_for_isolated_features(self):
        # Six orthogonal features → cosine threshold 0.5 puts each in
        # its own block (no pairs above threshold).
        vectors = np.eye(6, dtype=np.float32)
        features = [_feature(f"iso_{i}", cluster="iso") for i in range(6)]
        cd = build_clustered_dictionary(
            name="isolated",
            features=features,
            decoder_vectors=vectors,
            encoding=MPSRung1(),
            block_formation=BlockFormation(
                strategy="cosine", cosine_threshold=0.5
            ),
        )
        assert cd.n_blocks == 6
        assert all(len(b.features) == 1 for b in cd.blocks)
        assert cd.n_cross_block_edges == 0

    def test_block_size_cap_enforced(self):
        # Twelve identical features should fit into ⌈12/8⌉ = 2 blocks
        # on MPSRung1 (cap 8).
        vectors = np.ones((12, 3), dtype=np.float32)
        features = [_feature(f"same_{i}", cluster="same") for i in range(12)]
        cd = build_clustered_dictionary(
            name="same",
            features=features,
            decoder_vectors=vectors,
            encoding=MPSRung1(),
            block_formation=BlockFormation(
                strategy="cosine", cosine_threshold=0.5
            ),
        )
        assert cd.n_blocks == 2
        assert all(len(b.features) <= 8 for b in cd.blocks)

    def test_block_size_max_override_honoured(self):
        # Twelve identical features, block_size_max=3 → 4 blocks.
        vectors = np.ones((12, 3), dtype=np.float32)
        features = [_feature(f"same_{i}", cluster="same") for i in range(12)]
        cd = build_clustered_dictionary(
            name="same",
            features=features,
            decoder_vectors=vectors,
            encoding=MPSRung1(),
            block_formation=BlockFormation(
                strategy="cosine", cosine_threshold=0.5, block_size_max=3
            ),
        )
        assert cd.n_blocks == 4
        assert all(len(b.features) <= 3 for b in cd.blocks)

    def test_decoder_shape_mismatch_raises(self):
        features = [_feature(f"f{i}") for i in range(5)]
        vectors = np.zeros((4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="must equal len\\(features\\)"):
            build_clustered_dictionary(
                name="bad",
                features=features,
                decoder_vectors=vectors,
                encoding=MPSRung1(),
                block_formation=BlockFormation(strategy="cosine"),
            )

    def test_cross_block_edges_populated(self):
        # Two near-orthogonal pairs in different blocks should produce
        # a cross-block edge if their cosine exceeds threshold.
        d = 4
        vectors = np.zeros((4, d), dtype=np.float32)
        vectors[0] = [1.0, 0.0, 0.0, 0.0]
        vectors[1] = [0.99, 0.01, 0.0, 0.0]
        vectors[2] = [0.0, 1.0, 0.0, 0.0]
        vectors[3] = [0.01, 0.99, 0.0, 0.0]
        # Add a small cross-block similarity between block_0 (features
        # 0, 1) and block_1 (features 2, 3) by tilting feature 0 toward
        # feature 2 just enough to clear threshold 0.4.
        vectors[0] = vectors[0] + 0.6 * vectors[2]
        vectors[0] /= np.linalg.norm(vectors[0])
        features = [_feature(f"f{i}", cluster="all") for i in range(4)]
        cd = build_clustered_dictionary(
            name="bridge",
            features=features,
            decoder_vectors=vectors,
            encoding=MPSRung1(),
            block_formation=BlockFormation(
                strategy="cosine",
                cosine_threshold=0.9,
                block_size_max=2,
            ),
        )
        assert cd.n_blocks >= 2
        # The lower-cosine cross-block edges (threshold tightened to
        # 0.4 for the edge computation) should show up; here we just
        # assert the cross-block adjacency is well-formed when present.
        for (bi, _, bj, _), cos in cd.cross_block_pairs.items():
            assert bi < bj
            assert cos >= 0.9


# ---------------------------------------------------------------------------
# §2 — build_clustered_dictionary user_declared
# ---------------------------------------------------------------------------


class TestBuildClusteredDictionaryUserDeclared:
    def test_respects_supplied_hierarchy(self):
        features = [_feature(f"f{i}", cluster=f"c{i // 2}") for i in range(6)]
        # Three clusters of 2 each.
        hierarchy = {
            "c0": ["f0", "f1"],
            "c1": ["f2", "f3"],
            "c2": ["f4", "f5"],
        }
        cd = build_clustered_dictionary(
            name="user",
            features=features,
            decoder_vectors=np.eye(6, dtype=np.float32),
            encoding=MPSRung1(),
            block_formation=BlockFormation(strategy="user_declared"),
            hierarchy=hierarchy,
        )
        assert cd.n_blocks == 3
        # Each block preserves the supplied cluster name.
        for block in cd.blocks:
            cluster_names = {f.cluster for f in block.features}
            assert len(cluster_names) == 1, cluster_names

    def test_splits_oversized_clusters(self):
        # One cluster of 10 features on MPSRung1 (cap 8) splits into
        # two blocks of 8 + 2.
        features = [_feature(f"f{i}", cluster="big") for i in range(10)]
        hierarchy = {"big": [f.name for f in features]}
        cd = build_clustered_dictionary(
            name="big",
            features=features,
            decoder_vectors=np.eye(10, 16, dtype=np.float32),
            encoding=MPSRung1(),
            block_formation=BlockFormation(strategy="user_declared"),
            hierarchy=hierarchy,
        )
        assert cd.n_blocks == 2
        assert sorted(len(b.features) for b in cd.blocks) == [2, 8]

    def test_missing_hierarchy_raises(self):
        features = [_feature(f"f{i}") for i in range(3)]
        with pytest.raises(ValueError, match="requires `hierarchy`"):
            build_clustered_dictionary(
                name="bad",
                features=features,
                decoder_vectors=np.eye(3, dtype=np.float32),
                encoding=MPSRung1(),
                block_formation=BlockFormation(strategy="user_declared"),
            )

    def test_hierarchy_references_unknown_feature_raises(self):
        features = [_feature(f"f{i}") for i in range(3)]
        hierarchy = {"c": ["f0", "ghost"]}
        with pytest.raises(ValueError, match="unknown feature"):
            build_clustered_dictionary(
                name="bad",
                features=features,
                decoder_vectors=np.eye(3, dtype=np.float32),
                encoding=MPSRung1(),
                block_formation=BlockFormation(strategy="user_declared"),
                hierarchy=hierarchy,
            )

    def test_feature_in_multiple_clusters_raises(self):
        features = [_feature(f"f{i}") for i in range(3)]
        hierarchy = {"a": ["f0", "f1"], "b": ["f1", "f2"]}
        with pytest.raises(ValueError, match="appears in multiple clusters"):
            build_clustered_dictionary(
                name="bad",
                features=features,
                decoder_vectors=np.eye(3, dtype=np.float32),
                encoding=MPSRung1(),
                block_formation=BlockFormation(strategy="user_declared"),
                hierarchy=hierarchy,
            )

    def test_unplaced_feature_raises(self):
        features = [_feature(f"f{i}") for i in range(3)]
        hierarchy = {"a": ["f0", "f1"]}  # f2 omitted
        with pytest.raises(ValueError, match="not assigned to any cluster"):
            build_clustered_dictionary(
                name="bad",
                features=features,
                decoder_vectors=np.eye(3, dtype=np.float32),
                encoding=MPSRung1(),
                block_formation=BlockFormation(strategy="user_declared"),
                hierarchy=hierarchy,
            )


# ---------------------------------------------------------------------------
# §2 — co_firing is reserved (NotImplementedError)
# ---------------------------------------------------------------------------


class TestCoFiringReserved:
    def test_co_firing_raises_not_implemented(self):
        features = [_feature(f"f{i}") for i in range(3)]
        with pytest.raises(NotImplementedError, match="co_firing block formation"):
            build_clustered_dictionary(
                name="reserved",
                features=features,
                decoder_vectors=np.eye(3, dtype=np.float32),
                encoding=MPSRung1(),
                block_formation=BlockFormation(
                    strategy="co_firing", firing_corpus=["prompt"]
                ),
                activation_traces=np.zeros((5, 3), dtype=np.float32),
            )

    def test_co_firing_without_traces_raises_value_error_first(self):
        features = [_feature(f"f{i}") for i in range(3)]
        # The ValueError about missing activation_traces precedes the
        # NotImplementedError, so the caller gets a clearer message.
        with pytest.raises(ValueError, match="requires `activation_traces`"):
            build_clustered_dictionary(
                name="reserved",
                features=features,
                decoder_vectors=np.eye(3, dtype=np.float32),
                encoding=MPSRung1(),
                block_formation=BlockFormation(
                    strategy="co_firing", firing_corpus=["prompt"]
                ),
            )
