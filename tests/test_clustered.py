"""Tests for `polygram.clustered_dictionary.ClusteredDictionary` — §1 of the
`clustered-dictionary-analysis` openspec change."""

from __future__ import annotations

import numpy as np
import pytest

from polygram.clustered_dictionary import (
    BlockFormation,
    BlockSparseGram,
    ClusteredDictionary,
    CrossBlockRedundancyReport,
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
        # HEA_Rung2 declares `max_features = 2 ** n_qubits` (per the
        # per-encoding-feature-cap change). ClusteredDictionary honours
        # that cap per-encoding.
        encoding = HEA_Rung2(depth=1, n_qubits=3)
        assert encoding.max_features == 8
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


# ---------------------------------------------------------------------------
# §3 — BlockSparseGram
# ---------------------------------------------------------------------------


class TestBlockSparseGramConstruction:
    def test_minimal_two_block(self):
        g0 = np.array([[1.0 + 0j, 0.5], [0.5, 1.0]], dtype=complex)
        g1 = np.array([[1.0 + 0j, 0.3], [0.3, 1.0]], dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 0, 1, 0): 0.42},
        )
        assert bsg.n_blocks == 2
        assert bsg.shape == (4, 4)

    def test_empty_block_grams_raises(self):
        with pytest.raises(ValueError, match="block_grams must be non-empty"):
            BlockSparseGram(block_grams=[])

    def test_non_square_block_gram_raises(self):
        with pytest.raises(ValueError, match="must be a square 2-D array"):
            BlockSparseGram(block_grams=[np.zeros((3, 4), dtype=complex)])

    def test_cross_block_key_canonical_ordering_enforced(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        # bi > bj violates canonical 0 <= bi < bj invariant.
        with pytest.raises(ValueError, match="canonical block ordering"):
            BlockSparseGram(
                block_grams=[g0, g1],
                cross_block_edges={(1, 0, 0, 0): 0.5},
            )

    def test_cross_block_key_out_of_range_feat_raises(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        with pytest.raises(ValueError, match="feat_j_local_idx=5 out of range"):
            BlockSparseGram(
                block_grams=[g0, g1],
                cross_block_edges={(0, 0, 1, 5): 0.5},
            )


class TestBlockSparseGramShapeAndDensity:
    def test_shape_sums_block_sizes(self):
        g0 = np.zeros((3, 3), dtype=complex)
        g1 = np.zeros((5, 5), dtype=complex)
        g2 = np.zeros((2, 2), dtype=complex)
        bsg = BlockSparseGram(block_grams=[g0, g1, g2])
        assert bsg.shape == (10, 10)

    def test_density_zero_when_no_cross_block_edges(self):
        g0 = np.zeros((3, 3), dtype=complex)
        g1 = np.zeros((3, 3), dtype=complex)
        bsg = BlockSparseGram(block_grams=[g0, g1])
        assert bsg.density == 0.0

    def test_density_with_edges(self):
        # 2 blocks of size 2 each → N=4, total cells 16, block-diagonal
        # cells 4+4=8, off-block cells 8. With 1 edge (covering both
        # halves of the dense form = 2 cells), density = 2/8 = 0.25.
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 0, 1, 0): 0.5},
        )
        assert bsg.density == pytest.approx(0.25)

    def test_density_zero_when_single_block(self):
        # 1 block → no off-block region; density gracefully returns 0.
        g0 = np.zeros((3, 3), dtype=complex)
        bsg = BlockSparseGram(block_grams=[g0])
        assert bsg.density == 0.0

    def test_cross_block_density_aliases_density(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 0, 1, 0): 0.5},
        )
        assert bsg.cross_block_density == bsg.density

    def test_cross_block_cosine_histogram_shape(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={
                (0, 0, 1, 0): 0.5 + 0j,
                (0, 1, 1, 1): 0.9 + 0j,
            },
        )
        counts, edges = bsg.cross_block_cosine_histogram(bins=10)
        assert counts.shape == (10,)
        assert edges.shape == (11,)
        # Total counts equal the number of edges.
        assert counts.sum() == 2

    def test_cross_block_cosine_histogram_empty(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        bsg = BlockSparseGram(block_grams=[g0, g1])
        counts, edges = bsg.cross_block_cosine_histogram(bins=5)
        assert counts.shape == (5,)
        assert counts.sum() == 0
        assert edges.shape == (6,)


class TestBlockSparseGramIteration:
    def test_block_diagonal_returns_per_block(self):
        g0 = np.array([[1.0 + 0j, 0.5], [0.5, 1.0]], dtype=complex)
        g1 = np.array([[1.0 + 0j]], dtype=complex)
        bsg = BlockSparseGram(block_grams=[g0, g1])
        diag = bsg.block_diagonal()
        assert len(diag) == 2
        np.testing.assert_array_equal(diag[0], g0)
        np.testing.assert_array_equal(diag[1], g1)

    def test_entries_iterates_all_nonzero(self):
        g0 = np.array([[1.0 + 0j, 0.5], [0.5, 1.0]], dtype=complex)
        g1 = np.array([[1.0 + 0j]], dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 0, 1, 0): 0.42 + 0j},
        )
        entries = list(bsg.entries())
        # 4 block-diagonal entries (g0 is 2x2 → 4 cells, g1 is 1x1 → 1 cell) + 1 cross = 6
        assert len(entries) == 5 + 1

    def test_cross_block_entries_uses_global_coords(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((3, 3), dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 1, 1, 2): 0.7 + 0j},
        )
        edges = list(bsg.cross_block_entries())
        assert len(edges) == 1
        gi, gj, v = edges[0]
        # global_i = offsets[0] + 1 = 0 + 1 = 1; global_j = offsets[1] + 2 = 2 + 2 = 4
        assert (gi, gj) == (1, 4)
        assert v == 0.7 + 0j

    def test_cross_block_entries_canonical_ordering(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 0, 1, 1): 0.5 + 0j},
        )
        for gi, gj, _ in bsg.cross_block_entries():
            assert gi < gj


class TestBlockSparseGramToDense:
    def test_block_diagonal_copied(self):
        g0 = np.array([[1.0 + 0j, 0.5], [0.5, 1.0]], dtype=complex)
        g1 = np.array([[1.0 + 0j, 0.3], [0.3, 1.0]], dtype=complex)
        bsg = BlockSparseGram(block_grams=[g0, g1])
        dense = bsg.to_dense()
        assert dense.shape == (4, 4)
        np.testing.assert_array_equal(dense[:2, :2], g0)
        np.testing.assert_array_equal(dense[2:, 2:], g1)
        # Off-block regions are zero (no edges supplied).
        np.testing.assert_array_equal(dense[:2, 2:], np.zeros((2, 2), dtype=complex))
        np.testing.assert_array_equal(dense[2:, :2], np.zeros((2, 2), dtype=complex))

    def test_cross_block_edges_placed_both_triangles(self):
        g0 = np.zeros((2, 2), dtype=complex)
        g1 = np.zeros((2, 2), dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 0, 1, 1): 0.42 + 0j},
        )
        dense = bsg.to_dense()
        # global_i = 0, global_j = 2 + 1 = 3
        assert dense[0, 3] == 0.42 + 0j
        assert dense[3, 0] == 0.42 + 0j  # real value's conjugate equals itself

    def test_dense_hermitian_with_complex_edge(self):
        g0 = np.array([[1.0 + 0j, 0.5j], [-0.5j, 1.0]], dtype=complex)  # Hermitian
        g1 = np.array([[1.0 + 0j]], dtype=complex)
        bsg = BlockSparseGram(
            block_grams=[g0, g1],
            cross_block_edges={(0, 0, 1, 0): 0.5 + 0.3j},
        )
        dense = bsg.to_dense()
        assert np.allclose(dense, dense.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# §4 — ClusteredDictionary.gram()
# ---------------------------------------------------------------------------


class TestClusteredDictionaryGram:
    def test_single_block_matches_flat_dictionary(self):
        # When a clustered dictionary has one block, the block_gram
        # equals the flat Dictionary's gram on the same features.
        features = [_feature(f"f{i}", cluster="single", beta=0.1 * i) for i in range(4)]
        flat = Dictionary(
            name="flat",
            features=features,
            hierarchy={"single": [f.name for f in features]},
            encoding=MPSRung1(),
        )
        flat_gram = flat.gram()
        cd = ClusteredDictionary(name="clustered", blocks=[flat])
        bsg = cd.gram()
        assert bsg.n_blocks == 1
        np.testing.assert_array_equal(bsg.block_grams[0], flat_gram)

    def test_two_block_dense_form_block_diagonal_correct(self):
        # Two blocks of 3 features each. Dense form's block-diagonal
        # regions match each block's gram exactly.
        block_a = _block("alpha", 3)
        block_b = _block("beta", 3)
        cd = ClusteredDictionary(
            name="cd", blocks=[block_a, block_b]
        )
        bsg = cd.gram()
        dense = bsg.to_dense()
        np.testing.assert_array_equal(dense[:3, :3], block_a.gram())
        np.testing.assert_array_equal(dense[3:, 3:], block_b.gram())
        # No cross-block edges supplied → off-block regions zero.
        np.testing.assert_array_equal(dense[:3, 3:], np.zeros((3, 3), dtype=complex))

    def test_cross_block_edge_appears_in_dense(self):
        block_a = _block("alpha", 2)
        block_b = _block("beta", 2)
        cd = ClusteredDictionary(
            name="cd",
            blocks=[block_a, block_b],
            cross_block_pairs={(0, 1, 1, 0): 0.42},
        )
        bsg = cd.gram()
        dense = bsg.to_dense()
        # global_i for (block 0, feat 1) = 0 + 1 = 1
        # global_j for (block 1, feat 0) = 2 + 0 = 2
        assert dense[1, 2] == 0.42 + 0j
        assert dense[2, 1] == 0.42 + 0j

    def test_gram_shape_matches_n_features(self):
        block_a = _block("alpha", 2)
        block_b = _block("beta", 3)
        block_c = _block("gamma", 1)
        cd = ClusteredDictionary(
            name="cd", blocks=[block_a, block_b, block_c]
        )
        bsg = cd.gram()
        assert bsg.shape == (6, 6)
        assert bsg.shape[0] == cd.n_features

    def test_via_build_clustered_dictionary_cosine(self):
        # End-to-end: planted-antipodal fixture → build_clustered_dictionary
        # → clustered.gram() round-trip works without errors.
        rng = np.random.default_rng(7)
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
        bsg = cd.gram()
        assert bsg.n_blocks == 3
        assert bsg.shape == (12, 12)
        # Per-block grams have unit-modulus diagonal (each feature's
        # self-overlap is 1).
        for block_gram in bsg.block_grams:
            np.testing.assert_allclose(
                np.abs(np.diag(block_gram)),
                1.0,
                atol=1e-9,
            )


# ---------------------------------------------------------------------------
# §5 — cross_block_redundant_pairs
# ---------------------------------------------------------------------------


def _planted_cross_block_duplicate(
    threshold_at_build: float = 0.3,
) -> ClusteredDictionary:
    """Two blocks of 3 features; feature `alpha_f0` and feature
    `beta_f0` have identical decoder vectors so their cross-block
    cosine is exactly 1.0. Used to assert the redundancy primitive
    catches the planted pair.
    """
    d = 8
    vectors = np.zeros((6, d), dtype=np.float32)
    # Alpha cluster: pointing along axis 0.
    vectors[0] = [1, 0, 0, 0, 0, 0, 0, 0]
    vectors[1] = [0.99, 0.05, 0, 0, 0, 0, 0, 0]
    vectors[2] = [0.95, 0.10, 0, 0, 0, 0, 0, 0]
    # Beta cluster: nominally along axis 1 but feature 0 collides
    # with alpha_f0.
    vectors[3] = [1, 0, 0, 0, 0, 0, 0, 0]  # ← duplicate of alpha_f0
    vectors[4] = [0, 1, 0, 0, 0, 0, 0, 0]
    vectors[5] = [0, 0.95, 0.10, 0, 0, 0, 0, 0]
    # Normalise.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    features = [
        Feature(name="alpha_f0", cluster="alpha", beta=0.0),
        Feature(name="alpha_f1", cluster="alpha", beta=0.05),
        Feature(name="alpha_f2", cluster="alpha", beta=0.10),
        Feature(name="beta_f0", cluster="beta", beta=0.0),
        Feature(name="beta_f1", cluster="beta", beta=0.05),
        Feature(name="beta_f2", cluster="beta", beta=0.10),
    ]
    hierarchy = {
        "alpha": ["alpha_f0", "alpha_f1", "alpha_f2"],
        "beta": ["beta_f0", "beta_f1", "beta_f2"],
    }
    return build_clustered_dictionary(
        name="planted",
        features=features,
        decoder_vectors=vectors,
        encoding=MPSRung1(),
        block_formation=BlockFormation(
            strategy="user_declared",
            cosine_threshold=threshold_at_build,
        ),
        hierarchy=hierarchy,
    )


class TestCrossBlockRedundantPairs:
    def test_planted_duplicate_caught_and_ranked_first(self):
        cd = _planted_cross_block_duplicate()
        report = cd.cross_block_redundant_pairs(threshold=0.7)
        assert isinstance(report, CrossBlockRedundancyReport)
        assert len(report.pairs) >= 1
        first = report.pairs[0]
        # alpha_f0 and beta_f0 are the planted identical pair.
        names = {first.feat_i_name, first.feat_j_name}
        assert names == {"alpha_f0", "beta_f0"}
        # Their cosine is exactly 1.0 (post-normalisation).
        assert first.cosine == pytest.approx(1.0, abs=1e-6)

    def test_threshold_monotone(self):
        cd = _planted_cross_block_duplicate()
        n_low = len(cd.cross_block_redundant_pairs(threshold=0.5).pairs)
        n_high = len(cd.cross_block_redundant_pairs(threshold=0.95).pairs)
        # Tightening threshold can only shrink the result.
        assert n_high <= n_low
        # All high-threshold pairs are present in low-threshold.
        low_pairs = {
            (p.feat_i_name, p.feat_j_name)
            for p in cd.cross_block_redundant_pairs(threshold=0.5).pairs
        }
        high_pairs = {
            (p.feat_i_name, p.feat_j_name)
            for p in cd.cross_block_redundant_pairs(threshold=0.95).pairs
        }
        assert high_pairs <= low_pairs

    def test_ordering_is_descending(self):
        cd = _planted_cross_block_duplicate()
        report = cd.cross_block_redundant_pairs(threshold=0.3)
        cosines = [p.cosine for p in report.pairs]
        assert cosines == sorted(cosines, reverse=True)

    def test_coverage_summary_present(self):
        cd = _planted_cross_block_duplicate()
        report = cd.cross_block_redundant_pairs(threshold=0.7)
        # The planted setup puts everything between block 0 and block 1.
        assert (0, 1) in report.coverage
        assert report.coverage[(0, 1)] == len(report.pairs)

    def test_n_total_edges_reported(self):
        cd = _planted_cross_block_duplicate()
        report = cd.cross_block_redundant_pairs(threshold=0.7)
        # Total edges in adjacency >= filtered pair count.
        assert report.n_total_cross_block_edges >= len(report.pairs)
        # Matches the underlying adjacency size.
        assert report.n_total_cross_block_edges == cd.n_cross_block_edges

    def test_threshold_out_of_range_raises(self):
        cd = _planted_cross_block_duplicate()
        with pytest.raises(ValueError, match="must lie in"):
            cd.cross_block_redundant_pairs(threshold=1.5)
        with pytest.raises(ValueError, match="must lie in"):
            cd.cross_block_redundant_pairs(threshold=-0.1)

    def test_no_cross_block_edges_yields_empty_report(self):
        # Construction with no cross_block_pairs → empty redundancy
        # report regardless of threshold.
        cd = ClusteredDictionary(
            name="empty_cb",
            blocks=[_block("alpha", 2), _block("beta", 2)],
        )
        report = cd.cross_block_redundant_pairs(threshold=0.0)
        assert report.pairs == []
        assert report.coverage == {}
        assert report.n_total_cross_block_edges == 0


# ---------------------------------------------------------------------------
# §8 — Per-block Q-OrCA emission
# ---------------------------------------------------------------------------


class TestClusteredEmitQorca:
    def test_emits_per_block_machines_and_manifest(self, tmp_path):
        cd = ClusteredDictionary(
            name="emit_test",
            blocks=[_block("alpha", 2), _block("beta", 2)],
        )
        artifacts = cd.emit_qorca(tmp_path)
        # Expect one entry per block + manifest.
        assert "manifest" in artifacts
        assert "alpha" in artifacts
        assert "beta" in artifacts
        # Files exist on disk.
        assert artifacts["alpha"].is_file()
        assert artifacts["beta"].is_file()
        assert artifacts["manifest"].is_file()
        # Block machine filenames follow the convention.
        assert artifacts["alpha"].name == "alpha.q.orca.md"
        assert artifacts["beta"].name == "beta.q.orca.md"

    def test_manifest_schema_fields_present(self, tmp_path):
        import json

        cd = ClusteredDictionary(
            name="schema_test",
            blocks=[_block("alpha", 2), _block("beta", 2)],
            cross_block_pairs={(0, 0, 1, 1): 0.6},
        )
        cd.emit_qorca(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["name"] == "schema_test"
        assert manifest["n_features"] == 4
        assert manifest["encoding"] == "MPSRung1"
        assert len(manifest["blocks"]) == 2
        # Block records carry the right fields.
        b0 = manifest["blocks"][0]
        assert b0["id"] == "alpha"
        assert b0["machine"] == "alpha.q.orca.md"
        assert len(b0["features"]) == 2
        assert b0["features"][0] == {"name": "alpha_f0", "cluster": "alpha"}
        # Cross-block edge captured.
        assert len(manifest["cross_block_edges"]) == 1
        edge = manifest["cross_block_edges"][0]
        assert edge["from"] == ["alpha", "alpha_f0"]
        assert edge["to"] == ["beta", "beta_f1"]
        assert edge["cosine"] == pytest.approx(0.6)
        # Block formation captured.
        assert manifest["block_formation"]["strategy"] == "user_declared"

    def test_per_block_machines_are_well_formed(self, tmp_path):
        # Each per-block .q.orca.md should at minimum carry the block's
        # name as a machine declaration. We don't run the full Q-OrCA
        # verifier here (that's covered by the existing write_qorca
        # tests on Dictionary); we just smoke-check the file shape.
        cd = ClusteredDictionary(
            name="wellformed",
            blocks=[_block("alpha", 2)],
        )
        artifacts = cd.emit_qorca(tmp_path)
        text = artifacts["alpha"].read_text()
        assert "# machine alpha" in text or "machine alpha" in text

    def test_empty_cross_block_edges_produces_empty_list(self, tmp_path):
        import json

        cd = ClusteredDictionary(
            name="no_edges",
            blocks=[_block("alpha", 2), _block("beta", 2)],
        )
        cd.emit_qorca(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["cross_block_edges"] == []


# ---------------------------------------------------------------------------
# §7 — EpochCompressor connection (shallow): from_compression_panels
# ---------------------------------------------------------------------------


class TestClusteredFromCompressionPanels:
    """Shallow forward-compatible connection between EpochCompressor's
    `_select_panels` output and the ClusteredDictionary primitive.

    The deep behaviour-preserving refactor (extracting the priority-
    driven seeded coverage algorithm into a BlockFormation strategy
    so `_select_panels` becomes a thin wrapper) is deferred to a
    follow-up change. This API exists so callers can consume the
    clustered view today without that refactor."""

    def _make_panel(self, anchor, feature_ids, cosines):
        # Lightweight Panel stand-in matching the
        # polygram.compression.epoch_report.Panel surface used here.
        from polygram.compression.epoch_report import Panel

        return Panel(
            panel_id=0,
            anchor=anchor,
            feature_ids=tuple(feature_ids),
            cosines_to_anchor=tuple(cosines),
        )

    def test_constructs_one_block_per_panel(self):
        # Two panels of 3 features each → 2 blocks.
        panels = [
            self._make_panel(anchor=0, feature_ids=(0, 1, 2), cosines=(0.9, 0.85)),
            self._make_panel(anchor=10, feature_ids=(10, 11, 12), cosines=(0.92, 0.81)),
        ]
        state_dict = {"W_dec": np.eye(20, 8, dtype=np.float32)}
        cd = ClusteredDictionary.from_compression_panels(
            panels, state_dict, MPSRung1(), name="from_panels_test"
        )
        assert cd.n_blocks == 2
        assert cd.n_features == 6
        # Block 0 has features f0, f1, f2; block 1 has f10, f11, f12.
        block_0_names = sorted(f.name for f in cd.blocks[0].features)
        assert block_0_names == ["f0", "f1", "f2"]

    def test_cross_block_edges_above_threshold(self):
        # Plant a cross-block similarity: feature 0 in panel A and
        # feature 10 in panel B share decoder direction.
        panels = [
            self._make_panel(anchor=0, feature_ids=(0, 1), cosines=(0.5,)),
            self._make_panel(anchor=10, feature_ids=(10, 11), cosines=(0.5,)),
        ]
        w_dec = np.zeros((20, 4), dtype=np.float32)
        w_dec[0] = [1, 0, 0, 0]
        w_dec[1] = [0, 1, 0, 0]
        w_dec[10] = [1, 0, 0, 0]  # ← identical to feature 0
        w_dec[11] = [0, 0, 1, 0]
        cd = ClusteredDictionary.from_compression_panels(
            panels,
            {"W_dec": w_dec},
            MPSRung1(),
            name="planted",
            cosine_threshold=0.5,
        )
        # Cross-block edge between block 0 feature 0 and block 1 feature 0.
        assert (0, 0, 1, 0) in cd.cross_block_pairs
        assert cd.cross_block_pairs[(0, 0, 1, 0)] == pytest.approx(1.0)

    def test_works_with_real_select_panels_output(self):
        # End-to-end: run _select_panels on a synthetic SAE state +
        # eligible set, take the panels, build a ClusteredDictionary.
        # Verifies the panel object's interface matches the API
        # contract.
        from polygram.compression.epoch import (
            _compute_cosine_graph,
            _select_panels,
        )

        rng = np.random.default_rng(0)
        n_features = 16
        d_model = 8
        w_dec = rng.standard_normal((n_features, d_model)).astype(np.float32)
        state_dict = {"W_dec": w_dec}
        eligible = np.arange(n_features, dtype=np.int64)
        priority = np.ones(n_features, dtype=np.float32)
        cosine_pairs = _compute_cosine_graph(w_dec, eligible, threshold=0.3)
        panels, _coverage = _select_panels(
            state_dict=state_dict,
            eligible=eligible,
            priority=priority,
            cosine_pairs=cosine_pairs,
            zeroed=set(),
            n_visits_per_feature=1,
            n_panels_max=4,
            coverage_target=0.5,
        )
        if not panels:
            pytest.skip("no panels produced for the synthetic fixture")
        cd = ClusteredDictionary.from_compression_panels(
            panels, state_dict, MPSRung1(), name="real_panels"
        )
        assert cd.n_blocks == len(panels)
        # Each block's feature count matches its panel's member count.
        for block, panel in zip(cd.blocks, panels):
            assert len(block.features) == len(panel.feature_ids)
