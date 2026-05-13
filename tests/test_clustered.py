"""Tests for `polygram.clustered.ClusteredDictionary` — §1 of the
`clustered-dictionary-analysis` openspec change."""

from __future__ import annotations

import pytest

from polygram.clustered import (
    BlockFormation,
    ClusteredDictionary,
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
