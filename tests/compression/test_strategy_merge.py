"""Unit tests for `polygram.compression.strategies.merge.apply_merge`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from polygram.compression.report import ClusterPlan, CompressionPlan
from polygram.compression.strategies.merge import apply_merge


def _state(
    n_features: int = 4, d_model: int = 4, dec_norms: dict[int, float] | None = None
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    w_dec = rng.standard_normal((n_features, d_model)).astype(np.float32)
    if dec_norms:
        for fid, target in dec_norms.items():
            curr = float(np.linalg.norm(w_dec[fid]))
            if curr > 0.0:
                w_dec[fid] *= target / curr
    return {
        "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32),
        "b_enc": np.zeros((n_features,), dtype=np.float32),
        "W_dec": w_dec,
        "b_dec": np.zeros((d_model,), dtype=np.float32),
    }


def _two_member_plan(rep: int, other: int) -> CompressionPlan:
    return CompressionPlan(
        clusters=(
            ClusterPlan(
                cluster_id=0,
                members=tuple(sorted((rep, other))),
                representative=rep,
                zeroed=(other,),
            ),
        ),
        feature_ids=(0, 1, 2, 3),
    )


class TestFreqWeightedArithmetic:
    def test_equal_fires_yields_simple_mean(self):
        """Equal fires for both members → weighted mean reduces to
        the simple mean."""
        state = _state(dec_norms={0: 1.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        out, merged = apply_merge(
            state,
            plan,
            merge_mode="freq_weighted",
            n_fires_by_fid={0: 10, 1: 10},
        )
        np.testing.assert_allclose(merged[0], 2.0, atol=1e-5)
        np.testing.assert_allclose(
            np.linalg.norm(out["W_dec"][0]), 2.0, atol=1e-5
        )

    def test_unequal_fires_weights_norms(self):
        """norms 1.0 / 3.0, fires 30 / 10 → (1·30 + 3·10) / 40 = 1.5."""
        state = _state(dec_norms={0: 1.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        out, merged = apply_merge(
            state,
            plan,
            merge_mode="freq_weighted",
            n_fires_by_fid={0: 30, 1: 10},
        )
        np.testing.assert_allclose(merged[0], 1.5, atol=1e-5)
        np.testing.assert_allclose(
            np.linalg.norm(out["W_dec"][0]), 1.5, atol=1e-5
        )

    def test_zero_total_fires_falls_back_to_simple_mean(self):
        """If both fires are 0, we cannot weight by them — degrades
        gracefully to simple_mean rather than divide by zero."""
        state = _state(dec_norms={0: 1.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        _, merged = apply_merge(
            state,
            plan,
            merge_mode="freq_weighted",
            n_fires_by_fid={0: 0, 1: 0},
        )
        np.testing.assert_allclose(merged[0], 2.0, atol=1e-5)


class TestSimpleMean:
    def test_three_member_simple_mean(self):
        """norms [1, 2, 3] → mean = 2.0."""
        state = _state(dec_norms={0: 1.0, 1: 2.0, 2: 3.0})
        plan = CompressionPlan(
            clusters=(
                ClusterPlan(
                    cluster_id=0,
                    members=(0, 1, 2),
                    representative=1,
                    zeroed=(0, 2),
                ),
            ),
            feature_ids=(0, 1, 2, 3),
        )
        out, merged = apply_merge(state, plan, merge_mode="simple_mean")
        np.testing.assert_allclose(merged[0], 2.0, atol=1e-5)
        np.testing.assert_allclose(
            np.linalg.norm(out["W_dec"][1]), 2.0, atol=1e-5
        )

    def test_simple_mean_ignores_n_fires(self):
        state = _state(dec_norms={0: 1.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        _, merged_a = apply_merge(
            state,
            plan,
            merge_mode="simple_mean",
            n_fires_by_fid={0: 100, 1: 1},
        )
        _, merged_b = apply_merge(
            state,
            plan,
            merge_mode="simple_mean",
            n_fires_by_fid={0: 1, 1: 100},
        )
        assert merged_a == merged_b


class TestNonRepresentativesZeroed:
    def test_w_enc_b_enc_w_dec_all_zeroed_for_non_rep(self):
        state = _state(dec_norms={0: 1.0, 1: 3.0, 2: 5.0})
        plan = CompressionPlan(
            clusters=(
                ClusterPlan(
                    cluster_id=0,
                    members=(0, 1, 2),
                    representative=1,
                    zeroed=(0, 2),
                ),
            ),
            feature_ids=(0, 1, 2, 3),
        )
        out, _ = apply_merge(state, plan, merge_mode="simple_mean")
        for fid in (0, 2):
            assert np.all(out["W_enc"][:, fid] == 0)
            assert out["b_enc"][fid] == 0
            assert np.all(out["W_dec"][fid, :] == 0)

    def test_singleton_cluster_member_left_alone(self):
        """Members not in any cluster pass through unchanged."""
        state = _state(dec_norms={0: 1.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        original_2 = state["W_dec"][2].copy()
        out, _ = apply_merge(state, plan, merge_mode="simple_mean")
        np.testing.assert_array_equal(out["W_dec"][2], original_2)


class TestRepresentativeRescaling:
    def test_rep_direction_preserved(self):
        """Rescaling MUST preserve direction; only L2 norm changes."""
        state = _state(dec_norms={0: 1.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        original_dir = state["W_dec"][0] / np.linalg.norm(state["W_dec"][0])
        out, merged = apply_merge(state, plan, merge_mode="simple_mean")
        new_norm = float(np.linalg.norm(out["W_dec"][0]))
        new_dir = out["W_dec"][0] / new_norm
        np.testing.assert_allclose(new_dir, original_dir, atol=1e-5)
        np.testing.assert_allclose(new_norm, merged[0], atol=1e-5)

    def test_zero_norm_representative_left_zero(self):
        """If the representative's source norm is 0, no rescaling
        (avoids divide by zero); row remains zero."""
        state = _state(dec_norms={0: 0.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        out, merged = apply_merge(state, plan, merge_mode="simple_mean")
        # Source norm 0 / 3 → mean = 1.5 ; merged_norm[0] is recorded
        # but the actual W_dec[0] row stays zero.
        np.testing.assert_allclose(merged[0], 1.5, atol=1e-5)
        assert np.all(out["W_dec"][0] == 0)


class TestAllZeroCluster:
    def test_all_zero_norm_cluster_emits_warning(self):
        """Every member has zero-norm W_dec → merged_norm = 0.
        Must emit a UserWarning so the caller notices the cluster
        produced a silent rep."""
        state = _state(dec_norms={0: 0.0, 1: 0.0})
        plan = _two_member_plan(rep=0, other=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, merged = apply_merge(
                state, plan, merge_mode="simple_mean"
            )
        assert any(
            issubclass(w.category, UserWarning)
            and "merged_norm" in str(w.message)
            for w in caught
        ), f"expected merged_norm warning, got: {[str(w.message) for w in caught]}"
        assert merged[0] == 0.0
        # The rep row stays zero (rep_norm == 0 → rescaling skipped).
        assert np.all(out["W_dec"][0] == 0)

    def test_normal_cluster_emits_no_warning(self):
        """Sanity: a healthy cluster with positive norms does NOT
        emit the all-zero warning."""
        state = _state(dec_norms={0: 1.0, 1: 3.0})
        plan = _two_member_plan(rep=0, other=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_merge(state, plan, merge_mode="simple_mean")
        assert not any(
            "merged_norm" in str(w.message) for w in caught
        )


class TestErrorPaths:
    def test_invalid_merge_mode_raises(self):
        state = _state()
        plan = _two_member_plan(rep=0, other=1)
        with pytest.raises(ValueError, match="unsupported merge_mode"):
            apply_merge(state, plan, merge_mode="kl_weighted")

    def test_missing_required_key_raises(self):
        state = _state()
        del state["b_enc"]
        plan = _two_member_plan(rep=0, other=1)
        with pytest.raises(KeyError, match="b_enc"):
            apply_merge(state, plan, merge_mode="simple_mean")

    def test_out_of_range_fid_raises(self):
        state = _state(n_features=4)
        plan = CompressionPlan(
            clusters=(
                ClusterPlan(
                    cluster_id=0,
                    members=(0, 99),
                    representative=0,
                    zeroed=(99,),
                ),
            ),
            feature_ids=(0, 1, 2, 3),
        )
        with pytest.raises(IndexError, match="99"):
            apply_merge(state, plan, merge_mode="simple_mean")
