"""`polygram.compression._helpers` — `compute_rank_ratio` and
`informative_metric` edge-case coverage.

See `openspec/changes/add-cancellation-and-compression-diagnostics/`.
"""

from __future__ import annotations

import numpy as np

from polygram.compression._helpers import (
    compute_rank_ratio,
    floats_eq,
    informative_metric,
    json_finite,
)


class TestComputeRankRatio:
    def test_full_rank_square_matrix(self):
        """A full-rank n×n matrix → rank_ratio = 1.0."""
        w = np.eye(4)
        assert compute_rank_ratio(w, d_model=4) == 1.0

    def test_full_rank_tall_matrix(self):
        """n_kept > d_model: rank caps at d_model, ratio = 1.0."""
        w = np.eye(4)
        # Stack two identity rows on top — 6 rows but rank still 4.
        w = np.vstack([w, w[:2]])
        assert w.shape == (6, 4)
        assert compute_rank_ratio(w, d_model=4) == 1.0

    def test_rank_deficient_wide_matrix(self):
        """n_kept < d_model: rank capped at n_kept, ratio = n_kept/d_model."""
        w = np.eye(3, 8)  # 3 unique rows in 8-dim space → rank 3
        assert compute_rank_ratio(w, d_model=8) == 3 / 8

    def test_all_zero_rows_yield_zero_rank_ratio(self):
        """Pathologically all-zero decoder rows → rank_ratio = 0.0."""
        w = np.zeros((4, 8))
        assert compute_rank_ratio(w, d_model=8) == 0.0

    def test_empty_kept_matrix(self):
        """No kept features (n_kept=0) → rank_ratio = 0.0."""
        w = np.zeros((0, 8))
        assert compute_rank_ratio(w, d_model=8) == 0.0

    def test_collinear_rows_collapse_rank(self):
        """Rows that are scalar multiples of one direction → rank 1."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.stack([v, 2 * v, -0.5 * v])
        assert w.shape == (3, 4)
        assert compute_rank_ratio(w, d_model=4) == 1 / 4

    def test_near_singular_below_default_tol(self):
        """`matrix_rank` uses a default tolerance (max(M,N) * max_sv * eps);
        rows differing by < that tolerance collapse together."""
        v = np.array([1.0, 0.0, 0.0, 0.0])
        # Two rows that are equal modulo float noise — should rank as 1.
        w = np.stack([v, v + 1e-15])
        assert compute_rank_ratio(w, d_model=4) == 1 / 4


class TestInformativeMetric:
    def test_low_rank_says_trust_post_A(self):
        assert informative_metric(0.5) == "post_A"
        assert informative_metric(0.94999) == "post_A"

    def test_band_center_says_both(self):
        assert informative_metric(0.95) == "both"
        assert informative_metric(1.0) == "both"
        assert informative_metric(1.05) == "both"

    def test_above_band_says_forge_mse(self):
        assert informative_metric(1.06) == "forge_mse"
        assert informative_metric(2.0) == "forge_mse"

    def test_zero_ratio_says_post_A(self):
        """All-zero decoder rows produce rank_ratio=0, well below the
        post_A threshold."""
        assert informative_metric(0.0) == "post_A"


class TestJsonFinite:
    def test_none_passes_through(self):
        assert json_finite(None) is None

    def test_nan_becomes_none(self):
        assert json_finite(float("nan")) is None

    def test_inf_becomes_none(self):
        assert json_finite(float("inf")) is None
        assert json_finite(float("-inf")) is None

    def test_finite_rounds_to_6_sigfigs(self):
        """6-sig-fig quantization matches the project's report serialization."""
        assert json_finite(0.123456789) == 0.123457
        assert json_finite(1234567.89) == 1234570.0


class TestFloatsEq:
    def test_both_none(self):
        assert floats_eq(None, None) is True

    def test_one_none(self):
        assert floats_eq(None, 1.0) is False
        assert floats_eq(1.0, None) is False

    def test_both_nan(self):
        assert floats_eq(float("nan"), float("nan")) is True

    def test_finite_equal(self):
        assert floats_eq(1.5, 1.5) is True

    def test_finite_unequal(self):
        assert floats_eq(1.5, 1.6) is False
