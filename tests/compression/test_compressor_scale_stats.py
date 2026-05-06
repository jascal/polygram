"""Scale stats populated by `Compressor.apply()`:

- `ClusterPlan.cluster_norm_mean` / `cluster_norm_std` filled
- `ClusterPlan.merged_norm` is None for zero, positive for merge
- `CompressionReport.scale_compression_ratio` < 1 for zero,
  ≈ 1 for merge+simple_mean
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polygram import Compressor
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def _checkpoint(
    tmp_path: Path,
    n_features: int = 4,
    dec_norms: dict[int, float] | None = None,
) -> Path:
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(
        sae_path, n_features=n_features, d_model=4, dec_norms=dec_norms
    )
    return sae_path


class TestClusterNormStats:
    def test_zero_strategy_populates_mean_and_std(self, tmp_path: Path):
        sae_path = _checkpoint(
            tmp_path, n_features=4, dec_norms={0: 1.0, 1: 3.0}
        )
        report = build_report(n_features=4, confirmed=[(0, 1)])
        result = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(tmp_path / "out.safetensors")
        c0 = result.plan.clusters[0]
        assert c0.cluster_norm_mean is not None
        assert c0.cluster_norm_std is not None
        np.testing.assert_allclose(c0.cluster_norm_mean, 2.0, atol=1e-5)
        np.testing.assert_allclose(c0.cluster_norm_std, 1.0, atol=1e-5)

    def test_zero_strategy_merged_norm_is_none(self, tmp_path: Path):
        sae_path = _checkpoint(
            tmp_path, n_features=4, dec_norms={0: 1.0, 1: 3.0}
        )
        report = build_report(n_features=4, confirmed=[(0, 1)])
        result = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(tmp_path / "out.safetensors")
        for c in result.plan.clusters:
            assert c.merged_norm is None

    def test_merge_strategy_merged_norm_is_positive(self, tmp_path: Path):
        sae_path = _checkpoint(
            tmp_path, n_features=4, dec_norms={0: 1.0, 1: 3.0}
        )
        report = build_report(n_features=4, confirmed=[(0, 1)])
        result = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            strategy="merge",
            merge_mode="simple_mean",
        ).run(tmp_path / "out.safetensors")
        c0 = result.plan.clusters[0]
        assert c0.merged_norm is not None
        assert c0.merged_norm > 0


class TestScaleCompressionRatio:
    def test_zero_strategy_ratio_below_one(self, tmp_path: Path):
        """Zero only preserves the rep's own norm: ratio < 1."""
        sae_path = _checkpoint(
            tmp_path, n_features=4, dec_norms={0: 1.0, 1: 3.0}
        )
        report = build_report(
            n_features=4, confirmed=[(0, 1)], n_fires={0: 1, 1: 100}
        )
        result = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(tmp_path / "out.safetensors")
        # rep = 1 (norm 3); preserved = 3, total = 4 → ratio = 0.75.
        np.testing.assert_allclose(
            result.report.scale_compression_ratio, 0.75, atol=1e-5
        )
        assert result.report.scale_compression_ratio < 1.0

    def test_merge_simple_mean_ratio_approx_one(self, tmp_path: Path):
        """simple_mean preserves total norm mass conceptually:
        merged_norm × cluster_size = mean × N = sum_of_norms.
        So ratio = 1.0 exactly (modulo float)."""
        sae_path = _checkpoint(
            tmp_path, n_features=4, dec_norms={0: 1.0, 1: 3.0}
        )
        report = build_report(n_features=4, confirmed=[(0, 1)])
        result = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            strategy="merge",
            merge_mode="simple_mean",
        ).run(tmp_path / "out.safetensors")
        np.testing.assert_allclose(
            result.report.scale_compression_ratio, 1.0, atol=1e-5
        )

    def test_merge_simple_mean_ratio_one_three_member_cluster(
        self, tmp_path: Path
    ):
        """3-member cluster, norms [1, 2, 6]. simple_mean → 3.0;
        preserved = 3.0×3 = 9.0; total before = 9.0 → ratio 1.0."""
        sae_path = _checkpoint(
            tmp_path, n_features=4, dec_norms={0: 1.0, 1: 2.0, 2: 6.0}
        )
        report = build_report(
            n_features=4, confirmed=[(0, 1), (0, 2), (1, 2)]
        )
        result = Compressor(
            validation_report=report,
            sae_checkpoint=sae_path,
            strategy="merge",
            merge_mode="simple_mean",
        ).run(tmp_path / "out.safetensors")
        np.testing.assert_allclose(
            result.report.scale_compression_ratio, 1.0, atol=1e-5
        )

    def test_no_clusters_ratio_is_one(self, tmp_path: Path):
        sae_path = _checkpoint(tmp_path, n_features=4)
        report = build_report(n_features=4, confirmed=[])
        result = Compressor(
            validation_report=report, sae_checkpoint=sae_path
        ).run(tmp_path / "out.safetensors")
        np.testing.assert_allclose(
            result.report.scale_compression_ratio, 1.0, atol=1e-9
        )


class TestRoundTripWithScaleFields:
    def test_report_json_round_trip_preserves_scale_fields(
        self, tmp_path: Path
    ):
        sae_path = _checkpoint(
            tmp_path, n_features=4, dec_norms={0: 1.0, 1: 3.0}
        )
        report_in = build_report(n_features=4, confirmed=[(0, 1)])
        result = Compressor(
            validation_report=report_in,
            sae_checkpoint=sae_path,
            strategy="merge",
            merge_mode="simple_mean",
        ).run(tmp_path / "out.safetensors")
        from polygram.compression.report import CompressionReport

        roundtrip = CompressionReport.from_json(result.report.to_json())
        assert roundtrip == result.report
        np.testing.assert_allclose(
            roundtrip.scale_compression_ratio,
            result.report.scale_compression_ratio,
        )
        for orig, rt in zip(
            result.plan.clusters, roundtrip.plan.clusters
        ):
            assert orig.cluster_norm_mean == rt.cluster_norm_mean
            assert orig.cluster_norm_std == rt.cluster_norm_std
            assert orig.merged_norm == rt.merged_norm
