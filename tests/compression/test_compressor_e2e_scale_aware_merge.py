"""End-to-end integration test combining `strategy="merge"` with
`rep_selection="scale_aware"` on a synthetic 4-feature SAE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from polygram import Compressor
from tests._synth_sae import synth_sae
from tests.compression._fixtures import build_report


def test_scale_aware_merge_full_pipeline(tmp_path: Path):
    """
    Build a 4-feature SAE with two confirmed pairs forming two
    2-member clusters:
      cluster 0 = {0, 1}, norms 1.0 / 3.0
      cluster 1 = {2, 3}, norms 4.0 / 5.0

    With scale_aware, the median-proximate + ablation-rich member is
    chosen as rep; with merge+simple_mean each rep's row gets the
    cluster mean norm. End-state assertions cover:
    - rep choice on each cluster
    - rescaled W_dec norms
    - non-rep rows fully zeroed
    - scale_compression_ratio == 1.0 (simple_mean preserves mass)
    - merged_norm populated for both clusters
    """
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(
        sae_path,
        n_features=4,
        d_model=4,
        dec_norms={0: 1.0, 1: 3.0, 2: 4.0, 3: 5.0},
    )

    # Cluster 0: equal n_fires, fid 1 has higher kl_ablate → rep 1
    # Cluster 1: equal kl_ablate, fid 2 has higher n_fires → log_freq
    #            tilts the score; norms are close so norm_proximity
    #            ties; with kl_ablate equal, fid 2 should win.
    report = build_report(
        n_features=4,
        confirmed=[(0, 1), (2, 3)],
        n_fires={0: 50, 1: 50, 2: 100, 3: 50},
        kl_ablate={0: 0.1, 1: 1.0, 2: 0.5, 3: 0.5},
    )

    out_path = tmp_path / "out.safetensors"
    result = Compressor(
        validation_report=report,
        sae_checkpoint=sae_path,
        strategy="merge",
        rep_selection="scale_aware",
        merge_mode="simple_mean",
    ).run(out_path)

    # Rep choices
    cluster_by_id = {c.cluster_id: c for c in result.plan.clusters}
    assert cluster_by_id[0].representative == 1, (
        f"cluster 0 rep should be 1 (higher kl_ablate); "
        f"got {cluster_by_id[0].representative}"
    )
    assert cluster_by_id[1].representative == 2, (
        f"cluster 1 rep should be 2 (higher n_fires under tied "
        f"norm/ablation); got {cluster_by_id[1].representative}"
    )

    # Merged-norm arithmetic (simple_mean)
    np.testing.assert_allclose(cluster_by_id[0].merged_norm, 2.0, atol=1e-5)
    np.testing.assert_allclose(cluster_by_id[1].merged_norm, 4.5, atol=1e-5)

    # Cluster-norm stats
    np.testing.assert_allclose(
        cluster_by_id[0].cluster_norm_mean, 2.0, atol=1e-5
    )
    np.testing.assert_allclose(
        cluster_by_id[0].cluster_norm_std, 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        cluster_by_id[1].cluster_norm_mean, 4.5, atol=1e-5
    )
    np.testing.assert_allclose(
        cluster_by_id[1].cluster_norm_std, 0.5, atol=1e-5
    )

    # Output W_dec norms
    out = load_file(str(out_path))
    np.testing.assert_allclose(
        np.linalg.norm(out["W_dec"][1]), 2.0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(out["W_dec"][2]), 4.5, atol=1e-5
    )

    # Non-reps fully zeroed
    for fid in (0, 3):
        assert np.all(out["W_dec"][fid] == 0)
        assert np.all(out["W_enc"][:, fid] == 0)
        assert out["b_enc"][fid] == 0

    # scale_compression_ratio == 1.0 (simple_mean preserves total mass)
    np.testing.assert_allclose(
        result.report.scale_compression_ratio, 1.0, atol=1e-5
    )

    # Source bytes untouched
    # (re-read source post-run; should still be the original SAE)
    source = load_file(str(sae_path))
    np.testing.assert_allclose(
        np.linalg.norm(source["W_dec"][0]), 1.0, atol=1e-5
    )


def test_scale_aware_zero_strategy(tmp_path: Path):
    """Same setup but `strategy="zero"`: scale_aware still picks the
    rep but the merged_norm path is skipped."""
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(
        sae_path,
        n_features=4,
        d_model=4,
        dec_norms={0: 1.0, 1: 3.0, 2: 4.0, 3: 5.0},
    )
    report = build_report(
        n_features=4,
        confirmed=[(0, 1), (2, 3)],
        n_fires={0: 50, 1: 50, 2: 100, 3: 50},
        kl_ablate={0: 0.1, 1: 1.0, 2: 0.5, 3: 0.5},
    )
    out_path = tmp_path / "out.safetensors"
    result = Compressor(
        validation_report=report,
        sae_checkpoint=sae_path,
        strategy="zero",
        rep_selection="scale_aware",
    ).run(out_path)
    cluster_by_id = {c.cluster_id: c for c in result.plan.clusters}
    assert cluster_by_id[0].representative == 1
    assert cluster_by_id[1].representative == 2
    # Zero strategy: merged_norm is None
    for c in result.plan.clusters:
        assert c.merged_norm is None
    # Reps keep original norms; non-reps are zero
    out = load_file(str(out_path))
    np.testing.assert_allclose(
        np.linalg.norm(out["W_dec"][1]), 3.0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(out["W_dec"][2]), 4.0, atol=1e-5
    )
    for fid in (0, 3):
        assert np.all(out["W_dec"][fid] == 0)
