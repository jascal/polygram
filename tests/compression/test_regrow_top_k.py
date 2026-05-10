"""Tests for the `top_k` cap on per-call regrow count.

`top_k=None` (default) preserves byte-equivalence with the
pre-change behavior — every zeroed slot regrown. Integer values
cap the per-call regrowth in `RegrowPlan` (plan-order) order.

Acceptance gate: the byte-equivalence test under `None` is the
load-bearing check that downstream callers' default behavior is
unchanged. The remaining tests pin functional behavior under the
new lever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from polygram import RegrowConfig, Regrower
from tests._synth_sae import synth_sae


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _setup(tmp_path: Path, *, n_features: int = 16, d_model: int = 8,
           zeroed: tuple[int, ...] = (2, 5, 9, 13)) -> Path:
    """Build a synth SAE with `zeroed` rows blanked out."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    sae_path = tmp_path / "sae.safetensors"
    synth_sae(sae_path, n_features=n_features, d_model=d_model)
    state = load_file(str(sae_path))
    for fid in zeroed:
        state["W_enc"][:, fid] = 0
        state["b_enc"][fid] = 0
        state["W_dec"][fid, :] = 0
    from safetensors.numpy import save_file
    save_file(state, str(sae_path))
    return sae_path


def _build(sae_path: Path, *, zeroed: set[int], top_k: int | None = None,
           seed: int = 0) -> Regrower:
    residuals = np.random.default_rng(seed).standard_normal(
        (200, 8)
    ).astype(np.float32)
    return Regrower(
        sae_checkpoint=sae_path,
        strategy="residual_kmeans",
        zeroed=zeroed,
        cached_residuals=residuals,
        seed=seed,
        n_init=4,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# §4 — Byte-equivalence acceptance gate
# ---------------------------------------------------------------------------


class TestByteEquivalenceUnderNone:
    """`top_k=None` must take the same code path as the pre-change
    regrower. We pin this empirically: the output checkpoint of a
    `top_k=None` run is byte-identical to a control regrower built
    without ever touching the field.

    Pre-change behavior is the control: the regrower regrows every
    zeroed slot. With the new field defaulting to None, the rewritten
    state-dict bytes must match exactly.
    """

    def test_top_k_none_is_byte_identical_to_unset(self, tmp_path: Path):
        sae_path_a = _setup(tmp_path / "a")
        sae_path_b = _setup(tmp_path / "b")
        # Identical inputs except sae_path location (so output paths can
        # differ). The synth helper is deterministic, so the source
        # bytes are identical.
        assert sae_path_a.read_bytes() == sae_path_b.read_bytes()

        rng_seed = 7
        residuals = np.random.default_rng(rng_seed).standard_normal(
            (200, 8)
        ).astype(np.float32)

        # Control: do not pass top_k at all (uses class default).
        r_control = Regrower(
            sae_checkpoint=sae_path_a,
            strategy="residual_kmeans",
            zeroed={2, 5, 9, 13},
            cached_residuals=residuals,
            seed=rng_seed,
            n_init=4,
        )
        # Treatment: explicitly pass top_k=None.
        r_treatment = Regrower(
            sae_checkpoint=sae_path_b,
            strategy="residual_kmeans",
            zeroed={2, 5, 9, 13},
            cached_residuals=residuals,
            seed=rng_seed,
            n_init=4,
            top_k=None,
        )
        out_control = r_control.run(tmp_path / "out_control.safetensors")
        out_treatment = r_treatment.run(tmp_path / "out_treatment.safetensors")

        s_c = load_file(str(out_control.output_checkpoint))
        s_t = load_file(str(out_treatment.output_checkpoint))
        for k in s_c:
            assert np.array_equal(s_c[k], s_t[k]), (
                f"tensor {k} differs between top_k=None and unset"
            )
        assert (
            out_control.report.output_checkpoint_sha256
            == out_treatment.report.output_checkpoint_sha256
        )


# ---------------------------------------------------------------------------
# §5 — Functional tests
# ---------------------------------------------------------------------------


class TestTopKFunctional:
    def test_top_k_caps_population_count(self, tmp_path: Path):
        # 10-slot fixture; cap at 3.
        zeroed_fids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        sae_path = _setup(
            tmp_path, n_features=24, d_model=8, zeroed=zeroed_fids
        )
        r = _build(sae_path, zeroed=set(zeroed_fids), top_k=3)
        result = r.run(tmp_path / "out.safetensors")

        # Plan order is sorted ascending by feature_id → first 3 are
        # 1, 2, 3.
        regrown = sorted(int(s.feature_id) for s in result.plan.slots)
        assert regrown == [1, 2, 3]
        assert len(result.plan.slots) == 3
        assert result.report.n_slots_repopulated <= 3

        # The remaining 7 slots stay zero.
        s_out = load_file(str(result.output_checkpoint))
        un_regrown = [f for f in zeroed_fids if f not in {1, 2, 3}]
        for fid in un_regrown:
            assert (s_out["W_dec"][fid, :] == 0).all(), (
                f"un-regrown slot {fid} was modified in W_dec"
            )
            assert (s_out["W_enc"][:, fid] == 0).all(), (
                f"un-regrown slot {fid} was modified in W_enc"
            )

    def test_top_k_above_zeroed_count_is_no_op_cap(self, tmp_path: Path):
        zeroed_fids = (2, 5, 9, 13, 14)
        sae_path = _setup(
            tmp_path, n_features=16, d_model=8, zeroed=zeroed_fids
        )
        r = _build(sae_path, zeroed=set(zeroed_fids), top_k=999)
        result = r.run(tmp_path / "out.safetensors")
        # All 5 slots in the plan.
        assert len(result.plan.slots) == 5

    def test_top_k_zero_is_no_regrow(self, tmp_path: Path):
        zeroed_fids = (2, 5, 9, 13, 14)
        sae_path = _setup(
            tmp_path, n_features=16, d_model=8, zeroed=zeroed_fids
        )
        r = _build(sae_path, zeroed=set(zeroed_fids), top_k=0)
        result = r.run(tmp_path / "out.safetensors")

        # No slots in the plan → no rows changed in the output.
        assert len(result.plan.slots) == 0
        s_out = load_file(str(result.output_checkpoint))
        s_in = load_file(str(sae_path))
        for k in s_out:
            assert np.array_equal(s_out[k], s_in[k]), (
                f"tensor {k} unexpectedly changed under top_k=0"
            )

    def test_top_k_deterministic(self, tmp_path: Path):
        zeroed_fids = (2, 5, 9, 13)
        sae_path_a = _setup(tmp_path / "a", zeroed=zeroed_fids)
        sae_path_b = _setup(tmp_path / "b", zeroed=zeroed_fids)
        residuals = np.random.default_rng(42).standard_normal(
            (200, 8)
        ).astype(np.float32)
        r1 = Regrower(
            sae_checkpoint=sae_path_a,
            strategy="residual_kmeans",
            zeroed=set(zeroed_fids),
            cached_residuals=residuals,
            seed=0, n_init=4, top_k=3,
        )
        r2 = Regrower(
            sae_checkpoint=sae_path_b,
            strategy="residual_kmeans",
            zeroed=set(zeroed_fids),
            cached_residuals=residuals,
            seed=0, n_init=4, top_k=3,
        )
        out1 = r1.run(tmp_path / "out1.safetensors")
        out2 = r2.run(tmp_path / "out2.safetensors")
        s1 = load_file(str(out1.output_checkpoint))
        s2 = load_file(str(out2.output_checkpoint))
        for k in s1:
            assert np.array_equal(s1[k], s2[k]), (
                f"tensor {k} differs across deterministic top_k=3 runs"
            )
        assert (
            out1.report.output_checkpoint_sha256
            == out2.report.output_checkpoint_sha256
        )

    def test_top_k_negative_raises_in_config(self):
        with pytest.raises(ValueError, match=r"top_k.*-1"):
            RegrowConfig(model_name="pythia-160m", layer=10, top_k=-1)

    def test_top_k_negative_raises_in_regrower(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        residuals = np.random.default_rng(0).standard_normal(
            (200, 8)
        ).astype(np.float32)
        with pytest.raises(ValueError, match=r"top_k"):
            Regrower(
                sae_checkpoint=sae_path,
                strategy="residual_kmeans",
                zeroed={2, 5, 9, 13},
                cached_residuals=residuals,
                top_k=-1,
            )


# ---------------------------------------------------------------------------
# §3 — from_compression_report kwarg precedence
# ---------------------------------------------------------------------------


def _hand_built_compression_report(sae_path: Path, zeroed_fids: tuple[int, ...]):
    from polygram import ClusterPlan, CompressionPlan, CompressionReport

    clusters = tuple(
        ClusterPlan(
            cluster_id=i, members=(fid,), representative=fid, zeroed=(fid,)
        )
        for i, fid in enumerate(zeroed_fids)
    )
    return CompressionReport(
        schema_version=1,
        source_checkpoint="/somewhere/source.safetensors",
        source_checkpoint_sha256="a" * 64,
        output_checkpoint=str(sae_path),
        output_checkpoint_sha256="b" * 64,
        validation_report_dictionary_name="UpstreamDict",
        validation_report_schema_version=1,
        strategy="zero",
        plan=CompressionPlan(
            clusters=clusters,
            feature_ids=tuple(range(16)),
        ),
        n_features_zeroed=len(zeroed_fids),
        n_features_kept=16 - len(zeroed_fids),
        n_clusters=len(zeroed_fids),
    )


class TestFromCompressionReportTopK:
    def test_kwarg_only(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path, (2, 5, 9, 13))
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans",
            cached_residuals=residuals,
            model_name="gpt2", layer=10,
            top_k=3,
        )
        assert r.top_k == 3

    def test_config_only(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path, (2, 5, 9, 13))
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        cfg = RegrowConfig(model_name="gpt2", layer=10, top_k=4)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans",
            cached_residuals=residuals,
            config=cfg,
        )
        assert r.top_k == 4

    def test_kwarg_wins_over_config(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path, (2, 5, 9, 13))
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        cfg = RegrowConfig(model_name="gpt2", layer=10, top_k=5)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans",
            cached_residuals=residuals,
            config=cfg, top_k=2,
        )
        assert r.top_k == 2

    def test_neither_set_defaults_to_none(self, tmp_path: Path):
        sae_path = _setup(tmp_path)
        report = _hand_built_compression_report(sae_path, (2, 5, 9, 13))
        residuals = np.random.default_rng(0).standard_normal(
            (50, 8)
        ).astype(np.float32)
        r = Regrower.from_compression_report(
            report, sae_checkpoint=sae_path,
            strategy="residual_kmeans",
            cached_residuals=residuals,
            model_name="gpt2", layer=10,
        )
        assert r.top_k is None
