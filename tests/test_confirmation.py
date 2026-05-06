"""Tests for polygram.confirmation — Confirmer protocol, DecoderGeometryConfirmer,
ClusterConfirmer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polygram import (
    ClusterConfirmer,
    Confirmer,
    DecoderGeometryConfirmer,
    SAEFeatureRecord,
    SelectionReport,
    ValidationReport,
    from_sae_lens,
    load_toy_sae,
)
from polygram.behavioural import BehaviouralValidator

FIXTURE = Path(__file__).parent / "fixtures" / "toy_sae.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(*projections: list[float]) -> dict[int, SAEFeatureRecord]:
    return {
        i: SAEFeatureRecord(feature_id=i, name=f"feat_{i}", projection=np.array(p))
        for i, p in enumerate(projections)
    }


def _synth_sae_checkpoint(tmp_path: Path, records: dict[int, SAEFeatureRecord]) -> Path:
    from safetensors.numpy import save_file

    n = len(records)
    d = len(next(iter(records.values())).projection)
    rng = np.random.default_rng(0)
    W_dec = np.stack([records[i].projection.astype(np.float32) for i in range(n)])
    path = tmp_path / "sae.safetensors"
    save_file(
        {
            "W_dec": W_dec,
            "W_enc": rng.standard_normal((d, n)).astype(np.float32),
            "b_dec": rng.standard_normal(d).astype(np.float32),
            "b_enc": rng.standard_normal(n).astype(np.float32),
        },
        str(path),
    )
    return path


# ---------------------------------------------------------------------------
# 2.2 — Confirmer protocol
# ---------------------------------------------------------------------------


class TestConfirmerProtocol:
    def test_behaviouralvalidator_satisfies_confirmer(self):
        assert isinstance(BehaviouralValidator.__new__(BehaviouralValidator), Confirmer)

    def test_custom_class_satisfies_confirmer(self):
        class MyConfirmer:
            def run(self) -> ValidationReport:  # type: ignore[empty-body]
                ...

        assert isinstance(MyConfirmer(), Confirmer)

    def test_object_without_run_does_not_satisfy(self):
        class NotAConfirmer:
            pass

        assert not isinstance(NotAConfirmer(), Confirmer)


# ---------------------------------------------------------------------------
# 3.4 — DecoderGeometryConfirmer
# ---------------------------------------------------------------------------


class TestDecoderGeometryConfirmer:
    def _make(self, tmp_path, projs, threshold=0.8):
        records = _make_records(*projs)
        path = _synth_sae_checkpoint(tmp_path, records)
        feature_ids = list(records.keys())
        return DecoderGeometryConfirmer(
            records=records,
            sae_checkpoint=path,
            feature_ids=feature_ids,
            threshold=threshold,
        )

    def test_above_threshold_pair_confirmed(self, tmp_path):
        # Two nearly identical vectors — cosine² ≈ 1.0
        v = [1.0, 0.0, 0.0]
        w = [0.999, 0.045, 0.0]  # cosine² ≈ 0.998
        confirmer = self._make(tmp_path, [v, w], threshold=0.8)
        report = confirmer.run()
        assert (0, 1) in report.confirmed

    def test_below_threshold_pair_not_confirmed(self, tmp_path):
        # Orthogonal vectors — cosine² = 0.0
        v = [1.0, 0.0, 0.0]
        w = [0.0, 1.0, 0.0]
        confirmer = self._make(tmp_path, [v, w], threshold=0.8)
        report = confirmer.run()
        assert (0, 1) not in report.confirmed
        assert len(report.confirmed) == 0

    def test_exact_threshold_is_included(self, tmp_path):
        # threshold=0.5; two vectors with cosine² exactly 0.5
        v = [1.0, 0.0]
        w = [1.0, 1.0]  # cosine² = 0.5
        confirmer = self._make(tmp_path, [v, w], threshold=0.5)
        report = confirmer.run()
        assert (0, 1) in report.confirmed

    def test_no_torch_required(self, tmp_path, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        v = [1.0, 0.0]
        w = [1.0, 0.0]
        confirmer = self._make(tmp_path, [v, w], threshold=0.5)
        report = confirmer.run()  # must not raise
        assert isinstance(report, ValidationReport)

    def test_metadata_sentinel(self, tmp_path):
        records = _make_records([1.0, 0.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        confirmer = DecoderGeometryConfirmer(records=records, sae_checkpoint=path, feature_ids=[0])
        report = confirmer.run()
        assert report.model_name == "geometry:decoder_cosine2"
        assert report.n_prompts == 0
        assert report.n_tokens == 0

    def test_behavioural_fields_are_nan(self, tmp_path):
        records = _make_records([1.0, 0.0], [0.0, 1.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        confirmer = DecoderGeometryConfirmer(records=records, sae_checkpoint=path, feature_ids=[0, 1])
        report = confirmer.run()
        for pair in report.pairs:
            assert math.isnan(pair.jaccard)
            assert math.isnan(pair.pearson_activation)
            assert math.isnan(pair.kl_ablate_i)
            assert math.isnan(pair.kl_ablate_j)

    def test_pairs_ordered_i_lt_j(self, tmp_path):
        records = _make_records([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        confirmer = DecoderGeometryConfirmer(records=records, sae_checkpoint=path, feature_ids=[0, 1, 2])
        report = confirmer.run()
        for i, j in report.confirmed:
            assert i < j

    def test_missing_feature_id_raises(self, tmp_path):
        records = _make_records([1.0, 0.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        with pytest.raises(ValueError, match="not in records"):
            DecoderGeometryConfirmer(records=records, sae_checkpoint=path, feature_ids=[0, 99])


# ---------------------------------------------------------------------------
# 4.3 — ClusterConfirmer
# ---------------------------------------------------------------------------


class TestClusterConfirmer:
    def _make_report(self, assignments: dict[str, str], n_selected=None) -> SelectionReport:
        return SelectionReport(
            n_input_features=16,
            n_selected=n_selected or len(assignments),
            cluster_assignments=assignments,
            cluster_method="kmeans",
            beta_variance_explained=0.5,
        )

    def test_within_cluster_pairs_confirmed(self, tmp_path):
        records = _make_records([1.0, 0.0], [0.9, 0.1], [0.0, 1.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        sr = self._make_report({"feat_0": "A", "feat_1": "A", "feat_2": "B"})
        confirmer = ClusterConfirmer(selection_report=sr, sae_checkpoint=path, records=records)
        report = confirmer.run()
        assert (0, 1) in report.confirmed
        assert len(report.confirmed) == 1

    def test_cross_cluster_pairs_not_confirmed(self, tmp_path):
        records = _make_records([1.0, 0.0], [0.0, 1.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        sr = self._make_report({"feat_0": "A", "feat_1": "B"})
        confirmer = ClusterConfirmer(selection_report=sr, sae_checkpoint=path, records=records)
        report = confirmer.run()
        assert len(report.confirmed) == 0

    def test_singleton_clusters_empty_confirmed(self, tmp_path):
        records = _make_records([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        sr = self._make_report({"feat_0": "A", "feat_1": "B", "feat_2": "C"})
        confirmer = ClusterConfirmer(selection_report=sr, sae_checkpoint=path, records=records)
        report = confirmer.run()
        assert report.confirmed == ()

    def test_three_member_cluster_three_pairs(self, tmp_path):
        records = _make_records([1.0, 0.0], [0.9, 0.1], [0.8, 0.2])
        path = _synth_sae_checkpoint(tmp_path, records)
        sr = self._make_report({"feat_0": "A", "feat_1": "A", "feat_2": "A"})
        confirmer = ClusterConfirmer(selection_report=sr, sae_checkpoint=path, records=records)
        report = confirmer.run()
        assert set(report.confirmed) == {(0, 1), (0, 2), (1, 2)}

    def test_metadata_sentinel(self, tmp_path):
        records = _make_records([1.0, 0.0])
        path = _synth_sae_checkpoint(tmp_path, records)
        sr = self._make_report({"feat_0": "A"})
        confirmer = ClusterConfirmer(selection_report=sr, sae_checkpoint=path, records=records)
        report = confirmer.run()
        assert report.model_name == "geometry:cluster"
        assert report.n_prompts == 0
        assert report.n_tokens == 0

    def test_works_with_toy_sae_fixture(self, tmp_path):
        records = load_toy_sae(FIXTURE)
        from safetensors.numpy import save_file

        W_dec = np.stack(
            [records[i].projection.astype(np.float32) for i in range(len(records))]
        )
        rng = np.random.default_rng(1)
        d = W_dec.shape[1]
        path = tmp_path / "toy.safetensors"
        save_file(
            {
                "W_dec": W_dec,
                "W_enc": rng.standard_normal((d, len(records))).astype(np.float32),
                "b_dec": rng.standard_normal(d).astype(np.float32),
                "b_enc": rng.standard_normal(len(records)).astype(np.float32),
            },
            str(path),
        )
        _, sr = from_sae_lens(records, [0, 1, 4, 5], n_clusters=2)
        confirmer = ClusterConfirmer(selection_report=sr, sae_checkpoint=path, records=records)
        report = confirmer.run()
        # Toy fixture has clear cluster structure; at least one pair should be confirmed
        assert len(report.confirmed) >= 1
        for i, j in report.confirmed:
            assert i < j
