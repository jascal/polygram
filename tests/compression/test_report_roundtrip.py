"""`CompressionReport.to_json(...)` ↔ `from_json(...)` round-trip."""

from __future__ import annotations

from polygram.compression.report import (
    SCHEMA_VERSION,
    ClusterPlan,
    CompressionPlan,
    CompressionReport,
)


def _hand_built_report() -> CompressionReport:
    plan = CompressionPlan(
        clusters=(
            ClusterPlan(
                cluster_id=0,
                members=(0, 1),
                representative=1,
                zeroed=(0,),
            ),
            ClusterPlan(
                cluster_id=1,
                members=(3, 4, 5),
                representative=5,
                zeroed=(3, 4),
            ),
        ),
        feature_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    )
    return CompressionReport(
        schema_version=SCHEMA_VERSION,
        source_checkpoint="/path/to/sae.safetensors",
        source_checkpoint_sha256="a" * 64,
        output_checkpoint="/path/to/sae.compressed.safetensors",
        output_checkpoint_sha256="b" * 64,
        validation_report_dictionary_name="HandBuilt",
        validation_report_schema_version=1,
        strategy="zero",
        plan=plan,
        n_features_zeroed=3,
        n_features_kept=2,
        n_clusters=2,
    )


class TestRoundTrip:
    def test_string_round_trip(self):
        r = _hand_built_report()
        s = r.to_json()
        rt = CompressionReport.from_json(s)
        assert rt == r

    def test_path_round_trip(self, tmp_path):
        r = _hand_built_report()
        out = tmp_path / "report.json"
        r.to_json(out)
        assert out.is_file()
        rt = CompressionReport.from_json(out)
        assert rt == r

    def test_payload_carries_required_keys(self, tmp_path):
        r = _hand_built_report()
        import json

        payload = json.loads(r.to_json())
        for key in (
            "schema_version",
            "source_checkpoint",
            "source_checkpoint_sha256",
            "output_checkpoint",
            "output_checkpoint_sha256",
            "validation_report_dictionary_name",
            "validation_report_schema_version",
            "strategy",
            "feature_ids",
            "clusters",
            "n_features_zeroed",
            "n_features_kept",
            "n_clusters",
        ):
            assert key in payload, f"missing key {key!r}"

        # Cluster shape
        c0 = payload["clusters"][0]
        for key in ("cluster_id", "members", "representative", "zeroed"):
            assert key in c0
