from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from polygram.behavioural.report import (
    SCHEMA_VERSION,
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)
from polygram.sae_import import SAEFeatureRecord, SelectionReport

_NAN = float("nan")

_EMPTY_SUMMARY = ValidationSummary(
    spearman_polygram_jaccard=_NAN,
    spearman_decoder_jaccard=_NAN,
    spearman_polygram_log_kl_abs=_NAN,
    pearson_polygram_jaccard=_NAN,
    pearson_decoder_jaccard=_NAN,
    buckets={},
    outcome="geometry_only",
)


@dataclass
class ClusterConfirmer:
    """Confirms redundant feature pairs from ``SelectionReport`` cluster membership.

    All within-cluster ``(i, j)`` pairs (``i < j``) are written to
    ``ValidationReport.confirmed``.  Singleton clusters produce no confirmed
    pairs.  No torch or transformers import occurs.

    ``model_name`` is set to ``"geometry:cluster"`` as a sentinel.
    Behavioural fields are ``NaN`` throughout.
    """

    selection_report: SelectionReport
    sae_checkpoint: Path
    records: dict[int, SAEFeatureRecord]

    def __post_init__(self) -> None:
        self.sae_checkpoint = Path(self.sae_checkpoint)

    def run(self) -> ValidationReport:
        name_to_id: dict[str, int] = {
            r.name: r.feature_id for r in self.records.values()
        }

        cluster_to_ids: dict[str, list[int]] = defaultdict(list)
        for fname, cname in self.selection_report.cluster_assignments.items():
            fid = name_to_id[fname]
            cluster_to_ids[cname].append(fid)

        confirmed: list[tuple[int, int]] = []
        pairs: list[CandidatePair] = []

        for members in cluster_to_ids.values():
            members_sorted = sorted(members)
            for ii, fid_i in enumerate(members_sorted):
                for fid_j in members_sorted[ii + 1 :]:
                    confirmed.append((fid_i, fid_j))
                    pairs.append(
                        CandidatePair(
                            i=fid_i,
                            j=fid_j,
                            polygram_overlap=_NAN,
                            decoder_overlap=_NAN,
                            jaccard=_NAN,
                            pearson_activation=_NAN,
                            kl_ablate_i=_NAN,
                            kl_ablate_j=_NAN,
                            kl_ratio_paired=_NAN,
                            kl_log_ratio_abs=_NAN,
                            n_fires_i=0,
                            n_fires_j=0,
                            n_both_fire=0,
                            n_either_fire=0,
                            gate_pass=True,
                        )
                    )

        all_feature_ids = sorted(name_to_id[fname]
                                 for fname in self.selection_report.cluster_assignments)

        return ValidationReport(
            schema_version=SCHEMA_VERSION,
            dictionary_name="imported_sae",
            model_name="geometry:cluster",
            layer=0,
            n_prompts=0,
            n_tokens=0,
            polygram_overlap_threshold=0.0,
            jaccard_threshold=0.0,
            min_firing_rate=0.0,
            min_both_fire=0,
            feature_ids=tuple(all_feature_ids),
            pairs=tuple(pairs),
            summary=_EMPTY_SUMMARY,
            confirmed=tuple(confirmed),
        )
