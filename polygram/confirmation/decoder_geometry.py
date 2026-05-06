from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from polygram.behavioural.report import (
    SCHEMA_VERSION,
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)
from polygram.sae_import import SAEFeatureRecord

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
class DecoderGeometryConfirmer:
    """Confirms redundant feature pairs purely from decoder cosine².

    Pairs whose ``decoder_cosine²  ≥ threshold`` are written to
    ``ValidationReport.confirmed``.  No torch or transformers import
    occurs; the report is suitable for direct use with :class:`Compressor`.

    Behavioural fields (``jaccard``, ``pearson_activation``, ``kl_*``)
    are ``NaN`` throughout.  ``model_name`` is set to
    ``"geometry:decoder_cosine2"`` as a sentinel.
    """

    records: dict[int, SAEFeatureRecord]
    sae_checkpoint: Path
    feature_ids: list[int]
    threshold: float = 0.8

    def __post_init__(self) -> None:
        self.sae_checkpoint = Path(self.sae_checkpoint)
        missing = [fid for fid in self.feature_ids if fid not in self.records]
        if missing:
            raise ValueError(
                f"DecoderGeometryConfirmer: feature_ids not in records: {missing}"
            )

    def run(self) -> ValidationReport:
        rows = np.stack(
            [self.records[fid].projection for fid in self.feature_ids],
            dtype=np.float64,
        )
        norms_sq = np.einsum("id,id->i", rows, rows)

        pairs: list[CandidatePair] = []
        confirmed: list[tuple[int, int]] = []

        n = len(self.feature_ids)
        for ii in range(n):
            wi_sq = float(norms_sq[ii])
            for jj in range(ii + 1, n):
                wj_sq = float(norms_sq[jj])
                denom = wi_sq * wj_sq
                decoder_overlap = (
                    float(np.dot(rows[ii], rows[jj])) ** 2 / denom
                    if denom > 0
                    else 0.0
                )
                fid_i = self.feature_ids[ii]
                fid_j = self.feature_ids[jj]
                gate = decoder_overlap >= self.threshold
                pairs.append(
                    CandidatePair(
                        i=fid_i,
                        j=fid_j,
                        polygram_overlap=_NAN,
                        decoder_overlap=decoder_overlap,
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
                        gate_pass=gate,
                    )
                )
                if gate:
                    confirmed.append((fid_i, fid_j))

        return ValidationReport(
            schema_version=SCHEMA_VERSION,
            dictionary_name="imported_sae",
            model_name="geometry:decoder_cosine2",
            layer=0,
            n_prompts=0,
            n_tokens=0,
            polygram_overlap_threshold=self.threshold,
            jaccard_threshold=0.0,
            min_firing_rate=0.0,
            min_both_fire=0,
            feature_ids=tuple(self.feature_ids),
            pairs=tuple(pairs),
            summary=_EMPTY_SUMMARY,
            confirmed=tuple(confirmed),
        )
