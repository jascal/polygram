"""Hand-built ValidationReport fixtures for compression tests.

`build_report(...)` synthesizes a `ValidationReport` whose
``confirmed`` list induces the union-find component structure the
tests assert against.
"""

from __future__ import annotations

from typing import Iterable

from polygram.behavioural.report import (
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)


def build_report(
    *,
    n_features: int,
    confirmed: Iterable[tuple[int, int]],
    n_fires: dict[int, int] | None = None,
    kl_ablate: dict[int, float] | None = None,
    dictionary_name: str = "FixtureDict",
) -> ValidationReport:
    """Hand-build a ValidationReport with the specified ``confirmed``
    pair list.

    All other per-pair fields are filled with placeholder values; only
    ``i``, ``j``, ``n_fires_i``, ``n_fires_j``, ``kl_ablate_i``,
    ``kl_ablate_j``, and ``gate_pass`` are consulted by `Compressor`.
    ``n_fires`` is a per-feature override map; missing features fall
    back to ``10 * (fid + 1)``. ``kl_ablate`` is a per-feature
    override map; missing features default to ``NaN`` (geometry-only
    semantics).
    """
    confirmed_set = {tuple(sorted(p)) for p in confirmed}
    n_fires = n_fires or {}
    kl_ablate = kl_ablate or {}

    def fires(fid: int) -> int:
        return n_fires.get(fid, 10 * (fid + 1))

    def ablate(fid: int) -> float:
        return kl_ablate.get(fid, float("nan"))

    pairs: list[CandidatePair] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            is_confirmed = (i, j) in confirmed_set
            pairs.append(
                CandidatePair(
                    i=i,
                    j=j,
                    polygram_overlap=0.8 if is_confirmed else 0.1,
                    decoder_overlap=0.9 if is_confirmed else 0.1,
                    jaccard=0.5 if is_confirmed else 0.05,
                    pearson_activation=float("nan"),
                    kl_ablate_i=ablate(i),
                    kl_ablate_j=ablate(j),
                    kl_ratio_paired=float("nan"),
                    kl_log_ratio_abs=float("nan"),
                    n_fires_i=fires(i),
                    n_fires_j=fires(j),
                    n_both_fire=8 if is_confirmed else 1,
                    n_either_fire=15 if is_confirmed else 5,
                    gate_pass=is_confirmed,
                )
            )

    summary = ValidationSummary(
        spearman_polygram_jaccard=0.7,
        spearman_decoder_jaccard=0.8,
        spearman_polygram_log_kl_abs=float("nan"),
        pearson_polygram_jaccard=0.6,
        pearson_decoder_jaccard=0.7,
        buckets={},
        outcome="partial",
    )
    return ValidationReport(
        schema_version=1,
        dictionary_name=dictionary_name,
        model_name="gpt2",
        layer=10,
        n_prompts=1,
        n_tokens=10,
        polygram_overlap_threshold=0.7,
        jaccard_threshold=0.30,
        min_firing_rate=0.01,
        min_both_fire=5,
        feature_ids=tuple(range(n_features)),
        pairs=tuple(pairs),
        summary=summary,
        confirmed=tuple(sorted(confirmed_set)),
    )
