"""Cross-panel aggregation: synthetic ValidationReport + global rep selection."""

from __future__ import annotations

import math

import numpy as np

from polygram.behavioural.report import (
    BucketStats,
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)
from polygram.compression.epoch import (
    Panel,
    _compute_global_n_fires,
    _pick_representatives_global,
    _synthesize_validation_report,
)


def _build_report(
    *,
    feature_ids: tuple[int, ...],
    confirmed: list[tuple[int, int]],
    polygram_overrides: dict[tuple[int, int], float] | None = None,
    jaccard_overrides: dict[tuple[int, int], float] | None = None,
    n_fires: dict[int, int] | None = None,
    n_both_fire: dict[tuple[int, int], int] | None = None,
) -> ValidationReport:
    polygram_overrides = polygram_overrides or {}
    jaccard_overrides = jaccard_overrides or {}
    n_fires = n_fires or {}
    n_both_fire = n_both_fire or {}
    confirmed_set = {tuple(sorted(p)) for p in confirmed}

    pairs: list[CandidatePair] = []
    for i_idx in range(len(feature_ids)):
        for j_idx in range(i_idx + 1, len(feature_ids)):
            i, j = int(feature_ids[i_idx]), int(feature_ids[j_idx])
            key = (i, j)
            is_confirmed = key in confirmed_set
            pairs.append(
                CandidatePair(
                    i=i, j=j,
                    polygram_overlap=polygram_overrides.get(
                        key, 0.8 if is_confirmed else 0.1
                    ),
                    decoder_overlap=0.9 if is_confirmed else 0.1,
                    jaccard=jaccard_overrides.get(
                        key, 0.5 if is_confirmed else 0.05
                    ),
                    pearson_activation=float("nan"),
                    kl_ablate_i=float("nan"),
                    kl_ablate_j=float("nan"),
                    kl_ratio_paired=float("nan"),
                    kl_log_ratio_abs=float("nan"),
                    n_fires_i=n_fires.get(i, 10 * (i + 1)),
                    n_fires_j=n_fires.get(j, 10 * (j + 1)),
                    n_both_fire=n_both_fire.get(key, 8 if is_confirmed else 1),
                    n_either_fire=15 if is_confirmed else 5,
                    gate_pass=is_confirmed,
                )
            )
    summary = ValidationSummary(
        spearman_polygram_jaccard=float("nan"),
        spearman_decoder_jaccard=float("nan"),
        spearman_polygram_log_kl_abs=float("nan"),
        pearson_polygram_jaccard=float("nan"),
        pearson_decoder_jaccard=float("nan"),
        buckets={},
        outcome="test",
    )
    return ValidationReport(
        schema_version=1, dictionary_name="Test", model_name="gpt2",
        layer=10, n_prompts=1, n_tokens=10,
        polygram_overlap_threshold=0.7, jaccard_threshold=0.30,
        min_firing_rate=0.0, min_both_fire=5,
        feature_ids=tuple(int(f) for f in feature_ids),
        pairs=tuple(pairs), summary=summary,
        confirmed=tuple(sorted(confirmed_set)),
    )


class TestSynthesis:
    def test_union_across_overlapping_panels(self, tmp_path):
        # P1: features (0, 1, 2) confirms (0, 1)
        # P2: features (0, 1, 3) confirms (0, 1) again + (3 unrelated)
        # P3: features (5, 6, 7) confirms (5, 6)
        # Expected union: {(0, 1), (5, 6)}
        p1 = _build_report(feature_ids=(0, 1, 2), confirmed=[(0, 1)])
        p2 = _build_report(feature_ids=(0, 1, 3), confirmed=[(0, 1)])
        p3 = _build_report(feature_ids=(5, 6, 7), confirmed=[(5, 6)])

        panels = [
            Panel(0, 0, (0, 1, 2), (0.9, 0.8)),
            Panel(1, 0, (0, 1, 3), (0.9, 0.7)),
            Panel(2, 5, (5, 6, 7), (0.85, 0.75)),
        ]
        synth = _synthesize_validation_report(
            panels, [p1, p2, p3], tmp_path / "src.safetensors"
        )
        assert sorted(synth.confirmed) == [(0, 1), (5, 6)]

    def test_max_aggregation_on_polygram_overlap(self, tmp_path):
        # Pair (0, 1) gets polygram=0.71 in P1, 0.85 in P2.
        # Synthetic report should report 0.85.
        p1 = _build_report(
            feature_ids=(0, 1, 2), confirmed=[(0, 1)],
            polygram_overrides={(0, 1): 0.71},
        )
        p2 = _build_report(
            feature_ids=(0, 1, 3), confirmed=[(0, 1)],
            polygram_overrides={(0, 1): 0.85},
        )
        panels = [
            Panel(0, 0, (0, 1, 2), (0.9, 0.8)),
            Panel(1, 0, (0, 1, 3), (0.9, 0.7)),
        ]
        synth = _synthesize_validation_report(
            panels, [p1, p2], tmp_path / "src.safetensors"
        )
        pair = next(p for p in synth.pairs if (p.i, p.j) == (0, 1))
        assert pair.polygram_overlap == 0.85

    def test_max_aggregation_on_jaccard(self, tmp_path):
        p1 = _build_report(
            feature_ids=(0, 1, 2), confirmed=[(0, 1)],
            jaccard_overrides={(0, 1): 0.4},
        )
        p2 = _build_report(
            feature_ids=(0, 1, 3), confirmed=[(0, 1)],
            jaccard_overrides={(0, 1): 0.7},
        )
        panels = [
            Panel(0, 0, (0, 1, 2), (0.9, 0.8)),
            Panel(1, 0, (0, 1, 3), (0.9, 0.7)),
        ]
        synth = _synthesize_validation_report(
            panels, [p1, p2], tmp_path / "src.safetensors"
        )
        pair = next(p for p in synth.pairs if (p.i, p.j) == (0, 1))
        assert pair.jaccard == 0.7


class TestRepresentatives:
    def test_global_n_fires_picks_highest_firing(self, tmp_path):
        # Cluster {3, 4, 5} where firing rates × n_tokens give:
        # fid 3 → 100, fid 4 → 50, fid 5 → 60
        # Rep should be fid 3.
        firing_rates = np.zeros(8, dtype=np.float32)
        firing_rates[3] = 1.0
        firing_rates[4] = 0.5
        firing_rates[5] = 0.6
        panels = [Panel(0, 3, (3, 4, 5), (0.95, 0.9))]
        global_n = _compute_global_n_fires(panels, firing_rates, n_tokens=100)
        assert global_n == {3: 100, 4: 50, 5: 60}

        # Build a synth report with cluster {3, 4, 5}
        synth = _build_report(
            feature_ids=(3, 4, 5),
            confirmed=[(3, 4), (3, 5), (4, 5)],
        )
        reps = _pick_representatives_global(synth, global_n)
        assert reps == {0: 3}

    def test_lowest_fid_tiebreak(self):
        synth = _build_report(
            feature_ids=(2, 3),
            confirmed=[(2, 3)],
        )
        # Equal n_fires → lowest fid wins.
        reps = _pick_representatives_global(synth, {2: 50, 3: 50})
        assert reps == {0: 2}

    def test_no_confirmed_returns_none(self):
        synth = _build_report(feature_ids=(0, 1, 2), confirmed=[])
        assert _pick_representatives_global(synth, {0: 10, 1: 5, 2: 3}) is None
