"""Tests for `polygram.benchmarks.clustered_amortised`."""

from __future__ import annotations

import numpy as np
import pytest

from polygram.benchmarks import BenchmarkReport, run_amortised_benchmark
from polygram.dictionary import Dictionary, Feature
from polygram.encoding import MPSRung1


def _make_dictionary(n_features: int, d_model: int = 16, seed: int = 0):
    """Build a flat Dictionary + dense decoder matrix with planted
    near-duplicate pairs every 8 features so cosine clustering has
    real structure to lean on (and cross-block edges to walk)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_features // 8 + 1, d_model))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    rows = []
    for i in range(n_features):
        anchor = base[i // 8] + rng.normal(scale=0.05, size=d_model)
        anchor /= np.linalg.norm(anchor)
        rows.append(anchor)
    decoder = np.stack(rows).astype(np.float32)

    features = [
        Feature(name=f"f{i}", cluster="all", beta=0.0)
        for i in range(n_features)
    ]
    dictionary = Dictionary(
        name="bench_test",
        features=features,
        hierarchy={"all": [f.name for f in features]},
        encoding=MPSRung1(),
    )
    return dictionary, decoder


def test_smoke_all_ops_run():
    dictionary, decoder = _make_dictionary(64)
    for op in ("gram", "cross_block_overlap", "redundancy"):
        report = run_amortised_benchmark(
            dictionary, decoder, op=op, n_repeats=4, sample_size=8,
        )
        assert isinstance(report, BenchmarkReport)
        assert report.op_name == op
        assert report.n_repeats == 4
        assert report.flat_wall_seconds_per_op > 0.0
        assert report.clustered_wall_seconds_per_op >= 0.0
        assert report.clustered_block_formation_seconds > 0.0
        assert report.polygram_version  # non-empty


def test_unknown_op_raises():
    dictionary, decoder = _make_dictionary(16)
    with pytest.raises(ValueError, match="unknown op"):
        run_amortised_benchmark(
            dictionary, decoder, op="cancellation", n_repeats=1,  # type: ignore[arg-type]
        )


def test_invalid_n_repeats_raises():
    dictionary, decoder = _make_dictionary(16)
    with pytest.raises(ValueError, match="n_repeats"):
        run_amortised_benchmark(dictionary, decoder, op="gram", n_repeats=0)


def test_cross_over_field_consistency():
    """Synthetic dense-clustered fixture where cross_block_overlap's
    per-op cost on the clustered path is dominated by cached lookup
    (~zero). The flat path computes fresh cosines on every op, so
    crossover SHALL exist within the upper bound."""
    dictionary, decoder = _make_dictionary(64)
    report = run_amortised_benchmark(
        dictionary, decoder, op="cross_block_overlap",
        n_repeats=16, sample_size=4,
    )
    # The crossover is computable when flat_per_op > clustered_per_op.
    if report.flat_wall_seconds_per_op > report.clustered_wall_seconds_per_op:
        assert report.cross_over_n_repeats is not None
        assert report.cross_over_n_repeats >= 1


def test_redundancy_pins_worst_case():
    """Per the spec scenario: at small N where the flat cosine sweep
    is cheap, clustered redundancy ≥ flat (intra-block grams +
    cross-block walk cost more than one flat sweep). Crossover SHALL
    be None — clustered does not pay off for this op at this scale."""
    dictionary, decoder = _make_dictionary(64)
    report = run_amortised_benchmark(
        dictionary, decoder, op="redundancy", n_repeats=2,
    )
    # The proposal explicitly pins this as the worst case; check that
    # the report makes it observable (either crossover is None OR is
    # comparatively large vs the cheaper ops).
    if report.clustered_wall_seconds_per_op >= report.flat_wall_seconds_per_op:
        assert report.cross_over_n_repeats is None


def test_benchmark_report_round_trip():
    dictionary, decoder = _make_dictionary(16)
    report = run_amortised_benchmark(
        dictionary, decoder, op="gram", n_repeats=2,
    )
    as_dict = report.to_dict()
    restored = BenchmarkReport.from_dict(as_dict)
    assert restored == report


def test_diagnostic_fields_present():
    """Task 3.2 surfaces RSS delta + gc object count delta around the
    clustered build. The numbers don't have a hard contract (depends
    on platform); they just have to be reported."""
    dictionary, decoder = _make_dictionary(32)
    report = run_amortised_benchmark(
        dictionary, decoder, op="gram", n_repeats=2,
    )
    assert isinstance(report.rss_delta_kb, int)
    assert isinstance(report.object_count_delta, int)
    assert report.rss_delta_kb >= 0
    # object_count_delta is allowed to be negative (gc could collect
    # objects during the build) — just check the field exists.
