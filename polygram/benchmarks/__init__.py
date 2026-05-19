"""Benchmark utilities for measuring polygram primitives at SAE scale."""

from polygram.benchmarks.clustered_amortised import (
    BenchmarkReport,
    run_amortised_benchmark,
)

__all__ = ["BenchmarkReport", "run_amortised_benchmark"]
