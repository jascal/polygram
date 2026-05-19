"""Downstream-amortised benchmark for `ClusteredDictionary`.

PR #91 ran the §12 killer experiment from `clustered-dictionary-analysis`
at full SAE width (N=24,576) and produced a load-bearing mixed verdict:
clustering catches every redundant pair the flat O(N²) sweep finds
(recall = 1.000) but is ~0.9× the speed of flat for one-shot
redundancy enumeration. The honest framing landed in
`docs/research/clustered-dictionary-recall-vs-flat.md`: clustering's
savings live **downstream** of detection — per-block O(K²) ops
amortise against flat's O(N²) once you run more than a couple of
ops on one fixed `ClusteredDictionary`.

This module measures that downstream-amortisation claim directly.
`run_amortised_benchmark(...)` times two paths separately:

- **Flat**: pay the O(N²) cost once per operation. No block-formation
  cache; each repeat re-derives whatever pairwise structure the op
  needs from scratch.
- **Clustered**: pay block formation once (the amortised cost), then
  run the operation `n_repeats` times against the cached partition.

The crossover metric `cross_over_n_repeats` is the smallest `k` at
which the clustered total cost beats the flat total cost. The
benchmark is built around three ops with very different
expected behaviour:

- ``gram`` — clustering's strongest case. Flat doesn't have an
  exact analytic counterpart at SAE scale (the encoding cap forbids
  building a flat `Dictionary` past 8/16/32 features), so the flat
  baseline is a full N×N cosine matmul — the same shape of O(N²)
  work an unconstrained analytic gram would have to do.
- ``cross_block_overlap`` — sample-based cross-block edge cosines.
  Clustered amortises because `cross_block_pairs` is pre-computed
  at build time; flat resamples N choose 2 every op.
- ``redundancy`` — the worst case for clustering per #91; pinned
  here so the report explicitly records the regime where the
  amortisation thesis does *not* pay off.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

import polygram
from polygram.clustered_dictionary import (
    BlockFormation,
    ClusteredDictionary,
    build_clustered_dictionary,
    compute_cosine_pair_graph,
)
from polygram.dictionary import Dictionary


BenchmarkOp = Literal["gram", "cross_block_overlap", "redundancy"]
_SUPPORTED_OPS: tuple[BenchmarkOp, ...] = (
    "gram",
    "cross_block_overlap",
    "redundancy",
)


def _rss_kb() -> int:
    """Process-wide RSS in KB. macOS reports bytes, Linux reports KB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(raw // 1024)
    return int(raw)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class BenchmarkReport:
    """Result of one `run_amortised_benchmark` invocation.

    Fields with the ``_per_op`` suffix are per-repeat averages; the
    crossover metric and totals are derived from these and the one-time
    ``clustered_block_formation_seconds``.

    The diagnostic ``rss_delta_kb`` and ``object_count_delta`` fields
    are measured across the clustered build (NOT per op) and exist to
    serve Task 3.2's "if cross-over fails, surface the dominant cost"
    requirement.
    """

    op_name: BenchmarkOp
    n_repeats: int
    flat_wall_seconds_per_op: float
    clustered_wall_seconds_per_op: float
    clustered_block_formation_seconds: float
    cross_over_n_repeats: int | None
    rss_delta_kb: int
    object_count_delta: int
    sample_size_requested: int
    sample_size_effective: int
    polygram_version: str
    git_commit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkReport":
        return cls(**dict(data))


def _compute_cross_over(
    block_formation_s: float,
    flat_per_op: float,
    clustered_per_op: float,
    upper_bound: int,
) -> int | None:
    """Smallest `k` such that `block_formation + k*clustered < k*flat`.

    Returns ``None`` when clustered's per-op cost meets or exceeds
    flat's (no crossover possible), or when no `k <= upper_bound`
    satisfies the inequality.
    """
    if flat_per_op <= clustered_per_op:
        return None
    raw = block_formation_s / (flat_per_op - clustered_per_op)
    k = max(1, int(np.ceil(raw)))
    if k > upper_bound:
        return None
    return k


def _flat_op_gram(decoder_vectors: np.ndarray) -> int:
    """Flat baseline for `gram`: a full N×N decoder cosine matmul.

    Returns the count of off-diagonal entries (sanity; not used)."""
    norms = np.linalg.norm(decoder_vectors, axis=1)
    norms[norms == 0.0] = 1.0
    unit = decoder_vectors / norms[:, None]
    pairs = unit @ unit.T
    n = pairs.shape[0]
    return int(n * (n - 1) // 2)


def _clustered_op_gram(clustered: ClusteredDictionary) -> int:
    """Clustered: materialise every block's analytic gram once."""
    count = 0
    for block in clustered.blocks:
        g = block.gram()
        count += g.size
    return count


def _flat_op_cross_block_overlap(
    decoder_vectors: np.ndarray, sample_size: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Flat: sample `sample_size` index pairs, compute decoder cosines.

    Returns (requested, effective). Capped to N choose 2."""
    n = decoder_vectors.shape[0]
    max_pairs = n * (n - 1) // 2
    eff = min(sample_size, max_pairs)
    if eff == 0:
        return sample_size, 0
    # Reject self-pairs by drawing pairs of distinct indices.
    i = rng.integers(0, n, size=eff)
    j = rng.integers(0, n, size=eff)
    same = i == j
    while same.any():
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j
    a = decoder_vectors[i]
    b = decoder_vectors[j]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    _ = np.sum(a * b, axis=1) / (na * nb)
    return sample_size, eff


def _clustered_op_cross_block_overlap(
    clustered: ClusteredDictionary, sample_size: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Clustered: sample `sample_size` cached cross-block edges.

    Returns (requested, effective). Capped to total edge count."""
    edges = list(clustered.cross_block_pairs.items())
    eff = min(sample_size, len(edges))
    if eff == 0:
        return sample_size, 0
    idx = rng.choice(len(edges), size=eff, replace=False)
    _ = [edges[int(k)][1] for k in idx]
    return sample_size, eff


_REDUNDANCY_THRESHOLD = 0.7


def _flat_op_redundancy(decoder_vectors: np.ndarray) -> int:
    """Flat: full cosine pair graph filtered at the redundancy threshold."""
    pairs = compute_cosine_pair_graph(
        decoder_vectors, threshold=_REDUNDANCY_THRESHOLD
    )
    return len(pairs)


def _clustered_op_redundancy(clustered: ClusteredDictionary) -> int:
    """Clustered: cross-block + intra-block redundant pairs at 0.7."""
    cross = clustered.cross_block_redundant_pairs(_REDUNDANCY_THRESHOLD)
    intra = 0
    for block in clustered.blocks:
        g = np.abs(block.gram())
        n = g.shape[0]
        iu = np.triu_indices(n, k=1)
        intra += int((g[iu] >= _REDUNDANCY_THRESHOLD).sum())
    return len(cross.pairs) + intra


def run_amortised_benchmark(
    dictionary: Dictionary,
    decoder_vectors: np.ndarray,
    *,
    op: BenchmarkOp,
    n_repeats: int = 64,
    sample_size: int = 64,
    cosine_threshold: float = 0.3,
    block_size_max: int | None = None,
    seed: int = 0,
) -> BenchmarkReport:
    """Time flat vs clustered for a single op, return a `BenchmarkReport`.

    See module docstring for the framing of each op and why the flat
    baselines look the way they do.
    """
    if op not in _SUPPORTED_OPS:
        raise ValueError(
            f"run_amortised_benchmark: unknown op {op!r}; "
            f"supported: {_SUPPORTED_OPS}"
        )
    if n_repeats < 1:
        raise ValueError(
            f"run_amortised_benchmark: n_repeats must be >= 1; got {n_repeats}"
        )

    rng = np.random.default_rng(seed)

    # Block formation — the one-time clustered cost.
    gc.collect()
    obj_before = len(gc.get_objects())
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    clustered = build_clustered_dictionary(
        name=f"{dictionary.name}_benchmark",
        features=list(dictionary.features),
        decoder_vectors=decoder_vectors,
        encoding=dictionary.encoding,
        block_formation=BlockFormation(
            strategy="cosine",
            cosine_threshold=cosine_threshold,
            block_size_max=block_size_max,
        ),
    )
    block_formation_s = time.perf_counter() - t0
    rss_after = _rss_kb()
    obj_after = len(gc.get_objects())
    rss_delta_kb = max(0, rss_after - rss_before)
    object_count_delta = obj_after - obj_before

    # Per-op timing — flat first (single run; we extrapolate linearly
    # since the flat path has no state to share across repeats).
    requested = sample_size
    effective = sample_size
    if op == "gram":
        t_flat_start = time.perf_counter()
        _flat_op_gram(decoder_vectors)
        flat_per_op = time.perf_counter() - t_flat_start
    elif op == "cross_block_overlap":
        t_flat_start = time.perf_counter()
        requested, effective = _flat_op_cross_block_overlap(
            decoder_vectors, sample_size, rng,
        )
        flat_per_op = time.perf_counter() - t_flat_start
    else:  # redundancy
        t_flat_start = time.perf_counter()
        _flat_op_redundancy(decoder_vectors)
        flat_per_op = time.perf_counter() - t_flat_start

    # Per-op timing — clustered (average over n_repeats so the
    # measurement is stable; the relevant numbers are per-op anyway).
    clustered_walls: list[float] = []
    for _ in range(n_repeats):
        t_c_start = time.perf_counter()
        if op == "gram":
            _clustered_op_gram(clustered)
        elif op == "cross_block_overlap":
            requested, effective = _clustered_op_cross_block_overlap(
                clustered, sample_size, rng,
            )
        else:  # redundancy
            _clustered_op_redundancy(clustered)
        clustered_walls.append(time.perf_counter() - t_c_start)
    clustered_per_op = float(np.mean(clustered_walls))

    cross_over = _compute_cross_over(
        block_formation_s,
        flat_per_op,
        clustered_per_op,
        upper_bound=max(n_repeats, 1024),
    )

    return BenchmarkReport(
        op_name=op,
        n_repeats=int(n_repeats),
        flat_wall_seconds_per_op=float(flat_per_op),
        clustered_wall_seconds_per_op=float(clustered_per_op),
        clustered_block_formation_seconds=float(block_formation_s),
        cross_over_n_repeats=cross_over,
        rss_delta_kb=int(rss_delta_kb),
        object_count_delta=int(object_count_delta),
        sample_size_requested=int(requested),
        sample_size_effective=int(effective),
        polygram_version=polygram.__version__,
        git_commit=_git_commit(),
    )
