"""Behavioural-validator report dataclasses.

`CandidatePair` carries one row of the per-pair behavioural panel:
Polygram's predicted overlap, decoder cosine², the co-firing Jaccard
gate, the activation Pearson, the per-pair ablation-KL ratio, and the
fire-count counters. `ValidationSummary` carries the cross-set
Spearman / Pearson and the per-bucket Jaccard means with 95% bootstrap
CIs. `ValidationReport` glues them together with run-level metadata
and supports JSON round-trip + CSV emission.

The JSON layout matches `add-behavioural-validator-loop/design.md`
Decision 6; the CSV column order matches Decision 7 (the §4.4
`scaleup_pairs.csv` columns plus a `gate_pass` column).
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SIG_FIGS = 6


def _round_sig(v: float | None, sigfigs: int = SIG_FIGS) -> float | None:
    """Round a float to `sigfigs` significant figures via the
    `BatchResults` re-parse trick. Preserves NaN/Inf and `None`."""
    if v is None:
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return fv
    if fv == 0.0:
        return 0.0
    return float(format(fv, f".{sigfigs}g"))


def _json_finite(v: float | None) -> float | None:
    """JSON-encode a numeric field. NaN and `None` both serialize as
    JSON `null`; finite values are rounded to `SIG_FIGS`."""
    if v is None:
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return None
    return _round_sig(fv)


def _from_json_finite(v: Any) -> float:
    """Decode a JSON value back to a float; JSON `null` becomes NaN."""
    if v is None:
        return float("nan")
    return float(v)


@dataclass(frozen=True)
class CandidatePair:
    """One per-pair row of a `ValidationReport`.

    Behavioural fields (`jaccard`, `pearson_activation`, `kl_*`) are
    NaN on rows produced by `BehaviouralValidator.predict()` (the
    cheap torch-free stage). They are populated by `validate()`.

    `gate_pass` is `True` iff all three of `polygram_overlap >=
    polygram_overlap_threshold`, `jaccard >= jaccard_threshold`, and
    `n_both_fire >= min_both_fire`. `predict()` always sets it
    `False` (no behavioural data yet).
    """

    i: int
    j: int
    polygram_overlap: float
    decoder_overlap: float
    jaccard: float
    pearson_activation: float
    kl_ablate_i: float
    kl_ablate_j: float
    kl_ratio_paired: float
    kl_log_ratio_abs: float
    n_fires_i: int
    n_fires_j: int
    n_both_fire: int
    n_either_fire: int
    gate_pass: bool


@dataclass(frozen=True)
class BucketStats:
    """Per-Polygram-overlap-bucket Jaccard statistics."""

    polygram_range: str
    n_pairs: int
    jaccard_mean: float
    jaccard_ci_95: tuple[float, float]


@dataclass(frozen=True)
class ValidationSummary:
    """Cross-pair-set aggregates for a `ValidationReport`."""

    spearman_polygram_jaccard: float
    spearman_decoder_jaccard: float
    spearman_polygram_log_kl_abs: float
    pearson_polygram_jaccard: float
    pearson_decoder_jaccard: float
    buckets: dict[str, BucketStats]
    outcome: str


@dataclass(frozen=True, eq=False)
class ValidationReport:
    """Output of `BehaviouralValidator.validate()` / `run()`."""

    schema_version: int
    dictionary_name: str
    model_name: str
    layer: int
    n_prompts: int
    n_tokens: int
    polygram_overlap_threshold: float
    jaccard_threshold: float
    min_firing_rate: float
    min_both_fire: int
    feature_ids: tuple[int, ...]
    pairs: tuple[CandidatePair, ...]
    summary: ValidationSummary
    confirmed: tuple[tuple[int, int], ...]

    # ---- JSON ------------------------------------------------------

    def to_json(self, path: str | os.PathLike | None = None) -> str:
        text = self._serialize()
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike) -> "ValidationReport":
        if not isinstance(source, (str, os.PathLike)):
            raise TypeError(
                "ValidationReport.from_json: expected a path or JSON string"
            )
        text: str
        if isinstance(source, str) and source.lstrip().startswith("{"):
            text = source
        else:
            candidate = Path(source)
            try:
                if candidate.is_file():
                    text = candidate.read_text()
                else:
                    text = str(source)
            except OSError:
                text = str(source)
        payload = json.loads(text)

        for key in (
            "schema_version",
            "dictionary_name",
            "model_name",
            "layer",
            "n_prompts",
            "n_tokens",
            "polygram_overlap_threshold",
            "jaccard_threshold",
            "min_firing_rate",
            "min_both_fire",
            "feature_ids",
            "pairs",
            "summary",
            "confirmed",
        ):
            if key not in payload:
                raise ValueError(
                    f"ValidationReport.from_json: missing required key {key!r}"
                )

        pairs = tuple(_pair_from_dict(p) for p in payload["pairs"])
        summary = _summary_from_dict(payload["summary"])
        confirmed = tuple(
            (int(a), int(b)) for a, b in payload["confirmed"]
        )
        return cls(
            schema_version=int(payload["schema_version"]),
            dictionary_name=str(payload["dictionary_name"]),
            model_name=str(payload["model_name"]),
            layer=int(payload["layer"]),
            n_prompts=int(payload["n_prompts"]),
            n_tokens=int(payload["n_tokens"]),
            polygram_overlap_threshold=float(payload["polygram_overlap_threshold"]),
            jaccard_threshold=float(payload["jaccard_threshold"]),
            min_firing_rate=float(payload["min_firing_rate"]),
            min_both_fire=int(payload["min_both_fire"]),
            feature_ids=tuple(int(f) for f in payload["feature_ids"]),
            pairs=pairs,
            summary=summary,
            confirmed=confirmed,
        )

    # ---- CSV -------------------------------------------------------

    def to_csv(self, path: str | os.PathLike) -> Path:
        """Write the per-pair table to `path` in the §4.4 column order
        plus a `gate_pass` column.

        Returns the resolved `Path`. The first 14 columns and their
        order match `docs/research/data/scaleup_pairs.csv`.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sorted_pairs = sorted(self.pairs, key=lambda r: (r.i, r.j))
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_HEADER)
            for r in sorted_pairs:
                writer.writerow(_pair_to_csv_row(r))
        return p

    # ---- Equality (NaN-aware) -------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValidationReport):
            return NotImplemented
        return (
            self.schema_version == other.schema_version
            and self.dictionary_name == other.dictionary_name
            and self.model_name == other.model_name
            and self.layer == other.layer
            and self.n_prompts == other.n_prompts
            and self.n_tokens == other.n_tokens
            and _floats_eq(
                self.polygram_overlap_threshold,
                other.polygram_overlap_threshold,
            )
            and _floats_eq(self.jaccard_threshold, other.jaccard_threshold)
            and _floats_eq(self.min_firing_rate, other.min_firing_rate)
            and self.min_both_fire == other.min_both_fire
            and self.feature_ids == other.feature_ids
            and len(self.pairs) == len(other.pairs)
            and all(
                _pair_eq(a, b)
                for a, b in zip(
                    sorted(self.pairs, key=lambda r: (r.i, r.j)),
                    sorted(other.pairs, key=lambda r: (r.i, r.j)),
                )
            )
            and _summary_eq(self.summary, other.summary)
            and self.confirmed == other.confirmed
        )

    def __hash__(self) -> int:  # frozen dataclasses default-hash; keep
        return hash((
            self.schema_version,
            self.dictionary_name,
            self.model_name,
            self.layer,
            self.feature_ids,
            self.confirmed,
        ))

    # ---- Internal -------------------------------------------------

    def _serialize(self) -> str:
        sorted_pairs = sorted(self.pairs, key=lambda r: (r.i, r.j))
        payload = {
            "schema_version": int(self.schema_version),
            "dictionary_name": self.dictionary_name,
            "model_name": self.model_name,
            "layer": int(self.layer),
            "n_prompts": int(self.n_prompts),
            "n_tokens": int(self.n_tokens),
            "polygram_overlap_threshold": _json_finite(
                self.polygram_overlap_threshold
            ),
            "jaccard_threshold": _json_finite(self.jaccard_threshold),
            "min_firing_rate": _json_finite(self.min_firing_rate),
            "min_both_fire": int(self.min_both_fire),
            "feature_ids": [int(f) for f in self.feature_ids],
            "pairs": [_pair_to_dict(r) for r in sorted_pairs],
            "summary": _summary_to_dict(self.summary),
            "confirmed": [[int(a), int(b)] for a, b in self.confirmed],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ============================================================================
# JSON helpers (free functions to keep the dataclass body lean)
# ============================================================================


def _pair_to_dict(r: CandidatePair) -> dict[str, Any]:
    return {
        "i": int(r.i),
        "j": int(r.j),
        "polygram_overlap": _json_finite(r.polygram_overlap),
        "decoder_overlap": _json_finite(r.decoder_overlap),
        "jaccard": _json_finite(r.jaccard),
        "pearson_activation": _json_finite(r.pearson_activation),
        "kl_ablate_i": _json_finite(r.kl_ablate_i),
        "kl_ablate_j": _json_finite(r.kl_ablate_j),
        "kl_ratio_paired": _json_finite(r.kl_ratio_paired),
        "kl_log_ratio_abs": _json_finite(r.kl_log_ratio_abs),
        "n_fires_i": int(r.n_fires_i),
        "n_fires_j": int(r.n_fires_j),
        "n_both_fire": int(r.n_both_fire),
        "n_either_fire": int(r.n_either_fire),
        "gate_pass": bool(r.gate_pass),
    }


def _pair_from_dict(raw: dict[str, Any]) -> CandidatePair:
    return CandidatePair(
        i=int(raw["i"]),
        j=int(raw["j"]),
        polygram_overlap=_from_json_finite(raw["polygram_overlap"]),
        decoder_overlap=_from_json_finite(raw["decoder_overlap"]),
        jaccard=_from_json_finite(raw["jaccard"]),
        pearson_activation=_from_json_finite(raw["pearson_activation"]),
        kl_ablate_i=_from_json_finite(raw["kl_ablate_i"]),
        kl_ablate_j=_from_json_finite(raw["kl_ablate_j"]),
        kl_ratio_paired=_from_json_finite(raw["kl_ratio_paired"]),
        kl_log_ratio_abs=_from_json_finite(raw["kl_log_ratio_abs"]),
        n_fires_i=int(raw["n_fires_i"]),
        n_fires_j=int(raw["n_fires_j"]),
        n_both_fire=int(raw["n_both_fire"]),
        n_either_fire=int(raw["n_either_fire"]),
        gate_pass=bool(raw["gate_pass"]),
    )


def _summary_to_dict(s: ValidationSummary) -> dict[str, Any]:
    return {
        "spearman_polygram_jaccard": _json_finite(s.spearman_polygram_jaccard),
        "spearman_decoder_jaccard": _json_finite(s.spearman_decoder_jaccard),
        "spearman_polygram_log_kl_abs": _json_finite(
            s.spearman_polygram_log_kl_abs
        ),
        "pearson_polygram_jaccard": _json_finite(s.pearson_polygram_jaccard),
        "pearson_decoder_jaccard": _json_finite(s.pearson_decoder_jaccard),
        "buckets": {
            name: _bucket_to_dict(b) for name, b in s.buckets.items()
        },
        "outcome": s.outcome,
    }


def _summary_from_dict(raw: dict[str, Any]) -> ValidationSummary:
    return ValidationSummary(
        spearman_polygram_jaccard=_from_json_finite(
            raw["spearman_polygram_jaccard"]
        ),
        spearman_decoder_jaccard=_from_json_finite(
            raw["spearman_decoder_jaccard"]
        ),
        spearman_polygram_log_kl_abs=_from_json_finite(
            raw["spearman_polygram_log_kl_abs"]
        ),
        pearson_polygram_jaccard=_from_json_finite(
            raw["pearson_polygram_jaccard"]
        ),
        pearson_decoder_jaccard=_from_json_finite(
            raw["pearson_decoder_jaccard"]
        ),
        buckets={
            name: _bucket_from_dict(b) for name, b in raw["buckets"].items()
        },
        outcome=str(raw["outcome"]),
    )


def _bucket_to_dict(b: BucketStats) -> dict[str, Any]:
    lo, hi = b.jaccard_ci_95
    return {
        "polygram_range": b.polygram_range,
        "n_pairs": int(b.n_pairs),
        "jaccard_mean": _json_finite(b.jaccard_mean),
        "jaccard_ci_95": [_json_finite(lo), _json_finite(hi)],
    }


def _bucket_from_dict(raw: dict[str, Any]) -> BucketStats:
    lo, hi = raw["jaccard_ci_95"]
    return BucketStats(
        polygram_range=str(raw["polygram_range"]),
        n_pairs=int(raw["n_pairs"]),
        jaccard_mean=_from_json_finite(raw["jaccard_mean"]),
        jaccard_ci_95=(_from_json_finite(lo), _from_json_finite(hi)),
    )


# ============================================================================
# CSV helpers
# ============================================================================


_CSV_HEADER: tuple[str, ...] = (
    "i",
    "j",
    "polygram_overlap",
    "decoder_overlap",
    "jaccard",
    "pearson_activation",
    "n_fires_i",
    "n_fires_j",
    "n_both_fire",
    "n_either_fire",
    "kl_ablate_i_on_both_fire",
    "kl_ablate_j_on_both_fire",
    "kl_ratio_i_over_j",
    "kl_log_ratio_abs",
    "gate_pass",
)


def _csv_float(v: float) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return format(float(v), f".{SIG_FIGS}g")


def _pair_to_csv_row(r: CandidatePair) -> list[str]:
    return [
        str(int(r.i)),
        str(int(r.j)),
        _csv_float(r.polygram_overlap),
        _csv_float(r.decoder_overlap),
        _csv_float(r.jaccard),
        _csv_float(r.pearson_activation),
        str(int(r.n_fires_i)),
        str(int(r.n_fires_j)),
        str(int(r.n_both_fire)),
        str(int(r.n_either_fire)),
        _csv_float(r.kl_ablate_i),
        _csv_float(r.kl_ablate_j),
        _csv_float(r.kl_ratio_paired),
        _csv_float(r.kl_log_ratio_abs),
        "true" if r.gate_pass else "false",
    ]


# ============================================================================
# NaN-aware equality (helpers; module-private)
# ============================================================================


def _floats_eq(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _pair_eq(a: CandidatePair, b: CandidatePair) -> bool:
    return (
        a.i == b.i
        and a.j == b.j
        and _floats_eq(a.polygram_overlap, b.polygram_overlap)
        and _floats_eq(a.decoder_overlap, b.decoder_overlap)
        and _floats_eq(a.jaccard, b.jaccard)
        and _floats_eq(a.pearson_activation, b.pearson_activation)
        and _floats_eq(a.kl_ablate_i, b.kl_ablate_i)
        and _floats_eq(a.kl_ablate_j, b.kl_ablate_j)
        and _floats_eq(a.kl_ratio_paired, b.kl_ratio_paired)
        and _floats_eq(a.kl_log_ratio_abs, b.kl_log_ratio_abs)
        and a.n_fires_i == b.n_fires_i
        and a.n_fires_j == b.n_fires_j
        and a.n_both_fire == b.n_both_fire
        and a.n_either_fire == b.n_either_fire
        and a.gate_pass == b.gate_pass
    )


def _bucket_eq(a: BucketStats, b: BucketStats) -> bool:
    return (
        a.polygram_range == b.polygram_range
        and a.n_pairs == b.n_pairs
        and _floats_eq(a.jaccard_mean, b.jaccard_mean)
        and _floats_eq(a.jaccard_ci_95[0], b.jaccard_ci_95[0])
        and _floats_eq(a.jaccard_ci_95[1], b.jaccard_ci_95[1])
    )


def _summary_eq(a: ValidationSummary, b: ValidationSummary) -> bool:
    if set(a.buckets.keys()) != set(b.buckets.keys()):
        return False
    if not all(_bucket_eq(a.buckets[k], b.buckets[k]) for k in a.buckets):
        return False
    return (
        _floats_eq(
            a.spearman_polygram_jaccard, b.spearman_polygram_jaccard
        )
        and _floats_eq(
            a.spearman_decoder_jaccard, b.spearman_decoder_jaccard
        )
        and _floats_eq(
            a.spearman_polygram_log_kl_abs,
            b.spearman_polygram_log_kl_abs,
        )
        and _floats_eq(
            a.pearson_polygram_jaccard, b.pearson_polygram_jaccard
        )
        and _floats_eq(
            a.pearson_decoder_jaccard, b.pearson_decoder_jaccard
        )
        and a.outcome == b.outcome
    )
