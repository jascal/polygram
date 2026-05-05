"""Compression-epoch report dataclasses.

`Panel` describes one validator-sized feature window: anchor, the
8 (or fewer) feature ids, and the per-neighbour cosine to the
anchor.

`EpochIteration` carries one iteration's per-panel results,
counters, the cross-entropy quality delta, and the convergence
state observed at the end of the iteration.

`EpochReport` is the post-`run()` artifact: source + output sha256s,
convergence reason, total counters (features zeroed, panels run,
coverage achieved, wall time), and the iteration list. Round-
trippable JSON matches the `CompressionReport` discipline.

`EpochResult` is the in-process bundle: the report, the output
checkpoint path, and the final `Dictionary` rebuilt from the
rewritten checkpoint.

JSON layout matches `add-compression-epoch/design.md` Decision 8.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary


SCHEMA_VERSION = 1
SIG_FIGS = 6


def _round_sig(v: float | None, sigfigs: int = SIG_FIGS) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return fv
    if fv == 0.0:
        return 0.0
    return float(format(fv, f".{sigfigs}g"))


def _json_finite(v: float | None) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return None
    return _round_sig(fv)


@dataclass(frozen=True)
class Panel:
    """One validator-sized feature window inside an epoch iteration.

    `feature_ids` is sorted ascending for deterministic panel
    hashing. `cosines_to_anchor` carries the cosine of every non-
    anchor neighbour to the anchor's decoder direction (length =
    len(feature_ids) - 1, ordered to match `feature_ids` with the
    anchor's entry omitted).
    """

    panel_id: int
    anchor: int
    feature_ids: tuple[int, ...]
    cosines_to_anchor: tuple[float, ...]


@dataclass(frozen=True)
class EpochIteration:
    """One iteration of the epoch loop.

    `convergence_state` records what the orchestrator observed at
    the end of this iteration: `'continuing'` while iterating, or
    one of the terminal reasons (`'stable_clusters'`,
    `'max_iterations'`, `'quality_bound_breached'`,
    `'no_more_priority_candidates'`) on the final iteration.
    """

    iteration: int
    panels: tuple[Panel, ...]
    validation_report_paths: tuple[str, ...]
    confirmed_pair_count: int
    clusters_compressed: int
    features_zeroed_this_iteration: tuple[int, ...]
    cross_entropy_delta: float
    convergence_state: str


@dataclass(frozen=True, eq=False)
class EpochReport:
    """Post-`run()` artifact carrying provenance + the iteration list."""

    schema_version: int
    source_checkpoint: str
    source_checkpoint_sha256: str
    output_checkpoint: str
    output_checkpoint_sha256: str
    convergence_reason: str
    n_features_zeroed_total: int
    n_panels_total: int
    coverage_achieved: float
    wall_seconds: float
    iterations: tuple[EpochIteration, ...]

    # ---- JSON ------------------------------------------------------

    def to_json(self, path: str | os.PathLike | None = None) -> str:
        text = self._serialize()
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike) -> "EpochReport":
        if not isinstance(source, (str, os.PathLike)):
            raise TypeError(
                "EpochReport.from_json: expected a path or JSON string"
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
            "source_checkpoint",
            "source_checkpoint_sha256",
            "output_checkpoint",
            "output_checkpoint_sha256",
            "convergence_reason",
            "n_features_zeroed_total",
            "n_panels_total",
            "coverage_achieved",
            "wall_seconds",
            "iterations",
        ):
            if key not in payload:
                raise ValueError(
                    f"EpochReport.from_json: missing required key {key!r}"
                )

        iterations = tuple(
            _iteration_from_dict(it) for it in payload["iterations"]
        )
        return cls(
            schema_version=int(payload["schema_version"]),
            source_checkpoint=str(payload["source_checkpoint"]),
            source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
            output_checkpoint=str(payload["output_checkpoint"]),
            output_checkpoint_sha256=str(payload["output_checkpoint_sha256"]),
            convergence_reason=str(payload["convergence_reason"]),
            n_features_zeroed_total=int(payload["n_features_zeroed_total"]),
            n_panels_total=int(payload["n_panels_total"]),
            coverage_achieved=float(payload["coverage_achieved"]),
            wall_seconds=float(payload["wall_seconds"]),
            iterations=iterations,
        )

    # ---- Equality (NaN-aware on float fields) ----------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EpochReport):
            return NotImplemented
        return (
            self.schema_version == other.schema_version
            and self.source_checkpoint == other.source_checkpoint
            and self.source_checkpoint_sha256 == other.source_checkpoint_sha256
            and self.output_checkpoint == other.output_checkpoint
            and self.output_checkpoint_sha256 == other.output_checkpoint_sha256
            and self.convergence_reason == other.convergence_reason
            and self.n_features_zeroed_total == other.n_features_zeroed_total
            and self.n_panels_total == other.n_panels_total
            and _floats_eq(self.coverage_achieved, other.coverage_achieved)
            and _floats_eq(self.wall_seconds, other.wall_seconds)
            and len(self.iterations) == len(other.iterations)
            and all(
                _iter_eq(a, b)
                for a, b in zip(self.iterations, other.iterations)
            )
        )

    def __hash__(self) -> int:
        return hash((
            self.schema_version,
            self.source_checkpoint_sha256,
            self.output_checkpoint_sha256,
            self.convergence_reason,
            self.n_features_zeroed_total,
            self.n_panels_total,
        ))

    # ---- Internal --------------------------------------------------

    def _serialize(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "source_checkpoint": self.source_checkpoint,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "output_checkpoint": self.output_checkpoint,
            "output_checkpoint_sha256": self.output_checkpoint_sha256,
            "convergence_reason": self.convergence_reason,
            "n_features_zeroed_total": int(self.n_features_zeroed_total),
            "n_panels_total": int(self.n_panels_total),
            "coverage_achieved": _json_finite(self.coverage_achieved),
            "wall_seconds": _json_finite(self.wall_seconds),
            "iterations": [
                _iteration_to_dict(it) for it in self.iterations
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EpochResult:
    """Bundle returned by `EpochCompressor.run()`."""

    report: EpochReport
    output_checkpoint: Path
    final_dictionary: "Dictionary"


# ============================================================================
# JSON helpers
# ============================================================================


def _panel_to_dict(p: Panel) -> dict[str, Any]:
    return {
        "panel_id": int(p.panel_id),
        "anchor": int(p.anchor),
        "feature_ids": [int(f) for f in p.feature_ids],
        "cosines_to_anchor": [_json_finite(c) for c in p.cosines_to_anchor],
    }


def _panel_from_dict(raw: dict[str, Any]) -> Panel:
    return Panel(
        panel_id=int(raw["panel_id"]),
        anchor=int(raw["anchor"]),
        feature_ids=tuple(int(f) for f in raw["feature_ids"]),
        cosines_to_anchor=tuple(
            float(c if c is not None else 0.0)
            for c in raw["cosines_to_anchor"]
        ),
    )


def _iteration_to_dict(it: EpochIteration) -> dict[str, Any]:
    return {
        "iteration": int(it.iteration),
        "panels": [_panel_to_dict(p) for p in it.panels],
        "validation_report_paths": [str(p) for p in it.validation_report_paths],
        "confirmed_pair_count": int(it.confirmed_pair_count),
        "clusters_compressed": int(it.clusters_compressed),
        "features_zeroed_this_iteration": [
            int(f) for f in it.features_zeroed_this_iteration
        ],
        "cross_entropy_delta": _json_finite(it.cross_entropy_delta),
        "convergence_state": str(it.convergence_state),
    }


def _iteration_from_dict(raw: dict[str, Any]) -> EpochIteration:
    return EpochIteration(
        iteration=int(raw["iteration"]),
        panels=tuple(_panel_from_dict(p) for p in raw["panels"]),
        validation_report_paths=tuple(
            str(p) for p in raw["validation_report_paths"]
        ),
        confirmed_pair_count=int(raw["confirmed_pair_count"]),
        clusters_compressed=int(raw["clusters_compressed"]),
        features_zeroed_this_iteration=tuple(
            int(f) for f in raw["features_zeroed_this_iteration"]
        ),
        cross_entropy_delta=float(
            raw["cross_entropy_delta"]
            if raw["cross_entropy_delta"] is not None
            else float("nan")
        ),
        convergence_state=str(raw["convergence_state"]),
    )


# ============================================================================
# NaN-aware equality
# ============================================================================


def _floats_eq(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _panel_eq(a: Panel, b: Panel) -> bool:
    if (
        a.panel_id != b.panel_id
        or a.anchor != b.anchor
        or a.feature_ids != b.feature_ids
        or len(a.cosines_to_anchor) != len(b.cosines_to_anchor)
    ):
        return False
    return all(
        _floats_eq(x, y)
        for x, y in zip(a.cosines_to_anchor, b.cosines_to_anchor)
    )


def _iter_eq(a: EpochIteration, b: EpochIteration) -> bool:
    return (
        a.iteration == b.iteration
        and len(a.panels) == len(b.panels)
        and all(_panel_eq(x, y) for x, y in zip(a.panels, b.panels))
        and a.validation_report_paths == b.validation_report_paths
        and a.confirmed_pair_count == b.confirmed_pair_count
        and a.clusters_compressed == b.clusters_compressed
        and a.features_zeroed_this_iteration == b.features_zeroed_this_iteration
        and _floats_eq(a.cross_entropy_delta, b.cross_entropy_delta)
        and a.convergence_state == b.convergence_state
    )
