"""Pareto-path artifact for target-K compression sweeps.

Phase 2 of the `add-pareto-target-compression` openspec. Provides
the dataclasses returned by `Compressor.plan_pareto(targets)` — one
`ParetoOutcome` per requested K, all bundled in a `ParetoReport`
with provenance + score-field metadata.

Serialization mirrors `polygram.compression.report.CompressionReport`:
hand-coded `to_json` / `from_json` reusing `_cluster_to_dict` /
`_cluster_from_dict`. `CompressionPlan` itself doesn't have a
`to_dict`, so we flatten the plan's `(clusters, feature_ids)` pair
inline within each outcome.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polygram.compression.report import (
    CompressionPlan,
    _cluster_from_dict,
    _cluster_to_dict,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ParetoOutcome:
    """One target K's planning result inside a `ParetoReport`.

    `target_k` is the K the caller requested. `reached_target` is
    `True` iff the resulting plan's `n_features_kept <= target_k`.
    `plan` is the materialised `CompressionPlan`; callers can hand
    it to `Compressor.apply(plan=...)` without re-planning.
    """

    target_k: int
    reached_target: bool
    plan: CompressionPlan


@dataclass(frozen=True)
class ParetoReport:
    """Sequence of `ParetoOutcome` for a Pareto-path sweep.

    Provenance fields (`sae_checkpoint`, `sae_checkpoint_sha256`) and
    the score axis (`score_field`) record how the sweep was produced.
    `targets` is the deduplicated, descending-sorted K list;
    `outcomes` has the same length and ordering.

    The artifact is JSON-serialisable via `to_json` /
    `from_json` mirroring `CompressionReport`'s pattern. Callers can
    inspect the curve before paying SAE-rewrite cost; see the
    `--pareto` / `--pareto-materialize` CLI gating proposed in
    Phase 3.
    """

    schema_version: int
    sae_checkpoint: Path
    sae_checkpoint_sha256: str
    score_field: str
    targets: tuple[int, ...]
    outcomes: tuple[ParetoOutcome, ...]

    # ---- JSON ------------------------------------------------------

    def to_json(self, path: str | os.PathLike | None = None) -> str:
        text = self._serialize()
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike) -> "ParetoReport":
        if not isinstance(source, (str, os.PathLike)):
            raise TypeError(
                "ParetoReport.from_json: expected a path or JSON string"
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
            "sae_checkpoint",
            "sae_checkpoint_sha256",
            "score_field",
            "targets",
            "outcomes",
        ):
            if key not in payload:
                raise ValueError(
                    f"ParetoReport.from_json: missing required key {key!r}"
                )

        outcomes = tuple(
            _outcome_from_dict(raw) for raw in payload["outcomes"]
        )
        return cls(
            schema_version=int(payload["schema_version"]),
            sae_checkpoint=Path(payload["sae_checkpoint"]),
            sae_checkpoint_sha256=str(payload["sae_checkpoint_sha256"]),
            score_field=str(payload["score_field"]),
            targets=tuple(int(k) for k in payload["targets"]),
            outcomes=outcomes,
        )

    # ---- Equality --------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParetoReport):
            return NotImplemented
        if (
            self.schema_version != other.schema_version
            or str(self.sae_checkpoint) != str(other.sae_checkpoint)
            or self.sae_checkpoint_sha256 != other.sae_checkpoint_sha256
            or self.score_field != other.score_field
            or self.targets != other.targets
            or len(self.outcomes) != len(other.outcomes)
        ):
            return False
        for a, b in zip(self.outcomes, other.outcomes):
            if (
                a.target_k != b.target_k
                or a.reached_target != b.reached_target
                or a.plan.feature_ids != b.plan.feature_ids
                or a.plan.clusters != b.plan.clusters
            ):
                return False
        return True

    def __hash__(self) -> int:
        return hash((
            self.schema_version,
            self.sae_checkpoint_sha256,
            self.score_field,
            self.targets,
        ))

    # ---- Internal --------------------------------------------------

    def _serialize(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "sae_checkpoint": str(self.sae_checkpoint),
            "sae_checkpoint_sha256": self.sae_checkpoint_sha256,
            "score_field": self.score_field,
            "targets": [int(k) for k in self.targets],
            "outcomes": [_outcome_to_dict(o) for o in self.outcomes],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ============================================================================
# JSON helpers
# ============================================================================


def _outcome_to_dict(o: ParetoOutcome) -> dict[str, Any]:
    return {
        "target_k": int(o.target_k),
        "reached_target": bool(o.reached_target),
        "clusters": [_cluster_to_dict(c) for c in o.plan.clusters],
        "feature_ids": [int(f) for f in o.plan.feature_ids],
    }


def _outcome_from_dict(raw: dict[str, Any]) -> ParetoOutcome:
    plan = CompressionPlan(
        clusters=tuple(_cluster_from_dict(c) for c in raw["clusters"]),
        feature_ids=tuple(int(f) for f in raw["feature_ids"]),
    )
    return ParetoOutcome(
        target_k=int(raw["target_k"]),
        reached_target=bool(raw["reached_target"]),
        plan=plan,
    )
