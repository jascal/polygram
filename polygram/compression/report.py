"""Compression-action report dataclasses.

`ClusterPlan` describes one redundancy cluster (members, the chosen
representative, and the zero-list — the non-representative members
that will get their encoder/decoder rows zeroed).

`CompressionPlan` is the cheap stage's output: every cluster the
strategy will act on, plus the full `feature_ids` list inherited from
the source `ValidationReport`. No I/O, no checkpoint hashes.

`CompressionReport` is the post-`apply()` artifact. It carries
provenance (source + output checkpoint paths and sha256 hashes,
upstream `ValidationReport` schema version + dictionary name), the
strategy name, the actual cluster plan that was applied, and three
roll-up counters (`n_features_zeroed`, `n_features_kept`,
`n_clusters`). JSON round-trip matches `ValidationReport`'s six-
sigfig float formatting where applicable.

`CompressionResult` is the in-process bundle: the plan applied, the
report written, the output-checkpoint path, and the rebuilt
`Dictionary`.

JSON layout matches `add-compression-action/design.md` Decision 6.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClusterPlan:
    """One redundancy cluster's compression plan.

    `members` is the full set of feature ids forming the connected
    component. `representative` is the one feature id whose weights
    are kept. `zeroed` is the sorted list of all other members; the
    `zero` strategy zeroes those features' encoder columns + biases
    and decoder rows.

    `cluster_norm_mean` / `cluster_norm_std` are the L2-norm mean and
    standard deviation of the cluster members' source W_dec rows
    (populated by `apply()`; `None` after `plan()` alone).
    `merged_norm` is the post-rescale norm for `strategy="merge"`;
    `None` for `strategy="zero"` and before `apply()`.
    """

    cluster_id: int
    members: tuple[int, ...]
    representative: int
    zeroed: tuple[int, ...]
    cluster_norm_mean: float | None = None
    cluster_norm_std: float | None = None
    merged_norm: float | None = None


@dataclass(frozen=True)
class CompressionPlan:
    """Output of `Compressor.plan()`.

    Singletons (feature ids that never appeared in a confirmed pair)
    are excluded — there's nothing to compress for a singleton. Cluster
    ids are assigned in ascending min-fid order so two `plan()` calls
    on the same `ValidationReport` produce identical ids.
    """

    clusters: tuple[ClusterPlan, ...]
    feature_ids: tuple[int, ...]

    @property
    def n_features_kept(self) -> int:
        # Mirrors `CompressionReport.n_features_kept`, which the
        # `Compressor` derives as `sum(1 for _ in plan.clusters)`. Exposed
        # here so target-K planners can read the same value off the plan
        # before `apply()` runs (see openspec/changes/add-pareto-target-compression).
        return len(self.clusters)


@dataclass(frozen=True, eq=False)
class CompressionReport:
    """Post-`apply()` artifact carrying provenance + the applied plan."""

    schema_version: int
    source_checkpoint: str
    source_checkpoint_sha256: str
    output_checkpoint: str
    output_checkpoint_sha256: str
    validation_report_dictionary_name: str
    validation_report_schema_version: int
    strategy: str
    plan: CompressionPlan
    n_features_zeroed: int
    n_features_kept: int
    n_clusters: int
    scale_compression_ratio: float = 1.0

    # ---- JSON ------------------------------------------------------

    def to_json(self, path: str | os.PathLike | None = None) -> str:
        text = self._serialize()
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike) -> "CompressionReport":
        if not isinstance(source, (str, os.PathLike)):
            raise TypeError(
                "CompressionReport.from_json: expected a path or JSON string"
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
            "validation_report_dictionary_name",
            "validation_report_schema_version",
            "strategy",
            "feature_ids",
            "clusters",
            "n_features_zeroed",
            "n_features_kept",
            "n_clusters",
        ):
            if key not in payload:
                raise ValueError(
                    f"CompressionReport.from_json: missing required key {key!r}"
                )

        clusters = tuple(_cluster_from_dict(c) for c in payload["clusters"])
        plan = CompressionPlan(
            clusters=clusters,
            feature_ids=tuple(int(f) for f in payload["feature_ids"]),
        )
        return cls(
            schema_version=int(payload["schema_version"]),
            source_checkpoint=str(payload["source_checkpoint"]),
            source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
            output_checkpoint=str(payload["output_checkpoint"]),
            output_checkpoint_sha256=str(payload["output_checkpoint_sha256"]),
            validation_report_dictionary_name=str(
                payload["validation_report_dictionary_name"]
            ),
            validation_report_schema_version=int(
                payload["validation_report_schema_version"]
            ),
            strategy=str(payload["strategy"]),
            plan=plan,
            n_features_zeroed=int(payload["n_features_zeroed"]),
            n_features_kept=int(payload["n_features_kept"]),
            n_clusters=int(payload["n_clusters"]),
            scale_compression_ratio=float(
                payload.get("scale_compression_ratio", 1.0)
            ),
        )

    # ---- Equality --------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompressionReport):
            return NotImplemented
        return (
            self.schema_version == other.schema_version
            and self.source_checkpoint == other.source_checkpoint
            and self.source_checkpoint_sha256
            == other.source_checkpoint_sha256
            and self.output_checkpoint == other.output_checkpoint
            and self.output_checkpoint_sha256
            == other.output_checkpoint_sha256
            and self.validation_report_dictionary_name
            == other.validation_report_dictionary_name
            and self.validation_report_schema_version
            == other.validation_report_schema_version
            and self.strategy == other.strategy
            and self.plan.feature_ids == other.plan.feature_ids
            and self.plan.clusters == other.plan.clusters
            and self.n_features_zeroed == other.n_features_zeroed
            and self.n_features_kept == other.n_features_kept
            and self.n_clusters == other.n_clusters
            and self.scale_compression_ratio == other.scale_compression_ratio
        )

    def __hash__(self) -> int:
        return hash((
            self.schema_version,
            self.source_checkpoint_sha256,
            self.output_checkpoint_sha256,
            self.strategy,
            self.plan.feature_ids,
        ))

    # ---- Internal --------------------------------------------------

    def _serialize(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "source_checkpoint": self.source_checkpoint,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "output_checkpoint": self.output_checkpoint,
            "output_checkpoint_sha256": self.output_checkpoint_sha256,
            "validation_report_dictionary_name": self.validation_report_dictionary_name,
            "validation_report_schema_version": int(
                self.validation_report_schema_version
            ),
            "strategy": self.strategy,
            "feature_ids": [int(f) for f in self.plan.feature_ids],
            "clusters": [_cluster_to_dict(c) for c in self.plan.clusters],
            "n_features_zeroed": int(self.n_features_zeroed),
            "n_features_kept": int(self.n_features_kept),
            "n_clusters": int(self.n_clusters),
            "scale_compression_ratio": float(self.scale_compression_ratio),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CompressionResult:
    """Bundle returned by `Compressor.apply()` / `run()`."""

    plan: CompressionPlan
    report: CompressionReport
    output_checkpoint: Path
    dictionary: "Dictionary"


# ============================================================================
# JSON helpers (free functions to keep the dataclass body lean)
# ============================================================================


def _cluster_to_dict(c: ClusterPlan) -> dict[str, Any]:
    return {
        "cluster_id": int(c.cluster_id),
        "members": [int(f) for f in c.members],
        "representative": int(c.representative),
        "zeroed": [int(f) for f in c.zeroed],
        "cluster_norm_mean": (
            None if c.cluster_norm_mean is None else float(c.cluster_norm_mean)
        ),
        "cluster_norm_std": (
            None if c.cluster_norm_std is None else float(c.cluster_norm_std)
        ),
        "merged_norm": (
            None if c.merged_norm is None else float(c.merged_norm)
        ),
    }


def _cluster_from_dict(raw: dict[str, Any]) -> ClusterPlan:
    def _opt_float(v: Any) -> float | None:
        return None if v is None else float(v)

    return ClusterPlan(
        cluster_id=int(raw["cluster_id"]),
        members=tuple(int(f) for f in raw["members"]),
        representative=int(raw["representative"]),
        zeroed=tuple(int(f) for f in raw["zeroed"]),
        cluster_norm_mean=_opt_float(raw.get("cluster_norm_mean")),
        cluster_norm_std=_opt_float(raw.get("cluster_norm_std")),
        merged_norm=_opt_float(raw.get("merged_norm")),
    )
