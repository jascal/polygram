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
from typing import TYPE_CHECKING, Any, Literal

from polygram.compression._helpers import (
    floats_eq,
)

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary


SCHEMA_VERSION = 3
# Per-block global-cluster-id namespace cap. Each block's local
# cluster ids in [0, n_clusters_in_block) get encoded as
# `block_index * MAX_CLUSTERS_PER_BLOCK + local_id` in the top-level
# CompressionReport's per-feature cluster ids. See
# `openspec/changes/add-encoding-partition/design.md` Decision 2.
MAX_CLUSTERS_PER_BLOCK = 10_000


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
class BlockReport:
    """Per-block diagnostic record emitted by `Compressor.apply` when
    a `CompressionConfig.encoding_partition` is set.

    Mirrors the per-block subset of the top-level
    :class:`CompressionReport` fields (n_features_kept,
    n_features_zeroed, n_clusters, plus the diagnostic floats), keyed
    by the block's analyst-supplied ``block_id``.

    Added by ``add-encoding-partition``. v3 schema.
    """

    block_id: str
    encoding_class: str
    encoding_kwargs: dict
    learn_axis_assignment: bool
    feature_ids: tuple[int, ...]
    n_features_kept: int
    n_features_zeroed: int
    n_clusters: int
    cluster_assignments: tuple[int, ...] | None = None
    scale_compression_ratio: float = 1.0
    rank_ratio: float | None = None
    post_A: float | None = None
    forge_mse: float | None = None
    informative_metric: Literal["post_A", "both", "forge_mse"] | None = None


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
    rank_ratio: float | None = None
    post_A: float | None = None
    forge_mse: float | None = None
    informative_metric: Literal["post_A", "both", "forge_mse"] | None = None
    # v3: per-block heterogeneous-encoding diagnostics. When the
    # `Compressor` was driven by a `CompressionConfig.encoding_partition`,
    # `blocks` carries one BlockReport per partition block. When the
    # compression used a single top-level encoding (the historical path),
    # `blocks` is None. Schema bump from 2 to 3; from_json defaults
    # `blocks=None` for v2 payloads.
    blocks: tuple[BlockReport, ...] | None = None

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
        # v1 payloads lack the new diagnostic fields; v2 payloads may
        # carry them as JSON null when the upstream value was None/NaN.
        rank_ratio = (
            float(payload["rank_ratio"])
            if payload.get("rank_ratio") is not None
            else None
        )
        post_A = (
            float(payload["post_A"])
            if payload.get("post_A") is not None
            else None
        )
        forge_mse = (
            float(payload["forge_mse"])
            if payload.get("forge_mse") is not None
            else None
        )
        informative = payload.get("informative_metric")

        # v3 (add-encoding-partition): `blocks` is a list of BlockReport
        # dicts when populated, or absent/null in v2 payloads.
        blocks_raw = payload.get("blocks")
        blocks: tuple[BlockReport, ...] | None = None
        if blocks_raw is not None:
            blocks = tuple(_block_from_dict(b) for b in blocks_raw)

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
            rank_ratio=rank_ratio,
            post_A=post_A,
            forge_mse=forge_mse,
            informative_metric=informative,
            blocks=blocks,
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
            and floats_eq(self.scale_compression_ratio, other.scale_compression_ratio)
            and floats_eq(self.rank_ratio, other.rank_ratio)
            and floats_eq(self.post_A, other.post_A)
            and floats_eq(self.forge_mse, other.forge_mse)
            and self.informative_metric == other.informative_metric
            and _blocks_eq(self.blocks, other.blocks)
        )

    def __hash__(self) -> int:
        # blocks contains dict (encoding_kwargs) which is unhashable;
        # hash by structural identifiers (block_id, encoding_class)
        # of each block instead of the full BlockReport tuple.
        blocks_hash = (
            tuple((b.block_id, b.encoding_class) for b in self.blocks)
            if self.blocks is not None else None
        )
        return hash((
            self.schema_version,
            self.source_checkpoint_sha256,
            self.output_checkpoint_sha256,
            self.strategy,
            self.plan.feature_ids,
            self.rank_ratio,
            self.post_A,
            self.forge_mse,
            blocks_hash,
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
            # Pre-existing field; preserved at full float precision for
            # round-trip equality. The new diagnostic fields below use
            # `json_finite` (6-sigfig quantization) intentionally.
            "scale_compression_ratio": float(self.scale_compression_ratio),
            # Diagnostic fields: preserve full float precision so the
            # JSON round-trip is bit-exact. `json_finite`'s 6-sigfig
            # quantization breaks strict equality (e.g. post_A
            # 1.0596382e-07 → 1.05964e-07 on write, can't recover the
            # original on read). NaN/Inf are still rejected by the
            # dataclass on construction.
            "rank_ratio": (
                float(self.rank_ratio) if self.rank_ratio is not None else None
            ),
            "post_A": (
                float(self.post_A) if self.post_A is not None else None
            ),
            "forge_mse": (
                float(self.forge_mse) if self.forge_mse is not None else None
            ),
            "informative_metric": self.informative_metric,
            # v3 (add-encoding-partition): list of per-block dicts, or
            # null when the compression used a single top-level encoding.
            "blocks": (
                [_block_to_dict(b) for b in self.blocks]
                if self.blocks is not None else None
            ),
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


def _block_to_dict(b: BlockReport) -> dict[str, Any]:
    """Serialise a :class:`BlockReport` to a JSON-roundtrippable dict.
    Mirrors `_cluster_to_dict`'s shape for the new v3 partition fields.
    """
    return {
        "block_id":              str(b.block_id),
        "encoding_class":        str(b.encoding_class),
        "encoding_kwargs":       dict(b.encoding_kwargs),
        "learn_axis_assignment": bool(b.learn_axis_assignment),
        "feature_ids":           [int(f) for f in b.feature_ids],
        "n_features_kept":       int(b.n_features_kept),
        "n_features_zeroed":     int(b.n_features_zeroed),
        "n_clusters":            int(b.n_clusters),
        "cluster_assignments":   (
            list(b.cluster_assignments)
            if b.cluster_assignments is not None else None
        ),
        "scale_compression_ratio": float(b.scale_compression_ratio),
        "rank_ratio":            (
            float(b.rank_ratio) if b.rank_ratio is not None else None
        ),
        "post_A":                (
            float(b.post_A) if b.post_A is not None else None
        ),
        "forge_mse":             (
            float(b.forge_mse) if b.forge_mse is not None else None
        ),
        "informative_metric":    b.informative_metric,
    }


def _block_from_dict(raw: dict[str, Any]) -> BlockReport:
    def _opt_float(v: Any) -> float | None:
        return None if v is None else float(v)

    ca = raw.get("cluster_assignments")
    return BlockReport(
        block_id=str(raw["block_id"]),
        encoding_class=str(raw["encoding_class"]),
        encoding_kwargs=dict(raw.get("encoding_kwargs") or {}),
        learn_axis_assignment=bool(raw.get("learn_axis_assignment", False)),
        feature_ids=tuple(int(f) for f in raw["feature_ids"]),
        n_features_kept=int(raw["n_features_kept"]),
        n_features_zeroed=int(raw["n_features_zeroed"]),
        n_clusters=int(raw["n_clusters"]),
        cluster_assignments=(
            tuple(int(x) for x in ca) if ca is not None else None
        ),
        scale_compression_ratio=float(raw.get("scale_compression_ratio", 1.0)),
        rank_ratio=_opt_float(raw.get("rank_ratio")),
        post_A=_opt_float(raw.get("post_A")),
        forge_mse=_opt_float(raw.get("forge_mse")),
        informative_metric=raw.get("informative_metric"),
    )


def _blocks_eq(
    a: "tuple[BlockReport, ...] | None",
    b: "tuple[BlockReport, ...] | None",
) -> bool:
    """NaN-aware element-wise equality for the optional `blocks` field.
    Either both `None` (single-encoding compression) or same-length
    tuples whose corresponding BlockReports compare equal field-by-field.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not _block_report_eq(x, y):
            return False
    return True


def _block_report_eq(x: BlockReport, y: BlockReport) -> bool:
    return (
        x.block_id == y.block_id
        and x.encoding_class == y.encoding_class
        and x.encoding_kwargs == y.encoding_kwargs
        and x.learn_axis_assignment == y.learn_axis_assignment
        and x.feature_ids == y.feature_ids
        and x.n_features_kept == y.n_features_kept
        and x.n_features_zeroed == y.n_features_zeroed
        and x.n_clusters == y.n_clusters
        and x.cluster_assignments == y.cluster_assignments
        and floats_eq(x.scale_compression_ratio, y.scale_compression_ratio)
        and floats_eq(x.rank_ratio, y.rank_ratio)
        and floats_eq(x.post_A, y.post_A)
        and floats_eq(x.forge_mse, y.forge_mse)
        and x.informative_metric == y.informative_metric
    )
