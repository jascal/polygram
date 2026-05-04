"""Feature-graph artifacts derived from a `TriagePrediction`.

Two kinds of graph are emitted by this module:

- **Sharing graph** — edges connect pairs of features that look like
  *good sharing candidates*: pairs whose `cancellation_gap` is large
  relative to their `current_overlap`, whose `structural_floor` is
  below `FLOOR_BLOCK`, and (by default) which sit within a single
  cluster. Phase tuning at use time can drive these pairs apart.

- **Separation graph** — edges connect pairs whose squared overlap
  cannot be driven low by phase tuning alone (their `structural_floor`
  remains above the kept-edge threshold). Cross-cluster edges of this
  kind flag pairs the phase-only triage cannot disambiguate; a future
  disentanglement primitive operating on β/α/γ would target them.

Both graphs reuse the `FeatureGraph` shape with a `kind` discriminator
and a kind-specific edge-weight rule. No q-orca calls; no per-pair
`Cancellation` runs; signal comes entirely from the closed-form
`PairPrediction` fields produced by `predict_cancellation_depth`.

Edge `reason` vocabulary
------------------------
Each kept `FeatureEdge.reason` is one of the following stable
identifiers (closed set):

- ``"phase_headroom"`` — sharing graph: the pair passed every gate
  (cluster, floor) and the clipped headroom ratio met the threshold.
- ``"irreducible_cross_cluster"`` — separation graph: kept
  cross-cluster pair whose structural floor is above the threshold.
- ``"irreducible_within_cluster"`` — separation graph: kept
  within-cluster pair (only emitted when
  ``include_within_cluster=True``).

Internal weight helpers (`_sharing_weight`, `_separation_weight`) also
return one of the following dropped-pair reasons; these never appear
on a `FeatureEdge` because dropped pairs are absent from the edge list,
but they are documented here for completeness:

- ``"blocked_cross_cluster"`` — sharing graph: cross-cluster pair with
  ``allow_cross_cluster=False`` (gate zeroed the weight).
- ``"blocked_high_floor"`` — sharing graph: ``structural_floor >
  FLOOR_BLOCK`` (gate zeroed the weight).
- ``"low_phase_headroom"`` — sharing graph: pair survived every gate
  but the clipped headroom ratio was below the threshold.
- ``"blocked_within_cluster"`` — separation graph: within-cluster pair
  with ``include_within_cluster=False`` (gate zeroed the weight).
- ``"low_floor"`` — separation graph: structural floor below the
  threshold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from polygram.analysis.triage import PairPrediction, TriagePrediction

FLOOR_BLOCK = 0.5

SHARING_EDGE_FORMULA = (
    "ratio  = p.cancellation_gap / max(p.current_overlap, 1e-12)\n"
    "weight = clip(ratio, 0.0, 1.0)\n"
    "         × (0.0 if p.structural_floor > FLOOR_BLOCK else 1.0)\n"
    "         × (0.0 if (p.is_cross_cluster and not allow_cross_cluster)\n"
    "            else 1.0)"
)

SEPARATION_EDGE_FORMULA = (
    "weight = clip(p.structural_floor, 0.0, 1.0)\n"
    "         × (0.0 if (not p.is_cross_cluster\n"
    "                    and not include_within_cluster) else 1.0)"
)

_SHARING_KIND = "sharing"
_SEPARATION_KIND = "separation"


@dataclass(frozen=True)
class FeatureEdge:
    """One kept edge in a `FeatureGraph`.

    `reason` is one of the kept-edge identifiers documented in this
    module's docstring (`"phase_headroom"` for sharing graphs;
    `"irreducible_cross_cluster"` / `"irreducible_within_cluster"` for
    separation graphs).
    """

    source: str
    target: str
    weight: float
    floor: float
    gap: float
    is_cross_cluster: bool
    reason: str


@dataclass(frozen=True)
class FeatureGraph:
    """Frozen graph artifact summarizing pair-level triage signal.

    `kind` discriminates `"sharing"` vs `"separation"`. `nodes` is the
    full feature list in dictionary order; `edges` is the kept-edge
    list sorted by descending weight (ties broken by `(source, target)`
    lexicographically); `clusters` is the set of connected components
    over the kept-edge subgraph (descending size; ties broken
    lexicographically by first member); `metadata` carries the
    builder's selection method, total feature count, threshold,
    builder-specific flags, and the kind-specific `formula` string.
    """

    kind: str
    nodes: tuple[str, ...]
    edges: tuple[FeatureEdge, ...]
    clusters: tuple[tuple[str, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "kind": self.kind,
            "nodes": list(self.nodes),
            "edges": [asdict(e) for e in self.edges],
            "clusters": [list(c) for c in self.clusters],
            "metadata": dict(self.metadata),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sharing_weight(
    p: PairPrediction, allow_cross_cluster: bool
) -> tuple[float, str]:
    if p.is_cross_cluster and not allow_cross_cluster:
        return 0.0, "blocked_cross_cluster"
    if p.structural_floor > FLOOR_BLOCK:
        return 0.0, "blocked_high_floor"
    denom = max(p.current_overlap, 1e-12)
    ratio = p.cancellation_gap / denom
    weight = max(0.0, min(1.0, ratio))
    if weight <= 0.0:
        return 0.0, "low_phase_headroom"
    return weight, "phase_headroom"


def _separation_weight(
    p: PairPrediction, include_within_cluster: bool
) -> tuple[float, str]:
    if not p.is_cross_cluster and not include_within_cluster:
        return 0.0, "blocked_within_cluster"
    weight = max(0.0, min(1.0, p.structural_floor))
    if p.is_cross_cluster:
        return weight, "irreducible_cross_cluster"
    return weight, "irreducible_within_cluster"


def build_sharing_graph(
    prediction: TriagePrediction,
    *,
    threshold: float = 0.5,
    allow_cross_cluster: bool = False,
) -> FeatureGraph:
    """Build a sharing-kind `FeatureGraph` from a `TriagePrediction`.

    Edges connect feature pairs whose computed sharing weight is at or
    above `threshold`. Pure function over `prediction.pairs`; no
    quantum simulation, no `Cancellation` runs.
    """
    edges: list[FeatureEdge] = []
    for p in prediction.pairs:
        weight, reason = _sharing_weight(p, allow_cross_cluster)
        # Gate-blocked pairs land at weight == 0; they SHALL NOT appear
        # in the edge list regardless of how loose `threshold` is.
        if weight <= 0.0 or weight < threshold:
            continue
        edges.append(
            FeatureEdge(
                source=p.feature_a,
                target=p.feature_b,
                weight=float(weight),
                floor=float(p.structural_floor),
                gap=float(p.cancellation_gap),
                is_cross_cluster=p.is_cross_cluster,
                reason=reason,
            )
        )

    nodes = tuple(f.name for f in prediction.dictionary.features)
    sorted_edges = _sort_edges(edges)
    clusters = _connected_components(nodes, sorted_edges)
    metadata: dict[str, Any] = {
        "kind": _SHARING_KIND,
        "selection_method": prediction.selection_report.cluster_method,
        "total_features": len(nodes),
        "threshold": float(threshold),
        "allow_cross_cluster": bool(allow_cross_cluster),
        "formula": SHARING_EDGE_FORMULA,
    }
    return FeatureGraph(
        kind=_SHARING_KIND,
        nodes=nodes,
        edges=sorted_edges,
        clusters=clusters,
        metadata=metadata,
    )


def build_separation_graph(
    prediction: TriagePrediction,
    *,
    threshold: float = 0.2,
    include_within_cluster: bool = False,
) -> FeatureGraph:
    """Build a separation-kind `FeatureGraph` from a `TriagePrediction`.

    Edges connect feature pairs whose `structural_floor` is at or above
    `threshold` (i.e. pairs the phase-only triage cannot disambiguate).
    Pure function over `prediction.pairs`; no quantum simulation, no
    `Cancellation` runs.
    """
    edges: list[FeatureEdge] = []
    for p in prediction.pairs:
        weight, reason = _separation_weight(p, include_within_cluster)
        if weight <= 0.0 or weight < threshold:
            continue
        edges.append(
            FeatureEdge(
                source=p.feature_a,
                target=p.feature_b,
                weight=float(weight),
                floor=float(p.structural_floor),
                gap=float(p.cancellation_gap),
                is_cross_cluster=p.is_cross_cluster,
                reason=reason,
            )
        )

    nodes = tuple(f.name for f in prediction.dictionary.features)
    sorted_edges = _sort_edges(edges)
    clusters = _connected_components(nodes, sorted_edges)
    metadata: dict[str, Any] = {
        "kind": _SEPARATION_KIND,
        "selection_method": prediction.selection_report.cluster_method,
        "total_features": len(nodes),
        "threshold": float(threshold),
        "include_within_cluster": bool(include_within_cluster),
        "formula": SEPARATION_EDGE_FORMULA,
    }
    return FeatureGraph(
        kind=_SEPARATION_KIND,
        nodes=nodes,
        edges=sorted_edges,
        clusters=clusters,
        metadata=metadata,
    )


def render_feature_graph_section(graph: FeatureGraph) -> str:
    """Render a kind-aware markdown section for a `FeatureGraph`.

    Top-level heading is `## Sharing graph` or `## Separation graph`;
    body sections are `### Edges`, `### Components`, `### Formula`.
    Deterministic given the input — appends to `render_report` output
    when the caller wants to include a graph.
    """
    if graph.kind == _SHARING_KIND:
        heading = "## Sharing graph"
    elif graph.kind == _SEPARATION_KIND:
        heading = "## Separation graph"
    else:
        raise ValueError(
            f"FeatureGraph.kind must be 'sharing' or 'separation'; "
            f"got {graph.kind!r}"
        )

    threshold = graph.metadata.get("threshold")
    threshold_str = (
        f"{float(threshold):.4f}" if threshold is not None else "n/a"
    )
    lines: list[str] = [
        heading,
        "",
        f"- threshold: {threshold_str}",
        f"- kept edges: {len(graph.edges)}",
        f"- components: {len(graph.clusters)}",
        "",
        "### Edges",
        "",
        "| source | target | weight | floor | gap | cross_cluster | reason |",
        "|--------|--------|-------:|------:|----:|:-------------:|--------|",
    ]
    for e in graph.edges:
        lines.append(
            f"| {e.source} | {e.target} | {e.weight:.4f} | "
            f"{e.floor:.4f} | {e.gap:.4f} | "
            f"{'yes' if e.is_cross_cluster else 'no'} | {e.reason} |"
        )
    if not graph.edges:
        lines.append("| _(none)_ | | | | | | |")
    lines.extend(["", "### Components", ""])
    if graph.clusters:
        for c in graph.clusters:
            lines.append(f"- {{ {', '.join(c)} }}")
    else:
        lines.append("- _(none)_")
    lines.extend([
        "",
        "### Formula",
        "",
        "```",
        graph.metadata.get("formula", ""),
        "```",
        "",
    ])
    return "\n".join(lines)


def _sort_edges(edges: list[FeatureEdge]) -> tuple[FeatureEdge, ...]:
    return tuple(
        sorted(
            edges,
            key=lambda e: (-float(e.weight), e.source, e.target),
        )
    )


def _connected_components(
    nodes: tuple[str, ...], edges: tuple[FeatureEdge, ...]
) -> tuple[tuple[str, ...], ...]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Union by lexicographically smaller root for determinism.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for e in edges:
        if e.source in parent and e.target in parent:
            union(e.source, e.target)

    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)

    components = [tuple(sorted(members)) for members in groups.values()]
    components.sort(key=lambda c: (-len(c), c[0]))
    return tuple(components)
