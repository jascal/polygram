"""BatchExperiment — FeatureGraph-driven `Cancellation` runner.

Consumes a `FeatureGraph` (typically produced by `build_sharing_graph`
or `build_separation_graph`) plus a `Dictionary` and runs
`Cancellation` on the graph's top-K edges. Emits a `BatchResults`
artifact carrying the input graph verbatim alongside per-pair
empirical fields, so callers can compare the closed-form prediction
(`predicted_floor`, `predicted_gap`) with what φ/θ search actually
delivered (`achieved_overlap`, `cancellation_efficiency`).

The default `knobs="cluster_shared"` regime is the algebra-preserving
choice the `add-cluster-shared-knobs` archive validated for `MPSRung1`
`<cluster>.phi` paths. On `HEA_Rung2` the same string switches to
`<cluster>.theta[r,d,q]` paths; the `r, q` slot follows the canonical
`_default_hea_theta` layout (the φ slot is `[Rz_index, 0, 1]`).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polygram.analysis.feature_graph import FeatureEdge, FeatureGraph
from polygram.cancellation import Cancellation
from polygram.dictionary import Dictionary
from polygram.encoding import HEA_Rung2

SUPPORTED_KNOBS = ("cluster_shared", "per_feature")
TOP_K_MIN = 1
TOP_K_MAX = 16
SIG_FIGS = 6


def _round_sig(v: float | None, sigfigs: int = SIG_FIGS) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if fv == 0.0 or not math.isfinite(fv):
        return fv
    return float(format(fv, f".{sigfigs}g"))


def _round_knobs(knobs: dict[str, float]) -> dict[str, float]:
    return {k: float(_round_sig(v)) for k, v in knobs.items()}


@dataclass(frozen=True)
class BatchRun:
    """One pair-level row in a `BatchResults`."""

    source: str
    target: str
    predicted_floor: float
    predicted_gap: float
    current_overlap: float
    achieved_overlap: float
    cancellation_efficiency: float
    best_knobs: dict[str, float]
    tier_separation_after: float | None
    artifact_subpath: str | None


@dataclass(frozen=True)
class BatchResults:
    """Aggregated output of a `BatchExperiment.run()`."""

    source_graph: FeatureGraph
    dictionary_name: str
    knobs: str
    created_at: str
    runs: tuple[BatchRun, ...]

    def to_json(self, path: str | os.PathLike | None = None) -> str:
        """Serialize to deterministic JSON.

        Numeric fields are emitted at 6 significant figures (matching
        the in-memory rounding done at run-time), the nested
        `source_graph` is parsed via `FeatureGraph.to_json()` and
        re-attached as a nested object, and `runs` are emitted in
        input-graph edge order. When `path` is provided the JSON is
        written there in addition to being returned.
        """
        text = self._serialize()
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return text

    @classmethod
    def from_json(
        cls, source: str | os.PathLike
    ) -> "BatchResults":
        """Inverse of `to_json`: accepts either a path or a JSON string."""
        if isinstance(source, (str, bytes, os.PathLike)):
            candidate = Path(source) if not isinstance(source, bytes) else None
            if candidate is not None and candidate.is_file():
                payload = json.loads(candidate.read_text())
            else:
                payload = json.loads(source if isinstance(source, str) else source.decode())
        else:
            raise TypeError(
                "BatchResults.from_json: expected a path or JSON string"
            )
        for key in (
            "source_graph",
            "dictionary_name",
            "knobs",
            "created_at",
            "runs",
        ):
            if key not in payload:
                raise ValueError(
                    f"BatchResults.from_json: missing required key {key!r}"
                )
        graph = FeatureGraph.from_json(payload["source_graph"])
        runs = tuple(_run_from_dict(r) for r in payload["runs"])
        return cls(
            source_graph=graph,
            dictionary_name=str(payload["dictionary_name"]),
            knobs=str(payload["knobs"]),
            created_at=str(payload["created_at"]),
            runs=runs,
        )

    def _serialize(self) -> str:
        runs_payload = [
            {
                "source": r.source,
                "target": r.target,
                "predicted_floor": _round_sig(r.predicted_floor),
                "predicted_gap": _round_sig(r.predicted_gap),
                "current_overlap": _round_sig(r.current_overlap),
                "achieved_overlap": _round_sig(r.achieved_overlap),
                "cancellation_efficiency": _round_sig(r.cancellation_efficiency),
                "best_knobs": _round_knobs(r.best_knobs),
                "tier_separation_after": _round_sig(r.tier_separation_after),
                "artifact_subpath": r.artifact_subpath,
            }
            for r in self.runs
        ]
        payload = {
            "source_graph": json.loads(self.source_graph.to_json()),
            "dictionary_name": self.dictionary_name,
            "knobs": self.knobs,
            "created_at": self.created_at,
            "runs": runs_payload,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _run_from_dict(raw: dict[str, Any]) -> BatchRun:
    return BatchRun(
        source=str(raw["source"]),
        target=str(raw["target"]),
        predicted_floor=float(raw["predicted_floor"]),
        predicted_gap=float(raw["predicted_gap"]),
        current_overlap=float(raw["current_overlap"]),
        achieved_overlap=float(raw["achieved_overlap"]),
        cancellation_efficiency=float(raw["cancellation_efficiency"]),
        best_knobs={k: float(v) for k, v in raw["best_knobs"].items()},
        tier_separation_after=(
            None
            if raw["tier_separation_after"] is None
            else float(raw["tier_separation_after"])
        ),
        artifact_subpath=raw["artifact_subpath"],
    )


@dataclass
class BatchExperiment:
    """Run `Cancellation` on the top-K edges of a `FeatureGraph`."""

    feature_graph: FeatureGraph
    dictionary: Dictionary
    top_k: int = 8
    knobs: str = "cluster_shared"
    output_dir: Path | None = None
    cancellation_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.knobs not in SUPPORTED_KNOBS:
            raise ValueError(
                f"BatchExperiment.knobs must be one of {SUPPORTED_KNOBS!r}; "
                f"got {self.knobs!r}"
            )
        if self.top_k < TOP_K_MIN:
            raise ValueError(
                f"BatchExperiment.top_k must be >= {TOP_K_MIN}; "
                f"got {self.top_k}"
            )
        if self.top_k > TOP_K_MAX:
            raise ValueError(
                f"BatchExperiment.top_k must be <= {TOP_K_MAX} "
                f"(cluster-shared Cancellation runs at ~seconds per pair; "
                f"the cap exists to keep wall time bounded); got {self.top_k}"
            )
        feature_names = {f.name for f in self.dictionary.features}
        missing = [n for n in self.feature_graph.nodes if n not in feature_names]
        if missing:
            raise ValueError(
                f"BatchExperiment: feature_graph nodes not declared by "
                f"the dictionary: {missing}; dictionary feature names "
                f"are {sorted(feature_names)}"
            )

    def run(self) -> BatchResults:
        created_at = _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        edges = self.feature_graph.edges[: min(self.top_k, len(self.feature_graph.edges))]
        runs: list[BatchRun] = []
        out_root: Path | None = None
        if self.output_dir is not None:
            out_root = Path(self.output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
        total = len(edges)
        for i, edge in enumerate(edges, start=1):
            t0 = time.monotonic()
            run = self._run_one_edge(edge, out_root)
            runs.append(run)
            elapsed = time.monotonic() - t0
            print(
                f"polygram batch: {i}/{total}: "
                f"{edge.source} x {edge.target} — done in {elapsed:.1f}s"
            )
        result = BatchResults(
            source_graph=self.feature_graph,
            dictionary_name=self.dictionary.name,
            knobs=self.knobs,
            created_at=created_at,
            runs=tuple(runs),
        )
        if out_root is not None:
            result.to_json(out_root / "batch_results.json")
        return result

    def _run_one_edge(
        self, edge: FeatureEdge, out_root: Path | None
    ) -> BatchRun:
        knob_paths = self._resolve_knob_paths(edge)
        ck = dict(self.cancellation_kwargs or {})
        ck.setdefault("preserve_tiers", False)
        cancel = Cancellation(
            dictionary=self.dictionary,
            target_pair=(edge.source, edge.target),
            knobs=knob_paths,
            **ck,
        )
        result = cancel.run()
        artifact_subpath: str | None = None
        if out_root is not None:
            sub = out_root / f"{edge.source}_x_{edge.target}"
            sub.mkdir(parents=True, exist_ok=True)
            result.materialize(sub)
            artifact_subpath = f"{edge.source}_x_{edge.target}"

        try:
            tier_after = result.dictionary_at_optimum.tier_separation()
            tier_after_val: float | None = (
                None if tier_after is None else float(tier_after)
            )
        except Exception:
            tier_after_val = None

        predicted_gap = float(edge.gap)
        current_overlap = float(result.before_overlap)
        achieved_overlap = float(result.after_overlap)
        if predicted_gap > 1e-12:
            efficiency = (current_overlap - achieved_overlap) / predicted_gap
        else:
            efficiency = 0.0

        return BatchRun(
            source=edge.source,
            target=edge.target,
            predicted_floor=float(_round_sig(edge.floor)),
            predicted_gap=float(_round_sig(predicted_gap)),
            current_overlap=float(_round_sig(current_overlap)),
            achieved_overlap=float(_round_sig(achieved_overlap)),
            cancellation_efficiency=float(_round_sig(efficiency)),
            best_knobs=_round_knobs(result.optimized_knobs),
            tier_separation_after=_round_sig(tier_after_val),
            artifact_subpath=artifact_subpath,
        )

    def _resolve_knob_paths(self, edge: FeatureEdge) -> list[str]:
        encoding = self.dictionary.encoding
        is_hea = isinstance(encoding, HEA_Rung2)
        if self.knobs == "cluster_shared":
            ca = self.dictionary.feature(edge.source).cluster
            cb = self.dictionary.feature(edge.target).cluster
            clusters: list[str] = [ca]
            if cb != ca:
                clusters.append(cb)
            if is_hea:
                slot = _hea_phi_slot(encoding)
                return [f"{c}.theta[{slot[0]},{slot[1]},{slot[2]}]" for c in clusters]
            return [f"{c}.phi" for c in clusters]

        # per_feature
        if is_hea:
            slot = _hea_phi_slot(encoding)
            return [
                f"{edge.source}.theta[{slot[0]},{slot[1]},{slot[2]}]",
                f"{edge.target}.theta[{slot[0]},{slot[1]},{slot[2]}]",
            ]
        return [f"{edge.source}.phi", f"{edge.target}.phi"]


def _hea_phi_slot(encoding: HEA_Rung2) -> tuple[int, int, int]:
    """Pick the canonical "phi-equivalent" θ slot for a cluster-shared
    or per-feature HEA knob, mirroring `_default_hea_theta`'s layout.

    Prefers the `Rz` rotation when available; falls back to the first
    rotation. The qubit defaults to 1 (the inner-rung qubit on the
    canonical 3-qubit register) and clamps for narrower registers.
    """
    if "Rz" in encoding.rotations:
        r = encoding.rotations.index("Rz")
    else:
        r = 0
    q = 1 if encoding.n_qubits >= 2 else 0
    return (r, 0, q)
