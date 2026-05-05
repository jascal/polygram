"""Compression-regrow report dataclasses.

`SlotPopulation` describes one repopulated slot: which feature id it
occupies, how many residual tokens fed its centroid, and the post-
population decoder + encoder norms.

`RegrowPlan` is the cheap stage's output: the list of slot
populations, the zeroed-set the plan was built against, the strategy
name, the residual-token count, and the SAE-wide feature_ids list.
No I/O, no checkpoint hashes.

`RegrowReport` is the post-`apply()` artifact carrying source +
output sha256s, strategy parameters, the applied plan, slot counters,
and an optional `provenance` dict (populated by
`Regrower.from_compression_report`, empty for the direct
constructor).

`RegrowResult` is the in-process bundle: the plan applied, the
report written, the output-checkpoint path, and the rebuilt
`Dictionary`.

JSON layout matches `add-compression-regrow/design.md` Decision 8.
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
class SlotPopulation:
    """One repopulated slot's diagnostics.

    `feature_id` is the zeroed feature id this slot occupies.
    `cluster_size` is the count of residual tokens whose nearest
    centroid was this one (0 means the slot was left zero — the
    strategy chose not to populate it). `decoder_norm` and
    `encoder_norm` are the L2 norms of the populated tensors
    post-write (1.0 by construction for `residual_kmeans`).
    """

    feature_id: int
    cluster_size: int
    decoder_norm: float
    encoder_norm: float


@dataclass(frozen=True)
class RegrowPlan:
    """Output of `Regrower.plan()`.

    `slots` is the per-slot diagnostics list, sorted ascending by
    `feature_id`. Empty for `zeroed=set()`. `zeroed_input` is the
    input zeroed-set used to build the plan, sorted ascending — for
    audit. `feature_ids` is the SAE-wide feature-id list (matches the
    convention used by `CompressionPlan`).
    """

    strategy: str
    n_residual_tokens: int
    zeroed_input: tuple[int, ...]
    feature_ids: tuple[int, ...]
    slots: tuple[SlotPopulation, ...]


@dataclass(frozen=True, eq=False)
class RegrowReport:
    """Post-`apply()` artifact carrying provenance + the applied plan."""

    schema_version: int
    source_checkpoint: str
    source_checkpoint_sha256: str
    output_checkpoint: str
    output_checkpoint_sha256: str
    strategy: str
    plan: RegrowPlan
    n_slots_repopulated: int
    n_slots_left_zero: int
    strategy_params: dict[str, int | float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    # ---- JSON ------------------------------------------------------

    def to_json(self, path: str | os.PathLike | None = None) -> str:
        text = self._serialize()
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike) -> "RegrowReport":
        if not isinstance(source, (str, os.PathLike)):
            raise TypeError(
                "RegrowReport.from_json: expected a path or JSON string"
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
            "strategy",
            "plan",
            "n_slots_repopulated",
            "n_slots_left_zero",
            "strategy_params",
            "provenance",
        ):
            if key not in payload:
                raise ValueError(
                    f"RegrowReport.from_json: missing required key {key!r}"
                )

        plan_payload = payload["plan"]
        slots = tuple(_slot_from_dict(s) for s in plan_payload["slots"])
        plan = RegrowPlan(
            strategy=str(plan_payload["strategy"]),
            n_residual_tokens=int(plan_payload["n_residual_tokens"]),
            zeroed_input=tuple(
                int(f) for f in plan_payload["zeroed_input"]
            ),
            feature_ids=tuple(int(f) for f in plan_payload["feature_ids"]),
            slots=slots,
        )
        return cls(
            schema_version=int(payload["schema_version"]),
            source_checkpoint=str(payload["source_checkpoint"]),
            source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
            output_checkpoint=str(payload["output_checkpoint"]),
            output_checkpoint_sha256=str(payload["output_checkpoint_sha256"]),
            strategy=str(payload["strategy"]),
            plan=plan,
            n_slots_repopulated=int(payload["n_slots_repopulated"]),
            n_slots_left_zero=int(payload["n_slots_left_zero"]),
            strategy_params={
                k: _coerce_param(v)
                for k, v in payload["strategy_params"].items()
            },
            provenance={str(k): str(v) for k, v in payload["provenance"].items()},
        )

    # ---- Equality (NaN-aware on float diagnostics) -----------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegrowReport):
            return NotImplemented
        return (
            self.schema_version == other.schema_version
            and self.source_checkpoint == other.source_checkpoint
            and self.source_checkpoint_sha256 == other.source_checkpoint_sha256
            and self.output_checkpoint == other.output_checkpoint
            and self.output_checkpoint_sha256 == other.output_checkpoint_sha256
            and self.strategy == other.strategy
            and _plan_eq(self.plan, other.plan)
            and self.n_slots_repopulated == other.n_slots_repopulated
            and self.n_slots_left_zero == other.n_slots_left_zero
            and _params_eq(self.strategy_params, other.strategy_params)
            and self.provenance == other.provenance
        )

    def __hash__(self) -> int:
        return hash((
            self.schema_version,
            self.source_checkpoint_sha256,
            self.output_checkpoint_sha256,
            self.strategy,
            self.plan.feature_ids,
            self.n_slots_repopulated,
        ))

    # ---- Internal --------------------------------------------------

    def _serialize(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "source_checkpoint": self.source_checkpoint,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "output_checkpoint": self.output_checkpoint,
            "output_checkpoint_sha256": self.output_checkpoint_sha256,
            "strategy": self.strategy,
            "n_slots_repopulated": int(self.n_slots_repopulated),
            "n_slots_left_zero": int(self.n_slots_left_zero),
            "feature_ids": [int(f) for f in self.plan.feature_ids],
            "plan": {
                "strategy": self.plan.strategy,
                "n_residual_tokens": int(self.plan.n_residual_tokens),
                "zeroed_input": [int(f) for f in self.plan.zeroed_input],
                "feature_ids": [int(f) for f in self.plan.feature_ids],
                "slots": [_slot_to_dict(s) for s in self.plan.slots],
            },
            "strategy_params": {
                k: _serialize_param(v)
                for k, v in self.strategy_params.items()
            },
            "provenance": dict(self.provenance),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RegrowResult:
    """Bundle returned by `Regrower.apply()` / `run()`."""

    plan: RegrowPlan
    report: RegrowReport
    output_checkpoint: Path
    dictionary: "Dictionary"


# ============================================================================
# JSON helpers
# ============================================================================


def _slot_to_dict(s: SlotPopulation) -> dict[str, Any]:
    return {
        "feature_id": int(s.feature_id),
        "cluster_size": int(s.cluster_size),
        "decoder_norm": _json_finite(s.decoder_norm),
        "encoder_norm": _json_finite(s.encoder_norm),
    }


def _slot_from_dict(raw: dict[str, Any]) -> SlotPopulation:
    return SlotPopulation(
        feature_id=int(raw["feature_id"]),
        cluster_size=int(raw["cluster_size"]),
        decoder_norm=float(raw.get("decoder_norm") or 0.0),
        encoder_norm=float(raw.get("encoder_norm") or 0.0),
    )


def _serialize_param(v: Any) -> int | float:
    if isinstance(v, (int, bool)):
        return int(v)
    return _json_finite(float(v))


def _coerce_param(v: Any) -> int | float:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    return float(v)


# ============================================================================
# NaN-aware equality
# ============================================================================


def _floats_eq(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _slot_eq(a: SlotPopulation, b: SlotPopulation) -> bool:
    return (
        a.feature_id == b.feature_id
        and a.cluster_size == b.cluster_size
        and _floats_eq(a.decoder_norm, b.decoder_norm)
        and _floats_eq(a.encoder_norm, b.encoder_norm)
    )


def _plan_eq(a: RegrowPlan, b: RegrowPlan) -> bool:
    return (
        a.strategy == b.strategy
        and a.n_residual_tokens == b.n_residual_tokens
        and a.zeroed_input == b.zeroed_input
        and a.feature_ids == b.feature_ids
        and len(a.slots) == len(b.slots)
        and all(_slot_eq(x, y) for x, y in zip(a.slots, b.slots))
    )


def _params_eq(
    a: dict[str, int | float], b: dict[str, int | float]
) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if not _floats_eq(float(a[k]), float(b[k])):
            return False
    return True
