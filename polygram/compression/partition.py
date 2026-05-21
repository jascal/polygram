"""Per-block heterogeneous encoding partition for
:class:`polygram.compression.Compressor`.

Defines :class:`BlockSpec` (a single block's encoding + axis-assignment
+ feature-id list) and :func:`validate_partition_coverage` (the
disjointness + completeness check). Used by
:class:`polygram.config.CompressionConfig`'s ``encoding_partition``
field to drive per-block compression in
:meth:`polygram.compression.Compressor.apply`.

See ``openspec/changes/add-encoding-partition/proposal.md`` for the
full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


SUPPORTED_ENCODING_CLASSES: frozenset[str] = frozenset(
    {"MPSRung1", "Rung3", "Rung4", "Rung5", "HEA_Rung2"}
)


class PartitionCoverageError(ValueError):
    """Raised when an ``encoding_partition`` fails the disjointness or
    completeness coverage check against the input feature count.

    Subclasses :class:`ValueError` so existing
    ``except ValueError`` blocks still catch partition-validation
    failures; the dedicated class lets callers distinguish coverage
    errors from other ``ValueError``s when desired.
    """


def _validate_empty_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs:
        raise ValueError(
            f"BlockSpec: encoding_class accepts no kwargs; "
            f"got {sorted(kwargs)}"
        )


def _validate_rung5_kwargs(kwargs: dict[str, Any]) -> None:
    if "n_amp_qubits" not in kwargs:
        raise ValueError(
            "BlockSpec: encoding_class='Rung5' requires "
            "encoding_kwargs['n_amp_qubits']: int >= 1"
        )
    v = kwargs["n_amp_qubits"]
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(
            f"BlockSpec: encoding_kwargs['n_amp_qubits'] must be "
            f"int >= 1; got {v!r}"
        )
    extras = set(kwargs) - {"n_amp_qubits"}
    if extras:
        raise ValueError(
            f"BlockSpec: encoding_class='Rung5' got unexpected "
            f"kwargs: {sorted(extras)}"
        )


def _validate_hea_rung2_kwargs(kwargs: dict[str, Any]) -> None:
    if "n_qubits" not in kwargs:
        raise ValueError(
            "BlockSpec: encoding_class='HEA_Rung2' requires "
            "encoding_kwargs['n_qubits']: int >= 1"
        )
    v = kwargs["n_qubits"]
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(
            f"BlockSpec: encoding_kwargs['n_qubits'] must be "
            f"int >= 1; got {v!r}"
        )
    extras = set(kwargs) - {"n_qubits"}
    if extras:
        raise ValueError(
            f"BlockSpec: encoding_class='HEA_Rung2' got unexpected "
            f"kwargs: {sorted(extras)}"
        )


# Per-family validator registry. Adding a future family
# (Rung6, HEA_Rung3, …) is one new validator function + one
# registry entry, not a new branch in BlockSpec.__post_init__.
# See `openspec/changes/add-encoding-partition/design.md` Decision 2b.
_BLOCK_SPEC_KWARG_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "MPSRung1": _validate_empty_kwargs,
    "Rung3":    _validate_empty_kwargs,
    "Rung4":    _validate_empty_kwargs,
    "Rung5":    _validate_rung5_kwargs,
    "HEA_Rung2": _validate_hea_rung2_kwargs,
}


@dataclass(frozen=True)
class BlockSpec:
    """One block of a per-block heterogeneous encoding partition.

    Each block defines a subset of feature ids that SHALL be compressed
    with the same encoding family + kwargs + axis-assignment policy.
    A :class:`polygram.config.CompressionConfig`'s
    ``encoding_partition`` is a tuple of these.

    Fields:
        block_id: human-readable id (e.g. ``"heavy"``, ``"tail"``).
            Surfaces in the ``CompressionReport.blocks`` per-block
            report so analysts can correlate blocks with their
            partition-manifest entry.
        encoding_class: one of
            ``{"MPSRung1", "Rung3", "Rung4", "Rung5", "HEA_Rung2"}``.
        encoding_kwargs: per-family kwargs:
            - ``Rung5`` requires ``{"n_amp_qubits": int >= 1}``.
            - ``HEA_Rung2`` requires ``{"n_qubits": int >= 1}``.
            - Others MUST be empty.
        learn_axis_assignment: per-block axis-assignment policy
            (matches the top-level ``CompressionConfig`` field of
            the same name).
        feature_ids: tuple of non-negative ints, non-empty, no
            duplicates within this block. Disjointness across blocks
            + completeness vs n_features_input are checked separately
            at ``Compressor.apply`` time via
            :func:`validate_partition_coverage`.
    """

    block_id: str
    encoding_class: str
    encoding_kwargs: dict[str, Any] = field(default_factory=dict)
    learn_axis_assignment: bool = False
    feature_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError(
                f"BlockSpec: block_id must be a non-empty str; got "
                f"{self.block_id!r}"
            )
        if self.encoding_class not in SUPPORTED_ENCODING_CLASSES:
            raise ValueError(
                f"BlockSpec: encoding_class must be one of "
                f"{sorted(SUPPORTED_ENCODING_CLASSES)}; got "
                f"{self.encoding_class!r}"
            )
        if not isinstance(self.encoding_kwargs, dict):
            raise TypeError(
                f"BlockSpec: encoding_kwargs must be a dict; got "
                f"{type(self.encoding_kwargs).__name__}"
            )
        # Per-family validator (Decision 2b — registry, not elif chain).
        validator = _BLOCK_SPEC_KWARG_VALIDATORS[self.encoding_class]
        validator(self.encoding_kwargs)
        if not isinstance(self.learn_axis_assignment, bool):
            raise TypeError(
                f"BlockSpec: learn_axis_assignment must be bool; got "
                f"{type(self.learn_axis_assignment).__name__}"
            )
        if not isinstance(self.feature_ids, tuple):
            raise TypeError(
                f"BlockSpec: feature_ids must be a tuple; got "
                f"{type(self.feature_ids).__name__}"
            )
        if not self.feature_ids:
            raise ValueError("BlockSpec: feature_ids must be non-empty")
        seen: set[int] = set()
        for fid in self.feature_ids:
            if not isinstance(fid, int) or isinstance(fid, bool):
                raise TypeError(
                    f"BlockSpec: feature_ids must contain int; got "
                    f"{type(fid).__name__}"
                )
            if fid < 0:
                raise ValueError(
                    f"BlockSpec: feature_ids must be non-negative; "
                    f"got {fid}"
                )
            if fid in seen:
                raise ValueError(
                    f"BlockSpec: feature_ids contains duplicate id {fid} "
                    f"in block {self.block_id!r}"
                )
            seen.add(fid)

    # `frozen=True` provides default __eq__ / __hash__, but `dict` is
    # unhashable so __hash__ would crash on a populated encoding_kwargs.
    # Override to hash a frozenset of items.
    def __hash__(self) -> int:
        return hash((
            self.block_id,
            self.encoding_class,
            frozenset(self.encoding_kwargs.items()),
            self.learn_axis_assignment,
            self.feature_ids,
        ))


def validate_partition_coverage(
    partition: tuple[BlockSpec, ...],
    *,
    n_features_input: int,
) -> None:
    """Verify a partition's disjointness + completeness against the
    input feature count.

    Disjointness: no feature id appears in more than one block.
    Completeness: the union of all blocks' ``feature_ids`` is exactly
    ``set(range(n_features_input))``.

    On violation raises :class:`PartitionCoverageError` naming the
    offending ids (capped at first 10) so analysts can edit the
    manifest without scanning the whole list.

    Called by :meth:`polygram.compression.Compressor.apply` immediately
    after loading the input SAE (when ``n_features_input`` becomes
    known). Not called from ``BlockSpec.__post_init__`` because a
    BlockSpec is built before the input feature count is known.
    """
    if n_features_input < 0:
        raise ValueError(
            f"validate_partition_coverage: n_features_input must be "
            f"non-negative; got {n_features_input}"
        )

    seen: dict[int, str] = {}  # fid -> first-seen block_id
    duplicates: list[tuple[int, str, str]] = []  # (fid, block_a, block_b)
    for block in partition:
        for fid in block.feature_ids:
            if fid in seen:
                duplicates.append((fid, seen[fid], block.block_id))
            else:
                seen[fid] = block.block_id

    if duplicates:
        dupe_strs = [
            f"{fid} (in blocks {a!r} and {b!r})"
            for fid, a, b in duplicates[:10]
        ]
        more = ""
        if len(duplicates) > 10:
            more = f" ... and {len(duplicates) - 10} more"
        raise PartitionCoverageError(
            f"encoding_partition has overlapping feature_ids: "
            f"{', '.join(dupe_strs)}{more}"
        )

    expected = set(range(n_features_input))
    actual = set(seen.keys())
    missing = sorted(expected - actual)
    extras = sorted(actual - expected)

    if missing:
        truncated = missing[:10]
        more = f" ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise PartitionCoverageError(
            f"encoding_partition is incomplete: feature_ids "
            f"{truncated}{more} are not covered by any block "
            f"(n_features_input={n_features_input})"
        )
    if extras:
        truncated = extras[:10]
        more = f" ... and {len(extras) - 10} more" if len(extras) > 10 else ""
        raise PartitionCoverageError(
            f"encoding_partition has extra feature_ids "
            f"{truncated}{more} outside the input range "
            f"[0, {n_features_input})"
        )


def make_default_block(
    *,
    encoding_class: Literal["MPSRung1", "Rung3", "Rung4", "Rung5", "HEA_Rung2"],
    n_features_input: int,
    encoding_kwargs: dict[str, Any] | None = None,
    learn_axis_assignment: bool = False,
    excluded_feature_ids: set[int] | frozenset[int] = frozenset(),
    block_id: str = "default",
) -> BlockSpec:
    """Build a :class:`BlockSpec` covering all features in
    ``range(n_features_input)`` minus ``excluded_feature_ids``.

    Useful for "default + heavy override" partition patterns where
    a small set of heavy features get a custom block and the rest
    falls through to a default. See
    ``openspec/changes/add-encoding-partition/design.md`` Decision 2c.

    Example:
        >>> heavy = BlockSpec(
        ...     block_id="heavy", encoding_class="Rung5",
        ...     encoding_kwargs={"n_amp_qubits": 4},
        ...     feature_ids=(0, 1, 2, 3),
        ... )
        >>> tail = make_default_block(
        ...     encoding_class="MPSRung1",
        ...     n_features_input=128,
        ...     excluded_feature_ids={0, 1, 2, 3},
        ... )
        >>> partition = (heavy, tail)
    """
    excluded = frozenset(int(x) for x in excluded_feature_ids)
    feature_ids = tuple(
        fid for fid in range(int(n_features_input)) if fid not in excluded
    )
    if not feature_ids:
        raise ValueError(
            f"make_default_block: excluded_feature_ids covers every "
            f"feature in range({n_features_input}); the resulting "
            f"block would be empty"
        )
    return BlockSpec(
        block_id=block_id,
        encoding_class=encoding_class,
        encoding_kwargs=encoding_kwargs if encoding_kwargs is not None else {},
        learn_axis_assignment=learn_axis_assignment,
        feature_ids=feature_ids,
    )


__all__ = [
    "BlockSpec",
    "PartitionCoverageError",
    "SUPPORTED_ENCODING_CLASSES",
    "make_default_block",
    "validate_partition_coverage",
]
