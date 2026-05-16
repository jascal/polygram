"""Polygram tuning configuration dataclasses.

This module centralises every tunable knob exposed on a polygram public
constructor (`Compressor`, `EpochCompressor`, `Cancellation`,
`BehaviouralValidator`, `Regrower.from_compression_report`,
`from_sae_lens`) into a small set of frozen dataclasses with explicit
range validation, dict round-trip, and forward-compatible deserialization.

## Override precedence

Each affected constructor accepts an optional ``config`` keyword argument
in addition to its existing per-field kwargs. The resolution rule is:

1. An explicit per-field keyword argument wins (the field is "set" if its
   value differs from the constructor's sentinel ``None`` default).
2. Otherwise, when ``config`` is supplied, the matching field on the
   config instance applies.
3. Otherwise, the field's dataclass default applies.

In short: **per-field kwargs > config > defaults**.

When ``config=None`` and no per-field kwargs are supplied, behaviour is
identical to the constructor's pre-existing defaults.

## Dict round-trip

Every config implements ``to_dict()`` and ``from_dict(data)``:

- ``to_dict()`` returns a JSON-serializable mapping (tuples → lists,
  nested configs serialised recursively).
- ``from_dict(data)`` accepts dicts produced by ``to_dict()`` (or
  hand-rolled dicts), coerces tuple-typed fields from lists, recurses
  into nested-config keys, and emits a :class:`UserWarning` when an
  unknown key is supplied (forward-compat: a stored config from an older
  release survives a knob being added).

Downstream callers (e.g. sae-forge's FSM context dicts) can stash a
config bundle as a JSON-friendly dict via ``cfg.to_dict()`` and
reconstitute it via ``<Config>.from_dict(...)`` without manual marshalling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Mixin: dict round-trip with tuple coercion, nested-config recursion,
# and forward-compatible unknown-key handling.
# ---------------------------------------------------------------------------


class _ConfigMixin:
    """Adds ``to_dict()`` / ``from_dict()`` to a frozen dataclass.

    Subclasses must be ``@dataclass(frozen=True)``. Tuple fields are
    serialised as lists and reconstituted from lists; nested-config
    fields (any value that is itself a ``_ConfigMixin`` instance) are
    serialised recursively. Unknown keys in :meth:`from_dict` warn and
    are dropped.
    """

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if isinstance(v, _ConfigMixin):
                out[f.name] = v.to_dict()
            elif isinstance(v, tuple):
                out[f.name] = list(v)
            else:
                out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            raise TypeError(
                f"{cls.__name__}.from_dict expected a mapping, "
                f"got {type(data).__name__}"
            )
        known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                warnings.warn(
                    f"{cls.__name__}.from_dict: ignoring unknown key {key!r}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            f = known[key]
            kwargs[key] = _coerce_field(f.type, value)
        return cls(**kwargs)


def _coerce_field(declared_type: Any, value: Any) -> Any:
    """Coerce a deserialised value to match a dataclass field's type.

    Handles two cases that ``json.loads`` mangles:

    1. ``tuple[...]`` — JSON serialises tuples as lists, so a list comes
       back; we coerce to ``tuple`` when the declared type is a tuple.
    2. Nested ``_ConfigMixin`` configs — when a field's declared type is
       a config class and the value is a dict, recursively rebuild via
       ``from_dict``.

    Falls back to identity for everything else (the constructor's own
    ``__post_init__`` does range validation).
    """
    type_str = declared_type if isinstance(declared_type, str) else repr(declared_type)
    if "tuple" in type_str.lower() and isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        # Sort longest-first so e.g. ``EpochCompressionConfig`` matches
        # ahead of its substring ``CompressionConfig``.
        for nested_cls in sorted(
            _registered_configs, key=lambda c: -len(c.__name__)
        ):
            if nested_cls.__name__ in type_str:
                return nested_cls.from_dict(value)
    return value


# Registered config classes (filled below). Used by ``_coerce_field`` to
# recurse into nested-config-typed fields.
_registered_configs: list[type] = []


def _register(cls):
    _registered_configs.append(cls)
    return cls


# ---------------------------------------------------------------------------
# ValidationConfig — BehaviouralValidator knobs
# ---------------------------------------------------------------------------


@_register
@dataclass(frozen=True)
class ValidationConfig(_ConfigMixin):
    """Tuning for :class:`polygram.behavioural.BehaviouralValidator`.

    Defaults match the §4.4 GPT-2-small calibration that
    ``BehaviouralValidator`` shipped with.
    """

    polygram_overlap_threshold: float = 0.7
    jaccard_threshold: float = 0.30
    min_firing_rate: float = 0.01
    min_both_fire: int = 5
    allow_layer_zero: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.polygram_overlap_threshold <= 1.0):
            raise ValueError(
                f"ValidationConfig: polygram_overlap_threshold must be in "
                f"[0, 1]; got {self.polygram_overlap_threshold}"
            )
        if not (0.0 <= self.jaccard_threshold <= 1.0):
            raise ValueError(
                f"ValidationConfig: jaccard_threshold must be in [0, 1]; "
                f"got {self.jaccard_threshold}"
            )
        if not (0.0 <= self.min_firing_rate <= 1.0):
            raise ValueError(
                f"ValidationConfig: min_firing_rate must be in [0, 1]; "
                f"got {self.min_firing_rate}"
            )
        # Aligned with ``BehaviouralValidator.__post_init__``'s ``>= 1``
        # check — a config with ``min_both_fire=0`` would always fail at
        # the validator's own range check, so we surface it earlier.
        if int(self.min_both_fire) < 1:
            raise ValueError(
                f"ValidationConfig: min_both_fire must be >= 1; "
                f"got {self.min_both_fire}"
            )


# ---------------------------------------------------------------------------
# CancellationConfig — Cancellation tuning knobs
# ---------------------------------------------------------------------------


_DEFAULT_OPTIMIZE = {"method": "grid", "max_steps": 50}
_SUPPORTED_OPTIMIZE_METHODS = ("grid", "scipy")


@_register
@dataclass(frozen=True)
class CancellationConfig(_ConfigMixin):
    """Tuning for :class:`polygram.cancellation.Cancellation`.

    ``optimize`` is intentionally a plain dict (rather than its own
    dataclass) to preserve the ``{"method": "grid", "max_steps": 50}``
    shape that existing tests and callers use.
    """

    tolerance: float = 0.05
    preserve_tiers: bool = True
    optimize: dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_OPTIMIZE)
    )
    grid_outer: tuple[int, int] = (5, 5)
    min_amp_overlap: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.tolerance <= 1.0):
            raise ValueError(
                f"CancellationConfig: tolerance must be in [0, 1]; "
                f"got {self.tolerance}"
            )
        method = self.optimize.get("method", "grid")
        if method not in _SUPPORTED_OPTIMIZE_METHODS:
            raise ValueError(
                f"CancellationConfig: optimize['method'] must be one of "
                f"{_SUPPORTED_OPTIMIZE_METHODS}; got {method!r}"
            )
        if not (
            isinstance(self.grid_outer, tuple)
            and len(self.grid_outer) == 2
            and int(self.grid_outer[0]) >= 1
            and int(self.grid_outer[1]) >= 1
        ):
            raise ValueError(
                f"CancellationConfig: grid_outer must be a (M, N) tuple "
                f"with M >= 1 and N >= 1; got {self.grid_outer!r}"
            )
        if not (0.0 <= self.min_amp_overlap <= 1.0):
            raise ValueError(
                f"CancellationConfig: min_amp_overlap must be in [0, 1]; "
                f"got {self.min_amp_overlap}"
            )


# ---------------------------------------------------------------------------
# CompressionConfig — Compressor knobs
# ---------------------------------------------------------------------------


_SUPPORTED_STRATEGIES = ("merge", "zero")
_SUPPORTED_REP_SELECTIONS = ("n_fires", "scale_aware", "kl_attribution")
_SUPPORTED_MERGE_MODES = ("freq_weighted", "simple_mean")
_SUPPORTED_SCORE_FIELDS = ("polygram_overlap", "jaccard", "decoder_overlap")


@_register
@dataclass(frozen=True)
class CompressionConfig(_ConfigMixin):
    """Tuning for :class:`polygram.compression.Compressor`.

    Defaults mirror what current external callers (e.g. sae-forge) pass
    explicitly today.

    `target_n_features_kept` is the count of cluster *representatives*
    (matching `CompressionReport.n_features_kept`), not the SAE's total
    surviving feature count. Setting it activates
    :meth:`polygram.compression.Compressor.plan_with_target`; leaving it
    `None` keeps the historical threshold-driven `plan()` path
    byte-identical. See `openspec/changes/add-pareto-target-compression/design.md`
    Decision 1.

    `score_field` chooses which `CandidatePair` field orders the greedy
    union in target-K mode. Only the three bounded-`[0, 1]`
    similarity-like fields are accepted; KL- and count-based fields are
    excluded by Decision 3 of the same change.

    `rep_selection` chooses how cluster representatives are picked
    when more than one feature joins the same cluster. Three modes:

    - ``"n_fires"`` — highest summed firing count; tiebreak lowest
      feature id. Pure-frequency proxy.
    - ``"scale_aware"`` (default) — weighted mix of decoder-norm
      proximity (0.4), summed kl_ablate (0.4), and log firing count
      (0.2). Falls back gracefully when kl_ablate is all-NaN.
    - ``"kl_attribution"`` — opt-in; pure behavioural-ablation
      importance. Picks the rep with the highest mean ``kl_ablate``
      across the pairs containing it. Per-feature NaN falls back to a
      geometric proxy for that one feature only; all-NaN cluster
      raises ``ValueError`` (the report came from a geometry-only
      confirmer that doesn't compute behavioural fields). Added in
      ``add-kl-attribution-rep-selection`` (polygram 0.5.0).

    **When to prefer ``kl_attribution``**: behaviourally-rich
    ``ValidationReport`` (came through ``BehaviouralValidator``, not
    ``DecoderGeometryConfirmer`` / ``ClusterConfirmer``) AND a
    structurally-feasible forge regime (sae-forge's
    ``quality_tier in {"good", "saturated"}`` rows from
    ``add-forge-quality-diagnostics``). In ``degenerate``/``undersized``
    regimes the rep choice is curiosity-level noise — no
    rep_selection rescues a rank-1 basis against a 768-dim
    residual. The default remains ``scale_aware`` until empirical
    evidence on real Axis-4 forge sweeps establishes a
    Pareto-dominant choice.

    **When to prefer ``scale_aware`` or ``n_fires``**: geometry-only
    reports (e.g. ``DecoderGeometryConfirmer``) lack the
    ``kl_ablate_*`` columns ``kl_attribution`` requires — those paths
    SHALL use ``scale_aware`` (graceful all-NaN fallback) or
    ``n_fires`` (no behavioural input).
    """

    strategy: str = "merge"
    rep_selection: str = "scale_aware"
    merge_mode: str = "freq_weighted"
    confirmer: str | None = None
    target_n_features_kept: int | None = None
    score_field: str = "polygram_overlap"

    def __post_init__(self) -> None:
        if self.strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"CompressionConfig: strategy must be one of "
                f"{_SUPPORTED_STRATEGIES}; got {self.strategy!r}"
            )
        if self.rep_selection not in _SUPPORTED_REP_SELECTIONS:
            raise ValueError(
                f"CompressionConfig: rep_selection must be one of "
                f"{_SUPPORTED_REP_SELECTIONS}; got {self.rep_selection!r}"
            )
        if self.merge_mode not in _SUPPORTED_MERGE_MODES:
            raise ValueError(
                f"CompressionConfig: merge_mode must be one of "
                f"{_SUPPORTED_MERGE_MODES}; got {self.merge_mode!r}"
            )
        if self.target_n_features_kept is not None:
            if (
                not isinstance(self.target_n_features_kept, int)
                or isinstance(self.target_n_features_kept, bool)
                or self.target_n_features_kept < 1
            ):
                raise ValueError(
                    "CompressionConfig: target_n_features_kept must be "
                    "None or an integer >= 1; got "
                    f"{self.target_n_features_kept!r}"
                )
        if self.score_field not in _SUPPORTED_SCORE_FIELDS:
            raise ValueError(
                f"CompressionConfig: score_field must be one of "
                f"{_SUPPORTED_SCORE_FIELDS}; got {self.score_field!r}"
            )


# ---------------------------------------------------------------------------
# EpochCompressionConfig — EpochCompressor tuning knobs
# ---------------------------------------------------------------------------


@_register
@dataclass(frozen=True)
class EpochCompressionConfig(_ConfigMixin):
    """Tuning for :class:`polygram.compression.EpochCompressor`.

    Defaults are the iterative-loop preset that downstream callers
    (sae-forge's outer-loop FSM) use. The pre-change "exhaustive offline
    run" defaults are reachable via :meth:`EpochCompressor.thorough`.
    """

    coverage_target: float = 0.5
    cosine_threshold: float = 0.30
    n_visits_per_feature: int = 1
    max_iterations: int = 1
    quality_delta_multiplier: float = 2.0
    validation: ValidationConfig | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.coverage_target <= 1.0):
            raise ValueError(
                f"EpochCompressionConfig: coverage_target must be in "
                f"(0, 1]; got {self.coverage_target}"
            )
        if not (-1.0 <= self.cosine_threshold <= 1.0):
            raise ValueError(
                f"EpochCompressionConfig: cosine_threshold must be in "
                f"[-1, 1]; got {self.cosine_threshold}"
            )
        if int(self.n_visits_per_feature) < 1:
            raise ValueError(
                f"EpochCompressionConfig: n_visits_per_feature must be >= 1; "
                f"got {self.n_visits_per_feature}"
            )
        if int(self.max_iterations) < 1:
            raise ValueError(
                f"EpochCompressionConfig: max_iterations must be >= 1; "
                f"got {self.max_iterations}"
            )
        if float(self.quality_delta_multiplier) <= 0:
            raise ValueError(
                f"EpochCompressionConfig: quality_delta_multiplier must "
                f"be > 0; got {self.quality_delta_multiplier}"
            )
        if self.validation is not None and not isinstance(
            self.validation, ValidationConfig
        ):
            raise TypeError(
                f"EpochCompressionConfig: validation must be a "
                f"ValidationConfig or None; got {type(self.validation).__name__}"
            )


# ---------------------------------------------------------------------------
# RegrowConfig — Regrower.from_compression_report tuning knobs
# ---------------------------------------------------------------------------


@_register
@dataclass(frozen=True, kw_only=True)
class RegrowConfig(_ConfigMixin):
    """Tuning for :meth:`polygram.compression.Regrower.from_compression_report`.

    ``model_name`` and ``layer`` are required (no defaults) because the
    pre-change defaults silently assumed GPT-2; an incorrect layer index
    on a non-GPT-2 host produces nonsense regrowth. Every field is
    keyword-only so the required fields can declare no default while
    sitting alongside fields that do.
    """

    model_name: str
    layer: int
    strategy: str = "residual_kmeans"
    prompts: tuple[str, ...] | None = None
    seed: int = 0
    n_init: int = 4
    top_k: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError(
                f"RegrowConfig: model_name must be a non-empty string; "
                f"got {self.model_name!r}"
            )
        if int(self.layer) < 0:
            raise ValueError(
                f"RegrowConfig: layer must be >= 0; got {self.layer}"
            )
        if int(self.n_init) < 1:
            raise ValueError(
                f"RegrowConfig: n_init must be >= 1; got {self.n_init}"
            )
        if self.top_k is not None and int(self.top_k) < 0:
            raise ValueError(
                f"RegrowConfig: top_k must be None or a non-negative int; "
                f"got top_k={self.top_k}"
            )


# ---------------------------------------------------------------------------
# SAEImportConfig — from_sae_lens knobs
# ---------------------------------------------------------------------------


@_register
@dataclass(frozen=True)
class SAEImportConfig(_ConfigMixin):
    """Tuning for :func:`polygram.sae_import.from_sae_lens`.

    Note: ``assign_gamma`` defaults to ``True`` (the README's recommended
    setting on real SAEs). The legacy ``assign_gamma=False`` behaviour is
    reachable by passing ``assign_gamma=False`` explicitly.
    """

    assign_gamma: bool = True
    gamma_range: tuple[float, float] = (-0.25, 0.25)
    n_clusters: int = 2
    profile: str | None = None
    # encoding-aware-knob-assignment. Default False preserves byte-
    # identical behaviour. When True, the loader populates higher-rung
    # encodings' amp-branch knobs from decoder PCA (see
    # polygram/geometry/amp_assignment.py). No-op for MPSRung1 /
    # HEA_Rung2.
    assign_amp_knobs: bool = False
    # add-phase-knob-assignment. Default False preserves byte-identical
    # behaviour. When True, the loader populates MPS-substrate α and φ
    # knobs from decoder PCA (PC2 and PC3 — see
    # polygram/geometry/phase_assignment.py). Applies to MPSRung1,
    # Rung3, Rung4 (all share MPS-substrate phase knobs). No-op for
    # HEA_Rung2 (different knob structure).
    assign_phase_knobs: bool = False
    # add-learned-axis-assignment. Default `None` preserves byte-
    # identical behaviour. When `True`, the loader instantiates a
    # default `LearnedKnobAssignment` and uses it instead of the
    # hardcoded `assign_*_pca` helpers. Explicit `LearnedKnobAssignment`
    # instances are not config-serialisable — pass them directly to
    # `from_sae_lens(...)` instead. See
    # `polygram/geometry/learned_axis_assignment.py`.
    learn_axis_assignment: bool | None = None

    def __post_init__(self) -> None:
        if not (
            isinstance(self.gamma_range, tuple)
            and len(self.gamma_range) == 2
            and self.gamma_range[0] <= self.gamma_range[1]
        ):
            raise ValueError(
                f"SAEImportConfig: gamma_range must be a (lo, hi) tuple "
                f"with lo <= hi; got {self.gamma_range!r}"
            )
        if int(self.n_clusters) < 1:
            raise ValueError(
                f"SAEImportConfig: n_clusters must be >= 1; "
                f"got {self.n_clusters}"
            )
        if self.profile is not None and not isinstance(self.profile, str):
            raise ValueError(
                f"SAEImportConfig: profile must be a string or None; "
                f"got {type(self.profile).__name__}"
            )


__all__ = [
    "CancellationConfig",
    "CompressionConfig",
    "EpochCompressionConfig",
    "RegrowConfig",
    "SAEImportConfig",
    "ValidationConfig",
]
