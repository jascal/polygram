"""Cancellation primitive — find knob values that drive a target-pair
overlap below a tolerance, optionally constrained to preserve the
hierarchical-tier ordering.

Two backends ship: a deterministic per-axis grid scan (no extra deps;
``len(knobs) <= 4``) and a `scipy.optimize.differential_evolution`
backend behind the `polygram[opt]` extra (dimension-agnostic).

The search space defaults to the two `<feature>.phi` knobs of
`target_pair` (preserves the rung-1 2-φ behavior) but accepts any
list of knob paths in the `Dictionary.with_knob` grammar
(`<feature>.phi` on either encoding, `<feature>.theta[r,d,q]` on
`HEA_Rung2`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from polygram.config import CancellationConfig  # noqa: F401

from polygram._assertions import hierarchical_ordering_preserved
from polygram.dictionary import Dictionary, _parse_knob_path
from polygram.emit import write_qorca
from polygram.encoding import (
    HEA_Rung2,
    MPSRung1,
    Rung3,
    Rung4,
    Rung5,
    rung3_amp_overlap_squared,
    rung4_amp_overlap_squared,
    rung5_amp_overlap_squared,
)

SUPPORTED_METHODS = ("grid", "scipy")
SUPPORTED_PLOT_KINDS = ("grid", "scipy", "before_after")
SUPPORTED_ENCODINGS = ("mps", "hea", "rung3", "rung4", "rung5")
INFEASIBLE_PENALTY = 1.0
GRID_KNOB_LIMIT = 4

_PHI_BOUNDS = (0.0, float(2 * np.pi))
_THETA_BOUNDS = (-float(np.pi), float(np.pi))
_THETA_AMP_BOUNDS = (0.0, float(np.pi / 2))
_PSI_AUX_BOUNDS = (0.0, float(2 * np.pi))

_PLOT_INSTALL_HINT = (
    "matplotlib is required for CancellationResult.plot(); "
    "install with `pip install polygram[plot]`."
)
_SCIPY_INSTALL_HINT = (
    "scipy is required for method='scipy'; "
    "install with `pip install polygram[opt]`."
)


@dataclass(eq=False)
class CancellationResult:
    """Output of a `Cancellation.run()`.

    Fields:

    - `optimized_knobs: dict[str, float]` — keyed by knob path
      (`<feature>.phi` or `<feature>.theta[r,d,q]`); value at the
      optimum
    - `before_gram, after_gram: np.ndarray (N, N) complex`
    - `before_overlap, after_overlap: float`
    - `tolerance_met: bool`
    - `method: str` — `"grid"` or `"scipy"`
    - `trajectory: np.ndarray (M, len(knobs) + 1)` — every evaluation
      with one column per knob (in declaration order) plus the target
      overlap. For grid this is a row-major flattening of the per-axis
      grid.
    - `feasible_mask: np.ndarray (M,) bool`
    - `feasible_count: int`
    - `dictionary_at_optimum: Dictionary`
    - `target_pair: tuple[str, str]`
    - `knobs: list[str]` — declared knob paths in trajectory column
      order
    - `structural_floor: float` — analytic floor when defined per the
      `structural_floor()` contract; `float("nan")` otherwise. For
      `encoding="rung3"` results this carries the *MPS phase-only*
      floor `M − |V|` of the same (α, β, γ) — the baseline the rung-3
      optimizer is *trying to break*, NOT a bound the rung-3
      optimizer is constrained by.
    - `cancellation_efficiency: float | None` —
      `(before − after) / (before − floor)`, clamped to `[0, 1]`.
      `None` when (a) the floor is `NaN` (undefined for the
      configuration) or (b) `before − floor < 1e-9` (no gap).
    - `theta_amp_optimum: float` — final feature-B `θ_amp` value at
      the rung-3 optimum (feature A is anchored at the default
      `π/4`). `float("nan")` for `encoding ∈ {"mps", "hea"}`.
    - `psi_aux_optimum: float` — final feature-B `ψ_aux` value at
      the rung-3 optimum (feature A is anchored at the default
      `0`). `float("nan")` for `encoding ∈ {"mps", "hea"}`.
    """

    optimized_knobs: dict[str, float]
    before_gram: np.ndarray
    after_gram: np.ndarray
    before_overlap: float
    after_overlap: float
    tolerance_met: bool
    method: str
    trajectory: np.ndarray
    feasible_count: int
    dictionary_at_optimum: Dictionary
    target_pair: tuple[str, str]
    knobs: list[str] = field(default_factory=list)
    feasible_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )
    structural_floor: float = float("nan")
    cancellation_efficiency: float | None = None
    theta_amp_optimum: float = float("nan")
    psi_aux_optimum: float = float("nan")

    def plot(
        self, path: str | os.PathLike, kind: str | None = None
    ) -> Path:
        """Render a matplotlib figure.

        ``kind=None`` dispatches on the result's ``method`` (the
        existing per-method default). Recognized kinds:

        - ``"grid"`` — heatmap of target-pair overlap on a 2D grid
          with the infeasible region masked and the optimum starred.
          Defined only when ``len(knobs) == 2``;
          ``NotImplementedError`` otherwise.
        - ``"scipy"`` — line plot of objective vs evaluation count.
        - ``"before_after"`` — three-panel figure: before Gram, after
          Gram (shared colorbar), bar chart with `before/after` and
          (when defined) the structural floor. The Gram panels mark
          the `target_pair` cell.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_PLOT_INSTALL_HINT) from exc

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        chosen = kind or self.method
        if chosen not in SUPPORTED_PLOT_KINDS:
            raise NotImplementedError(
                f"plot() does not support kind={chosen!r}; "
                f"supported: {SUPPORTED_PLOT_KINDS}"
            )

        if chosen == "grid":
            return _plot_grid(self, plt, p)
        if chosen == "scipy":
            return _plot_scipy(self, plt, p)
        return _plot_before_after(self, plt, p)

    def materialize(self, output_dir: str | os.PathLike) -> dict[str, Path]:
        """Write `<name>.q.orca.md` (Dictionary at optimum knobs),
        `<name>_summary.md`, and `<name>_trajectory.csv`. Returns
        `dict[str, Path]`."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        d = self.dictionary_at_optimum
        artifacts: dict[str, Path] = {}

        artifacts["machine"] = write_qorca(d, out / f"{d.name}.q.orca.md")

        summary_path = out / f"{d.name}_summary.md"
        summary_path.write_text(_render_summary(self))
        artifacts["summary"] = summary_path

        traj_path = out / f"{d.name}_trajectory.csv"
        traj_path.write_text(_render_trajectory_csv(self))
        artifacts["trajectory"] = traj_path

        return artifacts


@dataclass
class Cancellation:
    """Knob-search primitive: drive `target_pair` overlap toward
    `tolerance`, optionally constrained to preserve hierarchical-tier
    ordering.

    Pass ``config=CancellationConfig(...)`` (see :mod:`polygram.config`)
    to supply a typed tuning bundle for ``tolerance``, ``preserve_tiers``,
    ``optimize``, ``grid_outer``, and ``min_amp_overlap``. Per-field
    kwargs win over ``config``, which wins over the dataclass defaults.
    Search-target fields (``dictionary``, ``target_pair``, ``knobs``,
    ``optimize_all``) remain explicit constructor inputs.

    The default search space is the two `<feature>.phi` knobs of the
    target pair (preserves the rung-1 2-φ behavior). For HEA
    dictionaries, callers can pass a richer `knobs` list mixing
    `<feature>.phi` and `<feature>.theta[r,d,q]` paths.

    The analytic structural floor `M − |V|` (from
    `|<A|B>|²(δ) = M + V·cos(δ)`) is exact only for the canonical
    rung-1 2-φ shape: `structural_floor()` raises
    `NotImplementedError` outside that shape, and `run()` reports
    `structural_floor=float("nan")` and
    `cancellation_efficiency=None` in that case.
    """

    dictionary: Dictionary
    target_pair: tuple[str, str]
    # Tuning fields default to ``None`` as a sentinel; ``__post_init__``
    # resolves them via the per-field-kwarg > config > dataclass-default
    # precedence rule documented in :mod:`polygram.config`. The resolved
    # values are always concrete (non-None) post-construction, matching
    # the pre-config-rewrite behaviour.
    tolerance: float | None = None
    preserve_tiers: bool | None = None
    optimize: dict | None = None
    optimize_all: bool = False
    knobs: list[str] | None = None
    encoding: str | None = None
    grid_outer: tuple[int, int] | None = None
    min_amp_overlap: float | None = None
    config: "CancellationConfig | None" = None

    def __post_init__(self) -> None:
        # ``config`` precedence resolution. Per-field kwargs (already set
        # to non-None on the instance) win; otherwise pull from ``config``
        # if supplied; otherwise fall through to ``CancellationConfig``'s
        # own defaults.
        from polygram.config import CancellationConfig

        cfg = self.config if self.config is not None else CancellationConfig()
        if self.tolerance is None:
            self.tolerance = cfg.tolerance
        if self.preserve_tiers is None:
            self.preserve_tiers = cfg.preserve_tiers
        if self.optimize is None:
            self.optimize = dict(cfg.optimize)
        if self.grid_outer is None:
            self.grid_outer = cfg.grid_outer
        if self.min_amp_overlap is None:
            self.min_amp_overlap = cfg.min_amp_overlap

        if self.optimize_all:
            raise NotImplementedError(
                "optimize_all=True is reserved for a future release; "
                "use the configurable `knobs` list to broaden the search"
            )
        a, b = self.target_pair
        feature_names = [f.name for f in self.dictionary.features]
        for n in (a, b):
            if n not in feature_names:
                raise ValueError(
                    f"target_pair feature {n!r} not declared in dictionary"
                )

        if self.encoding is None:
            self.encoding = _infer_encoding_string(self.dictionary.encoding)
        if self.encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(
                f"unknown encoding {self.encoding!r}; "
                f"supported: {SUPPORTED_ENCODINGS}"
            )
        if self.encoding == "rung3" and not isinstance(
            self.dictionary.encoding, Rung3
        ):
            raise ValueError(
                f"Cancellation(encoding='rung3') requires a Rung3 "
                f"dictionary; got encoding={self.dictionary.encoding!r}"
            )
        if self.encoding == "rung4" and not isinstance(
            self.dictionary.encoding, Rung4
        ):
            raise ValueError(
                f"Cancellation(encoding='rung4') requires a Rung4 "
                f"dictionary; got encoding={self.dictionary.encoding!r}"
            )
        if self.encoding == "rung5" and not isinstance(
            self.dictionary.encoding, Rung5
        ):
            raise ValueError(
                f"Cancellation(encoding='rung5') requires a Rung5 "
                f"dictionary; got encoding={self.dictionary.encoding!r}"
            )

        if self.encoding == "rung3":
            default_knobs = [
                f"{a}.phi", f"{b}.phi",
                f"{b}.theta_amp", f"{b}.psi_aux",
            ]
            if self.knobs is None:
                self.knobs = default_knobs
            else:
                self.knobs = list(self.knobs)
                if self.knobs != default_knobs:
                    raise ValueError(
                        f"Cancellation(encoding='rung3') requires the "
                        f"canonical 4-knob list {default_knobs!r}; got "
                        f"{self.knobs!r}. Custom rung3 knob lists are not "
                        f"supported in v0."
                    )
        elif self.encoding == "rung4":
            default_knobs = [
                f"{a}.phi", f"{b}.phi",
                f"{b}.theta_amp", f"{b}.psi_aux",
                f"{b}.theta_amp_b", f"{b}.psi_amp_b",
            ]
            if self.knobs is None:
                self.knobs = default_knobs
            else:
                self.knobs = list(self.knobs)
                if self.knobs != default_knobs:
                    raise ValueError(
                        f"Cancellation(encoding='rung4') requires the "
                        f"canonical 6-knob list {default_knobs!r}; got "
                        f"{self.knobs!r}. Custom rung4 knob lists are not "
                        f"supported in v0."
                    )
        elif self.encoding == "rung5":
            k = self.dictionary.encoding.n_amp_qubits
            default_knobs = [f"{a}.phi", f"{b}.phi"]
            for i in range(k):
                default_knobs.append(f"{b}.amp_knobs[{i}].theta")
                default_knobs.append(f"{b}.amp_knobs[{i}].psi")
            if self.knobs is None:
                self.knobs = default_knobs
            else:
                self.knobs = list(self.knobs)
                if self.knobs != default_knobs:
                    raise ValueError(
                        f"Cancellation(encoding='rung5') requires the "
                        f"canonical {2 + 2 * k}-knob list "
                        f"{default_knobs!r}; got {self.knobs!r}. "
                        f"Custom rung5 knob lists are not supported in v0."
                    )
        else:
            if self.knobs is None:
                self.knobs = [f"{a}.phi", f"{b}.phi"]
            else:
                self.knobs = list(self.knobs)

        for path in self.knobs:
            self._validate_knob(path)

        if self.encoding not in ("rung3", "rung4", "rung5"):
            method = self.optimize.get("method", "grid")
            if method not in SUPPORTED_METHODS:
                raise ValueError(
                    f"unknown method {method!r}; supported: {SUPPORTED_METHODS}"
                )
            if method == "grid" and len(self.knobs) > GRID_KNOB_LIMIT:
                raise ValueError(
                    f"grid backend supports at most {GRID_KNOB_LIMIT} knobs "
                    f"(got {len(self.knobs)}); use method='scipy' for richer "
                    "search spaces"
                )

        if not (
            len(self.grid_outer) == 2
            and self.grid_outer[0] >= 1
            and self.grid_outer[1] >= 1
        ):
            raise ValueError(
                f"grid_outer must be a (M, N) pair with M >= 1 and N >= 1; "
                f"got {self.grid_outer!r}"
            )

        if not (0.0 <= self.min_amp_overlap <= 1.0):
            raise ValueError(
                f"min_amp_overlap must be in [0, 1]; got {self.min_amp_overlap!r}"
            )
        if self.min_amp_overlap > 0.0 and self.encoding not in (
            "rung3", "rung4", "rung5"
        ):
            raise ValueError(
                f"min_amp_overlap > 0 is only meaningful for "
                f"encoding in {{'rung3', 'rung4', 'rung5'}}; got "
                f"encoding={self.encoding!r}"
            )

    def _validate_knob(self, path: str) -> None:
        name, kind, slot = _parse_knob_path(path)
        feature_names = [f.name for f in self.dictionary.features]
        cluster_names = list(self.dictionary.hierarchy.keys())
        if name not in feature_names and name not in cluster_names:
            raise ValueError(
                f"knob path {path!r}: identifier {name!r} not declared "
                f"(features={feature_names}, clusters={cluster_names})"
            )
        if kind == "theta":
            if not isinstance(self.dictionary.encoding, HEA_Rung2):
                raise ValueError(
                    f"knob path {path!r}: .theta[...] paths are HEA-only; "
                    f"this Dictionary uses encoding={self.dictionary.encoding!r}"
                )
            shape = self.dictionary.encoding.theta_shape
            r, d_, q = slot
            if not (
                0 <= r < shape[0] and 0 <= d_ < shape[1] and 0 <= q < shape[2]
            ):
                raise ValueError(
                    f"knob path {path!r}: slot {slot} is outside "
                    f"theta_shape={shape}"
                )
        if kind in ("theta_amp", "psi_aux"):
            if not isinstance(self.dictionary.encoding, (Rung3, Rung4)):
                raise ValueError(
                    f"knob path {path!r}: .{kind} paths are Rung3/Rung4-only; "
                    f"this Dictionary uses encoding={self.dictionary.encoding!r}"
                )
        if kind in ("theta_amp_b", "psi_amp_b"):
            if not isinstance(self.dictionary.encoding, Rung4):
                raise ValueError(
                    f"knob path {path!r}: .{kind} paths are Rung4-only; "
                    f"this Dictionary uses encoding={self.dictionary.encoding!r}"
                )
        if kind in ("amp_knobs_theta", "amp_knobs_psi"):
            if not isinstance(self.dictionary.encoding, Rung5):
                raise ValueError(
                    f"knob path {path!r}: .amp_knobs[...] paths are "
                    f"Rung5-only; this Dictionary uses "
                    f"encoding={self.dictionary.encoding!r}"
                )
            (amp_idx,) = slot
            k = self.dictionary.encoding.n_amp_qubits
            if not (0 <= amp_idx < k):
                raise ValueError(
                    f"knob path {path!r}: amp index {amp_idx} is outside "
                    f"[0, {k})"
                )

    def _knob_bounds(self, path: str) -> tuple[float, float]:
        _, kind, _ = _parse_knob_path(path)
        if kind == "phi":
            return _PHI_BOUNDS
        if kind in ("theta_amp", "theta_amp_b", "amp_knobs_theta"):
            return _THETA_AMP_BOUNDS
        if kind in ("psi_aux", "psi_amp_b", "amp_knobs_psi"):
            return _PSI_AUX_BOUNDS
        return _THETA_BOUNDS

    def _is_canonical_2phi(self) -> bool:
        a, b = self.target_pair
        return list(self.knobs) == [f"{a}.phi", f"{b}.phi"]

    def structural_floor(self) -> float:
        """Analytic floor `M − |V|` reachable by varying `(φ_A, φ_B)`
        on a rung-1 `MPSRung1` dictionary.

        Defined exactly when:

        1. `dictionary.encoding` is `MPSRung1` and
           `self.knobs == [f"{target_pair[0]}.phi",
           f"{target_pair[1]}.phi"]`, OR
        2. `dictionary.encoding` is `Rung3` (the rung-3 floor is the
           MPS-phase-only floor `M − |V|` of the same (α, β, γ); the
           rung-3 cancellation reports this as the *baseline being
           broken*, not as a bound).

        Outside that shape — every multi-knob configuration on
        `MPSRung1`, every non-canonical knob list, and every
        HEA-encoded dictionary — raises `NotImplementedError`. A
        defensible HEA bound (e.g. a Lipschitz upper bound on
        `|∂overlap/∂θ|`) is deferred to a follow-up research-track
        proposal.
        """
        encoding = self.dictionary.encoding
        if isinstance(encoding, (Rung3, Rung4, Rung5)):
            return _mps_equivalent_floor(self.dictionary, self.target_pair)
        if not isinstance(encoding, MPSRung1):
            raise NotImplementedError(
                f"structural_floor() is defined only for MPSRung1 with the "
                f"canonical 2-φ knob list; got encoding={encoding!r}, "
                f"knobs={self.knobs!r}. The analytic M ± |V| bound does "
                "not generalize to multi-knob HEA — a defensible bound is "
                "deferred to a future research-track proposal."
            )
        if not self._is_canonical_2phi():
            raise NotImplementedError(
                f"structural_floor() requires the canonical 2-φ knob list "
                f"[{self.target_pair[0]!r}.phi, {self.target_pair[1]!r}.phi]; "
                f"got knobs={self.knobs!r}. A general-knob analytic bound "
                "is deferred to a future research-track proposal."
            )

        a_name, b_name = self.target_pair
        a_idx = self.dictionary.feature_index(a_name)
        b_idx = self.dictionary.feature_index(b_name)
        anchor = float(self.dictionary.feature(a_name).phi)
        d_zero = self._dictionary_at(anchor, anchor)
        d_pi = self._dictionary_at(anchor, anchor + float(np.pi))
        m_zero = float(np.abs(d_zero.gram()[a_idx, b_idx]) ** 2)
        m_pi = float(np.abs(d_pi.gram()[a_idx, b_idx]) ** 2)
        return float(min(m_zero, m_pi))

    def run(self) -> CancellationResult:
        if self.encoding == "rung3":
            return self._run_rung3_joint()
        if self.encoding == "rung4":
            return self._run_rung4_joint()
        if self.encoding == "rung5":
            return self._run_rung5_joint()
        method = self.optimize.get("method", "grid")
        max_steps = int(self.optimize.get("max_steps", 50))

        before_dict = self.dictionary
        before_gram = before_dict.gram()
        a_idx = before_dict.feature_index(self.target_pair[0])
        b_idx = before_dict.feature_index(self.target_pair[1])
        before_overlap = float(np.abs(before_gram[a_idx, b_idx]) ** 2)

        try:
            floor = self.structural_floor()
        except NotImplementedError:
            floor = float("nan")

        if method == "grid":
            best_values, after_overlap, traj, feasible_mask = self._run_grid(
                max_steps
            )
        else:
            best_values, after_overlap, traj, feasible_mask = self._run_scipy(
                max_steps
            )

        optimized_dict = self._dictionary_at(*best_values)
        after_gram = optimized_dict.gram()
        efficiency = _compute_efficiency(
            before_overlap, float(after_overlap), floor
        )

        return CancellationResult(
            optimized_knobs={
                path: float(val)
                for path, val in zip(self.knobs, best_values)
            },
            before_gram=before_gram,
            after_gram=after_gram,
            before_overlap=before_overlap,
            after_overlap=float(after_overlap),
            tolerance_met=bool(after_overlap < self.tolerance),
            method=method,
            trajectory=traj,
            feasible_count=int(feasible_mask.sum()),
            dictionary_at_optimum=optimized_dict,
            target_pair=self.target_pair,
            knobs=list(self.knobs),
            feasible_mask=feasible_mask,
            structural_floor=float(floor),
            cancellation_efficiency=efficiency,
        )

    def _run_grid(
        self, res: int
    ) -> tuple[tuple[float, ...], float, np.ndarray, np.ndarray]:
        axes = [
            np.linspace(lo, hi, res)
            for lo, hi in (self._knob_bounds(p) for p in self.knobs)
        ]
        n_evals = res ** len(self.knobs)
        n_cols = len(self.knobs) + 1
        traj = np.zeros((n_evals, n_cols), dtype=float)
        feasible_mask = np.zeros(n_evals, dtype=bool)

        a_idx = self.dictionary.feature_index(self.target_pair[0])
        b_idx = self.dictionary.feature_index(self.target_pair[1])

        # Row-major iteration over the per-axis Cartesian product.
        grids = np.meshgrid(*axes, indexing="ij")
        flat = [g.reshape(-1) for g in grids]
        for k in range(n_evals):
            values = tuple(float(arr[k]) for arr in flat)
            d = self._dictionary_at(*values)
            g = d.gram()
            ov = float(np.abs(g[a_idx, b_idx]) ** 2)
            for col, v in enumerate(values):
                traj[k, col] = v
            traj[k, n_cols - 1] = ov
            if self.preserve_tiers:
                feasible_mask[k] = hierarchical_ordering_preserved(
                    g, d, self.target_pair
                )
            else:
                feasible_mask[k] = True

        if feasible_mask.any():
            search = np.where(
                feasible_mask, traj[:, n_cols - 1], np.inf
            )
            best = int(np.argmin(search))
        else:
            best = int(np.argmin(traj[:, n_cols - 1]))

        best_values = tuple(float(traj[best, col]) for col in range(len(self.knobs)))
        return (best_values, float(traj[best, n_cols - 1]), traj, feasible_mask)

    def _run_scipy(
        self, maxiter: int
    ) -> tuple[tuple[float, ...], float, np.ndarray, np.ndarray]:
        try:
            from scipy.optimize import differential_evolution
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_SCIPY_INSTALL_HINT) from exc

        a_idx = self.dictionary.feature_index(self.target_pair[0])
        b_idx = self.dictionary.feature_index(self.target_pair[1])
        bounds = [self._knob_bounds(p) for p in self.knobs]
        history: list[tuple[tuple[float, ...], float, bool]] = []

        def objective(x: np.ndarray) -> float:
            values = tuple(float(v) for v in x)
            d = self._dictionary_at(*values)
            g = d.gram()
            ov = float(np.abs(g[a_idx, b_idx]) ** 2)
            feasible = (
                hierarchical_ordering_preserved(g, d, self.target_pair)
                if self.preserve_tiers
                else True
            )
            history.append((values, ov, feasible))
            return ov + (0.0 if feasible else INFEASIBLE_PENALTY)

        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=0,
            maxiter=maxiter,
            polish=False,
        )

        n_cols = len(self.knobs) + 1
        traj = np.array(
            [list(values) + [ov] for values, ov, _ in history], dtype=float
        )
        feasible_mask = np.array([ok for *_, ok in history], dtype=bool)

        if self.preserve_tiers and feasible_mask.any():
            feasible_overlaps = np.where(
                feasible_mask, traj[:, n_cols - 1], np.inf
            )
            best = int(np.argmin(feasible_overlaps))
            best_values = tuple(
                float(traj[best, col]) for col in range(len(self.knobs))
            )
            return (best_values, float(traj[best, n_cols - 1]), traj, feasible_mask)

        best_values = tuple(float(v) for v in result.x)
        d = self._dictionary_at(*best_values)
        ov = float(np.abs(d.gram()[a_idx, b_idx]) ** 2)
        return (best_values, ov, traj, feasible_mask)

    def _dictionary_at(self, *values: float) -> Dictionary:
        if len(values) != len(self.knobs):
            raise ValueError(
                f"_dictionary_at expected {len(self.knobs)} values "
                f"(one per knob), got {len(values)}"
            )
        d = self.dictionary
        for path, val in zip(self.knobs, values):
            d = d.with_knob(path, float(val))
        return replace(d, name=f"{self.dictionary.name}_at_optimum")

    def _run_rung3_joint(self) -> CancellationResult:
        """Joint (φ_a, φ_b, θ_amp, ψ_aux) optimizer for rung-3 dicts.

        Pipeline per the spec:

        1. Outer grid (default 5×5) over (theta_amp, psi_aux) on
           feature B; feature A's amp knobs stay anchored at the
           Rung3 default (π/4, 0).
        2. Inner 2-φ MPS-equivalent phase grid at every outer cell —
           reuses the canonical phase optimizer's logic via
           ``_phi_only_grid_search``.
        3. Scipy `Nelder-Mead` refine over (φ_a, φ_b, θ_b, ψ_b)
           starting from the best outer cell.

        The reported `structural_floor` is the MPS-phase-only floor
        `M − |V|` of the same (α, β, γ) — the baseline this optimizer
        is trying to break, not a bound it is constrained by.

        When ``min_amp_overlap > 0``, outer-grid cells and scipy-
        refine candidates whose amp factor falls below the threshold
        are marked infeasible. This prevents the optimizer from
        winning trivially by driving ``|⟨amp_a|amp_b⟩|² → 0`` (the
        degenerate amp-zeroing solution at θ_b=π/4, ψ_b=π against
        the anchored A defaults) and forces it to find an amp
        configuration that combines non-trivially with the MPS-side
        phase knobs.
        """
        a_name, b_name = self.target_pair
        a_idx = self.dictionary.feature_index(a_name)
        b_idx = self.dictionary.feature_index(b_name)

        a_feature = self.dictionary.features[a_idx]
        theta_a = float(a_feature.theta_amp)
        psi_a = float(a_feature.psi_aux)

        before_gram = self.dictionary.gram()
        before_overlap = float(np.abs(before_gram[a_idx, b_idx]) ** 2)

        floor = _mps_equivalent_floor(self.dictionary, self.target_pair)

        M_outer, N_outer = self.grid_outer
        theta_axis = np.linspace(0.0, float(np.pi / 2), M_outer)
        psi_axis = np.linspace(0.0, float(2 * np.pi), N_outer, endpoint=False)
        inner_res = int(self.optimize.get("max_steps", 50))

        amp_threshold = float(self.min_amp_overlap)
        amp_constrained = amp_threshold > 0.0

        outer_evals: list[tuple[float, float, float, float, float, bool]] = []
        best_cell: tuple[float, float, float, float] | None = None
        best_overlap_outer = float("inf")

        for theta_b in theta_axis:
            for psi_b in psi_axis:
                d_cell = self.dictionary.with_knob(
                    f"{b_name}.theta_amp", float(theta_b)
                )
                d_cell = d_cell.with_knob(
                    f"{b_name}.psi_aux", float(psi_b)
                )
                phi_a, phi_b, cell_overlap, cell_feasible = (
                    _phi_only_grid_search(
                        d_cell,
                        self.target_pair,
                        self.preserve_tiers,
                        inner_res,
                    )
                )
                if amp_constrained:
                    amp_sq = rung3_amp_overlap_squared(
                        theta_a, psi_a, float(theta_b), float(psi_b)
                    )
                    if amp_sq < amp_threshold:
                        cell_feasible = False
                outer_evals.append(
                    (phi_a, phi_b, float(theta_b), float(psi_b),
                     cell_overlap, cell_feasible)
                )
                if cell_feasible and cell_overlap < best_overlap_outer:
                    best_overlap_outer = cell_overlap
                    best_cell = (phi_a, phi_b, float(theta_b), float(psi_b))

        if best_cell is None:
            # No feasible cell — fall back to the unconstrained best.
            unconstrained = min(outer_evals, key=lambda r: r[4])
            best_cell = (
                unconstrained[0], unconstrained[1],
                unconstrained[2], unconstrained[3],
            )
            best_overlap_outer = unconstrained[4]

        scipy_history: list[tuple[float, float, float, float, float, bool]] = []
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_SCIPY_INSTALL_HINT) from exc

        def _evaluate(phi_a: float, phi_b: float,
                      theta_b: float, psi_b: float) -> tuple[float, bool]:
            d = self.dictionary.with_knob(f"{a_name}.phi", phi_a)
            d = d.with_knob(f"{b_name}.phi", phi_b)
            d = d.with_knob(f"{b_name}.theta_amp", theta_b)
            d = d.with_knob(f"{b_name}.psi_aux", psi_b)
            g = d.gram()
            ov = float(np.abs(g[a_idx, b_idx]) ** 2)
            feasible = (
                hierarchical_ordering_preserved(g, d, self.target_pair)
                if self.preserve_tiers else True
            )
            if amp_constrained:
                amp_sq = rung3_amp_overlap_squared(
                    theta_a, psi_a, theta_b, psi_b
                )
                if amp_sq < amp_threshold:
                    feasible = False
            return ov, feasible

        def objective(x: np.ndarray) -> float:
            phi_a, phi_b, theta_b, psi_b = (float(v) for v in x)
            ov, feasible = _evaluate(phi_a, phi_b, theta_b, psi_b)
            scipy_history.append(
                (phi_a, phi_b, theta_b, psi_b, ov, feasible)
            )
            return ov + (0.0 if feasible else INFEASIBLE_PENALTY)

        x0 = np.array(best_cell, dtype=float)
        minimize(
            objective, x0=x0, method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200},
        )

        candidates: list[tuple[float, float, float, float, float, bool]] = []
        candidates.append(
            (best_cell[0], best_cell[1], best_cell[2], best_cell[3],
             best_overlap_outer,
             # Re-evaluate feasibility at outer best for the candidate set.
             _evaluate(*best_cell)[1])
        )
        if scipy_history:
            feasible_scipy = [r for r in scipy_history if r[5]]
            pool = feasible_scipy if feasible_scipy else scipy_history
            best_scipy = min(pool, key=lambda r: r[4])
            candidates.append(best_scipy)

        feasible_candidates = [c for c in candidates if c[5]]
        chosen_pool = feasible_candidates if feasible_candidates else candidates
        chosen = min(chosen_pool, key=lambda r: r[4])
        final_phi_a, final_phi_b, final_theta_b, final_psi_b, \
            final_overlap, _ = chosen

        optimized_dict = self._dictionary_at(
            final_phi_a, final_phi_b, final_theta_b, final_psi_b
        )
        after_gram = optimized_dict.gram()
        # Take the empirical post-overlap from the materialized dict so
        # the result's overlap is exactly consistent with the dict.
        after_overlap = float(np.abs(after_gram[a_idx, b_idx]) ** 2)
        efficiency = _compute_efficiency(before_overlap, after_overlap, floor)

        all_evals = list(outer_evals) + list(scipy_history)
        traj = np.array(
            [[r[0], r[1], r[2], r[3], r[4]] for r in all_evals],
            dtype=float,
        )
        feasible_mask = np.array([r[5] for r in all_evals], dtype=bool)

        return CancellationResult(
            optimized_knobs={
                f"{a_name}.phi": float(final_phi_a),
                f"{b_name}.phi": float(final_phi_b),
                f"{b_name}.theta_amp": float(final_theta_b),
                f"{b_name}.psi_aux": float(final_psi_b),
            },
            before_gram=before_gram,
            after_gram=after_gram,
            before_overlap=before_overlap,
            after_overlap=after_overlap,
            tolerance_met=bool(after_overlap < self.tolerance),
            method="rung3_joint",
            trajectory=traj,
            feasible_count=int(feasible_mask.sum()),
            dictionary_at_optimum=optimized_dict,
            target_pair=self.target_pair,
            knobs=list(self.knobs),
            feasible_mask=feasible_mask,
            structural_floor=float(floor),
            cancellation_efficiency=efficiency,
            theta_amp_optimum=float(final_theta_b),
            psi_aux_optimum=float(final_psi_b),
        )

    def _run_rung4_joint(self) -> CancellationResult:
        """Joint (φ_a, φ_b, θ_amp_a, ψ_amp_a, θ_amp_b, ψ_amp_b)
        optimizer for Rung4 dictionaries.

        Mirrors `_run_rung3_joint`'s three-stage pipeline (outer
        grid + inner 2-φ + scipy Nelder-Mead refine) with an extra
        two dimensions on the outer grid for the q4 amp knobs.
        With ``grid_outer=(M, N)``, the outer iteration explores
        ``M * N * M * N`` cells across (θ_3, ψ_3, θ_4, ψ_4). At
        the default ``grid_outer=(5, 5)`` that's 625 cells; callers
        may want to drop to e.g. ``grid_outer=(3, 3)`` (81 cells)
        for wall-clock parity with Rung3.

        Feature A's four amp knobs stay anchored at their current
        values (Rung4 defaults are 0 for all four — the |0⟩⊗|0⟩
        identity state). Feature B's four amp knobs are optimised.

        `min_amp_overlap` constraint applies to
        ``rung4_amp_overlap_squared`` (the product of the two
        single-qubit overlaps), preventing the trivial
        amp-zeroing degenerate solution.

        Returned `structural_floor` is the MPS-phase-only floor of
        (α, β, γ) — the baseline this optimizer is trying to break,
        not a bound it is constrained by.
        """
        a_name, b_name = self.target_pair
        a_idx = self.dictionary.feature_index(a_name)
        b_idx = self.dictionary.feature_index(b_name)

        a_feature = self.dictionary.features[a_idx]
        theta_a3 = float(a_feature.theta_amp)
        psi_a3 = float(a_feature.psi_aux)
        theta_a4 = float(a_feature.theta_amp_b)
        psi_a4 = float(a_feature.psi_amp_b)

        before_gram = self.dictionary.gram()
        before_overlap = float(np.abs(before_gram[a_idx, b_idx]) ** 2)

        floor = _mps_equivalent_floor(self.dictionary, self.target_pair)

        M_outer, N_outer = self.grid_outer
        theta_axis = np.linspace(0.0, float(np.pi / 2), M_outer)
        psi_axis = np.linspace(0.0, float(2 * np.pi), N_outer, endpoint=False)
        inner_res = int(self.optimize.get("max_steps", 50))

        amp_threshold = float(self.min_amp_overlap)
        amp_constrained = amp_threshold > 0.0

        # Outer evals: (phi_a, phi_b, theta_b3, psi_b3, theta_b4, psi_b4,
        # cell_overlap, cell_feasible).
        outer_evals: list[tuple[float, float, float, float, float, float, float, bool]] = []
        best_cell: tuple[float, float, float, float, float, float] | None = None
        best_overlap_outer = float("inf")

        for theta_b3 in theta_axis:
            for psi_b3 in psi_axis:
                for theta_b4 in theta_axis:
                    for psi_b4 in psi_axis:
                        d_cell = self.dictionary.with_knob(
                            f"{b_name}.theta_amp", float(theta_b3)
                        )
                        d_cell = d_cell.with_knob(
                            f"{b_name}.psi_aux", float(psi_b3)
                        )
                        d_cell = d_cell.with_knob(
                            f"{b_name}.theta_amp_b", float(theta_b4)
                        )
                        d_cell = d_cell.with_knob(
                            f"{b_name}.psi_amp_b", float(psi_b4)
                        )
                        phi_a, phi_b, cell_overlap, cell_feasible = (
                            _phi_only_grid_search(
                                d_cell,
                                self.target_pair,
                                self.preserve_tiers,
                                inner_res,
                            )
                        )
                        if amp_constrained:
                            amp_sq = rung4_amp_overlap_squared(
                                theta_a3, psi_a3, theta_a4, psi_a4,
                                float(theta_b3), float(psi_b3),
                                float(theta_b4), float(psi_b4),
                            )
                            if amp_sq < amp_threshold:
                                cell_feasible = False
                        outer_evals.append(
                            (phi_a, phi_b,
                             float(theta_b3), float(psi_b3),
                             float(theta_b4), float(psi_b4),
                             cell_overlap, cell_feasible)
                        )
                        if cell_feasible and cell_overlap < best_overlap_outer:
                            best_overlap_outer = cell_overlap
                            best_cell = (phi_a, phi_b,
                                         float(theta_b3), float(psi_b3),
                                         float(theta_b4), float(psi_b4))

        if best_cell is None:
            # No feasible cell — fall back to unconstrained best.
            unconstrained = min(outer_evals, key=lambda r: r[6])
            best_cell = (
                unconstrained[0], unconstrained[1],
                unconstrained[2], unconstrained[3],
                unconstrained[4], unconstrained[5],
            )
            best_overlap_outer = unconstrained[6]

        scipy_history: list[tuple[float, float, float, float, float, float, float, bool]] = []
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_SCIPY_INSTALL_HINT) from exc

        def _evaluate(phi_a: float, phi_b: float,
                      theta_b3: float, psi_b3: float,
                      theta_b4: float, psi_b4: float) -> tuple[float, bool]:
            d = self.dictionary.with_knob(f"{a_name}.phi", phi_a)
            d = d.with_knob(f"{b_name}.phi", phi_b)
            d = d.with_knob(f"{b_name}.theta_amp", theta_b3)
            d = d.with_knob(f"{b_name}.psi_aux", psi_b3)
            d = d.with_knob(f"{b_name}.theta_amp_b", theta_b4)
            d = d.with_knob(f"{b_name}.psi_amp_b", psi_b4)
            g = d.gram()
            ov = float(np.abs(g[a_idx, b_idx]) ** 2)
            feasible = (
                hierarchical_ordering_preserved(g, d, self.target_pair)
                if self.preserve_tiers else True
            )
            if amp_constrained:
                amp_sq = rung4_amp_overlap_squared(
                    theta_a3, psi_a3, theta_a4, psi_a4,
                    theta_b3, psi_b3, theta_b4, psi_b4,
                )
                if amp_sq < amp_threshold:
                    feasible = False
            return ov, feasible

        def objective(x: np.ndarray) -> float:
            phi_a, phi_b, theta_b3, psi_b3, theta_b4, psi_b4 = (
                float(v) for v in x
            )
            ov, feasible = _evaluate(
                phi_a, phi_b, theta_b3, psi_b3, theta_b4, psi_b4
            )
            scipy_history.append(
                (phi_a, phi_b, theta_b3, psi_b3, theta_b4, psi_b4,
                 ov, feasible)
            )
            return ov + (0.0 if feasible else INFEASIBLE_PENALTY)

        x0 = np.array(best_cell, dtype=float)
        minimize(
            objective, x0=x0, method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400},
        )

        candidates: list[tuple[float, ...]] = []
        candidates.append((
            best_cell[0], best_cell[1],
            best_cell[2], best_cell[3], best_cell[4], best_cell[5],
            best_overlap_outer,
            _evaluate(*best_cell)[1],
        ))
        if scipy_history:
            feasible_scipy = [r for r in scipy_history if r[7]]
            pool = feasible_scipy if feasible_scipy else scipy_history
            best_scipy = min(pool, key=lambda r: r[6])
            candidates.append(best_scipy)

        feasible_candidates = [c for c in candidates if c[7]]
        chosen_pool = feasible_candidates if feasible_candidates else candidates
        chosen = min(chosen_pool, key=lambda r: r[6])
        (final_phi_a, final_phi_b,
         final_theta_b3, final_psi_b3,
         final_theta_b4, final_psi_b4,
         final_overlap, _) = chosen

        optimized_dict = self._dictionary_at(
            final_phi_a, final_phi_b,
            final_theta_b3, final_psi_b3,
            final_theta_b4, final_psi_b4,
        )
        after_gram = optimized_dict.gram()
        after_overlap = float(np.abs(after_gram[a_idx, b_idx]) ** 2)
        efficiency = _compute_efficiency(before_overlap, after_overlap, floor)

        all_evals = list(outer_evals) + list(scipy_history)
        traj = np.array(
            [[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in all_evals],
            dtype=float,
        )
        feasible_mask = np.array([r[7] for r in all_evals], dtype=bool)

        return CancellationResult(
            optimized_knobs={
                f"{a_name}.phi": float(final_phi_a),
                f"{b_name}.phi": float(final_phi_b),
                f"{b_name}.theta_amp": float(final_theta_b3),
                f"{b_name}.psi_aux": float(final_psi_b3),
                f"{b_name}.theta_amp_b": float(final_theta_b4),
                f"{b_name}.psi_amp_b": float(final_psi_b4),
            },
            before_gram=before_gram,
            after_gram=after_gram,
            before_overlap=before_overlap,
            after_overlap=after_overlap,
            tolerance_met=bool(after_overlap < self.tolerance),
            method="rung4_joint",
            trajectory=traj,
            feasible_count=int(feasible_mask.sum()),
            dictionary_at_optimum=optimized_dict,
            target_pair=self.target_pair,
            knobs=list(self.knobs),
            feasible_mask=feasible_mask,
            structural_floor=float(floor),
            cancellation_efficiency=efficiency,
            theta_amp_optimum=float(final_theta_b3),
            psi_aux_optimum=float(final_psi_b3),
        )

    def _run_rung5_joint(self) -> CancellationResult:
        """Joint (φ_a, φ_b, [θ_i, ψ_i]_{i=0..k-1}) optimizer for Rung5
        dictionaries.

        Rung4's outer-grid approach explodes as ``(M*N)^k`` cells with
        k = ``n_amp_qubits``; at k ≥ 3 the wall-clock becomes
        impractical. Rung5 instead runs scipy
        ``differential_evolution`` over the full ``(2 + 2k)``-dim
        bounded space. ``differential_evolution`` is dimension-
        agnostic and handles bounded boxes natively, which matches
        sae-forge's pareto-sweep need to push k modestly without
        bespoke per-k tuning.

        Feature A's amp knobs stay anchored at their current values
        (Rung5 default is all-zeros under
        ``with_default_amp_knobs(encoding)`` — the |0⟩^⊗k state).
        Feature B's full ``(φ, θ_0, ψ_0, …, θ_{k-1}, ψ_{k-1})`` slate
        is optimised, plus feature A's φ.

        ``min_amp_overlap`` applies to
        ``rung5_amp_overlap_squared(a.amp_knobs, b.amp_knobs)`` — the
        full k-fold product — preventing the trivial amp-zeroing
        degenerate solution. Infeasible candidates receive
        ``INFEASIBLE_PENALTY``.

        Returned ``structural_floor`` is the MPS-phase-only floor of
        (α, β, γ) — independent of k.
        """
        try:
            from scipy.optimize import differential_evolution
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_SCIPY_INSTALL_HINT) from exc

        a_name, b_name = self.target_pair
        a_idx = self.dictionary.feature_index(a_name)
        b_idx = self.dictionary.feature_index(b_name)

        k = self.dictionary.encoding.n_amp_qubits
        a_feature = self.dictionary.features[a_idx]
        # Anchor A's amp knobs at their current values; default-pad
        # when the feature still holds the empty-tuple default.
        a_amp = a_feature.with_default_amp_knobs(
            self.dictionary.encoding
        ).amp_knobs

        before_gram = self.dictionary.gram()
        before_overlap = float(np.abs(before_gram[a_idx, b_idx]) ** 2)

        floor = _mps_equivalent_floor(self.dictionary, self.target_pair)

        amp_threshold = float(self.min_amp_overlap)
        amp_constrained = amp_threshold > 0.0

        bounds = [self._knob_bounds(p) for p in self.knobs]
        max_steps = int(self.optimize.get("max_steps", 50))

        history: list[tuple[tuple[float, ...], float, bool]] = []

        def _b_amp_from_x(x: np.ndarray) -> tuple[tuple[float, float], ...]:
            return tuple(
                (float(x[2 + 2 * i]), float(x[2 + 2 * i + 1]))
                for i in range(k)
            )

        def objective(x: np.ndarray) -> float:
            values = tuple(float(v) for v in x)
            d = self._dictionary_at(*values)
            g = d.gram()
            ov = float(np.abs(g[a_idx, b_idx]) ** 2)
            feasible = (
                hierarchical_ordering_preserved(g, d, self.target_pair)
                if self.preserve_tiers else True
            )
            if amp_constrained:
                amp_sq = rung5_amp_overlap_squared(a_amp, _b_amp_from_x(x))
                if amp_sq < amp_threshold:
                    feasible = False
            history.append((values, ov, feasible))
            return ov + (0.0 if feasible else INFEASIBLE_PENALTY)

        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=0,
            maxiter=max_steps,
            polish=False,
        )

        n_cols = len(self.knobs) + 1
        traj = np.array(
            [list(values) + [ov] for values, ov, _ in history], dtype=float
        )
        feasible_mask = np.array([ok for *_, ok in history], dtype=bool)

        if self.preserve_tiers and feasible_mask.any():
            feasible_overlaps = np.where(
                feasible_mask, traj[:, n_cols - 1], np.inf
            )
            best = int(np.argmin(feasible_overlaps))
            best_values = tuple(
                float(traj[best, col]) for col in range(len(self.knobs))
            )
        elif amp_constrained and feasible_mask.any():
            feasible_overlaps = np.where(
                feasible_mask, traj[:, n_cols - 1], np.inf
            )
            best = int(np.argmin(feasible_overlaps))
            best_values = tuple(
                float(traj[best, col]) for col in range(len(self.knobs))
            )
        else:
            best_values = tuple(float(v) for v in result.x)

        optimized_dict = self._dictionary_at(*best_values)
        after_gram = optimized_dict.gram()
        after_overlap = float(np.abs(after_gram[a_idx, b_idx]) ** 2)
        efficiency = _compute_efficiency(before_overlap, after_overlap, floor)

        return CancellationResult(
            optimized_knobs={
                path: float(val)
                for path, val in zip(self.knobs, best_values)
            },
            before_gram=before_gram,
            after_gram=after_gram,
            before_overlap=before_overlap,
            after_overlap=after_overlap,
            tolerance_met=bool(after_overlap < self.tolerance),
            method="rung5_joint",
            trajectory=traj,
            feasible_count=int(feasible_mask.sum()),
            dictionary_at_optimum=optimized_dict,
            target_pair=self.target_pair,
            knobs=list(self.knobs),
            feasible_mask=feasible_mask,
            structural_floor=float(floor),
            cancellation_efficiency=efficiency,
            # theta_amp_optimum / psi_aux_optimum carry the first
            # amp-qubit's pair for parity with Rung3/Rung4 result
            # reporting; downstream consumers querying these for
            # Rung5 should inspect optimized_knobs directly.
            theta_amp_optimum=float(best_values[2]),
            psi_aux_optimum=float(best_values[3]),
        )


def _is_hea(dictionary: Dictionary) -> bool:
    return isinstance(dictionary.encoding, HEA_Rung2)


def _infer_encoding_string(encoding: object) -> str:
    # Check rungs in descending order so the most specific class wins
    # if encodings ever subclass one another.
    if isinstance(encoding, Rung5):
        return "rung5"
    if isinstance(encoding, Rung4):
        return "rung4"
    if isinstance(encoding, Rung3):
        return "rung3"
    if isinstance(encoding, HEA_Rung2):
        return "hea"
    if isinstance(encoding, MPSRung1):
        return "mps"
    raise ValueError(
        f"unsupported dictionary.encoding type: {type(encoding).__name__}"
    )


def _phi_only_grid_search(
    dictionary: Dictionary,
    target_pair: tuple[str, str],
    preserve_tiers: bool,
    res: int,
) -> tuple[float, float, float, bool]:
    """Run the canonical 2-φ MPSRung1-equivalent phase grid on
    ``dictionary``, returning ``(phi_a, phi_b, overlap, feasible)``
    for the best (feasible if any, else infeasible) cell."""
    a_name, b_name = target_pair
    a_idx = dictionary.feature_index(a_name)
    b_idx = dictionary.feature_index(b_name)

    axis = np.linspace(_PHI_BOUNDS[0], _PHI_BOUNDS[1], res)
    grids = np.meshgrid(axis, axis, indexing="ij")
    flat_a = grids[0].reshape(-1)
    flat_b = grids[1].reshape(-1)
    n_evals = res * res

    best_feasible: tuple[float, float, float] | None = None
    best_unfeasible: tuple[float, float, float] | None = None

    for k in range(n_evals):
        phi_a = float(flat_a[k])
        phi_b = float(flat_b[k])
        d = dictionary.with_knob(f"{a_name}.phi", phi_a)
        d = d.with_knob(f"{b_name}.phi", phi_b)
        g = d.gram()
        ov = float(np.abs(g[a_idx, b_idx]) ** 2)
        feasible = (
            hierarchical_ordering_preserved(g, d, target_pair)
            if preserve_tiers else True
        )
        if feasible:
            if best_feasible is None or ov < best_feasible[2]:
                best_feasible = (phi_a, phi_b, ov)
        if best_unfeasible is None or ov < best_unfeasible[2]:
            best_unfeasible = (phi_a, phi_b, ov)

    if best_feasible is not None:
        phi_a, phi_b, ov = best_feasible
        return phi_a, phi_b, ov, True
    assert best_unfeasible is not None
    phi_a, phi_b, ov = best_unfeasible
    return phi_a, phi_b, ov, False


def _mps_equivalent_floor(
    dictionary: Dictionary, target_pair: tuple[str, str]
) -> float:
    """Return the MPS-phase-only floor `M − |V|` of the (α, β, γ)
    tuple carried by ``dictionary`` — the baseline a rung-3 cancel
    is trying to break.

    Constructs an MPSRung1-equivalent dictionary (same features, same
    hierarchy, encoding swapped) and calls its existing 2-φ
    structural_floor() helper.
    """
    mps_dict = replace(dictionary, encoding=MPSRung1())
    mps_canc = Cancellation(
        dictionary=mps_dict,
        target_pair=target_pair,
        preserve_tiers=False,
        encoding="mps",
    )
    return float(mps_canc.structural_floor())


def _compute_efficiency(
    before: float, after: float, floor: float
) -> float | None:
    if math.isnan(floor):
        return None
    gap = before - floor
    if gap < 1e-9:
        return None
    return float(np.clip((before - after) / gap, 0.0, 1.0))


def _interpret_efficiency(
    efficiency: float | None, floor: float
) -> str:
    if math.isnan(floor):
        return (
            "structural floor is encoding-bound; not yet defined for "
            "this configuration"
        )
    if efficiency is None:
        return "no cancellation gap available"
    if efficiency >= 0.99:
        return "phase search exhausted — encoding-bound"
    return "phase search underutilized"


def _plot_grid(result: CancellationResult, plt, p: Path) -> Path:
    if len(result.knobs) != 2:
        raise NotImplementedError(
            "kind='grid' is defined only when len(knobs) == 2 "
            f"(got knobs={result.knobs!r}); use kind='before_after' "
            "for higher-dimensional searches"
        )
    n = int(np.sqrt(result.trajectory.shape[0]))
    a_path, b_path = result.knobs
    a_lo, a_hi = float(result.trajectory[:, 0].min()), float(
        result.trajectory[:, 0].max()
    )
    b_lo, b_hi = float(result.trajectory[:, 1].min()), float(
        result.trajectory[:, 1].max()
    )
    overlaps = result.trajectory[:, 2].astype(float)
    display = np.where(result.feasible_mask, overlaps, np.nan).reshape(n, n)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        display.T,
        origin="lower",
        aspect="auto",
        extent=(a_lo, a_hi, b_lo, b_hi),
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="|<A|B>|²")
    ax.scatter(
        [result.optimized_knobs[a_path]],
        [result.optimized_knobs[b_path]],
        marker="*",
        s=180,
        c="red",
        edgecolors="white",
        label="optimum",
    )
    ax.set_xlabel(a_path)
    ax.set_ylabel(b_path)
    ax.set_title(
        f"Cancellation grid — {a_path} × {b_path}\n"
        f"after_overlap = {result.after_overlap:.4f}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_scipy(result: CancellationResult, plt, p: Path) -> Path:
    a_name, b_name = result.target_pair
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        np.arange(result.trajectory.shape[0]),
        result.trajectory[:, -1],
        marker=".",
        linewidth=1.0,
        color="tab:blue",
    )
    ax.set_xlabel("evaluation")
    ax.set_ylabel("|<A|B>|²")
    ax.set_title(
        f"Cancellation scipy — {a_name} × {b_name}\n"
        f"after_overlap = {result.after_overlap:.4f}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_before_after(result: CancellationResult, plt, p: Path) -> Path:
    a_name, b_name = result.target_pair
    a_idx = result.dictionary_at_optimum.feature_index(a_name)
    b_idx = result.dictionary_at_optimum.feature_index(b_name)
    before = np.abs(result.before_gram) ** 2
    after = np.abs(result.after_gram) ** 2
    vmax = float(max(before.max(), after.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, mat, title in zip(
        axes[:2], (before, after), ("before |<i|j>|²", "after |<i|j>|²")
    ):
        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(
            [b_idx, a_idx], [a_idx, b_idx],
            marker="s", s=80,
            facecolors="none", edgecolors="red", linewidths=1.5,
        )
    fig.colorbar(im, ax=axes[:2].tolist(), shrink=0.7)

    bar_ax = axes[2]
    labels = ["before", "after"]
    values = [result.before_overlap, result.after_overlap]
    if not math.isnan(result.structural_floor):
        labels.append("floor")
        values.append(float(result.structural_floor))
    bar_ax.bar(labels, values, color=["#888", "#1f77b4", "#2ca02c"][: len(labels)])
    bar_ax.set_ylabel("|<A|B>|²")
    bar_ax.set_title(f"{a_name} × {b_name}")
    bar_ax.grid(axis="y", alpha=0.3)

    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _render_summary(result: CancellationResult) -> str:
    a, b = result.target_pair
    eff = result.cancellation_efficiency
    eff_str = "n/a" if eff is None else f"{eff:.4f}"
    floor = result.structural_floor
    floor_str = (
        "undefined for this configuration"
        if math.isnan(floor)
        else f"{floor:.6f}"
    )
    lines = [
        f"# {result.dictionary_at_optimum.name}",
        "",
        f"- target pair: `{a}` × `{b}`",
        f"- method: `{result.method}`",
        f"- knobs: {', '.join(f'`{k}`' for k in result.knobs)}",
        f"- tolerance_met: {result.tolerance_met} "
        f"(after_overlap < tolerance)",
        f"- feasible evaluations: {result.feasible_count} of "
        f"{result.trajectory.shape[0]}",
        "",
        "## Optimum",
        "",
    ]
    for path in result.knobs:
        lines.append(f"- `{path}` = {result.optimized_knobs[path]:.6f}")
    # Surface the canonical 2-φ feature names too, for backwards-friendly
    # readability when the default knobs are in use.
    if result.knobs == [f"{a}.phi", f"{b}.phi"]:
        lines.extend([
            "",
            f"  - {a} = {result.optimized_knobs[f'{a}.phi']:.6f}",
            f"  - {b} = {result.optimized_knobs[f'{b}.phi']:.6f}",
        ])
    lines.extend([
        "",
        "## Overlap",
        "",
        f"- before: {result.before_overlap:.6f}",
        f"- after:  {result.after_overlap:.6f}",
        f"- delta:  {result.after_overlap - result.before_overlap:+.6f}",
        "",
        "## Structural floor",
        "",
        f"- structural_floor: {floor_str}",
        f"- cancellation_efficiency: {eff_str}",
        f"- interpretation: {_interpret_efficiency(eff, floor)}",
        "",
    ])
    if math.isnan(floor):
        caveat_paragraphs = _summary_caveat_paragraphs(result)
        lines.append("## Caveat")
        lines.append("")
        for para in caveat_paragraphs:
            lines.append(para)
            lines.append("")
    return "\n".join(lines)


def _summary_caveat_paragraphs(result: CancellationResult) -> list[str]:
    """Return the ``## Caveat`` body paragraphs for a NaN-floor result.

    Three knob-list shapes:

    - **pure cluster-shared** (every leading identifier is a cluster):
      one paragraph noting the within-cluster Gram entries are
      preserved exactly by unitarity.
    - **mixed** (both per-feature and cluster-shared paths): emit the
      multi-knob "best value found" warning *and* a note that mixed
      lists do NOT inherit the cluster-shared invariant.
    - **per-feature** (no cluster-shared paths): existing multi-knob
      warning, plus the θ-knob tier-separation addendum if any path is
      a ``.theta[...]`` slot.
    """
    cluster_names = set(result.dictionary_at_optimum.hierarchy.keys())
    leading = [k.split(".", 1)[0] for k in result.knobs]
    cluster_paths = [k for k, lead in zip(result.knobs, leading) if lead in cluster_names]
    feature_paths = [k for k, lead in zip(result.knobs, leading) if lead not in cluster_names]
    has_theta = any(".theta[" in k for k in result.knobs)
    is_mps = isinstance(result.dictionary_at_optimum.encoding, MPSRung1)
    pure_cluster_phi_only = not has_theta

    if is_mps and pure_cluster_phi_only:
        pure_cluster_note = (
            "**Note:** every knob is a cluster-shared `<cluster>.phi` path "
            "on `MPSRung1`. The final-Rz factorization makes the same outer "
            "rotation appear on every sibling branch, and the unitarity "
            "cancellation `<U_C a | U_C b> = <a|U_C†U_C|b> = <a|b>` "
            "preserves within-cluster Gram entries bit-for-bit (to numeric "
            "round-off) when sibling pre-mutation `phi` values agree."
        )
    else:
        pure_cluster_note = (
            "**Note:** every knob is a cluster-shared path. On HEA "
            "encodings (or any cluster-shared θ), this is a search-space "
            "dimensionality reduction — one axis per cluster instead of "
            "one per feature — which bounds optimizer leverage on each "
            "sibling but does NOT guarantee bit-for-bit Gram preservation. "
            "Within-cluster Gram entries MAY drift on diverse-sibling "
            "fixtures. Verify `concept_gram_tier_separation` on the "
            "materialized optimum to confirm tier ordering."
        )
    multi_knob_note = (
        f"**Note:** the reported `after` is the best value found by "
        f"`{result.method}`, not a guaranteed lower bound. Cross-term "
        "interactions among knobs can drive the true achievable "
        "overlap below this value"
    )
    if has_theta:
        multi_knob_note += (
            ", and θ knobs can break cluster invariants — verify "
            "`concept_gram_tier_separation` on the optimum"
        )
    multi_knob_note += "."
    mixed_note = (
        "**Note:** this knob list mixes per-feature and cluster-shared "
        "paths. The within-cluster Gram invariant that pure "
        "cluster-shared lists enjoy does NOT apply here — per-feature "
        "mutations on one branch break the matched unitarity."
    )

    if cluster_paths and not feature_paths:
        return [pure_cluster_note]
    if cluster_paths and feature_paths:
        return [multi_knob_note, mixed_note]
    return [multi_knob_note]


def _render_trajectory_csv(result: CancellationResult) -> str:
    header = ",".join(list(result.knobs) + ["overlap", "feasible"])
    out = [header]
    for row, feasible in zip(result.trajectory, result.feasible_mask):
        cells = [f"{float(v):.6f}" for v in row]
        cells.append("1" if feasible else "0")
        out.append(",".join(cells))
    return "\n".join(out) + "\n"
