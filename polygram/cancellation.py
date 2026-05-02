"""Cancellation primitive — find φ values that drive a target-pair
overlap below a tolerance, optionally constrained to preserve the
hierarchical-tier ordering.

Two backends ship: a deterministic 2D grid scan over `(φ_a, φ_b) ∈
[0, 2π]²` (no extra deps), and a `scipy.optimize.differential_evolution`
backend behind the `polygram[opt]` extra.

The search space in v0 is exactly the two φ values of the target pair.
`optimize_all=True` is reserved and raises `NotImplementedError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from polygram._assertions import hierarchical_ordering_preserved
from polygram.dictionary import Dictionary
from polygram.emit import write_qorca

SUPPORTED_METHODS = ("grid", "scipy")
INFEASIBLE_PENALTY = 1.0

_PLOT_INSTALL_HINT = (
    "matplotlib is required for CancellationResult.plot(); "
    "install with `pip install polygram[plot]`."
)
_SCIPY_INSTALL_HINT = (
    "scipy is required for method='scipy'; "
    "install with `pip install polygram[opt]`."
)


@dataclass
class CancellationResult:
    """Output of a `Cancellation.run()`.

    `Cancellation` only steers φ; the squared overlap as a function of
    δ = φ_A − φ_B factors as `|<A|B>|² = M + V·cos(δ)` and so is
    bounded below by `M − |V|`, the **structural floor**. Pure-phase
    search is a *constraint solver* against this floor, not a
    destructive-interference oracle. To break the floor you'd need
    to vary β/α/γ as well — outside the v0 search space.

    Fields:

    - `optimized_phis: dict[str, float]` — `{name_a: phi_a, name_b: phi_b}`
    - `before_gram, after_gram: np.ndarray (N, N) complex`
    - `before_overlap, after_overlap: float` — `|<A|B>|²` baseline /
      optimum
    - `tolerance_met: bool` — `after_overlap < tolerance`
    - `method: str` — `"grid"` or `"scipy"`
    - `trajectory: np.ndarray (M, 3)` — every evaluation
      `(phi_a, phi_b, overlap)` in evaluation order; for grid this is
      a row-major flattening of the (max_steps × max_steps) grid
    - `feasible_mask: np.ndarray (M,) bool` — per-evaluation
      feasibility flag (preserves hierarchical ordering); aligned with
      `trajectory`
    - `feasible_count: int`
    - `dictionary_at_optimum: Dictionary`
    - `target_pair: tuple[str, str]`
    - `structural_floor: float` — analytic minimum overlap reachable
      by varying only `(φ_A, φ_B)`; `M − |V|` for the cos-decomposition
      above. Encoding-bound; not affected by `preserve_tiers`.
    - `cancellation_efficiency: float | None` —
      `(before − after) / (before − floor)`, clamped to `[0, 1]`.
      `None` when `before − floor < 1e-9` (no cancellation gap to
      measure — already at the floor). 1.0 means phase search reached
      the floor; <1.0 means the constraint or optimizer left some gap.
    """

    optimized_phis: dict[str, float]
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
    feasible_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )
    structural_floor: float = 0.0
    cancellation_efficiency: float | None = None

    def plot(self, path: str | os.PathLike) -> Path:
        """Render a default matplotlib figure.

        - `method="grid"` → heatmap of target-pair overlap on the
          `(φ_a, φ_b)` grid with the infeasible region masked and the
          optimum starred.
        - `method="scipy"` → line plot of objective vs evaluation count.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_PLOT_INSTALL_HINT) from exc

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        a_name, b_name = self.target_pair

        if self.method == "grid":
            n = int(np.sqrt(self.trajectory.shape[0]))
            phis = np.linspace(0.0, 2 * np.pi, n)
            overlaps = self.trajectory[:, 2].astype(float)
            display = np.where(self.feasible_mask, overlaps, np.nan).reshape(n, n)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                display.T,
                origin="lower",
                aspect="auto",
                extent=(float(phis[0]), float(phis[-1]),
                        float(phis[0]), float(phis[-1])),
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, label="|<A|B>|²")
            ax.scatter(
                [self.optimized_phis[a_name]],
                [self.optimized_phis[b_name]],
                marker="*",
                s=180,
                c="red",
                edgecolors="white",
                label="optimum",
            )
            ax.set_xlabel(f"{a_name}.phi")
            ax.set_ylabel(f"{b_name}.phi")
            ax.set_title(
                f"Cancellation grid — {a_name} × {b_name}\n"
                f"after_overlap = {self.after_overlap:.4f}"
            )
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(p, dpi=120)
            plt.close(fig)
            return p

        if self.method == "scipy":
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                np.arange(self.trajectory.shape[0]),
                self.trajectory[:, 2],
                marker=".",
                linewidth=1.0,
                color="tab:blue",
            )
            ax.set_xlabel("evaluation")
            ax.set_ylabel("|<A|B>|²")
            ax.set_title(
                f"Cancellation scipy — {a_name} × {b_name}\n"
                f"after_overlap = {self.after_overlap:.4f}"
            )
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(p, dpi=120)
            plt.close(fig)
            return p

        raise NotImplementedError(f"plot() does not support method={self.method!r}")

    def materialize(self, output_dir: str | os.PathLike) -> dict[str, Path]:
        """Write `<name>.q.orca.md` (Dictionary at optimum φs),
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
    """φ-search primitive: drive `target_pair` overlap toward
    `tolerance`, optionally constrained to preserve hierarchical-tier
    ordering.

    The squared overlap `|<A|B>|²(δ)` factors as `M + V·cos(δ)` for
    `δ = φ_A − φ_B`, so pure-phase search is bounded below by the
    **structural floor** `M − |V|`. Use `structural_floor()` to query
    that limit before running, or read it off
    `CancellationResult.structural_floor` after; the result also
    reports `cancellation_efficiency` — the fraction of the
    cancellation gap the search closed. Driving overlap below the
    floor needs amplitude variation (β/α/γ), not just phase — outside
    the v0 search space.

    Fields:

    - `dictionary` — input `Dictionary`
    - `target_pair` — two declared feature names
    - `tolerance: float = 0.05`
    - `preserve_tiers: bool = True` — when True, candidates that violate
      `hierarchical_ordering_preserved` are infeasible
    - `optimize: dict = {"method": "grid", "max_steps": 50}` — backend
      and per-axis resolution / maxiter
    - `optimize_all: bool = False` — reserved; True raises
      `NotImplementedError` in v0
    """

    dictionary: Dictionary
    target_pair: tuple[str, str]
    tolerance: float = 0.05
    preserve_tiers: bool = True
    optimize: dict = field(
        default_factory=lambda: {"method": "grid", "max_steps": 50}
    )
    optimize_all: bool = False

    def __post_init__(self) -> None:
        if self.optimize_all:
            raise NotImplementedError(
                "optimize_all=True is reserved for a future release; v0 "
                "searches only the two target-pair φ values"
            )
        a, b = self.target_pair
        feature_names = [f.name for f in self.dictionary.features]
        for n in (a, b):
            if n not in feature_names:
                raise ValueError(
                    f"target_pair feature {n!r} not declared in dictionary"
                )
        method = self.optimize.get("method", "grid")
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unknown method {method!r}; supported: {SUPPORTED_METHODS}"
            )

    def structural_floor(self) -> float:
        """Analytic minimum target-pair `|<A|B>|²` reachable by varying
        only `(φ_A, φ_B)`, holding all other features fixed.

        Evaluates the overlap at two phase points: `(φ_anchor,
        φ_anchor)` (δ=0) and `(φ_anchor, φ_anchor + π)` (δ=π), where
        `φ_anchor` is the current `target_pair[0]` feature's φ.
        Returns `min(m_zero, m_pi)` — equivalent to `M − |V|` for the
        decomposition `|<A|B>|²(δ) = M + V·cos(δ)`.

        Two Gram evaluations regardless of backend; not affected by
        `preserve_tiers`.
        """
        m_zero, m_pi = self._floor_terms()
        return float(min(m_zero, m_pi))

    def _floor_terms(self) -> tuple[float, float]:
        a_name, b_name = self.target_pair
        a_idx = self.dictionary.feature_index(a_name)
        b_idx = self.dictionary.feature_index(b_name)
        anchor = float(self.dictionary.feature(a_name).phi)

        d_zero = self._dictionary_at(anchor, anchor)
        d_pi = self._dictionary_at(anchor, anchor + float(np.pi))
        m_zero = float(np.abs(d_zero.gram()[a_idx, b_idx]) ** 2)
        m_pi = float(np.abs(d_pi.gram()[a_idx, b_idx]) ** 2)
        return m_zero, m_pi

    def run(self) -> CancellationResult:
        method = self.optimize.get("method", "grid")
        max_steps = int(self.optimize.get("max_steps", 50))

        before_dict = self.dictionary
        before_gram = before_dict.gram()
        a_idx = before_dict.feature_index(self.target_pair[0])
        b_idx = before_dict.feature_index(self.target_pair[1])
        before_overlap = float(np.abs(before_gram[a_idx, b_idx]) ** 2)

        floor = self.structural_floor()

        if method == "grid":
            phi_a, phi_b, after_overlap, traj, feasible_mask = self._run_grid(
                max_steps
            )
        else:
            phi_a, phi_b, after_overlap, traj, feasible_mask = self._run_scipy(
                max_steps
            )

        optimized_dict = self._dictionary_at(phi_a, phi_b)
        after_gram = optimized_dict.gram()
        efficiency = _compute_efficiency(
            before_overlap, float(after_overlap), floor
        )

        return CancellationResult(
            optimized_phis={
                self.target_pair[0]: float(phi_a),
                self.target_pair[1]: float(phi_b),
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
            feasible_mask=feasible_mask,
            structural_floor=floor,
            cancellation_efficiency=efficiency,
        )

    def _run_grid(
        self, res: int
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        phis = np.linspace(0.0, 2 * np.pi, res)
        a_name, b_name = self.target_pair
        traj = np.zeros((res * res, 3), dtype=float)
        feasible_mask = np.zeros(res * res, dtype=bool)

        a_idx = self.dictionary.feature_index(a_name)
        b_idx = self.dictionary.feature_index(b_name)

        for i, pa in enumerate(phis):
            for j, pb in enumerate(phis):
                d = self._dictionary_at(float(pa), float(pb))
                g = d.gram()
                ov = float(np.abs(g[a_idx, b_idx]) ** 2)
                k = i * res + j
                traj[k, 0] = pa
                traj[k, 1] = pb
                traj[k, 2] = ov
                if self.preserve_tiers:
                    feasible_mask[k] = hierarchical_ordering_preserved(
                        g, d, self.target_pair
                    )
                else:
                    feasible_mask[k] = True

        if feasible_mask.any():
            search = np.where(feasible_mask, traj[:, 2], np.inf)
            best = int(np.argmin(search))
        else:
            best = int(np.argmin(traj[:, 2]))

        return (
            float(traj[best, 0]),
            float(traj[best, 1]),
            float(traj[best, 2]),
            traj,
            feasible_mask,
        )

    def _run_scipy(
        self, maxiter: int
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        try:
            from scipy.optimize import differential_evolution
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_SCIPY_INSTALL_HINT) from exc

        a_name, b_name = self.target_pair
        a_idx = self.dictionary.feature_index(a_name)
        b_idx = self.dictionary.feature_index(b_name)
        history: list[tuple[float, float, float, bool]] = []

        def objective(x: np.ndarray) -> float:
            pa, pb = float(x[0]), float(x[1])
            d = self._dictionary_at(pa, pb)
            g = d.gram()
            ov = float(np.abs(g[a_idx, b_idx]) ** 2)
            feasible = (
                hierarchical_ordering_preserved(g, d, self.target_pair)
                if self.preserve_tiers
                else True
            )
            history.append((pa, pb, ov, feasible))
            return ov + (0.0 if feasible else INFEASIBLE_PENALTY)

        result = differential_evolution(
            objective,
            bounds=[(0.0, 2 * np.pi), (0.0, 2 * np.pi)],
            seed=0,
            maxiter=maxiter,
            polish=False,
        )
        traj = np.array(
            [(pa, pb, ov) for pa, pb, ov, _ in history], dtype=float
        )
        feasible_mask = np.array([ok for *_, ok in history], dtype=bool)

        # If the unconstrained minimum is infeasible but feasible
        # candidates exist, return the best feasible one as the
        # reported optimum so `optimized_phis` is meaningful.
        if self.preserve_tiers and feasible_mask.any():
            feasible_overlaps = np.where(feasible_mask, traj[:, 2], np.inf)
            best = int(np.argmin(feasible_overlaps))
            return (
                float(traj[best, 0]),
                float(traj[best, 1]),
                float(traj[best, 2]),
                traj,
                feasible_mask,
            )
        return (
            float(result.x[0]),
            float(result.x[1]),
            float(np.abs(self._dictionary_at(
                float(result.x[0]), float(result.x[1])
            ).gram()[a_idx, b_idx]) ** 2),
            traj,
            feasible_mask,
        )

    def _dictionary_at(self, phi_a: float, phi_b: float) -> Dictionary:
        a_name, b_name = self.target_pair
        d = self.dictionary
        d = d.with_phi(a_name, float(phi_a))
        d = d.with_phi(b_name, float(phi_b))
        return replace(d, name=f"{self.dictionary.name}_at_optimum")


def _compute_efficiency(
    before: float, after: float, floor: float
) -> float | None:
    gap = before - floor
    if gap < 1e-9:
        return None
    return float(np.clip((before - after) / gap, 0.0, 1.0))


def _interpret_efficiency(efficiency: float | None) -> str:
    if efficiency is None:
        return "no cancellation gap available"
    if efficiency >= 0.99:
        return "phase search exhausted — encoding-bound"
    return "phase search underutilized"


def _render_summary(result: CancellationResult) -> str:
    a, b = result.target_pair
    eff = result.cancellation_efficiency
    eff_str = "n/a" if eff is None else f"{eff:.4f}"
    lines = [
        f"# {result.dictionary_at_optimum.name}",
        "",
        f"- target pair: `{a}` × `{b}`",
        f"- method: `{result.method}`",
        f"- tolerance_met: {result.tolerance_met} "
        f"(after_overlap < tolerance)",
        f"- feasible evaluations: {result.feasible_count} of "
        f"{result.trajectory.shape[0]}",
        "",
        "## Optimum",
        "",
        f"- `{a}.phi` = {result.optimized_phis[a]:.6f}",
        f"- `{b}.phi` = {result.optimized_phis[b]:.6f}",
        "",
        "## Overlap",
        "",
        f"- before: {result.before_overlap:.6f}",
        f"- after:  {result.after_overlap:.6f}",
        f"- delta:  {result.after_overlap - result.before_overlap:+.6f}",
        "",
        "## Structural floor",
        "",
        f"- structural_floor: {result.structural_floor:.6f}",
        f"- cancellation_efficiency: {eff_str}",
        f"- interpretation: {_interpret_efficiency(eff)}",
        "",
    ]
    return "\n".join(lines)


def _render_trajectory_csv(result: CancellationResult) -> str:
    out = ["phi_a,phi_b,overlap,feasible"]
    for row, feasible in zip(result.trajectory, result.feasible_mask):
        pa, pb, ov = float(row[0]), float(row[1]), float(row[2])
        out.append(f"{pa:.6f},{pb:.6f},{ov:.6f},{1 if feasible else 0}")
    return "\n".join(out) + "\n"
