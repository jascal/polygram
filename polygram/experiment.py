"""Experiment + InterferenceSweep — phase-sweep over a Polygram Dictionary."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from polygram._assertions import (
    SUPPORTED_ASSERTIONS,
    concept_gram_tier_separation_bound_holds,
    hierarchical_ordering_preserved,
    target_pair_destructive_at_endpoint,
)
from polygram._state import build_statevector, schmidt_rank
from polygram._tier_stats import TIER_NAMES, compute_tier_stats
from polygram.dictionary import Dictionary, Feature, _parse_knob_path
from polygram.emit import write_qorca
from polygram.encoding import HEA_Rung2

SUPPORTED_MEASURES = ("overlap", "gram_matrix", "schmidt_rank")
SUPPORTED_BACKENDS = ("analytic",)

_PLOT_INSTALL_HINT = (
    "matplotlib is required for ExperimentResult.plot(); "
    "install with `pip install polygram[plot]`."
)


@dataclass
class ExperimentResult:
    """Output of an `InterferenceSweep` run.

    Shapes assume sweep dims `(D_1, D_2, ..., D_K)` and `N` features:

    - `gram_matrices: complex array (*sweep_dims, N, N)`
    - `overlaps:     real array    (*sweep_dims,)` — `|<A|B>|²` for the
      target pair
    - `schmidt_ranks: int array    (*sweep_dims, N)` — Schmidt rank of
      each feature at the q0 | (q1,q2) bipartition
    - `assertion_pass: dict[str, bool array (*sweep_dims,)]`
    - `tier_stats:   dict[str, real array (*sweep_dims,)]` keyed by
      `TIER_NAMES` (`self`, `sibling`, `cross_cluster`)
    """

    sweep_axes: dict[str, np.ndarray]
    gram_matrices: np.ndarray
    overlaps: np.ndarray
    schmidt_ranks: np.ndarray
    assertion_pass: dict[str, np.ndarray] = field(default_factory=dict)
    tier_stats: dict[str, np.ndarray] = field(default_factory=dict)
    target_pair: tuple[str, str] | None = None
    dictionary_name: str | None = None
    tier_separation: np.ndarray | None = None

    def save(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        extra: dict[str, np.ndarray] = {}
        if self.tier_separation is not None:
            extra["tier_separation"] = self.tier_separation
        np.savez_compressed(
            p,
            gram_matrices=self.gram_matrices,
            overlaps=self.overlaps,
            schmidt_ranks=self.schmidt_ranks,
            sweep_keys=np.array(list(self.sweep_axes.keys())),
            **{f"axis_{k}": v for k, v in self.sweep_axes.items()},
            **{f"assert_{k}": v for k, v in self.assertion_pass.items()},
            **{f"tier_{k}": v for k, v in self.tier_stats.items()},
            **extra,
        )
        return p

    def to_csv(self, path: str | os.PathLike) -> Path:
        """Flatten overlaps + assertions + tier stats to a tabular CSV.

        Columns: each sweep axis value, then `overlap`, then one column
        per tier (`tier_<name>`), then one column per assertion. The
        full Gram tensor and Schmidt ranks are not flattened — use
        `.save()` (.npz) for those.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        axis_names = list(self.sweep_axes.keys())
        axes = [self.sweep_axes[k] for k in axis_names]
        shape = self.overlaps.shape
        tier_keys = list(self.tier_stats.keys())
        include_tier_sep = self.tier_separation is not None
        header = (
            axis_names
            + ["overlap"]
            + [f"tier_{k}" for k in tier_keys]
            + (["tier_separation"] if include_tier_sep else [])
            + list(self.assertion_pass.keys())
        )
        with p.open("w") as f:
            f.write(",".join(header) + "\n")
            for raw_idx in np.ndindex(*shape):
                row = [str(float(axes[d][raw_idx[d]])) for d in range(len(shape))]
                row.append(str(float(self.overlaps[raw_idx])))
                for k in tier_keys:
                    row.append(str(float(self.tier_stats[k][raw_idx])))
                if include_tier_sep:
                    row.append(str(float(self.tier_separation[raw_idx])))
                for a in self.assertion_pass.values():
                    row.append("1" if bool(a[raw_idx]) else "0")
                f.write(",".join(row) + "\n")
        return p

    def plot(
        self, path: str | os.PathLike, kind: str = "overlap"
    ) -> Path:
        """Render a default matplotlib figure.

        - 1D sweep → line plot of target-pair overlap with sibling and
          cross-cluster tier baselines.
        - 2D sweep → heatmap of target-pair overlap.
        - ≥3D sweep → `NotImplementedError`.

        `kind="overlap"` is the only kind supported in v0.
        """
        if kind != "overlap":
            raise ValueError(
                f"unknown plot kind {kind!r}; only 'overlap' is supported"
            )
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_PLOT_INSTALL_HINT) from exc

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ndim = len(self.sweep_axes)

        if ndim == 1:
            (axis_name,) = self.sweep_axes.keys()
            xs = self.sweep_axes[axis_name]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(xs, self.overlaps, marker="o", linewidth=1.5,
                    label=self._target_label())
            if "sibling" in self.tier_stats:
                ax.plot(xs, self.tier_stats["sibling"], linestyle="--",
                        color="tab:green", label="sibling tier (mean)")
            if "cross_cluster" in self.tier_stats:
                ax.plot(xs, self.tier_stats["cross_cluster"], linestyle=":",
                        color="tab:gray", label="cross-cluster tier (mean)")
            ax.set_xlabel(axis_name)
            ax.set_ylabel("|<A|B>|²")
            ax.set_title(self._plot_title())
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(p, dpi=120)
            plt.close(fig)
            return p

        if ndim == 2:
            keys = list(self.sweep_axes.keys())
            x_key, y_key = keys
            xs = self.sweep_axes[x_key]
            ys = self.sweep_axes[y_key]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                self.overlaps.T,
                origin="lower",
                aspect="auto",
                extent=(float(xs[0]), float(xs[-1]),
                        float(ys[0]), float(ys[-1])),
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, label="|<A|B>|²")
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(self._plot_title())
            fig.tight_layout()
            fig.savefig(p, dpi=120)
            plt.close(fig)
            return p

        raise NotImplementedError(
            f"plot() supports 1D and 2D sweeps; got {ndim} sweep axes. "
            f"Slice the result yourself for higher-dim landscapes."
        )

    def write_summary(self, path: str | os.PathLike) -> Path:
        """Append a tier rollup + assertion pass-rate table to a
        markdown summary file. If `path` already exists (e.g. written
        by `Experiment.materialize()`), the rollup is appended below
        the existing content."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = p.read_text() if p.exists() else ""
        body = "## Run results\n\n"
        body += "### Tier rollup (mean over sweep)\n\n"
        body += "| Tier | min | mean | max |\n|------|-----|------|-----|\n"
        for tier in TIER_NAMES:
            arr = self.tier_stats.get(tier)
            if arr is None or np.all(np.isnan(arr)):
                body += f"| {tier} | n/a | n/a | n/a |\n"
                continue
            finite = arr[~np.isnan(arr)]
            body += (
                f"| {tier} | {float(np.min(finite)):.4f} "
                f"| {float(np.mean(finite)):.4f} "
                f"| {float(np.max(finite)):.4f} |\n"
            )
        body += "\n### Target overlap\n\n"
        body += (
            f"- min: {float(self.overlaps.min()):.4f}\n"
            f"- mean: {float(self.overlaps.mean()):.4f}\n"
            f"- max: {float(self.overlaps.max()):.4f}\n"
        )
        if self.assertion_pass:
            body += "\n### Assertion pass-rate\n\n"
            body += "| Assertion | passes | total | rate |\n"
            body += "|-----------|--------|-------|------|\n"
            for name, arr in self.assertion_pass.items():
                passes = int(arr.sum())
                total = int(arr.size)
                rate = passes / total if total else 0.0
                body += f"| {name} | {passes} | {total} | {rate:.0%} |\n"
        p.write_text(existing + ("\n" if existing else "") + body)
        return p

    def _target_label(self) -> str:
        if self.target_pair is None:
            return "target pair"
        a, b = self.target_pair
        return f"({a}, {b})"

    def _plot_title(self) -> str:
        if self.dictionary_name is None:
            return "InterferenceSweep — target overlap"
        return f"{self.dictionary_name} — target overlap"


@dataclass
class Experiment:
    """A declarative phase-sweep experiment over a `Dictionary`.

    `sweep` keys are `<feature_name>.phi` — the only knob v0 sweeps. All
    sweep keys MUST reference declared features. `target_pair` is the
    pair of feature names whose overlap is the headline measure.
    """

    name: str
    dictionary: Dictionary
    target_pair: tuple[str, str]
    sweep: dict[str, np.ndarray]
    measures: list[str] = field(default_factory=lambda: ["overlap", "gram_matrix"])
    assertions: list[str] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        a, b = self.target_pair
        for n in (a, b):
            if n not in [f.name for f in self.dictionary.features]:
                raise ValueError(
                    f"target_pair feature {n!r} not declared in dictionary"
                )

        for m in self.measures:
            if m not in SUPPORTED_MEASURES:
                raise ValueError(
                    f"unknown measure {m!r}; supported: {SUPPORTED_MEASURES}"
                )

        for a_name in self.assertions:
            if a_name not in SUPPORTED_ASSERTIONS:
                raise ValueError(
                    f"unknown assertion {a_name!r}; supported: {SUPPORTED_ASSERTIONS}"
                )

        if "concept_gram_tier_separation_bound_holds" in self.assertions:
            bound = getattr(
                self.dictionary.encoding, "tier_separation_bound", None
            )
            if bound is None:
                raise ValueError(
                    f"assertion 'concept_gram_tier_separation_bound_holds' "
                    f"requires the dictionary's encoding to declare a "
                    f"non-None tier_separation_bound; got "
                    f"encoding={self.dictionary.encoding!r}"
                )

        for key in self.sweep:
            self._parse_sweep_key(key)

    def _parse_sweep_key(self, key: str) -> tuple[str, str, tuple[int, int, int] | None]:
        feature, kind, slot = _parse_knob_path(key)
        if feature not in [f.name for f in self.dictionary.features]:
            raise ValueError(
                f"sweep key {key!r} references unknown feature {feature!r}"
            )
        if kind == "theta":
            if not isinstance(self.dictionary.encoding, HEA_Rung2):
                raise ValueError(
                    f"sweep key {key!r}: .theta[...] paths are HEA-only; "
                    f"this Dictionary uses encoding={self.dictionary.encoding!r}"
                )
            shape = self.dictionary.encoding.theta_shape
            r, d_, q = slot
            if not (
                0 <= r < shape[0] and 0 <= d_ < shape[1] and 0 <= q < shape[2]
            ):
                raise ValueError(
                    f"sweep key {key!r}: slot {slot} is outside "
                    f"theta_shape={shape}"
                )
        return feature, kind, slot

    def run(self, backend: str = "analytic", shots: int = 0) -> ExperimentResult:
        if backend not in SUPPORTED_BACKENDS:
            raise NotImplementedError(
                f"backend {backend!r} not supported in v0; use 'analytic'. "
                f"Shot-based backends (qutip, qiskit-aer) are roadmap items."
            )
        return InterferenceSweep(self).run()

    def materialize(self, output_dir: str | os.PathLike) -> dict[str, Path]:
        """Write `<name>.q.orca.md` (the dictionary at sweep midpoint
        for reference), a self-contained `run_<name>.py`, and a
        human-readable `<name>_summary.md` describing the configuration."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}

        midpoint_dict = self._dictionary_at_sweep_index(
            tuple(len(v) // 2 for v in self.sweep.values())
        )
        artifacts["machine"] = write_qorca(
            midpoint_dict, out / f"{self.name}.q.orca.md"
        )

        runner = out / f"run_{self.name}.py"
        runner.write_text(_render_runner_script(self, runner.name))
        artifacts["runner"] = runner

        summary = out / f"{self.name}_summary.md"
        summary.write_text(_render_summary_header(self))
        artifacts["summary"] = summary

        return artifacts

    def _dictionary_at_sweep_index(self, idx: tuple[int, ...]) -> Dictionary:
        d = self.dictionary
        for (key, values), i in zip(self.sweep.items(), idx):
            d = d.with_knob(key, float(values[i]))
        return d


@dataclass
class InterferenceSweep:
    experiment: Experiment

    def run(self) -> ExperimentResult:
        exp = self.experiment
        d0 = exp.dictionary
        n = len(d0.features)
        sweep_dims = tuple(len(v) for v in exp.sweep.values())

        gram = np.zeros(sweep_dims + (n, n), dtype=complex)
        overlaps = np.zeros(sweep_dims, dtype=float)
        schmidt = np.zeros(sweep_dims + (n,), dtype=int)
        tier_arrays: dict[str, np.ndarray] = {
            t: np.zeros(sweep_dims, dtype=float) for t in TIER_NAMES
        }

        a_idx = d0.feature_index(exp.target_pair[0])
        b_idx = d0.feature_index(exp.target_pair[1])

        per_point_assertions: dict[str, np.ndarray] = {}
        if "hierarchical_ordering_preserved" in exp.assertions:
            per_point_assertions["hierarchical_ordering_preserved"] = np.zeros(
                sweep_dims, dtype=bool
            )
        if "concept_gram_tier_separation_bound_holds" in exp.assertions:
            per_point_assertions[
                "concept_gram_tier_separation_bound_holds"
            ] = np.zeros(sweep_dims, dtype=bool)

        all_singletons = all(len(m) == 1 for m in d0.hierarchy.values())
        tier_separation = (
            None if all_singletons else np.zeros(sweep_dims, dtype=float)
        )

        for raw_idx in product(*[range(d) for d in sweep_dims]):
            idx = raw_idx if sweep_dims else ()
            d = exp._dictionary_at_sweep_index(idx) if sweep_dims else d0
            g = d.gram()
            gram[idx] = g
            overlaps[idx] = float(np.abs(g[a_idx, b_idx]) ** 2)

            for i, f in enumerate(d.features):
                schmidt[idx + (i,)] = schmidt_rank(build_statevector(f))

            tiers = compute_tier_stats(g, d)
            for t in TIER_NAMES:
                tier_arrays[t][idx] = tiers[t]

            if tier_separation is not None:
                from q_orca.compiler.concept_gram_hea import (
                    compute_tier_separation,
                )

                sep = compute_tier_separation(
                    g, [feat.cluster for feat in d.features]
                )
                tier_separation[idx] = float(sep) if sep is not None else 0.0

            if "hierarchical_ordering_preserved" in exp.assertions:
                per_point_assertions["hierarchical_ordering_preserved"][idx] = (
                    hierarchical_ordering_preserved(g, d, exp.target_pair)
                )
            if "concept_gram_tier_separation_bound_holds" in exp.assertions:
                per_point_assertions[
                    "concept_gram_tier_separation_bound_holds"
                ][idx] = concept_gram_tier_separation_bound_holds(g, d)

        if "target_pair_destructive_at_endpoint" in exp.assertions:
            endpoint_idx = tuple(d - 1 for d in sweep_dims) if sweep_dims else ()
            endpoint_gram = gram[endpoint_idx]
            endpoint_dict = (
                exp._dictionary_at_sweep_index(endpoint_idx) if sweep_dims else d0
            )
            ok = target_pair_destructive_at_endpoint(
                endpoint_gram, endpoint_dict, exp.target_pair
            )
            per_point_assertions["target_pair_destructive_at_endpoint"] = np.full(
                sweep_dims if sweep_dims else (1,), ok, dtype=bool
            )

        return ExperimentResult(
            sweep_axes={k: np.asarray(v) for k, v in exp.sweep.items()},
            gram_matrices=gram,
            overlaps=overlaps,
            schmidt_ranks=schmidt,
            assertion_pass=per_point_assertions,
            tier_stats=tier_arrays,
            target_pair=exp.target_pair,
            dictionary_name=d0.name,
            tier_separation=tier_separation,
        )


def _render_summary_header(experiment: Experiment) -> str:
    """Configuration-only summary; run results are appended later via
    `ExperimentResult.write_summary(path)`."""
    d = experiment.dictionary
    lines = [
        f"# {experiment.name}",
        "",
        f"- dictionary: `{d.name}` ({len(d.features)} features, "
        f"{len(d.hierarchy)} clusters)",
        f"- target pair: `{experiment.target_pair[0]}` × "
        f"`{experiment.target_pair[1]}`",
        "- backend: analytic",
        f"- seed: {experiment.seed}",
        "",
        "## Sweep axes",
        "",
        "| Axis | n_points | min | max |",
        "|------|----------|-----|-----|",
    ]
    for key, values in experiment.sweep.items():
        arr = np.asarray(values)
        lines.append(
            f"| `{key}` | {len(arr)} | {float(arr.min()):.4f} "
            f"| {float(arr.max()):.4f} |"
        )
    lines += [
        "",
        "## Measures",
        "",
        ", ".join(f"`{m}`" for m in experiment.measures),
        "",
        "## Assertions",
        "",
        ", ".join(f"`{a}`" for a in experiment.assertions) or "_(none)_",
        "",
    ]
    return "\n".join(lines)


def _render_runner_script(experiment: Experiment, script_filename: str) -> str:
    """Emit a self-contained Python script that reconstructs the
    experiment, runs it, and saves the result npz."""
    d = experiment.dictionary
    feature_lines = ",\n        ".join(
        _feature_repr(f) for f in d.features
    )
    hierarchy_lines = ",\n        ".join(
        f"{cluster!r}: {members!r}" for cluster, members in d.hierarchy.items()
    )
    sweep_lines = ",\n        ".join(
        f"{k!r}: np.asarray({list(v)!r})" for k, v in experiment.sweep.items()
    )

    measures_repr = repr(experiment.measures)
    assertions_repr = repr(experiment.assertions)
    target_repr = repr(experiment.target_pair)
    if isinstance(d.encoding, HEA_Rung2):
        e = d.encoding
        encoding_repr = (
            f"HEA_Rung2(depth={e.depth!r}, entangler={e.entangler!r}, "
            f"rotations={e.rotations!r}, "
            f"tier_separation_bound={e.tier_separation_bound!r}, "
            f"n_qubits={e.n_qubits!r})"
        )
        encoding_import = "HEA_Rung2"
    else:
        encoding_repr = (
            f"MPSRung1(bond_dim={d.encoding.bond_dim}, "
            f"phase_knobs={d.encoding.phase_knobs})"
        )
        encoding_import = "MPSRung1"

    return (
        '"""Auto-generated by polygram.Experiment.materialize.\n\n'
        f"  source experiment: {experiment.name}\n"
        f"  source dictionary: {d.name}\n"
        '"""\n'
        "\n"
        "import numpy as np\n"
        "\n"
        f"from polygram import Dictionary, Experiment, Feature, {encoding_import}\n"
        "\n"
        "\n"
        "def build_dictionary() -> Dictionary:\n"
        "    return Dictionary(\n"
        f"        name={d.name!r},\n"
        "        features=[\n"
        f"            {feature_lines},\n"
        "        ],\n"
        "        hierarchy={\n"
        f"            {hierarchy_lines},\n"
        "        },\n"
        f"        encoding={encoding_repr},\n"
        "    )\n"
        "\n"
        "\n"
        "def main() -> None:\n"
        "    experiment = Experiment(\n"
        f"        name={experiment.name!r},\n"
        "        dictionary=build_dictionary(),\n"
        f"        target_pair={target_repr},\n"
        "        sweep={\n"
        f"            {sweep_lines},\n"
        "        },\n"
        f"        measures={measures_repr},\n"
        f"        assertions={assertions_repr},\n"
        f"        seed={experiment.seed},\n"
        "    )\n"
        "    result = experiment.run()\n"
        f"    result.save({experiment.name + '_result.npz'!r})\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def _feature_repr(f: Feature) -> str:
    return (
        f"Feature(name={f.name!r}, cluster={f.cluster!r}, "
        f"beta={f.beta!r}, alpha={f.alpha!r}, "
        f"gamma={f.gamma!r}, phi={f.phi!r})"
    )
