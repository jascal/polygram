"""Experiment + InterferenceSweep — phase-sweep over a Polygram Dictionary."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from polygram._assertions import (
    SUPPORTED_ASSERTIONS,
    hierarchical_ordering_preserved,
    target_pair_destructive_at_endpoint,
)
from polygram._state import build_statevector, schmidt_rank
from polygram.dictionary import Dictionary, Feature
from polygram.emit import write_qorca

SUPPORTED_MEASURES = ("overlap", "gram_matrix", "schmidt_rank")
SUPPORTED_BACKENDS = ("analytic",)


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
    """

    sweep_axes: dict[str, np.ndarray]
    gram_matrices: np.ndarray
    overlaps: np.ndarray
    schmidt_ranks: np.ndarray
    assertion_pass: dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            gram_matrices=self.gram_matrices,
            overlaps=self.overlaps,
            schmidt_ranks=self.schmidt_ranks,
            sweep_keys=np.array(list(self.sweep_axes.keys())),
            **{f"axis_{k}": v for k, v in self.sweep_axes.items()},
            **{f"assert_{k}": v for k, v in self.assertion_pass.items()},
        )
        return p

    def to_csv(self, path: str | os.PathLike) -> Path:
        """Flatten overlaps + assertions to a tabular CSV for plotting
        tools that don't speak ndarray.

        Columns: each sweep axis value, then `overlap`, then one column
        per assertion. The full Gram tensor and Schmidt ranks are not
        flattened — use `.save()` (.npz) for those.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        axis_names = list(self.sweep_axes.keys())
        axes = [self.sweep_axes[k] for k in axis_names]
        shape = self.overlaps.shape
        header = axis_names + ["overlap"] + list(self.assertion_pass.keys())
        with p.open("w") as f:
            f.write(",".join(header) + "\n")
            for raw_idx in np.ndindex(*shape):
                row = [str(float(axes[d][raw_idx[d]])) for d in range(len(shape))]
                row.append(str(float(self.overlaps[raw_idx])))
                for a in self.assertion_pass.values():
                    row.append("1" if bool(a[raw_idx]) else "0")
                f.write(",".join(row) + "\n")
        return p


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

        for key in self.sweep:
            self._parse_sweep_key(key)

    def _parse_sweep_key(self, key: str) -> tuple[str, str]:
        if "." not in key:
            raise ValueError(
                f"sweep key {key!r} must be of form '<feature>.phi'"
            )
        feature, knob = key.rsplit(".", 1)
        if knob != "phi":
            raise ValueError(
                f"sweep key {key!r}: only the `.phi` knob is supported in v0"
            )
        if feature not in [f.name for f in self.dictionary.features]:
            raise ValueError(
                f"sweep key {key!r} references unknown feature {feature!r}"
            )
        return feature, knob

    def run(self, backend: str = "analytic", shots: int = 0) -> ExperimentResult:
        if backend not in SUPPORTED_BACKENDS:
            raise NotImplementedError(
                f"backend {backend!r} not supported in v0; use 'analytic'. "
                f"Shot-based backends (qutip, qiskit-aer) are roadmap items."
            )
        return InterferenceSweep(self).run()

    def materialize(self, output_dir: str | os.PathLike) -> dict[str, Path]:
        """Write `<name>.q.orca.md` (the dictionary at sweep midpoint
        for reference) and a self-contained `run_<name>.py` runner."""
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

        return artifacts

    def _dictionary_at_sweep_index(self, idx: tuple[int, ...]) -> Dictionary:
        d = self.dictionary
        for (key, values), i in zip(self.sweep.items(), idx):
            feature, _ = self._parse_sweep_key(key)
            d = d.with_phi(feature, float(values[i]))
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

        a_idx = d0.feature_index(exp.target_pair[0])
        b_idx = d0.feature_index(exp.target_pair[1])

        per_point_assertions: dict[str, np.ndarray] = {}
        if "hierarchical_ordering_preserved" in exp.assertions:
            per_point_assertions["hierarchical_ordering_preserved"] = np.zeros(
                sweep_dims, dtype=bool
            )

        for raw_idx in product(*[range(d) for d in sweep_dims]):
            idx = raw_idx if sweep_dims else ()
            d = exp._dictionary_at_sweep_index(idx) if sweep_dims else d0
            g = d.gram()
            gram[idx] = g if sweep_dims else g
            overlaps[idx] = float(np.abs(g[a_idx, b_idx]) ** 2)

            for i, f in enumerate(d.features):
                schmidt[idx + (i,)] = schmidt_rank(build_statevector(f))

            if "hierarchical_ordering_preserved" in exp.assertions:
                per_point_assertions["hierarchical_ordering_preserved"][idx] = (
                    hierarchical_ordering_preserved(g, d, exp.target_pair)
                )

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
        )


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
    encoding_repr = (
        f"MPSRung1(bond_dim={d.encoding.bond_dim}, "
        f"phase_knobs={d.encoding.phase_knobs})"
    )

    return (
        '"""Auto-generated by polygram.Experiment.materialize.\n\n'
        f"  source experiment: {experiment.name}\n"
        f"  source dictionary: {d.name}\n"
        '"""\n'
        "\n"
        "import numpy as np\n"
        "\n"
        "from polygram import Dictionary, Experiment, Feature, MPSRung1\n"
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
