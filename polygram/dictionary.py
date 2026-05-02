"""Polysemantic feature dictionaries with shallow hierarchy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from polygram.encoding import MPSRung1


@dataclass(frozen=True)
class Feature:
    """One feature in a Polygram dictionary.

    Maps to one rung-1 MPS concept state |c_i⟩ via the cross-coupled
    staircase preparation. The four angles parametrize the encoder:

    - `alpha`, `gamma` — outer-rung Ry knobs (default 0)
    - `beta` — inner-rung Ry knob; carries the cluster identity
    - `phi` — optional `Rz(qs[1], phi)` phase knob (default 0)
    """

    name: str
    cluster: str
    beta: float
    alpha: float = 0.0
    gamma: float = 0.0
    phi: float = 0.0


_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Dictionary:
    """A polysemantic feature dictionary with a shallow hierarchy.

    `features` is the ordered list of features; `hierarchy` maps each
    cluster name to the names of its members. Every feature SHALL be
    listed under exactly one cluster.

    The order of `features` defines the row/column index of the Gram
    matrix; use `feature_index(name)` to look up an index.
    """

    name: str
    features: list[Feature]
    hierarchy: dict[str, list[str]]
    encoding: MPSRung1 = field(default_factory=MPSRung1)

    def __post_init__(self) -> None:
        if not _VALID_NAME_RE.match(self.name):
            raise ValueError(
                f"Dictionary name {self.name!r} must match {_VALID_NAME_RE.pattern}"
            )

        feature_names = [f.name for f in self.features]
        if len(feature_names) != len(set(feature_names)):
            dups = [n for n in feature_names if feature_names.count(n) > 1]
            raise ValueError(f"duplicate feature name(s): {sorted(set(dups))}")

        seen: dict[str, str] = {}
        for cluster, members in self.hierarchy.items():
            for m in members:
                if m in seen:
                    raise ValueError(
                        f"feature {m!r} listed in two clusters: "
                        f"{seen[m]!r} and {cluster!r}"
                    )
                seen[m] = cluster

        for f in self.features:
            if f.cluster not in self.hierarchy:
                raise ValueError(
                    f"feature {f.name!r} has cluster {f.cluster!r} "
                    f"which is not a key in hierarchy"
                )
            if f.name not in self.hierarchy[f.cluster]:
                raise ValueError(
                    f"feature {f.name!r} (cluster={f.cluster!r}) is not "
                    f"listed under hierarchy[{f.cluster!r}]"
                )

        for cluster, members in self.hierarchy.items():
            unknown = set(members) - set(feature_names)
            if unknown:
                raise ValueError(
                    f"hierarchy[{cluster!r}] references unknown feature(s): "
                    f"{sorted(unknown)}"
                )

    def feature_index(self, name: str) -> int:
        for i, f in enumerate(self.features):
            if f.name == name:
                return i
        raise KeyError(f"no feature named {name!r}; "
                       f"have {[f.name for f in self.features]}")

    def feature(self, name: str) -> Feature:
        return self.features[self.feature_index(name)]

    def with_phi(self, name: str, value: float) -> Dictionary:
        """Return a copy of this Dictionary with one feature's `phi` set."""
        idx = self.feature_index(name)
        new_feature = replace(self.features[idx], phi=value)
        new_features = list(self.features)
        new_features[idx] = new_feature
        return replace(self, features=new_features)

    def gram(self) -> np.ndarray:
        """Return the analytic N×N concept overlap matrix.

        Builds an in-memory Q-Orca machine in larql-animals-interference
        style and calls `q_orca.compute_concept_gram_mps(form="preparation")`
        — preparation form is required because the safe-`Rz` matcher
        rejects inverse-form effects with non-trivial `Rz` (the |0⟩
        fixed-point symmetry break).
        """
        from polygram._qorca_emit import build_machine
        from q_orca.compiler.concept_gram_mps import compute_concept_gram_mps

        machine = build_machine(self)
        return compute_concept_gram_mps(
            machine, concept_action_label="prepare_concept"
        )

    @classmethod
    def with_default_angles(
        cls,
        name: str,
        hierarchy: dict[str, list[str]],
        encoding: MPSRung1 | None = None,
    ) -> Dictionary:
        """Build a Dictionary from a `{cluster: [feature_names]}` mapping,
        spreading β evenly in `[-0.5, +0.5]` per cluster (α = γ = φ = 0)."""
        clusters = list(hierarchy.keys())
        betas = _default_betas(clusters)
        features: list[Feature] = []
        for cluster in clusters:
            for member in hierarchy[cluster]:
                features.append(
                    Feature(name=member, cluster=cluster, beta=betas[cluster])
                )
        return cls(
            name=name,
            features=features,
            hierarchy=hierarchy,
            encoding=encoding or MPSRung1(),
        )


def _default_betas(clusters: Iterable[str]) -> dict[str, float]:
    cs = list(clusters)
    n = len(cs)
    if n == 0:
        return {}
    if n == 1:
        return {cs[0]: 0.0}
    return {c: -0.5 + 1.0 * i / (n - 1) for i, c in enumerate(cs)}
