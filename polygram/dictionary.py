"""Polysemantic feature dictionaries with shallow hierarchy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from polygram.encoding import (
    HEA_Rung2,
    MPSRung1,
    RUNG3_DEFAULT_PSI_AUX,
    RUNG3_DEFAULT_THETA_AMP,
    RUNG4_DEFAULT_PSI_AMP_B,
    RUNG4_DEFAULT_THETA_AMP_B,
    Rung3,
    Rung4,
    Rung5,
    rung3_amp_overlap,
    rung4_amp_overlap,
    rung5_amp_overlap,
)


@dataclass(frozen=True, eq=False)
class Feature:
    """One feature in a Polygram dictionary.

    Maps to one rung-1 MPS concept state |c_i⟩ via the cross-coupled
    staircase preparation. The four angles parametrize the encoder:

    - `alpha`, `gamma` — outer-rung Ry knobs (default 0)
    - `beta` — inner-rung Ry knob; carries the cluster identity
    - `phi` — optional `Rz(qs[1], phi)` phase knob (default 0)

    For HEA encodings, an explicit ``theta`` of shape
    ``(|rotations|, depth, n_qubits)`` overrides the default tensor
    synthesized from ``(alpha, beta, gamma, phi)``.
    """

    name: str
    cluster: str
    beta: float
    alpha: float = 0.0
    gamma: float = 0.0
    phi: float = 0.0
    theta: np.ndarray | None = None
    theta_amp: float = RUNG3_DEFAULT_THETA_AMP
    psi_aux: float = RUNG3_DEFAULT_PSI_AUX
    # Rung4 q4 single-qubit amp knobs. Defaults (0.0, 0.0) make each
    # single-qubit amp factor reduce to ⟨0|0⟩ = 1 under Rung4's
    # product-amp interpretation. Rung3 ignores these fields — its
    # amp branch lives entirely on the (theta_amp, psi_aux) pair.
    # Field-level default values preserve Rung3 gram bit-for-bit.
    theta_amp_b: float = RUNG4_DEFAULT_THETA_AMP_B
    psi_amp_b: float = RUNG4_DEFAULT_PSI_AMP_B
    # Rung5 product-amp knobs as a length-k tuple of (theta, psi) pairs.
    # Empty tuple (the default) is the no-op shape for every non-Rung5
    # encoding; Dictionary.__post_init__ validates length against
    # encoding.n_amp_qubits when the encoding is Rung5. Stored as a
    # tuple-of-tuples so the dataclass remains hashable.
    amp_knobs: tuple[tuple[float, float], ...] = ()

    def with_default_amp_knobs(self, encoding: object) -> Feature:
        """Return a copy with ``amp_knobs`` padded to the encoding's width.

        For ``Rung5(n_amp_qubits=k)`` encodings, returns a copy with
        ``amp_knobs`` set to ``((0.0, 0.0),) * k`` whenever the field
        is currently the empty-tuple default. Already-populated
        ``amp_knobs`` are preserved unchanged (length mismatch is the
        caller's problem — ``Dictionary.__post_init__`` validates).

        For every non-Rung5 encoding, returns ``self`` unchanged.
        """
        if isinstance(encoding, Rung5) and self.amp_knobs == ():
            default = ((0.0, 0.0),) * encoding.n_amp_qubits
            return replace(self, amp_knobs=default)
        return self


def _default_hea_theta(feature: Feature, encoding: HEA_Rung2) -> np.ndarray:
    """Synthesize a default θ tensor from a feature's ``(α, β, γ, φ)`` knobs.

    Lays the three Ry-style knobs across the first rotation's first layer
    (qubits 0, 1, 2) and parks φ on the second rotation's first layer at
    qubit 1 when an ``Rz`` slot exists. Remaining slots are zero. Small
    ``(α, β, γ)`` keeps cluster cohesion high; outsider features pick up
    a magnitude shift via larger knobs.

    Heuristic boundary
    ------------------
    The placement assumes the canonical 3-qubit / ``("Ry", "Rz")``
    layout. When ``rotations == ("Rz",)`` and ``n_qubits < 2``, both
    ``α`` and ``φ`` map to the same slot ``(0, 0, 0)``: ``φ`` is written
    last and silently overwrites ``α``. That's a degenerate
    configuration for the rung-2 spike (which targets 3 qubits) and the
    helper does not raise — pass an explicit ``Feature.theta`` instead
    if you want full control over θ in such layouts.
    """
    shape = encoding.theta_shape
    theta = np.zeros(shape, dtype=float)
    n_rot, depth, n_qubits = shape
    knobs = [feature.alpha, feature.beta, feature.gamma]
    for q in range(min(n_qubits, len(knobs))):
        theta[0, 0, q] = knobs[q]
    if "Rz" in encoding.rotations and feature.phi != 0.0:
        rz_idx = encoding.rotations.index("Rz")
        if rz_idx != 0 and n_qubits >= 2:
            theta[rz_idx, 0, 1] = feature.phi
        elif rz_idx == 0:
            phi_q = 1 if n_qubits >= 2 else 0
            theta[rz_idx, 0, phi_q] = feature.phi
    return theta


_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KNOB_PHI_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.phi$")
# Order matters: `.theta_amp_b` must be tried BEFORE `.theta_amp`
# because the latter's regex matches the prefix of the former.
_KNOB_THETA_AMP_B_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.theta_amp_b$")
_KNOB_THETA_AMP_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.theta_amp$")
_KNOB_PSI_AMP_B_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.psi_amp_b$")
_KNOB_PSI_AUX_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.psi_aux$")
_KNOB_AMP_KNOBS_THETA_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\.amp_knobs\[(\d+)\]\.theta$"
)
_KNOB_AMP_KNOBS_PSI_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\.amp_knobs\[(\d+)\]\.psi$"
)
_KNOB_THETA_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\.theta\[(\d+),(\d+),(\d+)\]$"
)


def _parse_knob_path(path: str) -> tuple[str, str, tuple[int, ...] | None]:
    m = _KNOB_PHI_RE.match(path)
    if m:
        return m.group(1), "phi", None
    # Try Rung5 amp_knobs[i].{theta,psi} before the Rung3/Rung4 named
    # variants so the longer pattern wins.
    m = _KNOB_AMP_KNOBS_THETA_RE.match(path)
    if m:
        return m.group(1), "amp_knobs_theta", (int(m.group(2)),)
    m = _KNOB_AMP_KNOBS_PSI_RE.match(path)
    if m:
        return m.group(1), "amp_knobs_psi", (int(m.group(2)),)
    # Try the Rung4 q4-amp knobs first (longer match) so they don't
    # collide with the Rung3 q3-amp prefixes.
    m = _KNOB_THETA_AMP_B_RE.match(path)
    if m:
        return m.group(1), "theta_amp_b", None
    m = _KNOB_THETA_AMP_RE.match(path)
    if m:
        return m.group(1), "theta_amp", None
    m = _KNOB_PSI_AMP_B_RE.match(path)
    if m:
        return m.group(1), "psi_amp_b", None
    m = _KNOB_PSI_AUX_RE.match(path)
    if m:
        return m.group(1), "psi_aux", None
    m = _KNOB_THETA_RE.match(path)
    if m:
        name = m.group(1)
        slot = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        return name, "theta", slot
    raise ValueError(
        f"knob path {path!r} does not match expected grammar "
        f"'<feature>.phi', '<feature>.theta_amp', '<feature>.psi_aux', "
        f"'<feature>.theta[r,d,q]', or "
        f"'<feature>.amp_knobs[i].{{theta,psi}}'"
    )


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
    encoding: MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | Rung5 = field(
        default_factory=MPSRung1
    )

    def __post_init__(self) -> None:
        if not _VALID_NAME_RE.match(self.name):
            raise ValueError(
                f"Dictionary name {self.name!r} must match {_VALID_NAME_RE.pattern}"
            )

        feature_names = [f.name for f in self.features]
        if len(feature_names) != len(set(feature_names)):
            dups = [n for n in feature_names if feature_names.count(n) > 1]
            raise ValueError(f"duplicate feature name(s): {sorted(set(dups))}")

        collisions = sorted(set(self.hierarchy.keys()) & set(feature_names))
        if collisions:
            raise ValueError(
                f"name collision between feature(s) and cluster(s): "
                f"{collisions}; rename either the feature or the cluster so "
                f"with_knob paths resolve unambiguously"
            )

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

        if isinstance(self.encoding, HEA_Rung2):
            expected = self.encoding.theta_shape
            for f in self.features:
                if f.theta is None:
                    continue
                if f.theta.shape != expected:
                    raise ValueError(
                        f"feature {f.name!r}: theta has shape "
                        f"{tuple(f.theta.shape)}, expected {expected} for "
                        f"encoding={self.encoding!r}"
                    )

        if isinstance(self.encoding, Rung5):
            k = self.encoding.n_amp_qubits
            for f in self.features:
                if len(f.amp_knobs) != k:
                    raise ValueError(
                        f"feature {f.name!r}: amp_knobs has length "
                        f"{len(f.amp_knobs)}, expected {k} for "
                        f"encoding={self.encoding!r}; pad with "
                        f"`Feature.with_default_amp_knobs(encoding)` or "
                        f"supply the full tuple at construction"
                    )
                for i, pair in enumerate(f.amp_knobs):
                    if not (isinstance(pair, tuple) and len(pair) == 2):
                        raise ValueError(
                            f"feature {f.name!r}: amp_knobs[{i}] must be a "
                            f"2-tuple of (theta, psi); got {pair!r}"
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

    def with_knob(self, path: str, value: float) -> Dictionary:
        """Return a copy with one named parameter slot set.

        Path grammar:

        - ``<feature>.phi`` — sets ``Feature.phi`` (both encodings).
        - ``<feature>.theta[r,d,q]`` — sets the ``(r, d, q)`` slot of
          the named feature's θ tensor (HEA only). When the feature's
          ``theta`` is ``None``, the default tensor is materialized via
          ``_default_hea_theta(...)``, copied, and the slot is set on
          the copy.
        - ``<cluster>.phi`` — *cluster-shared*: applies ``phi=value`` to
          every feature in ``self.hierarchy[cluster]`` (both encodings).
        - ``<cluster>.theta[r,d,q]`` — *cluster-shared*: writes the
          ``(r, d, q)`` slot of every member's θ tensor (HEA only).

        The leading identifier resolves as a feature name first, then
        falls back to a cluster name. Construction-time uniqueness
        (``Dictionary.__post_init__`` rejects feature/cluster collisions)
        guarantees the resolution is unambiguous.
        """
        name, kind, slot = _parse_knob_path(path)
        feature_names = [f.name for f in self.features]

        if name in feature_names:
            target_indices = [self.feature_index(name)]
        elif name in self.hierarchy:
            target_indices = [
                self.feature_index(m) for m in self.hierarchy[name]
            ]
        else:
            raise ValueError(
                f"knob path {path!r}: identifier {name!r} matches neither "
                f"a feature name (have {feature_names}) nor a cluster name "
                f"(have {sorted(self.hierarchy.keys())})"
            )

        if kind == "theta":
            if not isinstance(self.encoding, HEA_Rung2):
                raise ValueError(
                    f"knob path {path!r}: .theta[...] paths are HEA-only; "
                    f"this Dictionary uses encoding={self.encoding!r}"
                )
            shape = self.encoding.theta_shape
            assert slot is not None
            r, d, q = slot
            if not (0 <= r < shape[0] and 0 <= d < shape[1] and 0 <= q < shape[2]):
                raise ValueError(
                    f"knob path {path!r}: slot {slot} is outside "
                    f"theta_shape={shape} for encoding={self.encoding!r}"
                )

        if kind in ("amp_knobs_theta", "amp_knobs_psi"):
            if not isinstance(self.encoding, Rung5):
                raise ValueError(
                    f"knob path {path!r}: .amp_knobs[...] paths require "
                    f"a Rung5 encoding; this Dictionary uses "
                    f"encoding={self.encoding!r}"
                )
            assert slot is not None
            (amp_idx,) = slot
            k = self.encoding.n_amp_qubits
            if not (0 <= amp_idx < k):
                raise ValueError(
                    f"knob path {path!r}: amp index {amp_idx} is outside "
                    f"[0, {k}) for encoding={self.encoding!r}"
                )

        new_features = list(self.features)
        for idx in target_indices:
            feature = new_features[idx]
            if kind == "phi":
                new_features[idx] = replace(feature, phi=value)
            elif kind == "theta_amp":
                new_features[idx] = replace(feature, theta_amp=float(value))
            elif kind == "psi_aux":
                new_features[idx] = replace(feature, psi_aux=float(value))
            elif kind == "theta_amp_b":
                new_features[idx] = replace(feature, theta_amp_b=float(value))
            elif kind == "psi_amp_b":
                new_features[idx] = replace(feature, psi_amp_b=float(value))
            elif kind in ("amp_knobs_theta", "amp_knobs_psi"):
                # Materialise the default-padded amp_knobs tuple when
                # the feature still holds the empty default; preserves
                # the "set the slot, leave everything else at default"
                # ergonomic for first-time mutation.
                padded = feature.with_default_amp_knobs(self.encoding)
                assert slot is not None
                (amp_idx,) = slot
                pair = padded.amp_knobs[amp_idx]
                if kind == "amp_knobs_theta":
                    new_pair = (float(value), pair[1])
                else:
                    new_pair = (pair[0], float(value))
                new_amp = list(padded.amp_knobs)
                new_amp[amp_idx] = new_pair
                new_features[idx] = replace(
                    padded, amp_knobs=tuple(new_amp)
                )
            else:
                base = (
                    feature.theta.copy()
                    if feature.theta is not None
                    else _default_hea_theta(feature, self.encoding).copy()
                )
                assert slot is not None
                r, d, q = slot
                base[r, d, q] = float(value)
                new_features[idx] = replace(feature, theta=base)

        return replace(self, features=new_features)

    def gram(self) -> np.ndarray:
        """Return the analytic N×N concept overlap matrix.

        Dispatches on ``self.encoding``: ``MPSRung1`` calls
        ``q_orca.compute_concept_gram_mps`` against an
        ``larql-animals-interference``-style preparation-form machine;
        ``HEA_Rung2`` calls ``q_orca.compute_concept_gram_hea`` against
        the new ``## encoding`` + ``## theta`` machine layout. ``Rung3``
        composes the MPSRung1-equivalent gram on (α, β, γ, φ) with the
        per-pair amplitude-branch overlap factor (analytic, closed form
        per ``polygram.encoding.rung3_amp_overlap_squared``). ``Rung4``
        uses the same elementwise-product factorisation pattern with
        the product-amp ``rung4_amp_overlap`` (two independent
        single-qubit overlaps on q3 and q4).
        """
        if isinstance(self.encoding, Rung5):
            mps_dict = replace(self, encoding=MPSRung1())
            mps_gram = mps_dict.gram()
            n = len(self.features)
            amp_factor = np.ones((n, n), dtype=complex)
            for i in range(n):
                fi = self.features[i]
                for j in range(n):
                    fj = self.features[j]
                    amp_factor[i, j] = rung5_amp_overlap(
                        fi.amp_knobs, fj.amp_knobs
                    )
            return mps_gram.astype(complex) * amp_factor

        if isinstance(self.encoding, Rung4):
            mps_dict = replace(self, encoding=MPSRung1())
            mps_gram = mps_dict.gram()
            n = len(self.features)
            amp_factor = np.ones((n, n), dtype=complex)
            for i in range(n):
                fi = self.features[i]
                for j in range(n):
                    fj = self.features[j]
                    amp_factor[i, j] = rung4_amp_overlap(
                        fi.theta_amp, fi.psi_aux,
                        fi.theta_amp_b, fi.psi_amp_b,
                        fj.theta_amp, fj.psi_aux,
                        fj.theta_amp_b, fj.psi_amp_b,
                    )
            return mps_gram.astype(complex) * amp_factor

        if isinstance(self.encoding, Rung3):
            mps_dict = replace(self, encoding=MPSRung1())
            mps_gram = mps_dict.gram()
            n = len(self.features)
            amp_factor = np.ones((n, n), dtype=complex)
            for i in range(n):
                fi = self.features[i]
                for j in range(n):
                    fj = self.features[j]
                    amp_factor[i, j] = rung3_amp_overlap(
                        fi.theta_amp, fi.psi_aux, fj.theta_amp, fj.psi_aux
                    )
            return mps_gram.astype(complex) * amp_factor

        from polygram._qorca_emit import build_machine

        machine = build_machine(self)
        if isinstance(self.encoding, HEA_Rung2):
            from q_orca.compiler.concept_gram_hea import compute_concept_gram_hea

            return compute_concept_gram_hea(
                machine, concept_action_label="query_concept"
            )

        from q_orca.compiler.concept_gram_mps import compute_concept_gram_mps

        return compute_concept_gram_mps(
            machine, concept_action_label="prepare_concept"
        )

    def tier_separation(self) -> float | None:
        """Return ``concept_gram_tier_separation`` for the current dictionary.

        Wraps ``q_orca.compiler.concept_gram_hea.compute_tier_separation``;
        the result is ``None`` when every cluster is a singleton (matching
        the helper's contract). Defined for both encodings — the cluster
        labels come from each ``Feature.cluster`` regardless.
        """
        from q_orca.compiler.concept_gram_hea import compute_tier_separation

        return compute_tier_separation(
            self.gram(), [f.cluster for f in self.features]
        )

    @classmethod
    def with_default_angles(
        cls,
        name: str,
        hierarchy: dict[str, list[str]],
        encoding: MPSRung1 | HEA_Rung2 | None = None,
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
