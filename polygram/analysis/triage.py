"""Pure-classical triage primitives for SAE feature subsets.

Given a small (≤8) selection of SAE features, this module predicts the
rung-1 cancellation potential of every pair using only the analytic
closed-form Gram (no quantum simulation, no Q-Orca runtime). The
underlying decomposition is

    |⟨A|B⟩|²(δ) = M + V·cos(δ)         where δ = φ_A − φ_B

so once we evaluate the squared overlap at two anchor points (δ = 0
and δ = π) we recover

    M = (m_zero + m_pi) / 2
    V = (m_zero − m_pi) / 2     (signed; structural floor uses |V|)
    structural_floor = min(m_zero, m_pi) = M − |V|
    cancellation_gap = current_overlap − structural_floor

Per-feature `feature_sensitivity[i]` is the average `|V_ij|` across all
pairs containing `i`, i.e. the average swing magnitude that feature
contributes to its pairwise overlaps. The composite
`encoding_suitability_score` is documented in `SUITABILITY_FORMULA`.

All assumptions inherit from `polygram.from_sae_lens`: rung-1 MPS
encoding with `α = 0`, `φ = 0` defaults, β/γ derived from cluster /
PCA on the selected projection vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from polygram.dictionary import Dictionary
from polygram.sae_import import (
    SAEFeatureRecord,
    SelectionReport,
    from_sae_lens,
)

SUITABILITY_FORMULA = (
    "encoding_suitability_score = mean_cancellation_gap × min_pairwise_separation\n"
    "  mean_cancellation_gap   = mean over pairs of "
    "(current_overlap − structural_floor)\n"
    "  min_pairwise_separation = 1 − max over pairs of current_overlap\n"
    "Intuition: rewards subsets that have phase headroom to drive overlaps "
    "down (large gaps) AND don't have any near-coincident pair before phase "
    "tuning (clean separation). Both factors live in [0, 1], so the score "
    "lives in [0, 1] too. Higher is better."
)

KNOB_SELECTION_GUIDANCE = (
    "**Default knob.** Start with a single `<feature>.phi` per target "
    "feature (`Cancellation`'s default). On `MPSRung1` this is the "
    "final `Rz(qs[1], phi)` — the highest-leverage axis. On "
    "`HEA_Rung2` it is also the cleanest single axis (`phi` factors "
    "out of the θ tensor regardless of depth).\n"
    "\n"
    "**Multi-feature binding.** When several features should be tuned "
    "coherently within a cluster, prefer the cluster-shared grammar "
    "(`<cluster>.phi`, `<cluster>.theta[r,d,q]`) over a list of "
    "per-feature paths. Bit-for-bit Gram preservation holds for "
    "`MPSRung1 <cluster>.phi` (final-Rz factorization). HEA "
    "cluster-shared paths ship as a search-space dimensionality "
    "reduction (one axis per cluster, bounding optimizer leverage), "
    "not an algebraic invariant.\n"
    "\n"
    "**Cluster-shatterer hazard.** Per-feature θ knobs on diverse-"
    "sibling HEA fixtures can drive a target pair to ≈0 while "
    "shattering the cluster ordering. Empirically: a 4-θ Ry knob "
    "set drove `(dog_poodle, bird_hawk)` overlap to ≈0 while "
    "inverting `(dog_poodle, dog_beagle)` from 0.9999 → 0.5735 and "
    "`tier_separation` from +0.22 to −0.20. Only reach for "
    "per-feature θ after verifying tier separation will not invert.\n"
    "\n"
    "**HEA Pauli leverage.** `Rz` at depth 0 has zero leverage on "
    "`|0⟩` initial states (`Rz |0⟩` is a global phase). `Rz` at "
    "later depths takes effect only after entanglers rotate states "
    "off the Z basis; leverage is depth- and entangler-dependent. "
    "`Ry` has across-the-board leverage but is the cluster-shatterer "
    "above. There is no defensible *per-feature* safe HEA knob; the "
    "principled answer is the cluster-shared grammar.\n"
    "\n"
    "**Structural floor.** Pure-φ search bounds the squared overlap "
    "to [M − |V|, M + |V|]. The floor at `M − |V|` cannot be pierced "
    "by any φ tuning. To reduce overlap below the floor, β/α/γ "
    "adjustment or a richer encoding is needed (deferred — see "
    "`docs/research/cancellation-phase-floor.md`).\n"
    "\n"
    "**Sensitivity ranking.** No `suggest_safe_knobs` helper is "
    "provided; sorting features by `feature_sensitivity` is two "
    "lines at the call site:\n"
    "\n"
    "    top = sorted(prediction.feature_sensitivity.items(),\n"
    "                 key=lambda kv: kv[1], reverse=True)[:n]"
)


@dataclass(frozen=True)
class PairPrediction:
    """Per-pair prediction of the rung-1 phase-only cancellation profile."""

    feature_a: str
    feature_b: str
    cluster_a: str
    cluster_b: str
    current_overlap: float  # |⟨A|B⟩|² at φ_A = φ_B = 0  (= m_zero)
    m_pi: float             # |⟨A|B⟩|² at φ_A = 0, φ_B = π
    M: float                # (m_zero + m_pi) / 2
    V: float                # (m_zero − m_pi) / 2 (signed)
    structural_floor: float
    cancellation_gap: float

    @property
    def is_cross_cluster(self) -> bool:
        return self.cluster_a != self.cluster_b


@dataclass(frozen=True)
class TriagePrediction:
    """Bundle of per-pair predictions plus aggregate diagnostics."""

    dictionary: Dictionary
    selection_report: SelectionReport
    pairs: list[PairPrediction]
    feature_sensitivity: dict[str, float]
    encoding_suitability_score: float
    suitability_formula: str = SUITABILITY_FORMULA


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def predict_cancellation_depth(
    sae_dict: dict[int, SAEFeatureRecord],
    feature_ids: list[int],
    **from_sae_lens_kwargs: Any,
) -> TriagePrediction:
    """Predict per-pair `(M, V, structural_floor, cancellation_gap)`
    plus per-feature sensitivity and an aggregate suitability score.

    No quantum simulation: builds the rung-1 Dictionary via
    `from_sae_lens` and evaluates the analytic Gram at exactly
    `n_features + 1` phase configurations (one all-zero baseline plus
    one with φ_i = π per feature). Each `PairPrediction.m_pi` is read
    off the row of the configuration where one of its endpoints is
    flipped to π — both endpoints give the same value because
    `cos(π) = cos(−π)`.
    """
    dictionary, report = from_sae_lens(
        sae_dict, feature_ids, **from_sae_lens_kwargs
    )
    pairs = _predict_pairs_from_dictionary(dictionary)
    sensitivity = _feature_sensitivity_from_pairs(dictionary, pairs)
    score = _suitability_score_from_pairs(pairs)
    return TriagePrediction(
        dictionary=dictionary,
        selection_report=report,
        pairs=pairs,
        feature_sensitivity=sensitivity,
        encoding_suitability_score=score,
    )


def feature_sensitivity(
    sae_dict: dict[int, SAEFeatureRecord],
    feature_ids: list[int],
    **from_sae_lens_kwargs: Any,
) -> dict[str, float]:
    """For each selected feature, return the mean `|V_ij|` across pairs
    containing it (average swing magnitude this feature contributes to
    its pairwise overlaps under the rung-1 projection)."""
    return predict_cancellation_depth(
        sae_dict, feature_ids, **from_sae_lens_kwargs
    ).feature_sensitivity


def encoding_suitability_score(
    sae_dict: dict[int, SAEFeatureRecord],
    feature_ids: list[int],
    **from_sae_lens_kwargs: Any,
) -> float:
    """Aggregate scalar score (formula in `SUITABILITY_FORMULA`)."""
    return predict_cancellation_depth(
        sae_dict, feature_ids, **from_sae_lens_kwargs
    ).encoding_suitability_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _predict_pairs_from_dictionary(
    dictionary: Dictionary,
) -> list[PairPrediction]:
    n = len(dictionary.features)
    g_zero = np.abs(dictionary.gram()) ** 2

    m_pi = np.zeros((n, n), dtype=float)
    for i, feat in enumerate(dictionary.features):
        flipped = dictionary.with_phi(feat.name, float(np.pi))
        gi = np.abs(flipped.gram()) ** 2
        for j in range(n):
            if j == i:
                continue
            # |⟨i|j⟩|² with φ_i=π, φ_j=0 ⇒ δ_ij = π. Symmetric in (i,j).
            m_pi[i, j] = gi[i, j]
            m_pi[j, i] = gi[i, j]

    pairs: list[PairPrediction] = []
    cluster_of = {f.name: f.cluster for f in dictionary.features}
    for i in range(n):
        for j in range(i + 1, n):
            mz = float(g_zero[i, j])
            mp = float(m_pi[i, j])
            big_m = 0.5 * (mz + mp)
            big_v = 0.5 * (mz - mp)
            floor = min(mz, mp)
            pairs.append(
                PairPrediction(
                    feature_a=dictionary.features[i].name,
                    feature_b=dictionary.features[j].name,
                    cluster_a=cluster_of[dictionary.features[i].name],
                    cluster_b=cluster_of[dictionary.features[j].name],
                    current_overlap=mz,
                    m_pi=mp,
                    M=big_m,
                    V=big_v,
                    structural_floor=floor,
                    cancellation_gap=mz - floor,
                )
            )
    return pairs


def _feature_sensitivity_from_pairs(
    dictionary: Dictionary, pairs: list[PairPrediction]
) -> dict[str, float]:
    sums = {f.name: 0.0 for f in dictionary.features}
    counts = {f.name: 0 for f in dictionary.features}
    for p in pairs:
        sums[p.feature_a] += abs(p.V)
        sums[p.feature_b] += abs(p.V)
        counts[p.feature_a] += 1
        counts[p.feature_b] += 1
    return {
        name: (sums[name] / counts[name] if counts[name] else 0.0)
        for name in sums
    }


def _suitability_score_from_pairs(pairs: list[PairPrediction]) -> float:
    if not pairs:
        return 0.0
    mean_gap = float(np.mean([p.cancellation_gap for p in pairs]))
    max_overlap = float(np.max([p.current_overlap for p in pairs]))
    separation = 1.0 - max_overlap
    return float(np.clip(mean_gap * separation, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_report(
    prediction: TriagePrediction,
    *,
    sae_path: str | None = None,
    feature_ids: list[int] | None = None,
) -> str:
    """Render a Markdown triage report for a `TriagePrediction`."""
    d = prediction.dictionary
    report = prediction.selection_report
    lines: list[str] = []
    lines.append(f"# Polygram analysis — {d.name}")
    lines.append("")
    if sae_path is not None:
        lines.append(f"- SAE source: `{sae_path}`")
    if feature_ids is not None:
        lines.append(f"- Selected feature_ids: {feature_ids}")
    lines.append(f"- Cluster method: `{report.cluster_method}`")
    lines.append(
        f"- β variance explained: {report.beta_variance_explained:.4f}"
    )
    if report.tier_preservation is not None:
        lines.append(
            f"- Projection→Polygram tier correlation: "
            f"{report.tier_preservation:.4f}"
        )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "All predictions assume the rung-1 MPS encoding with `α = 0`, "
        "`φ = 0` defaults from `polygram.from_sae_lens`. β values come "
        "from cluster centroids; γ defaults to 0 unless `assign_gamma=True` "
        "was used. Re-running with different encoding choices will shift "
        "M and V; the structural floor is invariant under uniform-φ shifts."
    )
    lines.append("")

    lines.append("## Pair predictions")
    lines.append("")
    lines.append(
        "| pair | tier | current | floor | gap | M | V |"
    )
    lines.append("|------|------|--------:|------:|----:|--:|--:|")
    for p in sorted(
        prediction.pairs,
        key=lambda x: x.cancellation_gap,
        reverse=True,
    ):
        tier = "cross" if p.is_cross_cluster else "intra"
        lines.append(
            f"| {p.feature_a} ↔ {p.feature_b} | {tier} | "
            f"{p.current_overlap:.4f} | {p.structural_floor:.4f} | "
            f"{p.cancellation_gap:.4f} | {p.M:.4f} | {p.V:+.4f} |"
        )
    lines.append("")

    lines.append("## Per-feature sensitivity (mean |V_ij|)")
    lines.append("")
    lines.append("| feature | cluster | sensitivity |")
    lines.append("|---------|---------|------------:|")
    cluster_of = {f.name: f.cluster for f in d.features}
    for name, sens in sorted(
        prediction.feature_sensitivity.items(),
        key=lambda kv: kv[1],
        reverse=True,
    ):
        lines.append(f"| {name} | {cluster_of[name]} | {sens:.4f} |")
    lines.append("")

    lines.append("## Choosing knobs")
    lines.append("")
    lines.append(KNOB_SELECTION_GUIDANCE)
    lines.append("")

    lines.append("## Encoding suitability")
    lines.append("")
    lines.append(
        f"**Score: {prediction.encoding_suitability_score:.4f}**  (range [0, 1])"
    )
    lines.append("")
    lines.append("```")
    lines.append(prediction.suitability_formula)
    lines.append("```")
    lines.append("")

    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)
