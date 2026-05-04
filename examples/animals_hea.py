"""Animals HEA example — Polygram emits a rung-2 HEA dictionary.

Same Animals shape as ``animals_interference.py``, but constructed with
``encoding=HEA_Rung2(depth=2)``. The default ``(α, β, γ, φ)`` knobs are
laid across the first HEA layer; siblings within a cluster pick up
small, near-identical rotations, and the two clusters are pulled apart
by a magnitude shift on β.

The ``main()`` walk emits, verifies, and then exercises both
sweep/cancellation primitives end-to-end:

1. Emit ``AnimalsHea.q.orca.md`` and verify it (Stage 4b green,
   including the declared ``concept_gram_tier_separation >= 0.025``
   invariant).
2. Run an ``InterferenceSweep`` over a single ``dog_poodle.phi`` axis
   (5 points in this coarsened example), materialize the result, and
   assert ``concept_gram_tier_separation_bound_holds`` across every
   sweep point.
3. Run a ``Cancellation`` on the ``(dog_poodle, bird_hawk)`` pair with
   the default 2-φ knobs and ``method="grid"``, materialize the
   ``.q.orca.md``, ``trajectory.csv``, ``summary.md``, and the
   ``before/after`` figure.
4. Run a *cluster-shared* ``Cancellation`` on the same pair with
   ``knobs=["dogs.theta[0,0,0]", "birds.theta[0,0,0]"]`` — one search
   axis per cluster, applied to every member of that cluster. Within-
   cluster Gram entries are preserved exactly by unitarity.
5. Print a small tier-separation rollup and a comparison row across
   both cancellation runs (target overlap, worst sibling overlap,
   tier-separation — before/after).

Output layout under ``output_dir / "animals_hea"``:

- ``AnimalsHea.q.orca.md`` — the emitted HEA dictionary at default knobs
- ``sweep/`` — InterferenceSweep artifacts
- ``cancellation/`` — per-feature 2-φ Cancellation artifacts
- ``cancellation/cluster_shared/`` — cluster-shared θ Cancellation
  artifacts (incl. ``before_after.png``)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polygram import (
    Cancellation,
    Dictionary,
    Experiment,
    Feature,
    HEA_Rung2,
    write_qorca,
)


def build_dictionary() -> Dictionary:
    return Dictionary(
        name="AnimalsHea",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.50, alpha=0.05, gamma=0.02),
            Feature("dog_beagle", "dogs", beta=-0.48, alpha=0.04, gamma=0.03),
            Feature("bird_hawk", "birds", beta=0.50, alpha=-0.04, gamma=0.02),
            Feature("bird_sparrow", "birds", beta=0.52, alpha=-0.03, gamma=0.01),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=HEA_Rung2(depth=2),
    )


def main(output_dir: str | Path = "examples/output") -> None:
    from q_orca.parser.markdown_parser import parse_q_orca_markdown
    from q_orca.verifier import VerifyOptions, verify

    out_dir = Path(output_dir) / "animals_hea"
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary = build_dictionary()
    out_path = out_dir / "AnimalsHea.q.orca.md"
    write_qorca(dictionary, out_path)

    parsed = parse_q_orca_markdown(out_path.read_text())
    if parsed.errors:
        raise SystemExit(f"parse errors: {parsed.errors}")
    machine = parsed.file.machines[0]

    result = verify(machine, VerifyOptions(skip_resource_bounds=True))
    forbidden = {
        "HEA_GRAM_INVALID",
        "HEA_TIER_INVARIANT_VIOLATED",
        "HEA_TIER_UNDEFINED",
    }
    offenders = [e for e in result.errors if e.code in forbidden]
    assert result.valid, [(e.code, e.message) for e in result.errors]
    assert not offenders, [(e.code, e.message) for e in offenders]

    sep = dictionary.tier_separation()
    print(f"emitted: {out_path}")
    print(f"encoding: {dictionary.encoding}")
    print(f"tier_separation: {sep:.4f} (declared bound: "
          f"{dictionary.encoding.tier_separation_bound})")
    print("verify.valid: True")

    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(
        name="AnimalsHeaSweep",
        dictionary=dictionary,
        target_pair=("dog_poodle", "bird_hawk"),
        sweep={"dog_poodle.phi": np.linspace(0.0, np.pi / 6, 5)},
        measures=["overlap", "gram_matrix"],
        assertions=[
            "hierarchical_ordering_preserved",
            "concept_gram_tier_separation_bound_holds",
        ],
    )
    experiment.materialize(sweep_dir)
    sweep_result = experiment.run()
    sweep_result.save(sweep_dir / "AnimalsHeaSweep_result.npz")
    sweep_result.to_csv(sweep_dir / "AnimalsHeaSweep_result.csv")
    sweep_result.write_summary(sweep_dir / "AnimalsHeaSweep_summary.md")
    bound_pass = sweep_result.assertion_pass[
        "concept_gram_tier_separation_bound_holds"
    ]
    assert bound_pass.all(), bound_pass

    canc_dir = out_dir / "cancellation"
    canc_dir.mkdir(parents=True, exist_ok=True)
    cancellation = Cancellation(
        dictionary=dictionary,
        target_pair=("dog_poodle", "bird_hawk"),
        optimize={"method": "grid", "max_steps": 12},
    )
    canc_result = cancellation.run()
    canc_result.materialize(canc_dir)
    try:
        canc_result.plot(canc_dir / "before_after.png", kind="before_after")
    except ImportError:
        # matplotlib is optional; example still completes without the figure.
        pass

    cluster_dir = canc_dir / "cluster_shared"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    cluster_cancellation = Cancellation(
        dictionary=dictionary,
        target_pair=("dog_poodle", "bird_hawk"),
        knobs=["dogs.theta[0,0,0]", "birds.theta[0,0,0]"],
        optimize={"method": "grid", "max_steps": 8},
    )
    cluster_result = cluster_cancellation.run()
    cluster_result.materialize(cluster_dir)
    try:
        cluster_result.plot(
            cluster_dir / "before_after.png", kind="before_after"
        )
    except ImportError:
        pass

    print(
        f"sweep tier_separation: min={float(sweep_result.tier_separation.min()):.4f} "
        f"max={float(sweep_result.tier_separation.max()):.4f} "
        f"(bound: {dictionary.encoding.tier_separation_bound})"
    )
    print(
        f"cancellation: before={canc_result.before_overlap:.4f} → "
        f"after={canc_result.after_overlap:.4f} "
        f"(method={canc_result.method!r}, "
        f"feasible={canc_result.feasible_count}/"
        f"{canc_result.trajectory.shape[0]})"
    )
    print(f"optimized knobs: {canc_result.optimized_knobs}")
    _print_comparison_row("per-feature 2-φ", dictionary, canc_result)
    _print_comparison_row(
        "cluster-shared θ", dictionary, cluster_result
    )


def _print_comparison_row(
    label: str, original: Dictionary, result
) -> None:
    """Print one comparison row: target / worst-sibling / tier-sep."""
    a, b = result.target_pair
    a_idx = original.feature_index(a)
    b_idx = original.feature_index(b)
    before_target = float(np.abs(result.before_gram[a_idx, b_idx]) ** 2)
    after_target = float(np.abs(result.after_gram[a_idx, b_idx]) ** 2)

    def _worst_sibling(mat: np.ndarray) -> float:
        """Smallest within-cluster overlap across all clusters of size ≥ 2."""
        worst = 1.0
        sq = np.abs(mat) ** 2
        for cluster, members in original.hierarchy.items():
            if len(members) < 2:
                continue
            for i, m_i in enumerate(members):
                for m_j in members[i + 1 :]:
                    idx_i = original.feature_index(m_i)
                    idx_j = original.feature_index(m_j)
                    worst = min(worst, float(sq[idx_i, idx_j]))
        return worst

    before_sibling = _worst_sibling(result.before_gram)
    after_sibling = _worst_sibling(result.after_gram)
    before_sep = original.tier_separation()
    after_sep = result.dictionary_at_optimum.tier_separation()
    sep_str = (
        f"{before_sep:+.4f}→{after_sep:+.4f}"
        if before_sep is not None and after_sep is not None
        else "n/a"
    )
    print(
        f"  [{label}] target {before_target:.4f}→{after_target:.4f}  "
        f"worst-sibling {before_sibling:.4f}→{after_sibling:.4f}  "
        f"tier-sep {sep_str}"
    )


if __name__ == "__main__":
    main()
