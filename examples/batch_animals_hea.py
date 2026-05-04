"""Batch-experiment walk-through on the Animals HEA dictionary.

Pipeline:

1. Build the same Animals dictionary as ``examples/animals_hea.py``
   (rung-2 HEA, depth-2, four animals across two clusters).
2. Run ``polygram.analysis.predict_cancellation_depth`` to get the
   closed-form per-pair `(M, V, structural_floor, cancellation_gap)`.
3. Build a separation `FeatureGraph` via
   ``build_separation_graph(prediction, threshold=0.0,
   include_within_cluster=True)``. Threshold 0.0 keeps every pair so
   the example exercises the full `top_k` budget; the Animals fixture
   is small.
4. Run ``BatchExperiment(top_k=4, knobs="cluster_shared")`` against
   that graph and the dictionary, materializing per-pair artifacts
   under ``output_dir/batch_animals_hea/<source>_x_<target>/`` and
   writing the aggregated ``batch_results.json`` at the top level.

Output layout under ``output_dir / "batch_animals_hea"``:

- ``input_separation_graph.json``  — the source FeatureGraph
- ``batch_results.json``           — aggregated BatchResults
- ``<source>_x_<target>/``         — per-pair Cancellation bundle
  (``.q.orca.md``, ``_summary.md``, ``_trajectory.csv``)

Prediction-vs-observation comparison: the user can read
``batch_results.json`` and compare each ``run.predicted_floor`` /
``predicted_gap`` (closed-form, from the input graph) against
``achieved_overlap`` / ``cancellation_efficiency`` (empirical, from
the per-pair Cancellation). Values near 1.0 of `cancellation_efficiency`
mean φ/θ search realized the predicted gap; values near 0.0 mean the
prediction was right that there was nothing to find (separation kind)
or the search got stuck (sharing kind).
"""

from __future__ import annotations

from pathlib import Path

from polygram import BatchExperiment
from polygram.analysis import build_separation_graph, triage_dictionary

from examples.animals_hea import build_dictionary


def main(output_dir: str | Path = "examples/output") -> None:
    out_dir = Path(output_dir) / "batch_animals_hea"
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary = build_dictionary()
    prediction = triage_dictionary(dictionary)
    graph = build_separation_graph(
        prediction, threshold=0.0, include_within_cluster=True
    )
    (out_dir / "input_separation_graph.json").write_text(graph.to_json())

    experiment = BatchExperiment(
        feature_graph=graph,
        dictionary=dictionary,
        top_k=4,
        knobs="cluster_shared",
        output_dir=out_dir,
    )
    results = experiment.run()
    print(f"wrote: {out_dir / 'batch_results.json'}")
    print(f"runs: {len(results.runs)}")
    for r in results.runs:
        eff = (
            f"{r.cancellation_efficiency:.4f}"
            if r.cancellation_efficiency is not None
            else "n/a"
        )
        print(
            f"  {r.source} x {r.target}: "
            f"predicted_floor={r.predicted_floor:.4f} "
            f"current={r.current_overlap:.4f} "
            f"achieved={r.achieved_overlap:.4f} "
            f"efficiency={eff}"
        )


if __name__ == "__main__":
    main()
