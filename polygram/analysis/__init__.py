"""Analytical triage layer — score real SAE feature subsets *before*
any quantum encoding or simulation, using only the rung-1 closed-form
Gram.

Useful when an SAE has 16k+ features and a researcher needs to pick
4–8 promising candidates for a `Cancellation` / `InterferenceSweep`
experiment without simulating every subset.
"""

from polygram.analysis.feature_graph import (
    FLOOR_BLOCK,
    SEPARATION_EDGE_FORMULA,
    SHARING_EDGE_FORMULA,
    FeatureEdge,
    FeatureGraph,
    build_separation_graph,
    build_sharing_graph,
    render_feature_graph_section,
)
from polygram.analysis.triage import (
    KNOB_SELECTION_GUIDANCE,
    PairPrediction,
    SUITABILITY_FORMULA,
    TriagePrediction,
    encoding_suitability_score,
    feature_sensitivity,
    predict_cancellation_depth,
    render_report,
    triage_dictionary,
)

__all__ = [
    "FLOOR_BLOCK",
    "FeatureEdge",
    "FeatureGraph",
    "KNOB_SELECTION_GUIDANCE",
    "PairPrediction",
    "SEPARATION_EDGE_FORMULA",
    "SHARING_EDGE_FORMULA",
    "SUITABILITY_FORMULA",
    "TriagePrediction",
    "build_separation_graph",
    "build_sharing_graph",
    "encoding_suitability_score",
    "feature_sensitivity",
    "predict_cancellation_depth",
    "render_feature_graph_section",
    "render_report",
    "triage_dictionary",
]
