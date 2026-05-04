"""Analytical triage layer — score real SAE feature subsets *before*
any quantum encoding or simulation, using only the rung-1 closed-form
Gram.

Useful when an SAE has 16k+ features and a researcher needs to pick
4–8 promising candidates for a `Cancellation` / `InterferenceSweep`
experiment without simulating every subset.
"""

from polygram.analysis.triage import (
    KNOB_SELECTION_GUIDANCE,
    PairPrediction,
    SUITABILITY_FORMULA,
    TriagePrediction,
    encoding_suitability_score,
    feature_sensitivity,
    predict_cancellation_depth,
    render_report,
)

__all__ = [
    "KNOB_SELECTION_GUIDANCE",
    "PairPrediction",
    "SUITABILITY_FORMULA",
    "TriagePrediction",
    "encoding_suitability_score",
    "feature_sensitivity",
    "predict_cancellation_depth",
    "render_report",
]
