"""Polygram behavioural validator — read-only four-constraint pipeline.

The `BehaviouralValidator` runs Polygram's predicted concept Gram, the
co-firing Jaccard gate, the per-feature ablation-KL probe, and the
layer ≥ 1 hook constraint against an SAE feature panel and emits a
structured `ValidationReport`. Two-stage API: `predict()` is cheap and
torch-free; `validate()` lazy-imports torch + transformers and pays
the model-bound cost; `run()` is `validate(predict())`.

See `openspec/changes/add-behavioural-validator-loop/` for the spec.
"""

from polygram.behavioural.report import (
    BucketStats,
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)
from polygram.behavioural.validator import BehaviouralValidator

__all__ = [
    "BehaviouralValidator",
    "BucketStats",
    "CandidatePair",
    "ValidationReport",
    "ValidationSummary",
]
