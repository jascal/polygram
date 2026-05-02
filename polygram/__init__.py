"""Polygram — quantum interference laboratory for polysemantic feature dictionaries."""

from polygram.cancellation import Cancellation, CancellationResult
from polygram.dictionary import Dictionary, Feature
from polygram.emit import write_qorca
from polygram.encoding import MPSRung1
from polygram.experiment import Experiment, ExperimentResult, InterferenceSweep
from polygram.sae_import import (
    SAEFeatureRecord,
    SelectionReport,
    from_sae_lens,
    load_toy_sae,
)

__version__ = "0.0.1"

__all__ = [
    "Cancellation",
    "CancellationResult",
    "Dictionary",
    "Experiment",
    "ExperimentResult",
    "Feature",
    "InterferenceSweep",
    "MPSRung1",
    "SAEFeatureRecord",
    "SelectionReport",
    "from_sae_lens",
    "load_toy_sae",
    "write_qorca",
]
