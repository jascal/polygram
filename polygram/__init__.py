"""Polygram — quantum interference laboratory for polysemantic feature dictionaries."""

from polygram.dictionary import Dictionary, Feature
from polygram.emit import write_qorca
from polygram.encoding import MPSRung1
from polygram.experiment import Experiment, ExperimentResult, InterferenceSweep

__version__ = "0.0.1"

__all__ = [
    "Dictionary",
    "Experiment",
    "ExperimentResult",
    "Feature",
    "InterferenceSweep",
    "MPSRung1",
    "write_qorca",
]
