"""Polygram — quantum interference laboratory for polysemantic feature dictionaries."""

from polygram.batch import BatchExperiment, BatchResults, BatchRun
from polygram.behavioural import (
    BehaviouralValidator,
    BucketStats,
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)
from polygram.cancellation import Cancellation, CancellationResult
from polygram.compression import (
    ClusterPlan,
    CompressionPlan,
    CompressionReport,
    CompressionResult,
    Compressor,
    EpochCompressor,
    EpochIteration,
    EpochReport,
    EpochResult,
    Panel,
    RegrowPlan,
    RegrowReport,
    RegrowResult,
    RegrowStrategy,
    Regrower,
    SlotPopulation,
)
from polygram.dictionary import Dictionary, Feature
from polygram.emit import write_qorca
from polygram.encoding import HEA_Rung2, MPSRung1, Rung3, Rung3State
from polygram.experiment import Experiment, ExperimentResult, InterferenceSweep
from polygram.sae_import import (
    SAEFeatureRecord,
    SelectionReport,
    from_sae_lens,
    load_sae_safetensors,
    load_toy_sae,
)

__version__ = "0.0.1"

__all__ = [
    "BatchExperiment",
    "BatchResults",
    "BatchRun",
    "BehaviouralValidator",
    "BucketStats",
    "CandidatePair",
    "Cancellation",
    "CancellationResult",
    "ClusterPlan",
    "CompressionPlan",
    "CompressionReport",
    "CompressionResult",
    "Compressor",
    "Dictionary",
    "EpochCompressor",
    "EpochIteration",
    "EpochReport",
    "EpochResult",
    "Experiment",
    "ExperimentResult",
    "Feature",
    "HEA_Rung2",
    "InterferenceSweep",
    "MPSRung1",
    "Panel",
    "RegrowPlan",
    "RegrowReport",
    "RegrowResult",
    "RegrowStrategy",
    "Regrower",
    "Rung3",
    "Rung3State",
    "SAEFeatureRecord",
    "SelectionReport",
    "SlotPopulation",
    "ValidationReport",
    "ValidationSummary",
    "from_sae_lens",
    "load_sae_safetensors",
    "load_toy_sae",
    "write_qorca",
]
