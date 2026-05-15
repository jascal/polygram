"""Polygram — laboratory for modelling polysemantic feature dictionaries.

Public surface (re-exported from this module):

- **Core**: ``Dictionary``, ``Feature``, ``MPSRung1``, ``HEA_Rung2``,
  ``Rung3``, ``Experiment``, ``Cancellation``, ``InterferenceSweep``.
- **SAE bridge**: ``from_sae_lens``, ``load_sae_safetensors``,
  ``load_toy_sae``, ``SAEFeatureRecord``, ``SelectionReport``.
- **Geometric profiles** (v0.2+): ``clustered``, ``uniform_sphere``,
  ``GeometricProfile``, ``register_profile``, ``get_profile``,
  ``available_profiles``. See ``polygram.geometry`` and
  ``docs/research/sae-geometry-regimes.md``.
- **Behavioural / compression**: ``BehaviouralValidator``,
  ``ValidationReport``, ``Compressor``, ``Regrower``,
  ``EpochCompressor``.
- **Configuration**: ``CancellationConfig``, ``CompressionConfig``,
  ``EpochCompressionConfig``, ``RegrowConfig``, ``SAEImportConfig``,
  ``ValidationConfig``.
"""

from polygram.batch import BatchExperiment, BatchResults, BatchRun
from polygram.config import (
    CancellationConfig,
    CompressionConfig,
    EpochCompressionConfig,
    RegrowConfig,
    SAEImportConfig,
    ValidationConfig,
)
from polygram.confirmation import ClusterConfirmer, Confirmer, DecoderGeometryConfirmer
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
    ParetoOutcome,
    ParetoReport,
    RegrowPlan,
    RegrowReport,
    RegrowResult,
    RegrowStrategy,
    Regrower,
    SlotPopulation,
)
from polygram.dictionary import Dictionary, Feature
from polygram.emit import write_qorca
from polygram.encoding import HEA_Rung2, MPSRung1, Rung3, Rung3State, Rung4, Rung4State
from polygram.experiment import Experiment, ExperimentResult, InterferenceSweep
from polygram.geometry import (
    GeometricProfile,
    available_profiles,
    clustered,
    get_profile,
    register_profile,
    uniform_sphere,
)
from polygram.sae_import import (
    SAEFeatureRecord,
    SelectionReport,
    from_sae_lens,
    load_sae_safetensors,
    load_toy_sae,
)

__version__ = "0.5.0"

__all__ = [
    "BatchExperiment",
    "BatchResults",
    "BatchRun",
    "BehaviouralValidator",
    "BucketStats",
    "CancellationConfig",
    "CompressionConfig",
    "EpochCompressionConfig",
    "RegrowConfig",
    "SAEImportConfig",
    "ValidationConfig",
    "ClusterConfirmer",
    "Confirmer",
    "DecoderGeometryConfirmer",
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
    "GeometricProfile",
    "HEA_Rung2",
    "InterferenceSweep",
    "MPSRung1",
    "Panel",
    "ParetoOutcome",
    "ParetoReport",
    "RegrowPlan",
    "RegrowReport",
    "RegrowResult",
    "RegrowStrategy",
    "Regrower",
    "Rung3",
    "Rung3State",
    "Rung4",
    "Rung4State",
    "SAEFeatureRecord",
    "SelectionReport",
    "SlotPopulation",
    "ValidationReport",
    "ValidationSummary",
    "available_profiles",
    "clustered",
    "from_sae_lens",
    "get_profile",
    "load_sae_safetensors",
    "load_toy_sae",
    "register_profile",
    "uniform_sphere",
    "write_qorca",
]
