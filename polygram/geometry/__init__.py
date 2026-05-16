"""Named geometric-regime profiles for SAE imports.

A `GeometricProfile` bundles a `KnobAssignment` strategy + a
`GeometricFidelity` metric + recommended `from_sae_lens` defaults
(`n_clusters`, `gamma_range`). Two built-in profiles ship:

- `clustered`: small dense LM SAEs at GPT-2-small scale.
- `uniform-sphere`: SAEs with `d_model ≥ ~1K`, `n_features ≥ ~16K`,
  empirically validated on audio + large-LM SAEs.

Third parties (sae-forge etc.) can register custom profiles via
`register_profile`. See `docs/research/sae-geometry-regimes.md`.
"""

from polygram.geometry.clustered import (
    ClusteredKnobAssignment,
    TierPreservationFidelity,
    clustered,
)
from polygram.geometry.learned_axis_assignment import LearnedKnobAssignment
from polygram.geometry.profile import GeometricProfile
from polygram.geometry.objectives import (
    behavioural_objective,
    pearson_objective,
    spearman_objective,
)
from polygram.geometry.protocols import (
    GeometricFidelity,
    KnobAssignment,
    KnobAssignmentResult,
    LearnedAxisObjective,
)
from polygram.geometry.registry import (
    available_profiles,
    get_profile,
    register_profile,
)
from polygram.geometry.uniform_sphere import (
    RankRecallAtKFidelity,
    UniformSphereKnobAssignment,
    uniform_sphere,
)

# Register built-ins at import time.
register_profile(clustered())
register_profile(uniform_sphere())

__all__ = [
    "ClusteredKnobAssignment",
    "GeometricFidelity",
    "GeometricProfile",
    "KnobAssignment",
    "KnobAssignmentResult",
    "LearnedAxisObjective",
    "LearnedKnobAssignment",
    "RankRecallAtKFidelity",
    "TierPreservationFidelity",
    "UniformSphereKnobAssignment",
    "available_profiles",
    "behavioural_objective",
    "clustered",
    "get_profile",
    "pearson_objective",
    "register_profile",
    "spearman_objective",
    "uniform_sphere",
]
