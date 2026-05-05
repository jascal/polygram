"""Polygram compression action — the loop's downstream half.

Consumes a `ValidationReport`'s `confirmed` candidate-pair list,
collapses it to redundancy clusters via union-find, picks one
representative per cluster, and rewrites an SAE checkpoint so the
non-representatives are zeroed.

Two-stage API on `Compressor`:

    plan() -> CompressionPlan         # cheap, no I/O
    apply(plan, output_checkpoint=)   # writes one new safetensors
        -> CompressionResult
    run(output_checkpoint=)           # apply(plan(), ...)

Initial release ships only the `zero` strategy; `merge` (decoder
centroid) is deferred to its own change. See
`openspec/changes/add-compression-action/` for the spec.

The `Regrower` primitive (see
`openspec/changes/add-compression-regrow/`) repopulates zeroed slots
with new directions extracted from the SAE's activation residuals.
Two-stage API mirrors `Compressor`'s.
"""

from polygram.compression.compressor import Compressor
from polygram.compression.epoch import EpochCompressor
from polygram.compression.epoch_report import (
    EpochIteration,
    EpochReport,
    EpochResult,
    Panel,
)
from polygram.compression.regrow import Regrower, RegrowStrategy
from polygram.compression.regrow_report import (
    RegrowPlan,
    RegrowReport,
    RegrowResult,
    SlotPopulation,
)
from polygram.compression.report import (
    ClusterPlan,
    CompressionPlan,
    CompressionReport,
    CompressionResult,
)

__all__ = [
    "ClusterPlan",
    "Compressor",
    "CompressionPlan",
    "CompressionReport",
    "CompressionResult",
    "EpochCompressor",
    "EpochIteration",
    "EpochReport",
    "EpochResult",
    "Panel",
    "RegrowPlan",
    "RegrowReport",
    "RegrowResult",
    "RegrowStrategy",
    "Regrower",
    "SlotPopulation",
]
