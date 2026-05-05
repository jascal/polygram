"""Compression strategies — modules that take a checkpoint state-dict
and a `CompressionPlan` and return a rewritten state-dict.

`zero` is the only strategy in the initial release. `merge` (decoder
centroid) is deferred to a follow-up change.
"""

from polygram.compression.strategies.zero import apply_zero

__all__ = ["apply_zero"]
