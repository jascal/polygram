"""Confirmation strategies — produce a ``ValidationReport`` with confirmed
redundant feature pairs via geometry rather than forward passes.

Three concrete strategies (all satisfy the ``Confirmer`` protocol):

- :class:`DecoderGeometryConfirmer` — pairs whose decoder cosine² ≥ threshold
- :class:`ClusterConfirmer` — all within-cluster pairs from a ``SelectionReport``
- :class:`~polygram.behavioural.BehaviouralValidator` — forward-pass behavioural gating (existing)
"""

from polygram.confirmation.cluster import ClusterConfirmer
from polygram.confirmation.decoder_geometry import DecoderGeometryConfirmer
from polygram.confirmation.protocol import Confirmer

__all__ = ["Confirmer", "DecoderGeometryConfirmer", "ClusterConfirmer"]
