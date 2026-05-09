"""`GeometricProfile`: a named bundle of strategy + metric + defaults."""

from __future__ import annotations

from dataclasses import dataclass

from polygram.geometry.protocols import GeometricFidelity, KnobAssignment


@dataclass(frozen=True)
class GeometricProfile:
    name: str
    knob_assignment: KnobAssignment
    geometric_fidelity: GeometricFidelity
    default_n_clusters: int | None
    default_gamma_range: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"GeometricProfile.name must be a non-empty string; "
                f"got {self.name!r}"
            )
        lo, hi = self.default_gamma_range
        if not lo <= hi:
            raise ValueError(
                f"GeometricProfile {self.name!r}: default_gamma_range "
                f"must satisfy lo <= hi; got {self.default_gamma_range!r}"
            )
