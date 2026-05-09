"""Profile registry — name → `GeometricProfile`."""

from __future__ import annotations

from polygram.geometry.profile import GeometricProfile

_REGISTRY: dict[str, GeometricProfile] = {}


def register_profile(profile: GeometricProfile) -> None:
    """Add a profile to the registry. Raises `ValueError` on duplicate
    names — no silent overrides."""
    if profile.name in _REGISTRY:
        raise ValueError(
            f"register_profile: name {profile.name!r} is already "
            f"registered. Use a different name or unregister first."
        )
    _REGISTRY[profile.name] = profile


def get_profile(name: str) -> GeometricProfile:
    """Look up a profile by name. Raises `KeyError` listing available
    names when not found."""
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"get_profile: no profile named {name!r}. "
            f"Registered profiles: {available}"
        )
    return _REGISTRY[name]


def available_profiles() -> list[str]:
    """Return the sorted list of registered profile names."""
    return sorted(_REGISTRY.keys())
