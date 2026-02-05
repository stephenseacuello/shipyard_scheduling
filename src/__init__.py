"""Top-level package for Shipyard Scheduling RL Framework."""

from importlib.metadata import version

__all__ = ["version"]


def get_version() -> str:
    """Return package version."""
    try:
        return version("shipyard_scheduling")
    except Exception:
        return "0.0.0"