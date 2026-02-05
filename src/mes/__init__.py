"""Manufacturing Execution System (MES) web dashboard.

The MES layer provides a Dash application for monitoring the simulation
state in real time. It exposes pages for block tracking, fleet
management, equipment health and KPI analytics.
"""

__all__ = ["create_app"]


def create_app():
    """Lazy import to avoid circular import warning when run as __main__."""
    from .app import create_app as _create_app
    return _create_app()
