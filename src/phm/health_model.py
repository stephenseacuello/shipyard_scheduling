"""Health indicator computation.

This module defines functions for computing health indicators from raw
sensors or equipment state. In this simplified implementation the
equipment classes already store normalized health values, so these
functions wrap the accessor methods.
"""

from __future__ import annotations

from typing import Dict, Any


def compute_health_indicator(equipment: Any) -> Dict[str, float]:
    """Return a dictionary of normalized health indicators for a piece of equipment.

    For SPMTs the indicators are hydraulic, tires and engine; for cranes
    the indicators are cable and motor. If other equipment types are
    passed they must implement `get_health_vector` returning a 1D array.
    """
    vec = equipment.get_health_vector()
    # Map length to names
    if len(vec) == 3:
        return {"hydraulic": float(vec[0]), "tires": float(vec[1]), "engine": float(vec[2])}
    elif len(vec) == 2:
        return {"cable": float(vec[0]), "motor": float(vec[1])}
    else:
        return {f"dim{i}": float(v) for i, v in enumerate(vec)}