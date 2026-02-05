"""Remaining useful life estimator for equipment.

This class estimates the expected remaining useful life (RUL) for
equipment using a provided degradation model. It exposes a callable
interface that accepts an equipment object and returns the estimated
remaining hours before failure.
"""

from __future__ import annotations

from typing import Any

from simulation.degradation import WienerDegradationModel


class RemainingUsefulLifeEstimator:
    def __init__(self, model: WienerDegradationModel | None = None) -> None:
        self.model = model or WienerDegradationModel()

    def __call__(self, equipment: Any) -> float:
        """Estimate RUL for equipment using its minimum health and load ratio 0.3."""
        try:
            health_vec = equipment.get_health_vector()
            min_health = float(min(health_vec)) * 100.0  # convert 0–1 scale to 0–100
        except Exception:
            min_health = 100.0
        return float(self.model.estimate_rul(min_health, load_ratio=0.3))