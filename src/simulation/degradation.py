"""Stochastic degradation model for equipment health.

This module implements a Wiener process degradation model, commonly used
for prognostics and health management (PHM). Each call to `step` simulates
the evolution of a health indicator over a small time interval. When the
health value falls below a failure threshold, the equipment is considered
failed. Maintenance can restore the health back to a predefined level.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class WienerDegradationModel:
    """Wiener process degradation model.

    Parameters
    ----------
    base_drift : float
        Baseline health decay per hour when the equipment is idle.
    load_drift_factor : float
        Additional decay proportional to the load ratio.
    volatility : float
        Standard deviation of the stochastic noise term.
    failure_threshold : float
        When health drops below this value the equipment is considered failed.
    pm_threshold : float
        Preventive maintenance is recommended when health is below this value.
    maintenance_restore : float
        Health level after maintenance.
    """

    def __init__(
        self,
        base_drift: float = 0.02,
        load_drift_factor: float = 0.03,
        volatility: float = 0.5,
        failure_threshold: float = 20.0,
        pm_threshold: float = 40.0,
        maintenance_restore: float = 95.0,
    ) -> None:
        self.base_drift = base_drift
        self.load_drift_factor = load_drift_factor
        self.volatility = volatility
        self.failure_threshold = failure_threshold
        self.pm_threshold = pm_threshold
        self.maintenance_restore = maintenance_restore

    def step(
        self,
        current_health: float,
        delta_t: float,
        load_ratio: float = 0.0,
        operating: bool = True,
    ) -> Tuple[float, bool]:
        """Advance health by `delta_t` hours.

        Parameters
        ----------
        current_health : float
            Current health level on a 0–100 scale.
        delta_t : float
            Time increment in hours.
        load_ratio : float, optional
            Ratio of the applied load to the equipment capacity. Defaults to 0.
        operating : bool, optional
            Whether the equipment is currently operating. If false, health
            remains unchanged. Defaults to True.

        Returns
        -------
        new_health : float
            Updated health value, bounded below by 0.
        failed : bool
            Indicates whether the equipment has failed at this step.
        """
        if not operating:
            return current_health, False
        drift = self.base_drift + self.load_drift_factor * load_ratio
        noise = np.random.normal(0.0, 1.0)
        delta_health = -drift * delta_t + self.volatility * np.sqrt(delta_t) * noise
        new_health = max(0.0, current_health + delta_health)
        failed = new_health < self.failure_threshold
        return new_health, failed

    def estimate_rul(self, current_health: float, load_ratio: float = 0.3) -> float:
        """Estimate remaining useful life (RUL) under nominal usage.

        Using the expected hitting time for a Wiener process, the RUL is
        approximated as `(H - H_fail) / μ`, where `μ` is the drift rate. If
        drift is non‑positive, the RUL is infinite (no degradation).
        """
        drift = self.base_drift + self.load_drift_factor * load_ratio
        if drift <= 0.0:
            return float("inf")
        return max(0.0, (current_health - self.failure_threshold) / drift)

    def perform_maintenance(self) -> float:
        """Return the health value after preventive maintenance."""
        return float(self.maintenance_restore)