"""Prognostics and Health Management (PHM) utilities.

This package provides health monitoring and remaining useful life (RUL)
estimation capabilities for shipyard equipment.

Components:
- Health indicator computation from sensor data
- RUL estimation (point estimates)
- Bayesian RUL estimation (uncertainty-aware)
- Feature engineering for ML-based prognostics
"""

from .health_model import compute_health_indicator
from .rul_estimator import RemainingUsefulLifeEstimator
from .feature_eng import extract_features
from .bayesian_rul import (
    BayesianRULEstimator,
    KalmanFilterRUL,
    ParticleFilterRUL,
    EnsembleRULEstimator,
    MaintenanceTrigger,
    RULPrediction,
    HealthState,
)

__all__ = [
    # Core PHM
    "compute_health_indicator",
    "RemainingUsefulLifeEstimator",
    "extract_features",
    # Bayesian RUL
    "BayesianRULEstimator",
    "KalmanFilterRUL",
    "ParticleFilterRUL",
    "EnsembleRULEstimator",
    "MaintenanceTrigger",
    "RULPrediction",
    "HealthState",
]
