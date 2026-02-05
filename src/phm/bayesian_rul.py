"""Bayesian Remaining Useful Life (RUL) estimation.

This module provides uncertainty-aware RUL estimation using Bayesian methods.
Instead of point estimates, it maintains posterior distributions over health
states and RUL predictions.

Methods implemented:
1. Kalman Filter: For linear-Gaussian health dynamics
2. Particle Filter: For nonlinear/non-Gaussian degradation models
3. Bayesian Neural Network: For data-driven uncertainty estimation

Benefits of uncertainty quantification:
- Risk-aware maintenance scheduling
- Confidence intervals for planning
- Optimal maintenance trigger points
- Integration with safe RL

Reference:
- Saha & Goebel "Model-based prognostics with concurrent damage progression"
- Zhu et al. "Bayesian approach for remaining useful life prediction"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy import stats


@dataclass
class RULPrediction:
    """RUL prediction with uncertainty quantification."""
    mean: float  # Expected RUL
    std: float  # Standard deviation
    percentile_10: float  # 10th percentile (optimistic bound)
    percentile_50: float  # Median RUL
    percentile_90: float  # 90th percentile (pessimistic bound)
    samples: Optional[np.ndarray] = None  # Raw samples (for particle filter)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 95% CI


@dataclass
class HealthState:
    """Estimated health state with uncertainty."""
    mean: float
    variance: float
    distribution: str = "gaussian"  # 'gaussian', 'beta', 'empirical'

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw samples from the health distribution."""
        if self.distribution == "gaussian":
            return np.random.normal(self.mean, self.std, n)
        elif self.distribution == "beta":
            # Convert mean/var to alpha/beta
            alpha = self.mean * (self.mean * (1 - self.mean) / self.variance - 1)
            beta = (1 - self.mean) * (self.mean * (1 - self.mean) / self.variance - 1)
            alpha = max(0.1, alpha)
            beta = max(0.1, beta)
            return np.random.beta(alpha, beta, n)
        else:
            return np.full(n, self.mean)


class BayesianRULEstimator(ABC):
    """Abstract base class for Bayesian RUL estimators."""

    @abstractmethod
    def update(self, measurement: float, time_delta: float) -> HealthState:
        """Update health estimate with new measurement.

        Args:
            measurement: Observed health indicator (e.g., vibration, temperature).
            time_delta: Time since last measurement.

        Returns:
            Updated health state with uncertainty.
        """
        pass

    @abstractmethod
    def predict_rul(self, failure_threshold: float = 0.0) -> RULPrediction:
        """Predict remaining useful life.

        Args:
            failure_threshold: Health level considered as failure.

        Returns:
            RUL prediction with uncertainty bounds.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset estimator to initial state."""
        pass


class KalmanFilterRUL(BayesianRULEstimator):
    """Kalman Filter for linear-Gaussian health dynamics.

    Assumes health evolves according to:
        h(t+1) = A * h(t) + B * u(t) + w(t),  w ~ N(0, Q)
        z(t) = C * h(t) + v(t),                v ~ N(0, R)

    For simple degradation without control:
        h(t+1) = h(t) - degradation_rate * dt + noise
        z(t) = h(t) + measurement_noise

    Args:
        initial_health: Initial health estimate (default: 1.0).
        initial_variance: Initial uncertainty in health.
        process_noise: Process noise variance (degradation uncertainty).
        measurement_noise: Measurement noise variance.
        degradation_rate: Expected degradation rate per unit time.
    """

    def __init__(
        self,
        initial_health: float = 1.0,
        initial_variance: float = 0.01,
        process_noise: float = 0.001,
        measurement_noise: float = 0.01,
        degradation_rate: float = 0.001,
    ):
        self.initial_health = initial_health
        self.initial_variance = initial_variance
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.degradation_rate = degradation_rate

        # State estimate
        self.health_mean = initial_health
        self.health_variance = initial_variance

        # Time tracking
        self.total_time = 0.0

    def update(self, measurement: float, time_delta: float) -> HealthState:
        """Kalman filter update step."""
        # Prediction step
        # x_pred = A * x + B * u (state transition)
        # For simple degradation: x_pred = x - degradation_rate * dt
        self.health_mean = self.health_mean - self.degradation_rate * time_delta
        self.health_variance = self.health_variance + self.Q * time_delta

        # Update step (with measurement)
        # Kalman gain: K = P_pred * C^T * (C * P_pred * C^T + R)^-1
        # For direct observation (C=1): K = P_pred / (P_pred + R)
        K = self.health_variance / (self.health_variance + self.R)

        # State update: x = x_pred + K * (z - C * x_pred)
        innovation = measurement - self.health_mean
        self.health_mean = self.health_mean + K * innovation

        # Covariance update: P = (1 - K * C) * P_pred
        self.health_variance = (1 - K) * self.health_variance

        # Clamp health to [0, 1]
        self.health_mean = np.clip(self.health_mean, 0, 1)

        self.total_time += time_delta

        return HealthState(
            mean=self.health_mean,
            variance=self.health_variance,
            distribution="gaussian",
        )

    def predict_rul(self, failure_threshold: float = 0.0) -> RULPrediction:
        """Predict RUL using current state estimate.

        Assumes linear degradation continues at current rate.
        """
        if self.degradation_rate <= 0:
            # No degradation, infinite RUL
            return RULPrediction(
                mean=float("inf"),
                std=0,
                percentile_10=float("inf"),
                percentile_50=float("inf"),
                percentile_90=float("inf"),
            )

        # Expected RUL = (current_health - threshold) / degradation_rate
        remaining_health = self.health_mean - failure_threshold
        if remaining_health <= 0:
            return RULPrediction(mean=0, std=0, percentile_10=0, percentile_50=0, percentile_90=0)

        mean_rul = remaining_health / self.degradation_rate

        # Uncertainty in RUL from uncertainty in health and degradation
        # Using first-order approximation
        rul_variance = self.health_variance / (self.degradation_rate ** 2)
        rul_std = np.sqrt(rul_variance)

        # Percentiles assuming Gaussian
        percentile_10 = max(0, mean_rul - 1.28 * rul_std)
        percentile_50 = max(0, mean_rul)
        percentile_90 = mean_rul + 1.28 * rul_std

        return RULPrediction(
            mean=mean_rul,
            std=rul_std,
            percentile_10=percentile_10,
            percentile_50=percentile_50,
            percentile_90=percentile_90,
            confidence_interval=(max(0, mean_rul - 1.96 * rul_std), mean_rul + 1.96 * rul_std),
        )

    def reset(self) -> None:
        """Reset to initial state."""
        self.health_mean = self.initial_health
        self.health_variance = self.initial_variance
        self.total_time = 0.0


class ParticleFilterRUL(BayesianRULEstimator):
    """Particle Filter for nonlinear/non-Gaussian health dynamics.

    Maintains a set of particles representing possible health states.
    Can model arbitrary degradation processes including:
    - Wiener process with drift
    - Jump diffusion (sudden damage events)
    - State-dependent degradation rates

    Args:
        n_particles: Number of particles.
        initial_health: Initial health level.
        process_noise: Process noise scale.
        measurement_noise: Measurement noise scale.
        degradation_model: 'linear', 'wiener', or 'nonlinear'.
        degradation_params: Parameters for degradation model.
    """

    def __init__(
        self,
        n_particles: int = 500,
        initial_health: float = 1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.02,
        degradation_model: str = "wiener",
        degradation_params: Optional[Dict[str, float]] = None,
    ):
        self.n_particles = n_particles
        self.initial_health = initial_health
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.degradation_model = degradation_model
        self.degradation_params = degradation_params or {
            "drift": 0.001,  # Mean degradation rate
            "volatility": 0.005,  # Random fluctuation
            "jump_rate": 0.01,  # Probability of sudden damage
            "jump_size": 0.05,  # Mean size of sudden damage
        }

        # Initialize particles
        self.particles = np.random.normal(
            initial_health, 0.01, n_particles
        )
        self.weights = np.ones(n_particles) / n_particles

        self.total_time = 0.0

    def _degradation_step(self, particles: np.ndarray, time_delta: float) -> np.ndarray:
        """Simulate degradation for one time step."""
        params = self.degradation_params

        if self.degradation_model == "linear":
            # Simple linear degradation with noise
            new_particles = particles - params["drift"] * time_delta
            new_particles += np.random.normal(0, self.process_noise * np.sqrt(time_delta), len(particles))

        elif self.degradation_model == "wiener":
            # Wiener process with drift (Brownian motion)
            drift = params["drift"] * time_delta
            diffusion = params["volatility"] * np.sqrt(time_delta) * np.random.randn(len(particles))
            new_particles = particles - drift + diffusion

            # Add jump process (sudden damage events)
            jumps = np.random.binomial(1, params["jump_rate"] * time_delta, len(particles))
            jump_sizes = np.random.exponential(params["jump_size"], len(particles))
            new_particles -= jumps * jump_sizes

        elif self.degradation_model == "nonlinear":
            # State-dependent degradation (faster degradation at lower health)
            rate_factor = 1 + (1 - particles) ** 2  # Accelerating degradation
            drift = params["drift"] * rate_factor * time_delta
            diffusion = params["volatility"] * np.sqrt(time_delta) * np.random.randn(len(particles))
            new_particles = particles - drift + diffusion

        else:
            raise ValueError(f"Unknown degradation model: {self.degradation_model}")

        # Clamp to [0, 1]
        return np.clip(new_particles, 0, 1)

    def update(self, measurement: float, time_delta: float) -> HealthState:
        """Particle filter update with new measurement."""
        # Prediction step: propagate particles through degradation model
        self.particles = self._degradation_step(self.particles, time_delta)

        # Update step: weight particles by likelihood of measurement
        likelihoods = stats.norm.pdf(
            measurement, loc=self.particles, scale=self.measurement_noise
        )
        self.weights = likelihoods * self.weights
        self.weights += 1e-10  # Avoid zero weights
        self.weights /= self.weights.sum()

        # Resample if effective sample size is low
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2:
            self._resample()

        self.total_time += time_delta

        # Return summary statistics
        mean = np.average(self.particles, weights=self.weights)
        variance = np.average((self.particles - mean) ** 2, weights=self.weights)

        return HealthState(mean=mean, variance=variance, distribution="empirical")

    def _resample(self) -> None:
        """Resample particles using systematic resampling."""
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # Ensure last element is exactly 1

        # Systematic resampling
        u = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        indices = np.searchsorted(cumsum, u)

        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict_rul(self, failure_threshold: float = 0.0) -> RULPrediction:
        """Predict RUL by simulating future trajectories."""
        # Simulate future trajectories from current particles
        n_simulations = 100
        time_step = 1.0  # Simulation time step
        max_time = 10000  # Maximum simulation time

        rul_samples = []

        for _ in range(n_simulations):
            # Sample a particle according to weights
            idx = np.random.choice(self.n_particles, p=self.weights)
            health = self.particles[idx]
            t = 0.0

            # Simulate until failure or max time
            while health > failure_threshold and t < max_time:
                health = self._degradation_step(np.array([health]), time_step)[0]
                t += time_step

            rul_samples.append(t)

        rul_samples = np.array(rul_samples)

        return RULPrediction(
            mean=np.mean(rul_samples),
            std=np.std(rul_samples),
            percentile_10=np.percentile(rul_samples, 10),
            percentile_50=np.percentile(rul_samples, 50),
            percentile_90=np.percentile(rul_samples, 90),
            samples=rul_samples,
            confidence_interval=(np.percentile(rul_samples, 2.5), np.percentile(rul_samples, 97.5)),
        )

    def reset(self) -> None:
        """Reset to initial state."""
        self.particles = np.random.normal(self.initial_health, 0.01, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.total_time = 0.0


class EnsembleRULEstimator(BayesianRULEstimator):
    """Ensemble of multiple RUL estimators for robust uncertainty.

    Combines predictions from multiple estimators using different models
    or parameters to capture model uncertainty in addition to state uncertainty.

    Args:
        estimators: List of BayesianRULEstimator instances.
        weights: Optional weights for combining predictions.
    """

    def __init__(
        self,
        estimators: Optional[List[BayesianRULEstimator]] = None,
        weights: Optional[List[float]] = None,
    ):
        if estimators is None:
            # Default ensemble with different degradation models
            estimators = [
                KalmanFilterRUL(degradation_rate=0.001),
                ParticleFilterRUL(degradation_model="wiener"),
                ParticleFilterRUL(degradation_model="nonlinear"),
            ]

        self.estimators = estimators
        self.weights = weights or [1.0 / len(estimators)] * len(estimators)

    def update(self, measurement: float, time_delta: float) -> HealthState:
        """Update all estimators with new measurement."""
        states = [e.update(measurement, time_delta) for e in self.estimators]

        # Combine estimates
        mean = sum(w * s.mean for w, s in zip(self.weights, states))

        # Total variance includes within-model and between-model variance
        within_variance = sum(w * s.variance for w, s in zip(self.weights, states))
        between_variance = sum(
            w * (s.mean - mean) ** 2 for w, s in zip(self.weights, states)
        )
        variance = within_variance + between_variance

        return HealthState(mean=mean, variance=variance, distribution="gaussian")

    def predict_rul(self, failure_threshold: float = 0.0) -> RULPrediction:
        """Combine RUL predictions from all estimators."""
        predictions = [e.predict_rul(failure_threshold) for e in self.estimators]

        # Combine means and variances
        mean = sum(w * p.mean for w, p in zip(self.weights, predictions))

        # Total variance
        within_var = sum(w * p.std ** 2 for w, p in zip(self.weights, predictions))
        between_var = sum(
            w * (p.mean - mean) ** 2 for w, p in zip(self.weights, predictions)
        )
        std = np.sqrt(within_var + between_var)

        return RULPrediction(
            mean=mean,
            std=std,
            percentile_10=mean - 1.28 * std,
            percentile_50=mean,
            percentile_90=mean + 1.28 * std,
            confidence_interval=(max(0, mean - 1.96 * std), mean + 1.96 * std),
        )

    def reset(self) -> None:
        """Reset all estimators."""
        for e in self.estimators:
            e.reset()


class MaintenanceTrigger:
    """Determines optimal maintenance timing based on RUL predictions.

    Uses uncertainty-aware triggering rules:
    - Threshold on expected RUL
    - Risk-based trigger (probability of failure within horizon)
    - Cost-optimal trigger (balancing maintenance vs failure costs)

    Args:
        trigger_type: 'threshold', 'risk', or 'cost_optimal'.
        rul_threshold: Trigger maintenance if expected RUL < threshold.
        risk_horizon: Time horizon for risk-based trigger.
        failure_risk_threshold: Maximum acceptable probability of failure.
        maintenance_cost: Cost of preventive maintenance.
        failure_cost: Cost of failure.
    """

    def __init__(
        self,
        trigger_type: str = "threshold",
        rul_threshold: float = 100.0,
        risk_horizon: float = 50.0,
        failure_risk_threshold: float = 0.1,
        maintenance_cost: float = 100.0,
        failure_cost: float = 1000.0,
    ):
        self.trigger_type = trigger_type
        self.rul_threshold = rul_threshold
        self.risk_horizon = risk_horizon
        self.failure_risk_threshold = failure_risk_threshold
        self.maintenance_cost = maintenance_cost
        self.failure_cost = failure_cost

    def should_trigger(self, prediction: RULPrediction) -> Tuple[bool, Dict[str, float]]:
        """Determine if maintenance should be triggered.

        Returns:
            Tuple of (should_trigger, diagnostics_dict).
        """
        if self.trigger_type == "threshold":
            trigger = prediction.mean < self.rul_threshold
            return trigger, {"expected_rul": prediction.mean}

        elif self.trigger_type == "risk":
            # Probability of failure within risk_horizon
            if prediction.std > 0:
                z_score = (self.risk_horizon - prediction.mean) / prediction.std
                failure_prob = stats.norm.cdf(z_score)
            else:
                failure_prob = 1.0 if prediction.mean < self.risk_horizon else 0.0

            trigger = failure_prob > self.failure_risk_threshold
            return trigger, {
                "failure_probability": failure_prob,
                "expected_rul": prediction.mean,
            }

        elif self.trigger_type == "cost_optimal":
            # Expected cost if we maintain now vs wait
            # Simplified model: compare immediate maintenance vs expected failure cost

            if prediction.std > 0:
                # Probability of failure if we wait one period
                z_score = (1.0 - prediction.mean) / prediction.std
                failure_prob = stats.norm.cdf(z_score)
            else:
                failure_prob = 1.0 if prediction.mean < 1.0 else 0.0

            expected_cost_wait = failure_prob * self.failure_cost
            maintain_now = self.maintenance_cost < expected_cost_wait

            return maintain_now, {
                "expected_failure_cost": expected_cost_wait,
                "maintenance_cost": self.maintenance_cost,
                "failure_probability": failure_prob,
            }

        else:
            raise ValueError(f"Unknown trigger type: {self.trigger_type}")
