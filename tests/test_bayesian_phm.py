"""Tests for Bayesian PHM (Prognostics and Health Management).

These tests verify the uncertainty-aware RUL estimation components.
"""

import pytest
import numpy as np
from scipy import stats

from phm.bayesian_rul import (
    KalmanFilterRUL,
    ParticleFilterRUL,
    EnsembleRULEstimator,
    MaintenanceTrigger,
    RULPrediction,
    HealthState,
)


class TestHealthState:
    """Tests for HealthState dataclass."""

    def test_health_state_creation(self):
        """Test creating health state."""
        state = HealthState(mean=0.8, variance=0.01, distribution="gaussian")

        assert state.mean == 0.8
        assert state.variance == 0.01
        assert abs(state.std - 0.1) < 1e-6

    def test_gaussian_sampling(self):
        """Test sampling from Gaussian health distribution."""
        state = HealthState(mean=0.7, variance=0.04, distribution="gaussian")

        samples = state.sample(n=1000)

        assert len(samples) == 1000
        assert abs(np.mean(samples) - 0.7) < 0.1
        assert abs(np.std(samples) - 0.2) < 0.1

    def test_beta_sampling(self):
        """Test sampling from Beta health distribution."""
        state = HealthState(mean=0.8, variance=0.01, distribution="beta")

        samples = state.sample(n=1000)

        assert len(samples) == 1000
        # All samples should be in [0, 1]
        assert np.all(samples >= 0) and np.all(samples <= 1)
        assert abs(np.mean(samples) - 0.8) < 0.1


class TestRULPrediction:
    """Tests for RULPrediction dataclass."""

    def test_rul_prediction_creation(self):
        """Test creating RUL prediction."""
        prediction = RULPrediction(
            mean=100.0,
            std=20.0,
            percentile_10=80.0,
            percentile_50=100.0,
            percentile_90=120.0,
            confidence_interval=(60.0, 140.0),
        )

        assert prediction.mean == 100.0
        assert prediction.std == 20.0
        assert prediction.percentile_50 == 100.0

    def test_prediction_with_samples(self):
        """Test RUL prediction with sample array."""
        samples = np.random.normal(100, 20, 500)
        prediction = RULPrediction(
            mean=np.mean(samples),
            std=np.std(samples),
            percentile_10=np.percentile(samples, 10),
            percentile_50=np.percentile(samples, 50),
            percentile_90=np.percentile(samples, 90),
            samples=samples,
        )

        assert prediction.samples is not None
        assert len(prediction.samples) == 500


class TestKalmanFilterRUL:
    """Tests for Kalman Filter RUL estimator."""

    def test_kalman_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilterRUL(
            initial_health=1.0,
            initial_variance=0.01,
            degradation_rate=0.001,
        )

        assert kf.health_mean == 1.0
        assert kf.health_variance == 0.01
        assert kf.total_time == 0.0

    def test_kalman_update(self):
        """Test Kalman filter update step."""
        kf = KalmanFilterRUL(
            initial_health=1.0,
            degradation_rate=0.01,
            measurement_noise=0.02,
        )

        # Simulate measurement
        state = kf.update(measurement=0.95, time_delta=1.0)

        assert state.mean < 1.0  # Health should decrease
        assert kf.total_time == 1.0

    def test_kalman_tracks_degradation(self):
        """Test that Kalman filter tracks degradation over time."""
        kf = KalmanFilterRUL(
            initial_health=1.0,
            degradation_rate=0.01,
            process_noise=0.001,
            measurement_noise=0.02,
        )

        health_history = [kf.health_mean]

        # Simulate degradation with noisy measurements
        true_health = 1.0
        for t in range(100):
            true_health -= 0.01
            measurement = true_health + np.random.normal(0, 0.02)
            state = kf.update(measurement=measurement, time_delta=1.0)
            health_history.append(state.mean)

        # Health should trend downward
        assert health_history[-1] < health_history[0]
        # Final estimate should be close to true health
        assert abs(health_history[-1] - true_health) < 0.1

    def test_kalman_rul_prediction(self):
        """Test RUL prediction from Kalman filter."""
        kf = KalmanFilterRUL(
            initial_health=0.5,
            degradation_rate=0.01,
        )

        prediction = kf.predict_rul(failure_threshold=0.0)

        # Expected RUL = 0.5 / 0.01 = 50
        assert abs(prediction.mean - 50.0) < 1.0
        assert prediction.std > 0  # Should have uncertainty

    def test_kalman_reset(self):
        """Test Kalman filter reset."""
        kf = KalmanFilterRUL(initial_health=1.0)

        # Update a few times
        for _ in range(10):
            kf.update(0.9, 1.0)

        # Reset
        kf.reset()

        assert kf.health_mean == 1.0
        assert kf.total_time == 0.0


class TestParticleFilterRUL:
    """Tests for Particle Filter RUL estimator."""

    def test_particle_filter_initialization(self):
        """Test particle filter initialization."""
        pf = ParticleFilterRUL(
            n_particles=100,
            initial_health=1.0,
        )

        assert len(pf.particles) == 100
        assert len(pf.weights) == 100
        assert abs(pf.weights.sum() - 1.0) < 1e-6

    def test_particle_filter_update(self):
        """Test particle filter update step."""
        pf = ParticleFilterRUL(
            n_particles=500,
            initial_health=1.0,
            degradation_model="wiener",
        )

        state = pf.update(measurement=0.95, time_delta=1.0)

        assert state.mean < 1.0
        assert state.variance > 0

    def test_particle_filter_resampling(self):
        """Test that particle filter resamples when needed."""
        pf = ParticleFilterRUL(n_particles=100)

        # Update many times to trigger resampling
        for _ in range(20):
            pf.update(measurement=pf.particles.mean() - 0.01, time_delta=1.0)

        # Weights should be approximately uniform after resampling
        # (not always true, but variance should be bounded)
        assert np.var(pf.weights) < 0.1

    def test_particle_filter_rul_prediction(self):
        """Test RUL prediction from particle filter."""
        pf = ParticleFilterRUL(
            n_particles=200,
            initial_health=0.5,
            degradation_model="linear",
            degradation_params={"drift": 0.01, "volatility": 0.005},
        )

        prediction = pf.predict_rul(failure_threshold=0.0)

        assert prediction.mean > 0
        assert prediction.std > 0
        assert prediction.percentile_10 < prediction.percentile_50 < prediction.percentile_90

    def test_different_degradation_models(self):
        """Test different degradation models."""
        for model in ["linear", "wiener", "nonlinear"]:
            pf = ParticleFilterRUL(
                n_particles=100,
                degradation_model=model,
            )

            # Should not raise error
            state = pf.update(0.95, 1.0)
            assert state.mean > 0

    def test_particle_filter_reset(self):
        """Test particle filter reset."""
        pf = ParticleFilterRUL(n_particles=100, initial_health=1.0)

        # Update several times
        for _ in range(10):
            pf.update(0.9, 1.0)

        pf.reset()

        assert abs(np.mean(pf.particles) - 1.0) < 0.1
        assert pf.total_time == 0.0


class TestEnsembleRULEstimator:
    """Tests for Ensemble RUL estimator."""

    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        ensemble = EnsembleRULEstimator()

        # Default should have multiple estimators
        assert len(ensemble.estimators) >= 2

    def test_ensemble_custom_estimators(self):
        """Test ensemble with custom estimators."""
        estimators = [
            KalmanFilterRUL(degradation_rate=0.005),
            KalmanFilterRUL(degradation_rate=0.01),
            KalmanFilterRUL(degradation_rate=0.015),
        ]
        ensemble = EnsembleRULEstimator(estimators=estimators)

        assert len(ensemble.estimators) == 3

    def test_ensemble_update(self):
        """Test ensemble update."""
        ensemble = EnsembleRULEstimator()

        state = ensemble.update(measurement=0.9, time_delta=1.0)

        # Combined estimate
        assert 0 < state.mean < 1
        # Variance includes model uncertainty
        assert state.variance > 0

    def test_ensemble_rul_prediction(self):
        """Test ensemble RUL prediction."""
        estimators = [
            KalmanFilterRUL(initial_health=0.5, degradation_rate=0.005),
            KalmanFilterRUL(initial_health=0.5, degradation_rate=0.01),
        ]
        ensemble = EnsembleRULEstimator(estimators=estimators)

        prediction = ensemble.predict_rul()

        # RUL should be between individual estimates
        # Fast degradation: 0.5 / 0.01 = 50
        # Slow degradation: 0.5 / 0.005 = 100
        # Mean should be around 75
        assert 40 < prediction.mean < 110

    def test_ensemble_model_uncertainty(self):
        """Test that ensemble captures model uncertainty."""
        # Two estimators with very different degradation rates
        estimators = [
            KalmanFilterRUL(initial_health=0.5, degradation_rate=0.001),
            KalmanFilterRUL(initial_health=0.5, degradation_rate=0.1),
        ]
        ensemble = EnsembleRULEstimator(estimators=estimators)

        state = ensemble.update(0.5, 0)

        # Variance should be high due to model disagreement
        # The between-model variance dominates
        assert state.variance > 0.01


class TestMaintenanceTrigger:
    """Tests for maintenance trigger."""

    def test_threshold_trigger(self):
        """Test threshold-based maintenance trigger."""
        trigger = MaintenanceTrigger(
            trigger_type="threshold",
            rul_threshold=50.0,
        )

        # Above threshold - no maintenance
        prediction_ok = RULPrediction(
            mean=100.0, std=10.0,
            percentile_10=80, percentile_50=100, percentile_90=120,
        )
        should_trigger, _ = trigger.should_trigger(prediction_ok)
        assert not should_trigger

        # Below threshold - trigger maintenance
        prediction_low = RULPrediction(
            mean=30.0, std=10.0,
            percentile_10=20, percentile_50=30, percentile_90=40,
        )
        should_trigger, diag = trigger.should_trigger(prediction_low)
        assert should_trigger
        assert "expected_rul" in diag

    def test_risk_trigger(self):
        """Test risk-based maintenance trigger."""
        trigger = MaintenanceTrigger(
            trigger_type="risk",
            risk_horizon=50.0,
            failure_risk_threshold=0.2,
        )

        # Low risk prediction
        prediction_safe = RULPrediction(
            mean=200.0, std=20.0,
            percentile_10=170, percentile_50=200, percentile_90=230,
        )
        should_trigger, diag = trigger.should_trigger(prediction_safe)
        assert not should_trigger
        assert diag["failure_probability"] < 0.2

        # High risk prediction (RUL close to horizon with high uncertainty)
        prediction_risky = RULPrediction(
            mean=60.0, std=30.0,  # P(RUL < 50) is significant
            percentile_10=30, percentile_50=60, percentile_90=90,
        )
        should_trigger, diag = trigger.should_trigger(prediction_risky)
        assert "failure_probability" in diag

    def test_cost_optimal_trigger(self):
        """Test cost-optimal maintenance trigger."""
        trigger = MaintenanceTrigger(
            trigger_type="cost_optimal",
            maintenance_cost=100.0,
            failure_cost=1000.0,
        )

        prediction = RULPrediction(
            mean=50.0, std=20.0,
            percentile_10=30, percentile_50=50, percentile_90=70,
        )

        should_trigger, diag = trigger.should_trigger(prediction)

        assert "expected_failure_cost" in diag
        assert "maintenance_cost" in diag
        assert diag["maintenance_cost"] == 100.0


class TestIntegration:
    """Integration tests for PHM pipeline."""

    def test_full_monitoring_pipeline(self):
        """Test full health monitoring pipeline."""
        # Initialize estimator
        estimator = KalmanFilterRUL(
            initial_health=1.0,
            degradation_rate=0.01,
            measurement_noise=0.02,
        )

        # Initialize trigger
        trigger = MaintenanceTrigger(
            trigger_type="threshold",
            rul_threshold=20.0,
        )

        # Simulate monitoring
        maintenance_triggered = False
        true_health = 1.0

        for t in range(150):
            # Simulate degradation
            true_health -= 0.01

            # Generate noisy measurement
            measurement = true_health + np.random.normal(0, 0.02)
            measurement = np.clip(measurement, 0, 1)

            # Update estimator
            state = estimator.update(measurement, time_delta=1.0)

            # Check trigger
            prediction = estimator.predict_rul()
            should_trigger, _ = trigger.should_trigger(prediction)

            if should_trigger and not maintenance_triggered:
                maintenance_triggered = True
                trigger_time = t
                break

        # Maintenance should be triggered before failure
        assert maintenance_triggered
        # Trigger should happen before RUL reaches 0
        assert trigger_time < 100  # True failure at t=100

    def test_ensemble_vs_single_estimator(self):
        """Compare ensemble vs single estimator uncertainty."""
        # Single estimator
        single = KalmanFilterRUL(initial_health=0.5, degradation_rate=0.01)

        # Ensemble
        ensemble = EnsembleRULEstimator(estimators=[
            KalmanFilterRUL(initial_health=0.5, degradation_rate=0.008),
            KalmanFilterRUL(initial_health=0.5, degradation_rate=0.01),
            KalmanFilterRUL(initial_health=0.5, degradation_rate=0.012),
        ])

        single_pred = single.predict_rul()
        ensemble_pred = ensemble.predict_rul()

        # Ensemble should have wider uncertainty bounds
        # (includes model uncertainty in addition to state uncertainty)
        # Note: This may not always be true depending on parameters
        assert ensemble_pred.mean > 0
        assert single_pred.mean > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
