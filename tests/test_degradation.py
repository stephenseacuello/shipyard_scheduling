"""Tests for degradation model."""

from shipyard_scheduling.simulation.degradation import WienerDegradationModel


def test_degradation_step_and_rul():
    model = WienerDegradationModel(base_drift=1.0, load_drift_factor=0.1, volatility=0.0, failure_threshold=20.0)
    health = 100.0
    # Step without noise
    new_health, failed = model.step(health, delta_t=1.0, load_ratio=0.0)
    assert new_health < health
    assert not failed
    # Step until failure
    for _ in range(100):
        new_health, failed = model.step(new_health, delta_t=1.0, load_ratio=0.0)
        if failed:
            break
    assert failed
    # RUL estimation should be positive for health above failure threshold
    rul = model.estimate_rul(50.0, load_ratio=0.3)
    assert rul > 0