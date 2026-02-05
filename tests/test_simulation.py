"""Tests for the shipyard simulation environment."""

from shipyard_scheduling.simulation.environment import ShipyardEnv
import yaml
import os


def load_cfg():
    with open(os.path.join(os.path.dirname(__file__), "..", "config", "small_instance.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        with open(os.path.join(os.path.dirname(__file__), "..", "config", inherit), "r") as g:
            base = yaml.safe_load(g)
        base.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        cfg = base
    return cfg


def test_env_reset_and_step():
    cfg = load_cfg()
    env = ShipyardEnv(cfg)
    obs, info = env.reset()
    # Observation length should match defined dimension
    assert len(obs) == env.observation_space.shape[0]
    # Take a hold action
    action = {"action_type": 3, "spmt_idx": 0, "request_idx": 0, "crane_idx": 0, "lift_idx": 0, "equipment_idx": 0}
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated