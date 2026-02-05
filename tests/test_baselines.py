"""Tests for baseline schedulers."""

import yaml
import os
from shipyard_scheduling.simulation.environment import ShipyardEnv
from shipyard_scheduling.baselines.rule_based import RuleBasedScheduler
from shipyard_scheduling.baselines.siloed_opt import SiloedOptimizationScheduler


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


def test_rule_based_decide():
    cfg = load_cfg()
    env = ShipyardEnv(cfg)
    env.reset()
    scheduler = RuleBasedScheduler()
    action = scheduler.decide(env)
    assert "action_type" in action
    assert action["action_type"] in [0, 1, 2, 3]


def test_siloed_decide():
    cfg = load_cfg()
    env = ShipyardEnv(cfg)
    env.reset()
    scheduler = SiloedOptimizationScheduler()
    action = scheduler.decide(env)
    assert "action_type" in action
    assert action["action_type"] in [0, 1, 2, 3]


def test_rule_based_runs_episode():
    cfg = load_cfg()
    env = ShipyardEnv(cfg)
    env.reset()
    scheduler = RuleBasedScheduler()
    for _ in range(20):
        action = scheduler.decide(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
