"""Integration test: run a mini training loop."""

from shipyard_scheduling.simulation.environment import ShipyardEnv
from shipyard_scheduling.agent.gnn_encoder import HeterogeneousGNNEncoder
from shipyard_scheduling.agent.policy import ActorCriticPolicy
from shipyard_scheduling.agent.ppo import PPOTrainer
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


def test_training_loop():
    cfg = load_cfg()
    env = ShipyardEnv(cfg)
    encoder = HeterogeneousGNNEncoder(env.block_features, env.spmt_features, env.crane_features, env.facility_features, hidden_dim=32, num_layers=1)
    state_dim = 32 * 4
    policy = ActorCriticPolicy(state_dim, 4, env.n_spmts, env.n_cranes, env.n_blocks, hidden_dim=64)
    trainer = PPOTrainer(policy, encoder, n_epochs=1, batch_size=32)
    rollout = trainer.collect_rollout(env, n_steps=50)
    metrics = trainer.update(rollout)
    assert "policy_loss" in metrics