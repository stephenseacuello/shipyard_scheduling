"""Tests for new baseline schedulers (Random, FIFO, CPM, MCTS)."""

import pytest
import yaml
import os

# Load config with inheritance
def load_cfg(path="config/small_instance.yaml"):
    full = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    with open(full) as f:
        cfg = yaml.safe_load(f)
    if "inherit_from" in cfg:
        base = load_cfg(os.path.join(os.path.dirname(path), cfg["inherit_from"]))
        base.update(cfg)
        return base
    return cfg


@pytest.fixture
def tiny_env():
    from simulation.shipyard_env import HHIShipyardEnv
    cfg = load_cfg("config/tiny_instance.yaml")
    env = HHIShipyardEnv(cfg)
    env.reset(seed=42)
    return env


class TestRandomScheduler:
    def test_decide_returns_valid_action(self, tiny_env):
        from baselines.random_policy import RandomScheduler
        sched = RandomScheduler(seed=42)
        action = sched.decide(tiny_env)
        assert isinstance(action, dict)
        assert "action_type" in action
        assert action["action_type"] in (0, 1, 2, 3)

    def test_multi_step_no_crash(self, tiny_env):
        from baselines.random_policy import RandomScheduler
        sched = RandomScheduler(seed=42)
        for _ in range(50):
            action = sched.decide(tiny_env)
            obs, reward, term, trunc, info = tiny_env.step(action)
            if term or trunc:
                break


class TestFIFOScheduler:
    def test_decide_returns_valid_action(self, tiny_env):
        from baselines.fifo_policy import FIFOScheduler
        sched = FIFOScheduler()
        action = sched.decide(tiny_env)
        assert isinstance(action, dict)
        assert "action_type" in action

    def test_multi_step_no_crash(self, tiny_env):
        from baselines.fifo_policy import FIFOScheduler
        sched = FIFOScheduler()
        for _ in range(50):
            action = sched.decide(tiny_env)
            obs, reward, term, trunc, info = tiny_env.step(action)
            if term or trunc:
                break


class TestCPMScheduler:
    def test_decide_returns_valid_action(self, tiny_env):
        from baselines.cpm_scheduler import CPMScheduler
        sched = CPMScheduler()
        action = sched.decide(tiny_env)
        assert isinstance(action, dict)
        assert "action_type" in action

    def test_multi_step_no_crash(self, tiny_env):
        from baselines.cpm_scheduler import CPMScheduler
        sched = CPMScheduler()
        for _ in range(50):
            action = sched.decide(tiny_env)
            obs, reward, term, trunc, info = tiny_env.step(action)
            if term or trunc:
                break


class TestMCTSScheduler:
    def test_decide_returns_valid_action(self, tiny_env):
        from agent.mcts import MCTSScheduler
        sched = MCTSScheduler(n_simulations=5, max_rollout_depth=10)
        action = sched.decide(tiny_env)
        assert isinstance(action, dict)
        assert "action_type" in action
        assert action["action_type"] in (0, 1, 2, 3)

    def test_multi_step_no_crash(self, tiny_env):
        from agent.mcts import MCTSScheduler
        sched = MCTSScheduler(n_simulations=3, max_rollout_depth=5)
        for _ in range(10):
            action = sched.decide(tiny_env)
            obs, reward, term, trunc, info = tiny_env.step(action)
            if term or trunc:
                break


class TestDAggerReEncoding:
    """Test that DAgger stores graph data, not pre-computed embeddings."""

    def test_dagger_stores_graph_data(self, tiny_env):
        """Verify DAgger trainer stores HeteroData, not Tensor embeddings."""
        from agent.gnn_encoder import HeterogeneousGNNEncoder
        from agent.policy import ActorCriticPolicy
        from baselines.rule_based import RuleBasedScheduler

        # Inline import to avoid circular
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments"))
        from train_dagger import DAggerTrainer

        encoder = HeterogeneousGNNEncoder(
            block_dim=tiny_env.block_features,
            spmt_dim=tiny_env.spmt_features,
            crane_dim=tiny_env.crane_features,
            facility_dim=tiny_env.facility_features,
            hidden_dim=32,
            num_layers=1,
        )
        policy = ActorCriticPolicy(
            state_dim=32 * 4,
            n_action_types=4,
            n_spmts=2,
            n_cranes=1,
            max_requests=10,
            hidden_dim=32,
        )
        expert = RuleBasedScheduler()
        trainer = DAggerTrainer(tiny_env, encoder, policy, expert)
        trainer.collect_expert_demos(n_episodes=1, max_steps=10)

        # Verify we stored graph data, not tensors
        assert len(trainer.graph_data_list) > 0
        first_item = trainer.graph_data_list[0]
        # Should be HeteroData, not a plain Tensor
        assert hasattr(first_item, "node_types") or hasattr(first_item, "x_dict"), \
            f"Expected HeteroData but got {type(first_item)}"


class TestGNNAttentionPooling:
    """Test that GNN encoder uses attention pooling for blocks."""

    def test_encoder_has_attention(self):
        from agent.gnn_encoder import HeterogeneousGNNEncoder
        import torch.nn as nn

        encoder = HeterogeneousGNNEncoder(
            block_dim=12,
            spmt_dim=9,
            crane_dim=11,
            facility_dim=4,
            hidden_dim=32,
            num_layers=1,
        )
        assert hasattr(encoder, "block_attn"), "Missing block attention layer"
        assert hasattr(encoder, "output_norm"), "Missing output LayerNorm"
        assert isinstance(encoder.block_attn, nn.Linear)
        assert isinstance(encoder.output_norm, nn.LayerNorm)
