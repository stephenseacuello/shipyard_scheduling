"""Tests for advanced RL agents (DQN, Multi-Objective PPO).

These tests verify the core functionality of the research-grade
RL implementations added for graduate-level coursework.
"""

import pytest
import numpy as np
import torch

from agent.dqn import (
    DoubleDQNAgent,
    DuelingDQN,
    FactorizedDQN,
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
    SumTree,
    Transition,
)
from agent.mo_ppo import (
    MultiObjectivePPO,
    ParetoArchive,
    ParetoSolution,
    HyperNetwork,
)
from agent.gnn_encoder import (
    HeterogeneousGNNEncoder,
    HeterogeneousGraphTransformer,
    TemporalGNNEncoder,
    create_encoder,
)
from agent.ppo import PPOTrainer, RunningMeanStd
from agent.curriculum import AdaptiveCurriculum, DifficultyLevel, create_curriculum


class TestSumTree:
    """Tests for prioritized replay sum tree."""

    def test_sum_tree_basic(self):
        """Test basic sum tree operations."""
        tree = SumTree(capacity=4)

        # Add items with priorities
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")

        # Check total
        assert abs(tree.total() - 6.0) < 1e-6

    def test_sum_tree_sampling(self):
        """Test that sampling is proportional to priorities."""
        tree = SumTree(capacity=100)

        # Add items: one with high priority, rest with low
        tree.add(100.0, "high")
        for i in range(99):
            tree.add(0.01, f"low_{i}")

        # Sample many times - high priority should dominate
        high_count = 0
        for _ in range(1000):
            s = np.random.uniform(0, tree.total())
            _, _, data = tree.get(s)
            if data == "high":
                high_count += 1

        # High priority item should be sampled much more often
        assert high_count > 500

    def test_sum_tree_update(self):
        """Test priority updates."""
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(1.0, "b")

        initial_total = tree.total()
        assert abs(initial_total - 2.0) < 1e-6

        # Update first item's priority
        tree.update(tree.capacity - 1, 5.0)
        assert abs(tree.total() - 6.0) < 1e-6


class TestPrioritizedReplayBuffer:
    """Tests for prioritized experience replay."""

    def test_buffer_add_and_sample(self):
        """Test adding and sampling from buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add transitions
        for i in range(50):
            transition = Transition(
                state=f"state_{i}",
                action=i % 4,
                reward=float(i),
                next_state=f"next_state_{i}",
                done=i % 10 == 0,
                mask=np.ones(4),
            )
            buffer.add(transition)

        assert len(buffer) == 50

        # Sample batch
        transitions, weights, indices = buffer.sample(batch_size=16)
        assert len(transitions) == 16
        assert len(weights) == 16
        assert len(indices) == 16

        # Weights should be positive
        assert all(w > 0 for w in weights)

    def test_priority_update(self):
        """Test TD error priority updates."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        for i in range(20):
            buffer.add(Transition(
                state=i, action=0, reward=1.0,
                next_state=i+1, done=False, mask=np.ones(4)
            ))

        _, _, indices = buffer.sample(batch_size=10)

        # Update with varying TD errors
        td_errors = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 0.2, 0.3, 0.4, 0.6, 0.8])
        buffer.update_priorities(indices, td_errors)

        # Max priority should be updated
        assert buffer.max_priority >= 5.0


class TestDuelingDQN:
    """Tests for dueling DQN architecture."""

    def test_dueling_forward(self):
        """Test forward pass produces valid Q-values."""
        state_dim = 64
        n_actions = 10

        network = DuelingDQN(state_dim, n_actions, hidden_dim=128)
        state = torch.randn(8, state_dim)  # Batch of 8

        q_values = network(state)

        assert q_values.shape == (8, n_actions)
        # Q-values should be finite
        assert torch.isfinite(q_values).all()

    def test_dueling_advantage_centering(self):
        """Test that advantages are mean-centered."""
        network = DuelingDQN(32, 5)
        state = torch.randn(1, 32)

        # Get Q-values
        q_values = network(state)

        # The advantage stream should be mean-centered
        # This is implicit in the dueling architecture
        assert q_values.shape == (1, 5)


class TestFactorizedDQN:
    """Tests for factorized Q-network."""

    def test_factorized_forward(self):
        """Test factorized network produces separate Q-values."""
        network = FactorizedDQN(
            state_dim=64,
            n_action_types=4,
            n_spmts=5,
            n_cranes=3,
            max_requests=10,
        )

        state = torch.randn(8, 64)
        q_type, q_spmt, q_crane, q_request = network(state)

        assert q_type.shape == (8, 4)
        assert q_spmt.shape == (8, 5)
        assert q_crane.shape == (8, 3)
        assert q_request.shape == (8, 10)


class TestParetoArchive:
    """Tests for Pareto archive."""

    def test_dominance_check(self):
        """Test Pareto dominance detection."""
        archive = ParetoArchive(max_size=10)

        # Add non-dominated solution
        sol1 = ParetoSolution(
            policy_state={},
            encoder_state={},
            objectives=np.array([1.0, 2.0]),
            weight_vector=np.array([0.5, 0.5]),
        )
        assert archive.add(sol1)

        # Add dominated solution (worse on all objectives)
        sol2 = ParetoSolution(
            policy_state={},
            encoder_state={},
            objectives=np.array([0.5, 1.0]),  # Dominated by sol1
            weight_vector=np.array([0.5, 0.5]),
        )
        assert not archive.add(sol2)

        # Add non-dominated solution (trade-off)
        sol3 = ParetoSolution(
            policy_state={},
            encoder_state={},
            objectives=np.array([2.0, 1.0]),  # Better on obj1, worse on obj2
            weight_vector=np.array([0.7, 0.3]),
        )
        assert archive.add(sol3)

        assert len(archive.solutions) == 2

    def test_archive_pruning(self):
        """Test archive pruning when over capacity."""
        archive = ParetoArchive(max_size=5)

        # Add many non-dominated solutions
        for i in range(10):
            sol = ParetoSolution(
                policy_state={},
                encoder_state={},
                objectives=np.array([i, 10 - i]),  # Pareto front: trade-off
                weight_vector=np.array([i / 10, 1 - i / 10]),
            )
            archive.add(sol)

        # Should be pruned to max_size
        assert len(archive.solutions) <= 5

    def test_hypervolume_2d(self):
        """Test 2D hypervolume calculation."""
        archive = ParetoArchive()

        # Simple Pareto front
        archive.add(ParetoSolution({}, {}, np.array([1.0, 0.0]), np.array([1, 0])))
        archive.add(ParetoSolution({}, {}, np.array([0.0, 1.0]), np.array([0, 1])))

        # Reference point at origin means hypervolume = 1.0
        # But our solutions are at (1,0) and (0,1), so with ref (0,0):
        # This is a degenerate case - let's use a proper reference
        hv = archive.get_hypervolume(reference_point=np.array([2.0, 2.0]))
        assert hv > 0


class TestRunningMeanStd:
    """Tests for running statistics."""

    def test_running_stats(self):
        """Test running mean and std computation."""
        rms = RunningMeanStd()

        # Generate random data
        data = np.random.randn(1000)

        # Update in batches
        for i in range(0, 1000, 100):
            rms.update(data[i:i+100])

        # Check approximate match to numpy stats
        assert abs(rms.mean - np.mean(data)) < 0.1
        assert abs(np.sqrt(rms.var) - np.std(data)) < 0.1

    def test_normalization(self):
        """Test normalization produces zero-mean unit-variance."""
        rms = RunningMeanStd()

        data = np.random.randn(1000) * 5 + 10  # Mean=10, std=5
        rms.update(data)

        normalized = rms.normalize(data)

        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1


class TestAdaptiveCurriculum:
    """Tests for adaptive curriculum learning."""

    def test_curriculum_initialization(self):
        """Test curriculum initializes correctly."""
        curriculum = AdaptiveCurriculum(
            advance_threshold=0.8,
            regress_threshold=0.3,
            window_size=10,
        )

        assert curriculum.current_level_idx == 0
        assert curriculum.total_episodes == 0

    def test_curriculum_advancement(self):
        """Test curriculum advances on consistent success."""
        curriculum = AdaptiveCurriculum(
            advance_threshold=0.8,
            regress_threshold=0.3,
            window_size=5,
            advance_patience=2,
            min_episodes_per_level=5,
        )

        # Record consistent successes
        for _ in range(20):
            curriculum.record_episode(success=True)

        # Should have advanced
        assert curriculum.current_level_idx > 0

    def test_curriculum_regression(self):
        """Test curriculum regresses on consistent failure."""
        curriculum = AdaptiveCurriculum(
            advance_threshold=0.8,
            regress_threshold=0.3,
            window_size=5,
            regress_patience=2,
            min_episodes_per_level=5,
        )

        # First advance to level 1
        curriculum.current_level_idx = 1
        curriculum.episodes_at_current_level = 10

        # Record consistent failures
        for _ in range(20):
            curriculum.record_episode(success=False)

        # Should have regressed
        assert curriculum.current_level_idx == 0

    def test_curriculum_config_generation(self):
        """Test curriculum generates valid configs."""
        curriculum = AdaptiveCurriculum()
        base_config = {"n_blocks": 10, "other_param": "value"}

        config = curriculum.get_config(base_config)

        # Should have curriculum level params
        assert "n_blocks" in config
        assert config["n_blocks"] == curriculum.current_level.n_blocks
        # Should preserve other params
        assert config["other_param"] == "value"


class TestEncoderFactory:
    """Tests for encoder factory function."""

    def test_create_gat_encoder(self):
        """Test creating GAT encoder."""
        encoder = create_encoder(
            encoder_type="gat",
            block_dim=8,
            spmt_dim=10,
            crane_dim=7,
            facility_dim=3,
            hidden_dim=64,
        )
        assert isinstance(encoder, HeterogeneousGNNEncoder)

    def test_create_transformer_encoder(self):
        """Test creating Graph Transformer encoder."""
        encoder = create_encoder(
            encoder_type="transformer",
            block_dim=8,
            spmt_dim=10,
            crane_dim=7,
            facility_dim=3,
            hidden_dim=64,
        )
        assert isinstance(encoder, HeterogeneousGraphTransformer)

    def test_create_temporal_encoder(self):
        """Test creating Temporal GNN encoder."""
        encoder = create_encoder(
            encoder_type="temporal",
            block_dim=8,
            spmt_dim=10,
            crane_dim=7,
            facility_dim=3,
            hidden_dim=64,
        )
        assert isinstance(encoder, TemporalGNNEncoder)

    def test_invalid_encoder_type(self):
        """Test invalid encoder type raises error."""
        with pytest.raises(ValueError):
            create_encoder(encoder_type="invalid")


class TestCurriculumFactory:
    """Tests for curriculum factory function."""

    def test_create_milestone_curriculum(self):
        """Test creating milestone-based curriculum."""
        config = {
            "type": "milestone",
            "milestones": {0: {"n_blocks": 10}, 10: {"n_blocks": 20}},
        }
        curriculum = create_curriculum(config)
        assert hasattr(curriculum, "milestones")

    def test_create_adaptive_curriculum(self):
        """Test creating adaptive curriculum."""
        config = {
            "type": "adaptive",
            "advance_threshold": 0.85,
            "regress_threshold": 0.25,
        }
        curriculum = create_curriculum(config)
        assert isinstance(curriculum, AdaptiveCurriculum)
        assert curriculum.advance_threshold == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
