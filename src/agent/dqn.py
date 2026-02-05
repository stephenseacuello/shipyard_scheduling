"""Double DQN with Prioritized Experience Replay for shipyard scheduling.

This module implements Double DQN (van Hasselt et al., 2016) with:
- Dueling architecture (Wang et al., 2016)
- Prioritized experience replay (Schaul et al., 2015)
- n-step returns
- Target network with soft updates

This serves as a strong off-policy baseline for comparison with PPO.

Note: The shipyard scheduling problem has a hierarchical action space
(action_type, spmt_idx, crane_idx, request_idx). For DQN, we flatten
this into a single discrete action space or use a factorized Q-network.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from .action_masking import (
    flatten_env_mask_to_policy_mask,
    to_torch_mask,
)


class Transition(NamedTuple):
    """A single transition in the replay buffer."""
    state: Any  # Graph data
    action: int  # Flattened action index
    reward: float
    next_state: Any  # Graph data
    done: bool
    mask: np.ndarray  # Valid action mask


class SumTree:
    """Sum tree for efficient priority-based sampling.

    A binary tree where each parent node's value is the sum of its children.
    Leaf nodes store priorities, enabling O(log n) sampling proportional to priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for a given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total sum of priorities."""
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        """Add data with given priority."""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at given tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get leaf index, priority, and data for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer using sum tree.

    Implements proportional prioritization where transition i is sampled
    with probability p_i^alpha / sum(p_k^alpha).

    Args:
        capacity: Maximum buffer size.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta: Importance sampling exponent (annealed from beta to 1).
        beta_annealing_steps: Steps to anneal beta to 1.
        epsilon: Small constant to ensure non-zero priorities.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: int = 100000,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.step_count = 0

    def add(self, transition: Transition) -> None:
        """Add transition with max priority (ensures new experiences are sampled)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, List[int]]:
        """Sample a batch of transitions with importance sampling weights.

        Returns:
            transitions: List of sampled transitions.
            weights: Importance sampling weights (normalized).
            indices: Tree indices for priority updates.
        """
        transitions = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Anneal beta
        self.step_count += 1
        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.step_count / self.beta_annealing_steps
        )

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            transitions.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights = weights / weights.max()  # Normalize

        return transitions, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


class UniformReplayBuffer:
    """Simple uniform replay buffer for comparison."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, List[int]]:
        transitions = random.sample(self.buffer, batch_size)
        weights = np.ones(batch_size)  # Uniform weights
        indices = list(range(batch_size))  # Dummy indices
        return transitions, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        pass  # No-op for uniform buffer

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams.

    The Q-value is decomposed as: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    This decomposition allows the network to learn the state value
    independently from the action advantages.

    Args:
        state_dim: Dimension of the encoded state.
        n_actions: Total number of discrete actions.
        hidden_dim: Hidden layer dimension.
        n_layers: Number of hidden layers in each stream.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()

        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        value_layers = []
        for _ in range(n_layers - 1):
            value_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        value_layers.append(nn.Linear(hidden_dim, 1))
        self.value_stream = nn.Sequential(*value_layers)

        # Advantage stream
        advantage_layers = []
        for _ in range(n_layers - 1):
            advantage_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        advantage_layers.append(nn.Linear(hidden_dim, n_actions))
        self.advantage_stream = nn.Sequential(*advantage_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            state: Encoded state tensor [batch, state_dim].

        Returns:
            Q-values tensor [batch, n_actions].
        """
        shared_features = self.shared(state)
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)

        # Combine value and advantage (subtract mean advantage for identifiability)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class FactorizedDQN(nn.Module):
    """Factorized Q-network for hierarchical action spaces.

    Instead of a single Q-value for all action combinations, this network
    outputs separate Q-values for each action dimension, which are then
    combined. This is more tractable for large combinatorial action spaces.

    For shipyard scheduling:
    - Q(s, action_type)
    - Q(s, spmt_idx | action_type)
    - Q(s, crane_idx | action_type)
    - Q(s, request_idx | action_type)
    """

    def __init__(
        self,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 10,
        n_cranes: int = 5,
        max_requests: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.n_action_types = n_action_types
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes
        self.max_requests = max_requests

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for each action dimension
        self.action_type_head = nn.Linear(hidden_dim, n_action_types)
        self.spmt_head = nn.Linear(hidden_dim, n_spmts)
        self.crane_head = nn.Linear(hidden_dim, n_cranes)
        self.request_head = nn.Linear(hidden_dim, max_requests)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q-values for each action dimension.

        Args:
            state: Encoded state tensor [batch, state_dim].

        Returns:
            Tuple of Q-values for (action_type, spmt, crane, request).
        """
        features = self.shared(state)
        return (
            self.action_type_head(features),
            self.spmt_head(features),
            self.crane_head(features),
            self.request_head(features),
        )


class DoubleDQNAgent:
    """Double DQN agent with prioritized replay and dueling architecture.

    Features:
    - Double DQN: Uses online network to select actions, target network to evaluate
    - Dueling architecture: Separates value and advantage estimation
    - Prioritized replay: Samples important transitions more frequently
    - Soft target updates: Gradual target network updates (tau)
    - Factorized Q-network option for hierarchical actions

    Args:
        encoder: GNN encoder for graph observations.
        state_dim: Dimension of encoded state.
        n_action_types: Number of action types.
        n_spmts: Number of SPMTs.
        n_cranes: Number of cranes.
        max_requests: Maximum number of move requests.
        lr: Learning rate.
        gamma: Discount factor.
        tau: Soft update coefficient.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate.
        epsilon_decay_steps: Steps to decay epsilon.
        buffer_size: Replay buffer capacity.
        batch_size: Training batch size.
        n_step: N-step returns (1 for standard TD).
        update_freq: Steps between network updates.
        target_update_freq: Steps between target network syncs.
        prioritized: Whether to use prioritized replay.
        dueling: Whether to use dueling architecture.
        factorized: Whether to use factorized Q-network.
        device: Computation device.
    """

    def __init__(
        self,
        encoder: nn.Module,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 10,
        n_cranes: int = 5,
        max_requests: int = 100,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        n_step: int = 3,
        update_freq: int = 4,
        target_update_freq: int = 1,  # Soft updates every step
        prioritized: bool = True,
        dueling: bool = True,
        factorized: bool = True,
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_step = n_step
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.factorized = factorized

        # Action space info
        self.n_action_types = n_action_types
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes
        self.max_requests = max_requests

        # Total actions for non-factorized version
        self.n_actions = n_action_types * n_spmts * n_cranes * max_requests

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        if factorized:
            self.q_network = FactorizedDQN(
                state_dim, n_action_types, n_spmts, n_cranes, max_requests
            ).to(device)
            self.target_network = FactorizedDQN(
                state_dim, n_action_types, n_spmts, n_cranes, max_requests
            ).to(device)
        else:
            # Use smaller action space for dueling (just action types for simplicity)
            self.q_network = DuelingDQN(state_dim, n_action_types).to(device)
            self.target_network = DuelingDQN(state_dim, n_action_types).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            lr=lr,
        )

        # Replay buffer
        if prioritized:
            self.buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.buffer = UniformReplayBuffer(buffer_size)

        # N-step buffer
        self.n_step_buffer: deque = deque(maxlen=n_step)

        # Counters
        self.step_count = 0
        self.update_count = 0

    def select_action(
        self,
        graph_data,
        mask: Dict[str, np.ndarray],
        training: bool = True,
    ) -> Dict[str, int]:
        """Select action using epsilon-greedy policy.

        Args:
            graph_data: Graph observation from environment.
            mask: Action mask dictionary.
            training: Whether in training mode (use exploration).

        Returns:
            Action dictionary with keys: action_type, spmt_idx, crane_idx, request_idx.
        """
        # Decay epsilon
        if training:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end)
                * self.step_count / self.epsilon_decay_steps
            )

        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return self._random_action(mask)

        # Greedy action selection
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            state = self.encoder(graph_data)

            if self.factorized:
                q_type, q_spmt, q_crane, q_request = self.q_network(state)

                # Apply masks
                policy_mask = flatten_env_mask_to_policy_mask(
                    mask, self.n_spmts, self.n_cranes, self.max_requests
                )

                # Select best action for each head (with masking)
                type_mask = torch.tensor(
                    policy_mask["action_type"], device=self.device, dtype=torch.bool
                )
                q_type = q_type.masked_fill(~type_mask, float("-inf"))
                action_type = q_type.argmax(dim=-1).item()

                spmt_mask = torch.tensor(
                    policy_mask["spmt"], device=self.device, dtype=torch.bool
                )
                q_spmt = q_spmt.masked_fill(~spmt_mask, float("-inf"))
                spmt_idx = q_spmt.argmax(dim=-1).item()

                crane_mask = torch.tensor(
                    policy_mask["crane"], device=self.device, dtype=torch.bool
                )
                q_crane = q_crane.masked_fill(~crane_mask, float("-inf"))
                crane_idx = q_crane.argmax(dim=-1).item()

                request_mask = torch.tensor(
                    policy_mask["request"], device=self.device, dtype=torch.bool
                )
                q_request = q_request.masked_fill(~request_mask, float("-inf"))
                request_idx = q_request.argmax(dim=-1).item()

                return {
                    "action_type": action_type,
                    "spmt_idx": spmt_idx,
                    "crane_idx": crane_idx,
                    "request_idx": request_idx,
                }
            else:
                # Non-factorized: just select action type
                q_values = self.q_network(state)
                policy_mask = flatten_env_mask_to_policy_mask(
                    mask, self.n_spmts, self.n_cranes, self.max_requests
                )
                type_mask = torch.tensor(
                    policy_mask["action_type"], device=self.device, dtype=torch.bool
                )
                q_values = q_values.masked_fill(~type_mask, float("-inf"))
                action_type = q_values.argmax(dim=-1).item()

                # Random selection for other dimensions (simplified)
                return self._random_action_given_type(mask, action_type)

    def _random_action(self, mask: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Select random valid action."""
        policy_mask = flatten_env_mask_to_policy_mask(
            mask, self.n_spmts, self.n_cranes, self.max_requests
        )

        valid_types = np.where(policy_mask["action_type"])[0]
        action_type = np.random.choice(valid_types) if len(valid_types) > 0 else 0

        valid_spmts = np.where(policy_mask["spmt"])[0]
        spmt_idx = np.random.choice(valid_spmts) if len(valid_spmts) > 0 else 0

        valid_cranes = np.where(policy_mask["crane"])[0]
        crane_idx = np.random.choice(valid_cranes) if len(valid_cranes) > 0 else 0

        valid_requests = np.where(policy_mask["request"])[0]
        request_idx = np.random.choice(valid_requests) if len(valid_requests) > 0 else 0

        return {
            "action_type": int(action_type),
            "spmt_idx": int(spmt_idx),
            "crane_idx": int(crane_idx),
            "request_idx": int(request_idx),
        }

    def _random_action_given_type(
        self, mask: Dict[str, np.ndarray], action_type: int
    ) -> Dict[str, int]:
        """Select random action for non-type dimensions."""
        policy_mask = flatten_env_mask_to_policy_mask(
            mask, self.n_spmts, self.n_cranes, self.max_requests
        )

        valid_spmts = np.where(policy_mask["spmt"])[0]
        spmt_idx = np.random.choice(valid_spmts) if len(valid_spmts) > 0 else 0

        valid_cranes = np.where(policy_mask["crane"])[0]
        crane_idx = np.random.choice(valid_cranes) if len(valid_cranes) > 0 else 0

        valid_requests = np.where(policy_mask["request"])[0]
        request_idx = np.random.choice(valid_requests) if len(valid_requests) > 0 else 0

        return {
            "action_type": action_type,
            "spmt_idx": int(spmt_idx),
            "crane_idx": int(crane_idx),
            "request_idx": int(request_idx),
        }

    def store_transition(
        self,
        state,
        action: Dict[str, int],
        reward: float,
        next_state,
        done: bool,
        mask: np.ndarray,
    ) -> None:
        """Store transition in n-step buffer and potentially in replay buffer."""
        # Flatten action to single index for storage
        flat_action = self._flatten_action(action)

        self.n_step_buffer.append((state, flat_action, reward, next_state, done, mask))

        # Only add to replay buffer when we have n transitions
        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            n_step_reward = 0
            for i, (_, _, r, _, d, _) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break

            first = self.n_step_buffer[0]
            last = self.n_step_buffer[-1]

            transition = Transition(
                state=first[0],
                action=first[1],
                reward=n_step_reward,
                next_state=last[3],
                done=last[4],
                mask=first[5],
            )
            self.buffer.add(transition)

        self.step_count += 1

    def _flatten_action(self, action: Dict[str, int]) -> int:
        """Flatten hierarchical action to single index."""
        return (
            action["action_type"] * self.n_spmts * self.n_cranes * self.max_requests
            + action["spmt_idx"] * self.n_cranes * self.max_requests
            + action["crane_idx"] * self.max_requests
            + action["request_idx"]
        )

    def _unflatten_action(self, flat_action: int) -> Dict[str, int]:
        """Unflatten single index to hierarchical action."""
        request_idx = flat_action % self.max_requests
        flat_action //= self.max_requests
        crane_idx = flat_action % self.n_cranes
        flat_action //= self.n_cranes
        spmt_idx = flat_action % self.n_spmts
        action_type = flat_action // self.n_spmts

        return {
            "action_type": action_type,
            "spmt_idx": spmt_idx,
            "crane_idx": crane_idx,
            "request_idx": request_idx,
        }

    def update(self) -> Optional[Dict[str, float]]:
        """Perform one update step if conditions are met.

        Returns:
            Training metrics or None if no update performed.
        """
        if len(self.buffer) < self.batch_size:
            return None

        if self.step_count % self.update_freq != 0:
            return None

        # Sample batch
        transitions, weights, indices = self.buffer.sample(self.batch_size)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

        # Prepare batch
        states = Batch.from_data_list([t.state for t in transitions]).to(self.device)
        next_states = Batch.from_data_list([t.next_state for t in transitions]).to(self.device)
        actions = torch.tensor([t.action for t in transitions], device=self.device)
        rewards = torch.tensor(
            [t.reward for t in transitions], device=self.device, dtype=torch.float32
        )
        dones = torch.tensor(
            [t.done for t in transitions], device=self.device, dtype=torch.float32
        )

        # Encode states
        state_embeddings = self.encoder(states)
        next_state_embeddings = self.encoder(next_states)

        if self.factorized:
            # Factorized Q-learning
            q_type, q_spmt, q_crane, q_request = self.q_network(state_embeddings)

            # Get actions for each dimension
            actions_unflat = [self._unflatten_action(a.item()) for a in actions]
            type_actions = torch.tensor([a["action_type"] for a in actions_unflat], device=self.device)
            spmt_actions = torch.tensor([a["spmt_idx"] for a in actions_unflat], device=self.device)
            crane_actions = torch.tensor([a["crane_idx"] for a in actions_unflat], device=self.device)
            request_actions = torch.tensor([a["request_idx"] for a in actions_unflat], device=self.device)

            # Current Q-values
            q_type_current = q_type.gather(1, type_actions.unsqueeze(1)).squeeze(1)
            q_spmt_current = q_spmt.gather(1, spmt_actions.unsqueeze(1)).squeeze(1)
            q_crane_current = q_crane.gather(1, crane_actions.unsqueeze(1)).squeeze(1)
            q_request_current = q_request.gather(1, request_actions.unsqueeze(1)).squeeze(1)

            # Combined Q-value (sum of factorized values)
            q_current = q_type_current + q_spmt_current + q_crane_current + q_request_current

            # Double DQN: use online network to select actions, target to evaluate
            with torch.no_grad():
                # Online network selects best action
                next_q_type, next_q_spmt, next_q_crane, next_q_request = self.q_network(
                    next_state_embeddings
                )
                best_type = next_q_type.argmax(dim=1)
                best_spmt = next_q_spmt.argmax(dim=1)
                best_crane = next_q_crane.argmax(dim=1)
                best_request = next_q_request.argmax(dim=1)

                # Target network evaluates
                target_q_type, target_q_spmt, target_q_crane, target_q_request = self.target_network(
                    next_state_embeddings
                )
                q_next = (
                    target_q_type.gather(1, best_type.unsqueeze(1)).squeeze(1)
                    + target_q_spmt.gather(1, best_spmt.unsqueeze(1)).squeeze(1)
                    + target_q_crane.gather(1, best_crane.unsqueeze(1)).squeeze(1)
                    + target_q_request.gather(1, best_request.unsqueeze(1)).squeeze(1)
                )

                # N-step target
                q_target = rewards + (self.gamma ** self.n_step) * q_next * (1 - dones)
        else:
            # Standard Q-learning (action types only)
            q_values = self.q_network(state_embeddings)
            # For simplified version, just use action_type
            type_actions = torch.tensor(
                [self._unflatten_action(a.item())["action_type"] for a in actions],
                device=self.device,
            )
            q_current = q_values.gather(1, type_actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.q_network(next_state_embeddings)
                best_actions = next_q_values.argmax(dim=1)
                target_q_values = self.target_network(next_state_embeddings)
                q_next = target_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                q_target = rewards + (self.gamma ** self.n_step) * q_next * (1 - dones)

        # TD error for priority update
        td_errors = (q_current - q_target).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # Huber loss with importance sampling weights
        loss = (weights * F.smooth_l1_loss(q_current, q_target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        # Soft update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self._soft_update()

        return {
            "loss": loss.item(),
            "q_value": q_current.mean().item(),
            "epsilon": self.epsilon,
            "td_error": np.abs(td_errors).mean(),
            "buffer_size": len(self.buffer),
        }

    def _soft_update(self) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to checkpoint."""
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
