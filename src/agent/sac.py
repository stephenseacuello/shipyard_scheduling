"""Soft Actor-Critic (SAC) implementation for discrete action spaces.

Implements entropy-regularized RL with:
- Twin Q-networks to reduce overestimation
- Automatic temperature (alpha) tuning
- Soft target network updates

Adapted for discrete action spaces using categorical policies.

Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import random

from .entropy_tuning import EntropyTuner


class QNetwork(nn.Module):
    """State-action value network for SAC.

    For discrete actions, outputs Q-values for all actions given a state.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            state: State tensor [batch, state_dim]

        Returns:
            Q-values [batch, n_actions]
        """
        return self.network(state)


class DiscretePolicy(nn.Module):
    """Categorical policy for discrete action SAC.

    Outputs action probabilities with temperature scaling.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action distribution.

        Args:
            state: State tensor [batch, state_dim]
            mask: Optional action mask [batch, n_actions]

        Returns:
            action: Sampled action [batch]
            log_prob: Log probability of action [batch]
            probs: Action probabilities [batch, n_actions]
        """
        logits = self.network(state)

        # Apply mask if provided
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, probs

    def evaluate(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate policy for all actions (for Q-value computation).

        Args:
            state: State tensor [batch, state_dim]
            mask: Optional action mask [batch, n_actions]

        Returns:
            probs: Action probabilities [batch, n_actions]
            log_probs: Log probabilities [batch, n_actions]
        """
        logits = self.network(state)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        return probs, log_probs


class ReplayBuffer:
    """Experience replay buffer for SAC."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        mask: Optional[torch.Tensor] = None,
    ):
        """Add a transition to the buffer."""
        self.buffer.append({
            "state": state.cpu(),
            "action": action,
            "reward": reward,
            "next_state": next_state.cpu(),
            "done": done,
            "mask": mask.cpu() if mask is not None else None,
        })

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.stack([t["state"] for t in batch])
        actions = torch.tensor([t["action"] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32)
        next_states = torch.stack([t["next_state"] for t in batch])
        dones = torch.tensor([t["done"] for t in batch], dtype=torch.float32)

        masks = None
        if batch[0]["mask"] is not None:
            masks = torch.stack([t["mask"] for t in batch])

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "masks": masks,
        }

    def __len__(self):
        return len(self.buffer)


class DiscreteSACTrainer:
    """Soft Actor-Critic trainer for discrete action spaces.

    Features:
    - Twin Q-networks with soft target updates
    - Automatic temperature (alpha) tuning
    - Experience replay

    Args:
        policy: Discrete policy network.
        encoder: GNN encoder for graph observations.
        n_actions: Number of discrete actions.
        lr_policy: Learning rate for policy.
        lr_q: Learning rate for Q-networks.
        lr_alpha: Learning rate for temperature.
        gamma: Discount factor.
        tau: Soft update coefficient.
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size for updates.
        target_entropy: Target entropy (None for automatic).
        device: Computation device.
    """

    def __init__(
        self,
        policy: DiscretePolicy,
        encoder: nn.Module,
        n_actions: int,
        state_dim: int,
        lr_policy: float = 3e-4,
        lr_q: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        target_entropy: Optional[float] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Networks
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)

        self.q1 = QNetwork(state_dim, n_actions).to(device)
        self.q2 = QNetwork(state_dim, n_actions).to(device)
        self.q1_target = QNetwork(state_dim, n_actions).to(device)
        self.q2_target = QNetwork(state_dim, n_actions).to(device)

        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.encoder.parameters()),
            lr=lr_policy,
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr_q)

        # Entropy tuning
        self.entropy_tuner = EntropyTuner(
            action_dim=n_actions,
            device=device,
            lr=lr_alpha,
            target_entropy=target_entropy,
        )

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Statistics
        self._update_count = 0

    @property
    def alpha(self) -> float:
        """Current temperature value."""
        return self.entropy_tuner.get_alpha()

    def select_action(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> int:
        """Select an action given a state.

        Args:
            state: Encoded state tensor.
            mask: Optional action mask.
            deterministic: If True, select argmax action.

        Returns:
            Selected action index.
        """
        with torch.no_grad():
            if deterministic:
                logits = self.policy.network(state)
                if mask is not None:
                    logits = logits.masked_fill(~mask, -1e9)
                action = logits.argmax(dim=-1)
            else:
                action, _, _ = self.policy(state, mask)

        return action.item()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single SAC update step.

        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones, masks.

        Returns:
            Dictionary of training metrics.
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        masks = batch["masks"].to(self.device) if batch["masks"] is not None else None

        # Get current alpha
        alpha = self.entropy_tuner.alpha

        # Compute target Q-values
        with torch.no_grad():
            next_probs, next_log_probs = self.policy.evaluate(next_states, masks)
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)

            # V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        # Update Q-networks
        current_q1 = self.q1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        current_q2 = self.q2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        probs, log_probs = self.policy.evaluate(states, masks)
        q1_values = self.q1(states)
        q2_values = self.q2(states)
        min_q = torch.min(q1_values, q2_values)

        # Policy loss: maximize E[Q - alpha * log pi]
        policy_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        alpha_loss = self.entropy_tuner.update(entropy)

        # Soft update target networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self._update_count += 1

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": entropy.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform a training step if buffer has enough samples."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        return self.update(batch)

    def collect_and_train(
        self,
        env,
        n_steps: int,
        updates_per_step: int = 1,
    ) -> Dict[str, Any]:
        """Collect experience and train.

        Args:
            env: The environment.
            n_steps: Number of environment steps.
            updates_per_step: Number of gradient updates per step.

        Returns:
            Dictionary with episode statistics.
        """
        obs, info = env.reset()
        total_reward = 0.0
        episode_rewards = []
        metrics_history = []

        for step in range(n_steps):
            # Get graph and encode
            graph_data = env.get_graph_data().to(self.device)
            with torch.no_grad():
                state = self.encoder(graph_data)

            # Get action mask
            env_mask = env.get_action_mask()
            # For SAC, we need just the action_type mask
            action_type_mask = torch.tensor(
                env_mask.get("action_type", [True] * self.n_actions),
                device=self.device,
            )

            # Select action
            action = self.select_action(state, action_type_mask)

            # Step environment (simplified - just action type)
            action_dict = {"action_type": action, "spmt": 0, "request": 0, "crane": 0, "lift": 0, "equipment": 0}
            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated

            # Get next state
            next_graph = env.get_graph_data().to(self.device)
            with torch.no_grad():
                next_state = self.encoder(next_graph)

            # Store transition
            self.buffer.push(state, action, reward, next_state, done, action_type_mask)

            total_reward += reward

            # Train
            for _ in range(updates_per_step):
                metrics = self.train_step()
                if metrics:
                    metrics_history.append(metrics)

            if done:
                episode_rewards.append(total_reward)
                total_reward = 0.0
                obs, info = env.reset()

        return {
            "episode_rewards": episode_rewards,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "metrics": metrics_history[-1] if metrics_history else {},
        }

    def save_checkpoint(self, path: str):
        """Save trainer state."""
        torch.save({
            "policy": self.policy.state_dict(),
            "encoder": self.encoder.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "entropy_tuner": self.entropy_tuner.state_dict(),
            "update_count": self._update_count,
        }, path)

    def load_checkpoint(self, path: str):
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.entropy_tuner.load_state_dict(checkpoint["entropy_tuner"])
        self._update_count = checkpoint["update_count"]
