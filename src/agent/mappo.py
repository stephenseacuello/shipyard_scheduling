"""Multi-Agent PPO (MAPPO) for coordinated SPMT scheduling.

Implements centralized training with decentralized execution (CTDE)
where each SPMT is treated as an agent. The agents share an encoder
but have individual action heads.

Reference: Yu et al., "The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games" (2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class AgentPolicy(nn.Module):
    """Individual agent policy for a single SPMT.

    Takes local observations and outputs actions for transport
    and maintenance decisions.
    """

    def __init__(
        self,
        local_obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        agent_id: int = 0,
    ):
        super().__init__()

        self.agent_id = agent_id

        # Agent-specific encoder
        self.encoder = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        local_obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """Compute action distribution.

        Args:
            local_obs: Local observation for this agent.
            mask: Action mask.

        Returns:
            Action distribution.
        """
        features = self.encoder(local_obs)
        logits = self.action_head(features)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    """Centralized critic that sees global state.

    Used during training to provide value estimates with
    full state information.
    """

    def __init__(
        self,
        global_state_dim: int,
        n_agents: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.n_agents = n_agents

        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),  # Value for each agent
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Compute value estimates for all agents.

        Args:
            global_state: Full environment state.

        Returns:
            Value estimates [batch, n_agents].
        """
        return self.network(global_state)


class MAPPOPolicy(nn.Module):
    """Multi-agent policy with shared encoder.

    Each SPMT has its own policy head but shares the
    graph encoder for efficiency.
    """

    def __init__(
        self,
        state_dim: int,
        local_obs_dim: int,
        n_agents: int,
        n_actions: int,
        hidden_dim: int = 128,
        share_encoder: bool = True,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.share_encoder = share_encoder

        # Shared feature encoder
        if share_encoder:
            self.shared_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.shared_encoder = None

        # Individual agent policies
        self.agent_policies = nn.ModuleList([
            AgentPolicy(
                local_obs_dim=hidden_dim if share_encoder else local_obs_dim,
                n_actions=n_actions,
                hidden_dim=hidden_dim // 2,
                agent_id=i,
            )
            for i in range(n_agents)
        ])

        # Centralized critic
        self.critic = CentralizedCritic(
            global_state_dim=state_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
        )

    def get_local_obs(
        self,
        global_state: torch.Tensor,
        agent_id: int,
    ) -> torch.Tensor:
        """Extract local observation for an agent.

        Args:
            global_state: Full state [batch, state_dim].
            agent_id: Agent index.

        Returns:
            Local observation for the agent.
        """
        if self.share_encoder:
            return self.shared_encoder(global_state)
        else:
            # In practice, this would extract agent-specific features
            # For now, just return full state
            return global_state

    def forward(
        self,
        global_state: torch.Tensor,
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[Categorical], torch.Tensor]:
        """Forward pass for all agents.

        Args:
            global_state: Full environment state.
            masks: List of action masks per agent.

        Returns:
            action_dists: List of action distributions.
            values: Value estimates for each agent.
        """
        action_dists = []

        for i, policy in enumerate(self.agent_policies):
            local_obs = self.get_local_obs(global_state, i)
            mask = masks[i] if masks is not None else None
            dist = policy(local_obs, mask)
            action_dists.append(dist)

        values = self.critic(global_state)

        return action_dists, values

    def get_actions(
        self,
        global_state: torch.Tensor,
        masks: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Sample actions for all agents.

        Args:
            global_state: Full environment state.
            masks: List of action masks.
            deterministic: If True, take argmax actions.

        Returns:
            actions: List of actions per agent.
            log_probs: Log probabilities per agent.
            values: Value estimates.
        """
        action_dists, values = self.forward(global_state, masks)

        actions = []
        log_probs = []

        for dist in action_dists:
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)

        return actions, log_probs, values


class MAPPOTrainer:
    """Multi-Agent PPO trainer.

    Implements CTDE training where:
    - Critic sees global state (centralized training)
    - Policies see local observations (decentralized execution)
    """

    def __init__(
        self,
        policy: MAPPOPolicy,
        encoder: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.n_agents = policy.n_agents

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(encoder.parameters()),
            lr=lr,
        )

        # Per-agent experience buffers
        self.buffers = {
            "states": [],
            "actions": [[] for _ in range(self.n_agents)],
            "rewards": [[] for _ in range(self.n_agents)],
            "dones": [],
            "log_probs": [[] for _ in range(self.n_agents)],
            "values": [[] for _ in range(self.n_agents)],
            "masks": [[] for _ in range(self.n_agents)],
        }

    def collect_rollout(
        self,
        env,
        n_steps: int,
        reset: bool = True,
    ) -> Dict[str, Any]:
        """Collect multi-agent rollout.

        Args:
            env: Environment (should support multi-agent interface).
            n_steps: Number of steps to collect.
            reset: Whether to reset environment.

        Returns:
            Rollout data.
        """
        if reset:
            obs, info = env.reset()

        for step in range(n_steps):
            # Get global state
            graph_data = env.get_graph_data().to(self.device)
            with torch.no_grad():
                global_state = self.encoder(graph_data)

            # Get action masks for each agent
            env_mask = env.get_action_mask()
            agent_masks = self._get_agent_masks(env_mask)

            # Get actions for all agents
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_actions(
                    global_state, agent_masks
                )

            # Convert to environment action
            env_action = self._actions_to_env_action(actions)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            # Get per-agent rewards (simplified - use global reward)
            agent_rewards = self._distribute_reward(reward, actions)

            # Store experience
            self.buffers["states"].append(global_state.cpu())
            self.buffers["dones"].append(done)

            for i in range(self.n_agents):
                self.buffers["actions"][i].append(actions[i])
                self.buffers["rewards"][i].append(agent_rewards[i])
                self.buffers["log_probs"][i].append(log_probs[i])
                self.buffers["values"][i].append(values[:, i])
                if agent_masks:
                    self.buffers["masks"][i].append(agent_masks[i])

            if done:
                obs, info = env.reset()

        return self._compute_returns_and_advantages()

    def _get_agent_masks(
        self,
        env_mask: Dict[str, Any],
    ) -> Optional[List[torch.Tensor]]:
        """Convert environment mask to per-agent masks."""
        # Simplified: all agents share same mask
        if "spmt" in env_mask:
            mask = torch.tensor(env_mask["spmt"], device=self.device, dtype=torch.bool)
            return [mask for _ in range(self.n_agents)]
        return None

    def _actions_to_env_action(
        self,
        actions: List[torch.Tensor],
    ) -> Dict[str, int]:
        """Convert per-agent actions to environment action.

        For shipyard scheduling, we aggregate agent decisions.
        The "active" agent (e.g., first available) determines the action.
        """
        # Simple aggregation: use first agent's decision
        return {
            "action_type": 0,  # SPMT dispatch
            "spmt": int(actions[0].item()),
            "request": 0,
            "crane": 0,
            "lift": 0,
            "equipment": 0,
        }

    def _distribute_reward(
        self,
        global_reward: float,
        actions: List[torch.Tensor],
    ) -> List[float]:
        """Distribute global reward to agents.

        Can use:
        - Equal split
        - Counterfactual baselines
        - Attention-based credit assignment
        """
        # Simple equal split
        return [global_reward / self.n_agents] * self.n_agents

    def _compute_returns_and_advantages(self) -> Dict[str, Any]:
        """Compute GAE for each agent."""
        dones = self.buffers["dones"]

        all_advantages = []
        all_returns = []

        for agent_id in range(self.n_agents):
            rewards = self.buffers["rewards"][agent_id]
            values = torch.stack(self.buffers["values"][agent_id])

            advantages = []
            returns = []
            gae = 0.0

            for t in reversed(range(len(rewards))):
                next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])

            advantages = torch.tensor(advantages, device=self.device)
            returns_t = torch.tensor(returns, device=self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            all_advantages.append(advantages)
            all_returns.append(returns_t)

        return {
            "states": self.buffers["states"],
            "actions": self.buffers["actions"],
            "log_probs": self.buffers["log_probs"],
            "advantages": all_advantages,
            "returns": all_returns,
            "masks": self.buffers["masks"],
        }

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform MAPPO update."""
        states = rollout_data["states"]
        n_samples = len(states)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]

                # Batch states
                batch_states = torch.stack([states[i] for i in batch_indices]).to(self.device)

                total_policy_loss = 0.0
                total_value_loss = 0.0
                total_entropy = 0.0

                for agent_id in range(self.n_agents):
                    # Get agent-specific data
                    batch_actions = torch.stack([
                        rollout_data["actions"][agent_id][i]
                        for i in batch_indices
                    ])
                    batch_old_log_probs = torch.stack([
                        rollout_data["log_probs"][agent_id][i]
                        for i in batch_indices
                    ])
                    batch_advantages = rollout_data["advantages"][agent_id][batch_indices]
                    batch_returns = rollout_data["returns"][agent_id][batch_indices]

                    # Forward pass
                    action_dists, values = self.policy.forward(batch_states)
                    dist = action_dists[agent_id]
                    agent_values = values[:, agent_id]

                    # Log probs and entropy
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()

                    # PPO loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(agent_values, batch_returns)

                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    total_entropy += entropy.mean()

                # Average across agents
                total_loss = (
                    total_policy_loss / self.n_agents
                    + self.value_coef * total_value_loss / self.n_agents
                    - self.entropy_coef * total_entropy / self.n_agents
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.encoder.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"].append(total_policy_loss.item() / self.n_agents)
                metrics["value_loss"].append(total_value_loss.item() / self.n_agents)
                metrics["entropy"].append(total_entropy.item() / self.n_agents)

        # Clear buffers
        self.buffers = {
            "states": [],
            "actions": [[] for _ in range(self.n_agents)],
            "rewards": [[] for _ in range(self.n_agents)],
            "dones": [],
            "log_probs": [[] for _ in range(self.n_agents)],
            "values": [[] for _ in range(self.n_agents)],
            "masks": [[] for _ in range(self.n_agents)],
        }

        return {k: float(np.mean(v)) for k, v in metrics.items()}
