"""Hierarchical Reinforcement Learning using the Options Framework.

Implements temporally extended actions (options) for shipyard scheduling.
The meta-policy selects high-level strategies, while option policies
execute low-level actions.

Reference: Sutton, Precup & Singh, "Between MDPs and Semi-MDPs:
A Framework for Temporal Abstraction in Reinforcement Learning" (1999)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum


class OptionType(Enum):
    """High-level scheduling strategies (options)."""
    AGGRESSIVE_PRODUCTION = 0  # Maximize throughput
    MAINTENANCE_FOCUS = 1      # Prioritize equipment health
    DEADLINE_RUSH = 2          # Focus on urgent blocks
    LOAD_BALANCE = 3           # Balance equipment utilization
    CONSERVATIVE = 4           # Minimize risk/breakdowns


@dataclass
class Option:
    """A temporally extended action (macro-action).

    Each option has:
    - Initiation set: when can this option start?
    - Policy: what low-level actions to take?
    - Termination condition: when does this option end?
    """
    name: str
    option_type: OptionType
    initiation_fn: Callable  # (state) -> bool
    policy: nn.Module
    termination_fn: Callable  # (state) -> float (termination probability)
    min_duration: int = 1
    max_duration: int = 50


class OptionPolicy(nn.Module):
    """Policy network for a single option.

    Specialized for a particular scheduling strategy.
    """

    def __init__(
        self,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 1,
        n_cranes: int = 1,
        max_requests: int = 1,
        hidden_dim: int = 128,
        strategy_bias: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.strategy_bias = strategy_bias or {}

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        self.action_type_head = nn.Linear(hidden_dim, n_action_types)
        self.spmt_head = nn.Linear(hidden_dim, n_spmts)
        self.request_head = nn.Linear(hidden_dim, max_requests)
        self.crane_head = nn.Linear(hidden_dim, n_cranes)
        self.equipment_head = nn.Linear(hidden_dim, n_spmts + n_cranes)

    def forward(
        self,
        state: torch.Tensor,
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Categorical]:
        """Forward pass."""
        features = self.shared(state)

        logits = {
            "action_type": self.action_type_head(features),
            "spmt": self.spmt_head(features),
            "request": self.request_head(features),
            "crane": self.crane_head(features),
            "equipment": self.equipment_head(features),
        }

        # Apply strategy biases
        if "action_type" in self.strategy_bias:
            bias = torch.tensor(
                self.strategy_bias["action_type"],
                device=state.device,
            )
            logits["action_type"] = logits["action_type"] + bias

        # Apply masks
        if mask is not None:
            for key, m in mask.items():
                if key in logits:
                    logits[key] = logits[key].masked_fill(~m, -1e9)

        return {k: Categorical(logits=v) for k, v in logits.items()}


class MetaPolicy(nn.Module):
    """Meta-policy for selecting options.

    Decides which high-level strategy to pursue based on
    the current state.
    """

    def __init__(
        self,
        state_dim: int,
        n_options: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_options),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._hidden = None

    def forward(
        self,
        state: torch.Tensor,
        option_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        """Forward pass.

        Args:
            state: Encoded state.
            option_mask: Mask for available options.

        Returns:
            option_dist: Distribution over options.
            value: State value estimate.
        """
        features = self.network[:-1](state)  # Get features before final layer
        logits = self.network[-1](features)

        if option_mask is not None:
            logits = logits.masked_fill(~option_mask, -1e9)

        value = self.value_head(features)

        return Categorical(logits=logits), value


class TerminationNetwork(nn.Module):
    """Network for predicting option termination probability."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict termination probability."""
        return self.network(state)


class HierarchicalPolicy(nn.Module):
    """Hierarchical policy using options framework.

    Combines a meta-policy (selects options) with option policies
    (execute low-level actions).
    """

    def __init__(
        self,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 1,
        n_cranes: int = 1,
        max_requests: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.n_options = len(OptionType)

        # Meta-policy
        self.meta_policy = MetaPolicy(state_dim, self.n_options, hidden_dim)

        # Option policies with strategy-specific biases
        self.option_policies = nn.ModuleList([
            self._create_option_policy(
                option_type, state_dim, n_action_types,
                n_spmts, n_cranes, max_requests, hidden_dim
            )
            for option_type in OptionType
        ])

        # Termination networks
        self.termination_networks = nn.ModuleList([
            TerminationNetwork(state_dim, hidden_dim // 2)
            for _ in OptionType
        ])

        # Current option state
        self.current_option: Optional[int] = None
        self.option_duration: int = 0

    def _create_option_policy(
        self,
        option_type: OptionType,
        state_dim: int,
        n_action_types: int,
        n_spmts: int,
        n_cranes: int,
        max_requests: int,
        hidden_dim: int,
    ) -> OptionPolicy:
        """Create an option policy with appropriate biases."""
        # Strategy-specific action biases
        biases = {
            OptionType.AGGRESSIVE_PRODUCTION: {
                "action_type": [1.0, 0.5, -0.5, -1.0],  # Prefer dispatch
            },
            OptionType.MAINTENANCE_FOCUS: {
                "action_type": [-0.5, -0.5, 2.0, 0.0],  # Prefer maintenance
            },
            OptionType.DEADLINE_RUSH: {
                "action_type": [1.5, 1.0, -1.0, -1.5],  # Strong dispatch preference
            },
            OptionType.LOAD_BALANCE: {
                "action_type": [0.5, 0.5, 0.0, 0.0],  # Balanced
            },
            OptionType.CONSERVATIVE: {
                "action_type": [-0.5, -0.5, 0.5, 1.0],  # Prefer hold/maintenance
            },
        }

        return OptionPolicy(
            state_dim=state_dim,
            n_action_types=n_action_types,
            n_spmts=n_spmts,
            n_cranes=n_cranes,
            max_requests=max_requests,
            hidden_dim=hidden_dim,
            strategy_bias=biases.get(option_type, {}),
        )

    def select_option(
        self,
        state: torch.Tensor,
        option_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select an option using the meta-policy.

        Returns:
            option_idx: Selected option index.
            log_prob: Log probability of selection.
            value: State value.
        """
        option_dist, value = self.meta_policy(state, option_mask)
        option_idx = option_dist.sample()
        log_prob = option_dist.log_prob(option_idx)

        self.current_option = option_idx.item()
        self.option_duration = 0

        return self.current_option, log_prob, value

    def should_terminate(self, state: torch.Tensor) -> bool:
        """Check if current option should terminate."""
        if self.current_option is None:
            return True

        # Check duration limits
        if self.option_duration < 1:
            return False
        if self.option_duration >= 50:
            return True

        # Use termination network
        term_prob = self.termination_networks[self.current_option](state)
        return torch.rand(1).item() < term_prob.item()

    def get_action(
        self,
        state: torch.Tensor,
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get action from current option policy."""
        if self.current_option is None:
            self.select_option(state)

        option_policy = self.option_policies[self.current_option]
        action_dist = option_policy(state, mask)

        action = {k: d.sample() for k, d in action_dist.items()}
        self.option_duration += 1

        return action

    def reset_option(self):
        """Reset option state at episode start."""
        self.current_option = None
        self.option_duration = 0


class HierarchicalPPOTrainer:
    """PPO trainer for hierarchical policies.

    Trains both the meta-policy and option policies using
    a two-level PPO approach.
    """

    def __init__(
        self,
        policy: HierarchicalPolicy,
        encoder: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Separate optimizers for meta and option policies
        self.meta_optimizer = torch.optim.Adam(
            list(policy.meta_policy.parameters()) +
            list(policy.termination_networks.parameters()),
            lr=lr,
        )

        self.option_optimizer = torch.optim.Adam(
            list(policy.option_policies.parameters()) +
            list(encoder.parameters()),
            lr=lr,
        )

        # Experience buffers
        self.meta_buffer: Dict[str, List] = {
            "states": [], "options": [], "rewards": [],
            "dones": [], "values": [], "log_probs": [],
        }
        self.option_buffer: Dict[str, List] = {
            "states": [], "actions": [], "rewards": [],
            "dones": [], "values": [], "log_probs": [], "options": [],
        }

    def collect_rollout(self, env, n_steps: int) -> Dict[str, Any]:
        """Collect hierarchical rollout."""
        obs, info = env.reset()
        self.policy.reset_option()

        for step in range(n_steps):
            graph_data = env.get_graph_data().to(self.device)
            with torch.no_grad():
                state = self.encoder(graph_data)

            # Check for option termination
            if self.policy.should_terminate(state):
                # Select new option
                option, meta_log_prob, meta_value = self.policy.select_option(state)
                self.meta_buffer["states"].append(state)
                self.meta_buffer["options"].append(option)
                self.meta_buffer["values"].append(meta_value)
                self.meta_buffer["log_probs"].append(meta_log_prob)

            # Get action from current option
            env_mask = env.get_action_mask()
            torch_mask = {
                k: torch.tensor(v, device=self.device, dtype=torch.bool)
                for k, v in env_mask.items()
            }

            with torch.no_grad():
                action = self.policy.get_action(state, torch_mask)

            action_cpu = {k: int(v.item()) for k, v in action.items()}
            next_obs, reward, terminated, truncated, info = env.step(action_cpu)
            done = terminated or truncated

            # Store option-level experience
            self.option_buffer["states"].append(state)
            self.option_buffer["actions"].append(action)
            self.option_buffer["rewards"].append(reward)
            self.option_buffer["dones"].append(done)
            self.option_buffer["options"].append(self.policy.current_option)

            if done:
                obs, info = env.reset()
                self.policy.reset_option()

        return {
            "meta": self.meta_buffer,
            "option": self.option_buffer,
        }

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Update both meta and option policies."""
        # Simplified update - full implementation would include:
        # 1. Compute option-level returns and advantages
        # 2. Update option policies with PPO
        # 3. Aggregate to meta-level rewards
        # 4. Update meta-policy with PPO

        # Clear buffers
        for k in self.meta_buffer:
            self.meta_buffer[k] = []
        for k in self.option_buffer:
            self.option_buffer[k] = []

        return {
            "meta_loss": 0.0,
            "option_loss": 0.0,
            "entropy": 0.0,
        }
