"""Recurrent actor-critic policy with LSTM for temporal memory.

Implements PPO-LSTM for capturing temporal dependencies in shipyard
scheduling. The LSTM maintains hidden state across timesteps to remember
past actions and observations.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
with LSTM extensions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional, List


class RecurrentActorCriticPolicy(nn.Module):
    """Actor-critic policy with LSTM for temporal memory.

    The LSTM processes the sequence of encoded states and maintains
    hidden state to capture temporal patterns in scheduling.

    Args:
        state_dim: Dimension of encoded state (from GNN).
        n_action_types: Number of high-level action types.
        n_spmts: Number of SPMTs.
        n_cranes: Number of cranes.
        max_requests: Maximum transport requests.
        hidden_dim: Policy MLP hidden dimension.
        lstm_hidden_dim: LSTM hidden dimension.
        n_lstm_layers: Number of LSTM layers.
        epsilon: Epsilon for epsilon-greedy exploration.
    """

    def __init__(
        self,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 1,
        n_cranes: int = 1,
        max_requests: int = 1,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        n_lstm_layers: int = 1,
        epsilon: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.n_action_types = n_action_types
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes
        self.max_requests = max_requests
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.epsilon = epsilon

        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
        )

        # Shared layers after LSTM
        self.shared = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action heads (hierarchical)
        self.action_type_head = nn.Linear(hidden_dim, n_action_types)
        self.spmt_head = nn.Linear(hidden_dim, n_spmts)
        self.request_head = nn.Linear(hidden_dim, max_requests)
        self.crane_head = nn.Linear(hidden_dim, n_cranes)
        self.lift_head = nn.Linear(hidden_dim, max_requests)
        self.equipment_head = nn.Linear(hidden_dim, n_spmts + n_cranes)

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Hidden state storage
        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(self, batch_size: int = 1, device: str = "cpu"):
        """Reset LSTM hidden state at episode boundaries.

        Args:
            batch_size: Batch size for hidden state.
            device: Device for hidden state tensors.
        """
        self._hidden_state = (
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_dim, device=device),
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_dim, device=device),
        )

    def get_hidden_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get current hidden state (for storing in buffer)."""
        if self._hidden_state is None:
            return None
        return (
            self._hidden_state[0].detach().clone(),
            self._hidden_state[1].detach().clone(),
        )

    def set_hidden_state(self, hidden: Tuple[torch.Tensor, torch.Tensor]):
        """Set hidden state (for restoring from buffer)."""
        self._hidden_state = hidden

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply action mask to logits."""
        if mask.dtype != torch.bool:
            mask = mask.bool()
        return logits.masked_fill(~mask, -1e9)

    def forward(
        self,
        state: torch.Tensor,
        mask: Optional[Dict[str, torch.Tensor]] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, Categorical], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the recurrent policy.

        Args:
            state: Encoded state [batch, state_dim] or [batch, seq_len, state_dim]
            mask: Per-head action masks
            hidden: Optional hidden state (uses internal if None)

        Returns:
            action_dist: Dictionary of action distributions
            value: State value estimate
            new_hidden: Updated hidden state
        """
        # Handle input dimensions
        if state.dim() == 2:
            # Single timestep: [batch, state_dim] -> [batch, 1, state_dim]
            state = state.unsqueeze(1)

        batch_size = state.size(0)

        # Use provided hidden state or internal state
        if hidden is not None:
            h, c = hidden
        elif self._hidden_state is not None:
            h, c = self._hidden_state
        else:
            # Initialize hidden state
            device = state.device
            h = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
            c = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_dim, device=device)

        # LSTM forward
        lstm_out, (h_new, c_new) = self.lstm(state, (h, c))

        # Take last timestep output
        lstm_out = lstm_out[:, -1, :]  # [batch, lstm_hidden_dim]

        # Update internal hidden state
        self._hidden_state = (h_new.detach(), c_new.detach())

        # Shared layers
        shared_out = self.shared(lstm_out)

        # Action heads
        action_type_logits = self.action_type_head(shared_out)
        spmt_logits = self.spmt_head(shared_out)
        request_logits = self.request_head(shared_out)
        crane_logits = self.crane_head(shared_out)
        lift_logits = self.lift_head(shared_out)
        equipment_logits = self.equipment_head(shared_out)

        # Apply masks
        if mask is not None:
            if "action_type" in mask:
                action_type_logits = self._apply_mask(action_type_logits, mask["action_type"])
            if "spmt" in mask:
                spmt_logits = self._apply_mask(spmt_logits, mask["spmt"])
            if "request" in mask:
                request_logits = self._apply_mask(request_logits, mask["request"])
            if "crane" in mask:
                crane_logits = self._apply_mask(crane_logits, mask["crane"])
            if "lift" in mask:
                lift_logits = self._apply_mask(lift_logits, mask["lift"])
            if "equipment" in mask:
                equipment_logits = self._apply_mask(equipment_logits, mask["equipment"])

        # Create distributions
        action_dist = {
            "action_type": Categorical(logits=action_type_logits),
            "spmt": Categorical(logits=spmt_logits),
            "request": Categorical(logits=request_logits),
            "crane": Categorical(logits=crane_logits),
            "lift": Categorical(logits=lift_logits),
            "equipment": Categorical(logits=equipment_logits),
        }

        # Value estimate
        value = self.critic(shared_out)

        return action_dist, value, (h_new, c_new)

    def get_action(
        self,
        state: torch.Tensor,
        mask: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample an action from the policy.

        Args:
            state: Encoded state tensor.
            mask: Per-head action masks.
            deterministic: If True, select argmax action.
            hidden: Optional hidden state.

        Returns:
            action: Dictionary of action tensors.
            log_prob: Log probability of the action.
            value: State value estimate.
            new_hidden: Updated hidden state.
        """
        action_dist, value, new_hidden = self.forward(state, mask, hidden)

        if deterministic:
            action = {k: d.probs.argmax(dim=-1) for k, d in action_dist.items()}
        else:
            # Epsilon-greedy exploration
            if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
                action = {}
                for k, d in action_dist.items():
                    probs = d.probs
                    valid_mask = probs > 1e-8
                    if valid_mask.any():
                        uniform_probs = valid_mask.float() / valid_mask.sum()
                        action[k] = torch.multinomial(uniform_probs.squeeze(0), 1).squeeze()
                    else:
                        action[k] = d.sample().squeeze(0)
            else:
                action = {k: d.sample().squeeze(0) for k, d in action_dist.items()}

        log_prob = self._compute_log_prob(action_dist, action)

        return action, log_prob, value.squeeze(0), new_hidden

    def _compute_log_prob(
        self,
        action_dist: Dict[str, Categorical],
        action: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute log probability of action under distribution."""
        log_probs = []
        for key in ["action_type", "spmt", "request", "crane", "lift", "equipment"]:
            if key in action_dist and key in action:
                lp = action_dist[key].log_prob(action[key])
                log_probs.append(lp)

        return sum(log_probs)

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate actions for PPO update.

        Args:
            state: Batch of encoded states.
            action: Batch of actions.
            mask: Per-head action masks.
            hidden: Optional hidden states.

        Returns:
            log_prob: Log probabilities.
            entropy: Policy entropy.
            value: Value estimates.
            new_hidden: Updated hidden states.
        """
        action_dist, value, new_hidden = self.forward(state, mask, hidden)

        log_prob = self._compute_log_prob(action_dist, action)
        entropy = self._compute_entropy(action_dist, action)

        return log_prob, entropy, value.squeeze(-1), new_hidden

    def _compute_entropy(
        self,
        action_dist: Dict[str, Categorical],
        action: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute entropy of action distribution."""
        # Only count entropy from relevant heads based on action_type
        action_type = action.get("action_type", torch.zeros(1, dtype=torch.long))

        entropy = action_dist["action_type"].entropy()

        # Add entropy from relevant sub-heads
        if action_type.item() == 0:  # SPMT dispatch
            entropy = entropy + action_dist["spmt"].entropy() + action_dist["request"].entropy()
        elif action_type.item() == 1:  # Crane dispatch
            entropy = entropy + action_dist["crane"].entropy() + action_dist["lift"].entropy()
        elif action_type.item() == 2:  # Maintenance
            entropy = entropy + action_dist["equipment"].entropy()
        # action_type == 3 (hold): no additional heads

        return entropy


class RecurrentPPOTrainer:
    """PPO trainer with recurrent policy support.

    Handles sequence-based rollouts and hidden state management
    for training recurrent policies.
    """

    def __init__(
        self,
        policy: RecurrentActorCriticPolicy,
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
        sequence_length: int = 8,
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
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.encoder.parameters()),
            lr=lr,
        )

        # Buffer for sequences
        self.buffer: Dict[str, List] = {
            "graph_data": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "masks": [],
            "hidden_states": [],
        }

    def collect_rollout(self, env, n_steps: int, reset: bool = True) -> Dict:
        """Collect rollout with hidden state tracking."""
        if reset:
            obs, info = env.reset()
            self.policy.reset_hidden(batch_size=1, device=self.device)

        for step in range(n_steps):
            graph_data = env.get_graph_data().to(self.device)

            with torch.no_grad():
                state = self.encoder(graph_data)

            # Store hidden state before action
            hidden_state = self.policy.get_hidden_state()

            # Get action mask
            env_mask = env.get_action_mask()
            # Convert to torch masks (simplified)
            torch_mask = {
                k: torch.tensor(v, device=self.device, dtype=torch.bool)
                for k, v in env_mask.items()
            }

            with torch.no_grad():
                action, log_prob, value, _ = self.policy.get_action(state, torch_mask)

            action_cpu = {k: int(v.item()) for k, v in action.items()}
            next_obs, reward, terminated, truncated, info = env.step(action_cpu)
            done = terminated or truncated

            # Store experience
            self.buffer["graph_data"].append(graph_data.cpu())
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(reward)
            self.buffer["dones"].append(done)
            self.buffer["log_probs"].append(log_prob)
            self.buffer["values"].append(value.detach())
            self.buffer["masks"].append(torch_mask)
            self.buffer["hidden_states"].append(hidden_state)

            if done:
                obs, info = env.reset()
                self.policy.reset_hidden(batch_size=1, device=self.device)

        return self._compute_returns_and_advantages()

    def _compute_returns_and_advantages(self) -> Dict:
        """Compute GAE advantages."""
        rewards = self.buffer["rewards"]
        values = torch.stack(self.buffer["values"])
        dones = self.buffer["dones"]

        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "graph_data": self.buffer["graph_data"],
            "actions": self.buffer["actions"],
            "log_probs": torch.stack(self.buffer["log_probs"]),
            "returns": returns_tensor,
            "advantages": advantages,
            "masks": self.buffer["masks"],
            "hidden_states": self.buffer["hidden_states"],
            "values": values,
        }

    def update(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform PPO update with sequence handling."""
        # Simplified update - full implementation would handle sequences properly
        # Clear buffer after update
        for k in self.buffer:
            self.buffer[k] = []

        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
