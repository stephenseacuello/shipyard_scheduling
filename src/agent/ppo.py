"""Proximal Policy Optimization (PPO) trainer for shipyard scheduling RL.

This class manages experience collection, return and advantage computation
using Generalized Advantage Estimation (GAE) and performs mini-batch updates
on the actor and critic networks. The encoder is re-run on stored graph data
during updates so that encoder gradients flow through backprop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Any

from .action_masking import (
    flatten_env_mask_to_policy_mask,
    batch_masks,
    to_torch_mask,
)


class PPOTrainer:
    def __init__(
        self,
        policy: nn.Module,
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
    ) -> None:
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.encoder.parameters()),
            lr=lr,
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # Experience storage
        self.buffer: Dict[str, List] = {
            "graph_data": [],  # raw HeteroData objects for encoder re-encoding
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "masks": [],  # per-head policy masks (flattened)
        }

    def collect_rollout(self, env, n_steps: int) -> Dict[str, Any]:
        """Collect `n_steps` interactions from the environment."""
        obs, info = env.reset()
        for _ in range(n_steps):
            graph_data = env.get_graph_data()
            graph_data = graph_data.to(self.device)

            # Encode state
            with torch.no_grad():
                state_emb = self.encoder(graph_data)

            # Build per-head policy masks from environment mask
            env_mask = env.get_action_mask()
            policy_mask = flatten_env_mask_to_policy_mask(
                env_mask,
                n_spmts=self.policy.n_spmts,
                n_cranes=self.policy.n_cranes,
                max_requests=self.policy.max_requests,
            )
            torch_mask = to_torch_mask(policy_mask, device=self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_emb, torch_mask)

            # Convert action tensors to python ints for env.step
            action_cpu = {k: int(v.item()) for k, v in action.items()}
            next_obs, reward, terminated, truncated, info = env.step(action_cpu)
            done = terminated or truncated

            # Store experience — keep raw graph data for re-encoding during update
            self.buffer["graph_data"].append(graph_data.cpu())
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(reward)
            self.buffer["dones"].append(done)
            self.buffer["log_probs"].append(log_prob)
            self.buffer["values"].append(value.squeeze(0).detach())
            self.buffer["masks"].append(torch_mask)

            if done:
                obs, info = env.reset()
        return self._compute_returns_and_advantages()

    def _compute_returns_and_advantages(self) -> Dict[str, Any]:
        """Compute GAE advantages and discounted returns."""
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
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "graph_data": self.buffer["graph_data"],
            "actions": self.buffer["actions"],
            "log_probs": torch.stack(self.buffer["log_probs"]),
            "returns": returns,
            "advantages": advantages,
            "masks": self.buffer["masks"],
        }

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform PPO updates using collected rollout data."""
        graph_data_list = rollout_data["graph_data"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        returns = rollout_data["returns"]
        advantages = rollout_data["advantages"]
        masks_list = rollout_data["masks"]

        n_samples = len(graph_data_list)
        metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Batch graph data and re-encode through encoder (gradients flow)
                batch_graphs = Batch.from_data_list(
                    [graph_data_list[i] for i in batch_indices]
                ).to(self.device)
                batch_states = self.encoder(batch_graphs)

                # Batch actions
                batch_actions = {
                    k: torch.stack([actions[i][k] for i in batch_indices])
                    for k in actions[0]
                }
                # Batch masks
                batch_mask = batch_masks([masks_list[i] for i in batch_indices])

                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Policy evaluate with masks
                log_probs, entropy, values = self.policy.evaluate_action(
                    batch_states, batch_actions, batch_mask
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.encoder.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []
        return {k: float(np.mean(v)) for k, v in metrics.items()}
