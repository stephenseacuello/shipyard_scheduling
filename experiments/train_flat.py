#!/usr/bin/env python3
"""Flat Action Space Training for Shipyard Scheduling.

Instead of hierarchical action heads, uses a single flat action space:
- Actions 0 to n_spmts*n_blocks-1: SPMT dispatches
- Actions n_spmts*n_blocks to n_spmts*n_blocks+n_cranes*n_blocks-1: Crane dispatches
- Action n_spmts*n_blocks+n_cranes*n_blocks: Maintenance (first equipment)
- Last action: HOLD

This simplifies the policy learning problem at the cost of a larger action space.

Usage:
    python experiments/train_flat.py --config config/small_instance.yaml --epochs 100
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import yaml
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


class FlatPolicy(nn.Module):
    """Single-head policy with flat action space."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_actions = n_actions

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        features = self.network(state)
        logits = self.actor(features)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        dist = Categorical(logits=logits)
        value = self.critic(features)
        return dist, value

    def get_action(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False):
        dist, value = self.forward(state, mask)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, value, entropy


class FlatActionMapper:
    """Maps flat action indices to hierarchical environment actions."""

    def __init__(self, n_spmts: int, n_cranes: int, n_blocks: int, n_equipment: int):
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes
        self.n_blocks = n_blocks
        self.n_equipment = n_equipment

        # Action space layout:
        # [SPMT dispatches] [Crane dispatches] [Maintenance] [HOLD]
        self.spmt_offset = 0
        self.crane_offset = n_spmts * n_blocks
        self.maint_offset = self.crane_offset + n_cranes * n_blocks
        self.hold_action = self.maint_offset + n_equipment

        self.n_actions = self.hold_action + 1

    def to_env_action(self, flat_action: int) -> Dict[str, int]:
        """Convert flat action index to environment action dict."""
        if flat_action < self.crane_offset:
            # SPMT dispatch
            spmt_idx = flat_action // self.n_blocks
            req_idx = flat_action % self.n_blocks
            return {
                "action_type": 0,
                "spmt_idx": spmt_idx,
                "request_idx": req_idx,
            }
        elif flat_action < self.maint_offset:
            # Crane dispatch
            adj = flat_action - self.crane_offset
            crane_idx = adj // self.n_blocks
            lift_idx = adj % self.n_blocks
            return {
                "action_type": 1,
                "crane_idx": crane_idx,
                "lift_idx": lift_idx,
            }
        elif flat_action < self.hold_action:
            # Maintenance
            equip_idx = flat_action - self.maint_offset
            return {
                "action_type": 2,
                "equipment_idx": equip_idx,
            }
        else:
            # HOLD
            return {"action_type": 3}

    def get_flat_mask(self, env) -> np.ndarray:
        """Convert environment mask to flat action mask."""
        env_mask = env.get_action_mask()
        flat_mask = np.zeros(self.n_actions, dtype=bool)

        # SPMT dispatch actions
        if env_mask["action_type"][0]:
            spmt_mask = env_mask.get("spmt_dispatch", np.zeros((self.n_spmts, self.n_blocks)))
            for i in range(self.n_spmts):
                for j in range(min(self.n_blocks, spmt_mask.shape[1] if len(spmt_mask.shape) > 1 else 0)):
                    if spmt_mask[i, j] if len(spmt_mask.shape) > 1 else False:
                        flat_idx = self.spmt_offset + i * self.n_blocks + j
                        if flat_idx < self.n_actions:
                            flat_mask[flat_idx] = True

        # Crane dispatch actions
        if env_mask["action_type"][1]:
            crane_mask = env_mask.get("crane_dispatch", np.zeros((self.n_cranes, self.n_blocks)))
            for i in range(self.n_cranes):
                for j in range(min(self.n_blocks, crane_mask.shape[1] if len(crane_mask.shape) > 1 else 0)):
                    if crane_mask[i, j] if len(crane_mask.shape) > 1 else False:
                        flat_idx = self.crane_offset + i * self.n_blocks + j
                        if flat_idx < self.n_actions:
                            flat_mask[flat_idx] = True

        # Maintenance actions
        if env_mask["action_type"][2]:
            maint_mask = env_mask.get("maintenance", np.zeros(self.n_equipment))
            for i in range(min(self.n_equipment, len(maint_mask))):
                if maint_mask[i]:
                    flat_idx = self.maint_offset + i
                    if flat_idx < self.n_actions:
                        flat_mask[flat_idx] = True

        # HOLD is always valid
        if env_mask["action_type"][3]:
            flat_mask[self.hold_action] = True

        # Ensure at least one action is valid
        if not flat_mask.any():
            flat_mask[self.hold_action] = True

        return flat_mask


class FlatPPOTrainer:
    """Simple PPO trainer for flat action space."""

    def __init__(
        self,
        policy: FlatPolicy,
        encoder: HeterogeneousGNNEncoder,
        action_mapper: FlatActionMapper,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.1,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.action_mapper = action_mapper
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(encoder.parameters()), lr=lr
        )

        self.buffer = {
            "states": [], "actions": [], "rewards": [], "dones": [],
            "log_probs": [], "values": [], "masks": [],
        }

    def collect_rollout(self, env, n_steps: int):
        """Collect experience from environment."""
        self.buffer = {k: [] for k in self.buffer}
        obs, info = env.reset()

        for _ in range(n_steps):
            graph_data = env.get_graph_data().to(self.device)
            flat_mask = self.action_mapper.get_flat_mask(env)
            mask_tensor = torch.tensor(flat_mask, dtype=torch.bool, device=self.device)

            with torch.no_grad():
                state = self.encoder(graph_data)
                action, log_prob, value, entropy = self.policy.get_action(state, mask_tensor)

            # Convert to environment action
            env_action = self.action_mapper.to_env_action(action.item())
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            self.buffer["states"].append(state.cpu())
            self.buffer["actions"].append(action.cpu())
            self.buffer["rewards"].append(reward)
            self.buffer["dones"].append(done)
            self.buffer["log_probs"].append(log_prob.cpu())
            self.buffer["values"].append(value.squeeze().cpu())
            self.buffer["masks"].append(mask_tensor.cpu())

            if done:
                obs, info = env.reset()

        return self._compute_advantages()

    def _compute_advantages(self):
        """Compute GAE advantages."""
        rewards = self.buffer["rewards"]
        values = torch.stack(self.buffer["values"])
        dones = self.buffer["dones"]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": self.buffer["states"],
            "actions": torch.stack(self.buffer["actions"]),
            "log_probs": torch.stack(self.buffer["log_probs"]),
            "returns": returns,
            "advantages": advantages,
            "masks": self.buffer["masks"],
        }

    def update(self, rollout, n_epochs: int = 4, batch_size: int = 64):
        """PPO update."""
        states = rollout["states"]
        actions = rollout["actions"].to(self.device)
        old_log_probs = rollout["log_probs"].to(self.device)
        returns = rollout["returns"].to(self.device)
        advantages = rollout["advantages"].to(self.device)
        masks = rollout["masks"]

        n_samples = len(states)
        indices = list(range(n_samples))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            random.shuffle(indices)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]

                # Batch data
                batch_states = torch.cat([states[j] for j in batch_idx]).to(self.device)
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Forward pass
                dist, values = self.policy.forward(batch_states)

                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.encoder.parameters()), 0.5
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="Flat Action PPO for Shipyard Scheduling")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="data/checkpoints/flat/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config and create environment
    cfg = load_config(args.config)
    env = ShipyardEnv(cfg)

    # Create action mapper
    n_equipment = env.n_spmts + env.n_cranes
    action_mapper = FlatActionMapper(
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        n_blocks=env.n_blocks,
        n_equipment=n_equipment,
    )

    print(f"Flat action space size: {action_mapper.n_actions}")
    print(f"  SPMT dispatches: {env.n_spmts * env.n_blocks}")
    print(f"  Crane dispatches: {env.n_cranes * env.n_blocks}")
    print(f"  Maintenance: {n_equipment}")
    print(f"  HOLD: 1")

    # Create networks
    hidden_dim = 128
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=2,
    )

    state_dim = hidden_dim * 4
    policy = FlatPolicy(
        state_dim=state_dim,
        n_actions=action_mapper.n_actions,
        hidden_dim=256,
    )

    trainer = FlatPPOTrainer(
        policy=policy,
        encoder=encoder,
        action_mapper=action_mapper,
        entropy_coef=0.1,  # Higher entropy for large action space
        device=args.device,
    )

    print("=" * 60)
    print("Flat Action PPO Training")
    print("=" * 60)

    for epoch in range(args.epochs):
        rollout = trainer.collect_rollout(env, args.steps)
        metrics = trainer.update(rollout)

        throughput = env.metrics["blocks_completed"] / max(env.sim_time, 1.0)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"PolicyLoss: {metrics['policy_loss']:.3f}, "
                  f"Entropy: {metrics['entropy']:.3f}, "
                  f"Throughput: {throughput:.4f}")

    # Save
    os.makedirs(args.save, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "encoder": encoder.state_dict(),
    }, os.path.join(args.save, "flat_final.pt"))
    print(f"\nCheckpoint saved to {args.save}")


if __name__ == "__main__":
    main()
