#!/usr/bin/env python3
"""SAC Training Script for Shipyard Scheduling.

Uses Soft Actor-Critic with automatic entropy tuning for more robust exploration.
Adapts the hierarchical action space to work with SAC's off-policy learning.

Usage:
    python experiments/train_sac.py --config config/small_instance.yaml --epochs 100
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
from collections import deque
import yaml
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.action_masking import flatten_env_mask_to_policy_mask, to_torch_mask


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


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return states, actions, rewards, next_states, dones, masks

    def __len__(self):
        return len(self.buffer)


class SACPolicy(nn.Module):
    """SAC-style policy for hierarchical action space."""

    def __init__(
        self,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 1,
        n_cranes: int = 1,
        max_requests: int = 50,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_action_types = n_action_types

        # Shared encoder
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
        self.lift_head = nn.Linear(hidden_dim, max_requests)
        self.equipment_head = nn.Linear(hidden_dim, n_spmts + n_cranes)

    def forward(self, state: torch.Tensor, mask: Optional[Dict] = None):
        features = self.shared(state)

        logits = {
            "action_type": self.action_type_head(features),
            "spmt": self.spmt_head(features),
            "request": self.request_head(features),
            "crane": self.crane_head(features),
            "lift": self.lift_head(features),
            "equipment": self.equipment_head(features),
        }

        # Apply masking
        if mask is not None:
            for key in logits:
                m = mask.get(key)
                if m is not None:
                    if m.dtype != torch.bool:
                        m = m.bool()
                    logits[key] = logits[key].masked_fill(~m, -20.0)

        # Convert to distributions
        dists = {k: Categorical(logits=v) for k, v in logits.items()}
        return dists, logits

    def get_action(self, state: torch.Tensor, mask: Optional[Dict] = None):
        dists, logits = self.forward(state, mask)

        # Sample from each head
        action = {k: d.sample() for k, d in dists.items()}

        # Compute log probs and entropy
        log_prob = sum(dists[k].log_prob(action[k]) for k in action)
        entropy = sum(d.entropy() for d in dists.values())

        return action, log_prob, entropy


class SACQNetwork(nn.Module):
    """Q-network for SAC - outputs Q-value for action type."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class SACTrainer:
    """SAC trainer for shipyard scheduling."""

    def __init__(
        self,
        policy: SACPolicy,
        encoder: HeterogeneousGNNEncoder,
        state_dim: int,
        n_action_types: int = 4,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Twin Q-networks
        self.q1 = SACQNetwork(state_dim, n_action_types, hidden_dim).to(device)
        self.q2 = SACQNetwork(state_dim, n_action_types, hidden_dim).to(device)
        self.q1_target = SACQNetwork(state_dim, n_action_types, hidden_dim).to(device)
        self.q2_target = SACQNetwork(state_dim, n_action_types, hidden_dim).to(device)

        # Copy weights to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Automatic entropy tuning
        self.target_entropy = -np.log(1.0 / n_action_types) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(encoder.parameters()), lr=lr
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, env, deterministic: bool = False):
        """Select action using current policy."""
        graph_data = env.get_graph_data().to(self.device)

        with torch.no_grad():
            state = self.encoder(graph_data)

        env_mask = env.get_action_mask()
        policy_mask = flatten_env_mask_to_policy_mask(
            env_mask, self.policy.spmt_head.out_features,
            self.policy.crane_head.out_features,
            self.policy.request_head.out_features,
        )
        torch_mask = to_torch_mask(policy_mask, device=self.device)

        with torch.no_grad():
            action, _, _ = self.policy.get_action(state, torch_mask)

        action_cpu = {k: int(v.item()) for k, v in action.items()}
        return action_cpu, state.cpu(), torch_mask

    def update(self):
        """Perform one SAC update step."""
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones, masks = self.buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.cat(states, dim=0).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        action_types = torch.tensor(
            [a["action_type"] for a in actions], device=self.device
        )

        # Compute target Q-values
        with torch.no_grad():
            next_dists, _ = self.policy.forward(next_states)
            next_probs = next_dists["action_type"].probs
            next_log_probs = torch.log(next_probs + 1e-8)

            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)

            # Soft value = E[Q - alpha * log(pi)]
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        # Update Q-networks
        q1_values = self.q1(states).gather(1, action_types.unsqueeze(1)).squeeze()
        q2_values = self.q2(states).gather(1, action_types.unsqueeze(1)).squeeze()

        q1_loss = F.mse_loss(q1_values, target_q)
        q2_loss = F.mse_loss(q2_values, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        dists, _ = self.policy.forward(states)
        probs = dists["action_type"].probs
        log_probs = torch.log(probs + 1e-8)

        q1_values = self.q1(states)
        q2_values = self.q2(states)
        q_values = torch.min(q1_values, q2_values)

        policy_loss = (probs * (self.alpha.detach() * log_probs - q_values)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update alpha (entropy coefficient)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft update targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": entropy.item(),
        }


def main():
    parser = argparse.ArgumentParser(description="SAC Training for Shipyard Scheduling")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="data/checkpoints/sac/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config and create environment
    cfg = load_config(args.config)
    env = HHIShipyardEnv(cfg)

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
    policy = SACPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2)),
        max_requests=env.n_blocks,
        hidden_dim=256,
    )

    trainer = SACTrainer(
        policy=policy,
        encoder=encoder,
        state_dim=state_dim,
        n_action_types=4,
        hidden_dim=256,
        device=args.device,
    )

    print("=" * 60)
    print("SAC Training for Shipyard Scheduling")
    print("=" * 60)

    for epoch in range(args.epochs):
        obs, info = env.reset()
        epoch_reward = 0.0

        for step in range(args.steps):
            # Select action
            action, state, mask = trainer.select_action(env)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get next state
            next_graph = env.get_graph_data().to(args.device)
            with torch.no_grad():
                next_state = trainer.encoder(next_graph).cpu()

            # Store transition
            trainer.buffer.push(state, action, reward, next_state, done, mask)

            epoch_reward += reward

            # Update
            if len(trainer.buffer) >= trainer.batch_size:
                trainer.update()

            if done:
                obs, info = env.reset()

        # Compute metrics
        throughput = env.metrics["blocks_completed"] / max(env.sim_time, 1.0)
        metrics = trainer.update() if len(trainer.buffer) >= trainer.batch_size else {}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Reward: {epoch_reward:.1f}, "
                  f"Throughput: {throughput:.4f}, "
                  f"Alpha: {trainer.alpha.item():.3f}, "
                  f"Entropy: {metrics.get('entropy', 0):.3f}")

    # Save
    os.makedirs(args.save, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "encoder": encoder.state_dict(),
        "q1": trainer.q1.state_dict(),
        "q2": trainer.q2.state_dict(),
    }, os.path.join(args.save, "sac_final.pt"))
    print(f"\nCheckpoint saved to {args.save}")


if __name__ == "__main__":
    main()
