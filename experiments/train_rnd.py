#!/usr/bin/env python3
"""PPO + RND (Random Network Distillation) for Curiosity-Driven Exploration.

Uses intrinsic motivation from prediction error on a random target network
to encourage exploration of novel states.

Reference: Burda et al., "Exploration by Random Network Distillation" (2019)

Usage:
    python experiments/train_rnd.py --config config/small_instance.yaml --epochs 100
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
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


class RNDNetwork(nn.Module):
    """Random Network Distillation predictor/target network."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RNDModule:
    """Random Network Distillation for intrinsic motivation."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device

        # Target network (fixed, random initialization)
        self.target = RNDNetwork(state_dim, hidden_dim, output_dim).to(device)
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network (trained to match target)
        self.predictor = RNDNetwork(state_dim, hidden_dim, output_dim).to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def compute_intrinsic_reward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward as prediction error."""
        with torch.no_grad():
            target_features = self.target(state)
            predictor_features = self.predictor(state)

        # MSE between predictor and target
        intrinsic_reward = ((target_features - predictor_features) ** 2).mean(dim=-1)

        return intrinsic_reward

    def update(self, states: torch.Tensor) -> float:
        """Train predictor to match target."""
        with torch.no_grad():
            target_features = self.target(states)

        predictor_features = self.predictor(states)
        loss = F.mse_loss(predictor_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize intrinsic reward using running statistics."""
        # Update running stats
        batch_mean = reward.mean().item()
        batch_std = reward.std().item() + 1e-8

        self.reward_count += reward.numel()
        alpha = min(1.0, reward.numel() / self.reward_count)
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
        self.reward_std = (1 - alpha) * self.reward_std + alpha * batch_std

        # Normalize
        normalized = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return normalized.clamp(-5.0, 5.0)


class RNDPPOTrainer:
    """PPO trainer with RND intrinsic motivation."""

    def __init__(
        self,
        policy: ActorCriticPolicy,
        encoder: HeterogeneousGNNEncoder,
        state_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.1,
        value_coef: float = 0.5,
        intrinsic_coef: float = 0.5,  # Weight for intrinsic reward
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
        self.intrinsic_coef = intrinsic_coef

        # RND module
        self.rnd = RNDModule(state_dim, device=device)

        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(encoder.parameters()), lr=lr
        )

        self.buffer = {
            "graph_data": [], "actions": [], "rewards": [], "intrinsic_rewards": [],
            "dones": [], "log_probs": [], "values": [], "masks": [], "states": [],
        }

    def collect_rollout(self, env, n_steps: int):
        """Collect experience with intrinsic rewards."""
        self.buffer = {k: [] for k in self.buffer}
        obs, info = env.reset()

        for _ in range(n_steps):
            graph_data = env.get_graph_data().to(self.device)

            # Get action mask
            env_mask = env.get_action_mask()
            policy_mask = flatten_env_mask_to_policy_mask(
                env_mask, self.policy.n_spmts, self.policy.n_cranes, self.policy.max_requests
            )
            torch_mask = to_torch_mask(policy_mask, device=self.device)

            # Encode state
            with torch.no_grad():
                state = self.encoder(graph_data)
                action, log_prob, value = self.policy.get_action(state, torch_mask)

                # Compute intrinsic reward
                intrinsic_reward = self.rnd.compute_intrinsic_reward(state)

            # Step environment
            action_cpu = {k: int(v.item()) for k, v in action.items()}
            next_obs, extrinsic_reward, terminated, truncated, info = env.step(action_cpu)
            done = terminated or truncated

            # Store experience
            self.buffer["graph_data"].append(graph_data.cpu())
            self.buffer["states"].append(state.cpu())
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(extrinsic_reward)
            self.buffer["intrinsic_rewards"].append(intrinsic_reward.item())
            self.buffer["dones"].append(done)
            self.buffer["log_probs"].append(log_prob.cpu())
            self.buffer["values"].append(value.squeeze().cpu())
            self.buffer["masks"].append(torch_mask)

            if done:
                obs, info = env.reset()

        return self._compute_advantages()

    def _compute_advantages(self):
        """Compute GAE with combined rewards."""
        extrinsic = self.buffer["rewards"]
        intrinsic = self.buffer["intrinsic_rewards"]
        values = torch.stack(self.buffer["values"])
        dones = self.buffer["dones"]

        # Combine rewards
        intrinsic_tensor = torch.tensor(intrinsic, dtype=torch.float32)
        intrinsic_normalized = self.rnd.normalize_reward(intrinsic_tensor)
        combined_rewards = [
            extrinsic[i] + self.intrinsic_coef * intrinsic_normalized[i].item()
            for i in range(len(extrinsic))
        ]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(combined_rewards))):
            next_value = 0.0 if t == len(combined_rewards) - 1 else values[t + 1]
            delta = combined_rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "graph_data": self.buffer["graph_data"],
            "states": self.buffer["states"],
            "actions": self.buffer["actions"],
            "log_probs": torch.stack(self.buffer["log_probs"]),
            "returns": returns,
            "advantages": advantages,
            "masks": self.buffer["masks"],
            "intrinsic_mean": np.mean(intrinsic),
        }

    def update(self, rollout, n_epochs: int = 4, batch_size: int = 64):
        """PPO update with RND training."""
        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"].to(self.device)
        returns = rollout["returns"].to(self.device)
        advantages = rollout["advantages"].to(self.device)
        masks = rollout["masks"]

        n_samples = len(states)
        indices = list(range(n_samples))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_rnd_loss = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            random.shuffle(indices)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                if len(batch_idx) < 2:
                    continue

                # Batch data
                batch_states = torch.cat([states[j] for j in batch_idx]).to(self.device)
                batch_actions = {
                    k: torch.stack([actions[j][k] for j in batch_idx]).to(self.device)
                    for k in actions[0]
                }
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Re-encode and evaluate
                log_probs, entropy, values = self.policy.evaluate_action(
                    batch_states, batch_actions
                )

                # PPO objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy_loss = -self.entropy_coef * entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.encoder.parameters()), 0.5
                )
                self.optimizer.step()

                # Update RND predictor
                rnd_loss = self.rnd.update(batch_states.detach())

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_rnd_loss += rnd_loss
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "rnd_loss": total_rnd_loss / max(n_updates, 1),
            "intrinsic_mean": rollout.get("intrinsic_mean", 0),
        }


def main():
    parser = argparse.ArgumentParser(description="PPO + RND for Shipyard Scheduling")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--intrinsic-coef", type=float, default=0.5)
    parser.add_argument("--save", type=str, default="data/checkpoints/rnd/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config and create environment
    cfg = load_config(args.config)
    env = ShipyardEnv(cfg)

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
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=256,
        epsilon=0.1,
    )

    trainer = RNDPPOTrainer(
        policy=policy,
        encoder=encoder,
        state_dim=state_dim,
        intrinsic_coef=args.intrinsic_coef,
        device=args.device,
    )

    print("=" * 60)
    print("PPO + RND (Curiosity-Driven) Training")
    print("=" * 60)
    print(f"Intrinsic reward coefficient: {args.intrinsic_coef}")

    for epoch in range(args.epochs):
        rollout = trainer.collect_rollout(env, args.steps)
        metrics = trainer.update(rollout)

        throughput = env.metrics["blocks_completed"] / max(env.sim_time, 1.0)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"PolicyLoss: {metrics['policy_loss']:.3f}, "
                  f"Entropy: {metrics['entropy']:.3f}, "
                  f"Intrinsic: {metrics['intrinsic_mean']:.4f}, "
                  f"Throughput: {throughput:.4f}")

    # Save
    os.makedirs(args.save, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "encoder": encoder.state_dict(),
        "rnd_predictor": trainer.rnd.predictor.state_dict(),
    }, os.path.join(args.save, "rnd_final.pt"))
    print(f"\nCheckpoint saved to {args.save}")


if __name__ == "__main__":
    main()
