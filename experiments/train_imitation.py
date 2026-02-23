#!/usr/bin/env python3
"""Imitation Learning: Pretrain policy on rule-based expert demonstrations.

This script collects expert trajectories from the rule-based (EDD) scheduler,
then trains the policy via behavioral cloning before fine-tuning with PPO.

Usage:
    python experiments/train_imitation.py --config config/small_instance.yaml --epochs 50
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from agent.ppo import PPOTrainer
from baselines.rule_based import RuleBasedScheduler


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


class ExpertDataset(Dataset):
    """Dataset of (state, action) pairs from expert demonstrations."""

    def __init__(self, states: List[torch.Tensor], actions: List[Dict[str, int]],
                 masks: List[Dict[str, torch.Tensor]]):
        self.states = states
        self.actions = actions
        self.masks = masks

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.masks[idx]


def collect_expert_demos(
    env: ShipyardEnv,
    encoder: HeterogeneousGNNEncoder,
    n_episodes: int = 10,
    max_steps: int = 500,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[Dict[str, int]], List[Dict[str, torch.Tensor]]]:
    """Collect demonstrations from rule-based expert."""
    expert = RuleBasedScheduler()
    states = []
    actions = []
    masks = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        for step in range(max_steps):
            # Get graph data and encode state
            graph_data = env.get_graph_data().to(device)
            with torch.no_grad():
                state_emb = encoder(graph_data)

            # Get action mask
            env_mask = env.get_action_mask()

            # Get expert action (RuleBasedScheduler uses 'decide' method)
            expert_action = expert.decide(env)

            # Convert mask to torch tensors
            torch_mask = {}
            for k, v in env_mask.items():
                if isinstance(v, np.ndarray):
                    torch_mask[k] = torch.tensor(v, dtype=torch.bool, device=device)

            # Store demonstration
            states.append(state_emb.cpu())
            actions.append(expert_action)
            masks.append({k: v.cpu() for k, v in torch_mask.items()})

            # Step environment
            obs, reward, terminated, truncated, info = env.step(expert_action)
            if terminated or truncated:
                break

    print(f"Collected {len(states)} expert demonstrations from {n_episodes} episodes")
    return states, actions, masks


def behavioral_cloning(
    policy: ActorCriticPolicy,
    states: List[torch.Tensor],
    actions: List[Dict[str, int]],
    masks: List[Dict[str, torch.Tensor]],
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """Train policy via behavioral cloning on expert demonstrations."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []

    # Prepare data
    n_samples = len(states)
    indices = list(range(n_samples))

    for epoch in range(n_epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < 2:
                continue

            # Batch states
            batch_states = torch.cat([states[j] for j in batch_indices], dim=0).to(device)

            # Get policy distributions (no mask during BC for simplicity)
            action_dist, _ = policy.forward(batch_states)

            # Compute cross-entropy loss for each action head
            loss = 0.0
            for head_name in ["action_type", "spmt", "request", "crane", "lift", "equipment"]:
                target = torch.tensor(
                    [actions[j].get(head_name.replace("spmt", "spmt_idx")
                                   .replace("request", "request_idx")
                                   .replace("crane", "crane_idx")
                                   .replace("lift", "lift_idx")
                                   .replace("equipment", "equipment_idx"), 0)
                     for j in batch_indices],
                    device=device
                )
                # Clamp target to valid range
                max_idx = action_dist[head_name].probs.shape[-1] - 1
                target = target.clamp(0, max_idx)

                head_loss = F.cross_entropy(action_dist[head_name].logits, target)
                loss += head_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"BC Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")

    return losses


def evaluate_policy(
    env: ShipyardEnv,
    policy: ActorCriticPolicy,
    encoder: HeterogeneousGNNEncoder,
    n_episodes: int = 3,
    max_steps: int = 500,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate trained policy."""
    total_reward = 0.0
    total_throughput = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            graph_data = env.get_graph_data().to(device)
            with torch.no_grad():
                state_emb = encoder(graph_data)
                action, _, _ = policy.get_action(state_emb, deterministic=True)

            action_cpu = {k: int(v.item()) for k, v in action.items()}
            obs, reward, terminated, truncated, info = env.step(action_cpu)
            ep_reward += reward

            if terminated or truncated:
                break

        total_reward += ep_reward
        if env.sim_time > 0:
            total_throughput += env.metrics["blocks_completed"] / env.sim_time

    return {
        "avg_reward": total_reward / n_episodes,
        "avg_throughput": total_throughput / n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Imitation Learning for Shipyard Scheduling")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--bc-epochs", type=int, default=100, help="Behavioral cloning epochs")
    parser.add_argument("--rl-epochs", type=int, default=50, help="RL fine-tuning epochs")
    parser.add_argument("--demo-episodes", type=int, default=20, help="Expert demo episodes")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="data/checkpoints/imitation/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config and create environment
    cfg = load_config(args.config)
    env = ShipyardEnv(cfg)

    # Create encoder and policy
    hidden_dim = 128
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=2,
    ).to(args.device)

    state_dim = hidden_dim * 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=256,
        epsilon=0.1,  # Low epsilon after BC
    ).to(args.device)

    print("=" * 60)
    print("PHASE 1: Collecting Expert Demonstrations")
    print("=" * 60)

    states, actions, masks = collect_expert_demos(
        env, encoder, n_episodes=args.demo_episodes, device=args.device
    )

    print("\n" + "=" * 60)
    print("PHASE 2: Behavioral Cloning")
    print("=" * 60)

    bc_losses = behavioral_cloning(
        policy, states, actions, masks,
        n_epochs=args.bc_epochs,
        device=args.device,
    )

    # Evaluate after BC
    print("\nEvaluating after Behavioral Cloning:")
    bc_metrics = evaluate_policy(env, policy, encoder, device=args.device)
    print(f"  Avg Reward: {bc_metrics['avg_reward']:.2f}")
    print(f"  Avg Throughput: {bc_metrics['avg_throughput']:.4f}")

    print("\n" + "=" * 60)
    print("PHASE 3: RL Fine-tuning with PPO")
    print("=" * 60)

    # Create PPO trainer for fine-tuning
    trainer = PPOTrainer(
        policy=policy,
        encoder=encoder,
        lr=1e-4,  # Lower LR for fine-tuning
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.5,  # Lower entropy since we have good initialization
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        device=args.device,
        total_epochs=args.rl_epochs,
        entropy_schedule="cosine",
        entropy_coef_final=0.05,
    )

    for epoch in range(args.rl_epochs):
        rollout = trainer.collect_rollout(env, n_steps=200)
        metrics = trainer.update(rollout)

        throughput = env.metrics["blocks_completed"] / max(env.sim_time, 1.0)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.rl_epochs} - "
                  f"PolicyLoss: {metrics['policy_loss']:.3f}, "
                  f"Entropy: {metrics['entropy']:.3f}, "
                  f"Throughput: {throughput:.4f}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_metrics = evaluate_policy(env, policy, encoder, device=args.device)
    print(f"  Avg Reward: {final_metrics['avg_reward']:.2f}")
    print(f"  Avg Throughput: {final_metrics['avg_throughput']:.4f}")

    # Save checkpoint
    os.makedirs(args.save, exist_ok=True)
    trainer.save_checkpoint(os.path.join(args.save, "imitation_final.pt"))
    print(f"\nCheckpoint saved to {args.save}")


if __name__ == "__main__":
    main()
