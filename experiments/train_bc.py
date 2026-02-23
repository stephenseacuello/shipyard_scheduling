#!/usr/bin/env python3
"""Pure Behavioral Cloning for Shipyard Scheduling.

Trains policy purely via behavioral cloning from expert demonstrations.
No RL fine-tuning - this was found to degrade performance due to entropy collapse.

Key findings:
- BC after 30 epochs achieved throughput 0.033 (vs 0.085 for rule-based)
- RL fine-tuning degraded throughput to 0.013 due to entropy collapse
- More expert demos and longer training should improve BC performance

Usage:
    python experiments/train_bc.py --config config/small_instance.yaml --epochs 100 --demo-episodes 50
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
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


def collect_expert_demos(
    env: HHIShipyardEnv,
    encoder: HeterogeneousGNNEncoder,
    n_episodes: int = 50,
    max_steps: int = 1000,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[Dict[str, int]]]:
    """Collect demonstrations from rule-based expert."""
    expert = RuleBasedScheduler()
    states = []
    actions = []

    total_reward = 0.0
    total_throughput = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            # Get graph data and encode state
            graph_data = env.get_graph_data().to(device)
            with torch.no_grad():
                state_emb = encoder(graph_data)

            # Get expert action
            expert_action = expert.decide(env)

            # Store demonstration
            states.append(state_emb.cpu())
            actions.append(expert_action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(expert_action)
            ep_reward += reward

            if terminated or truncated:
                break

        total_reward += ep_reward
        if env.sim_time > 0:
            total_throughput += env.metrics["blocks_completed"] / env.sim_time

        if (ep + 1) % 10 == 0:
            print(f"  Collected episode {ep + 1}/{n_episodes}, "
                  f"avg throughput: {total_throughput / (ep + 1):.4f}")

    print(f"\nCollected {len(states)} demonstrations from {n_episodes} episodes")
    print(f"Expert avg throughput: {total_throughput / n_episodes:.4f}")

    return states, actions


def train_behavioral_cloning(
    policy: ActorCriticPolicy,
    states: List[torch.Tensor],
    actions: List[Dict[str, int]],
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: str = "cpu",
    weight_decay: float = 1e-5,
) -> List[float]:
    """Train policy via behavioral cloning."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    n_samples = len(states)
    indices = list(range(n_samples))

    # Action key mapping
    action_keys = {
        "action_type": "action_type",
        "spmt": "spmt_idx",
        "request": "request_idx",
        "crane": "crane_idx",
        "lift": "lift_idx",
        "equipment": "equipment_idx",
    }

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

            # Get policy distributions
            action_dist, _ = policy.forward(batch_states)

            # Compute cross-entropy loss for each action head
            loss = 0.0
            for head_name, action_key in action_keys.items():
                # Get target actions, defaulting to 0 if key not present
                target = torch.tensor(
                    [actions[j].get(action_key, 0) for j in batch_indices],
                    device=device
                )
                # Clamp target to valid range
                max_idx = action_dist[head_name].probs.shape[-1] - 1
                target = target.clamp(0, max_idx)

                head_loss = F.cross_entropy(action_dist[head_name].logits, target)

                # Weight action_type head more heavily (it's the primary decision)
                if head_name == "action_type":
                    loss += 2.0 * head_loss
                else:
                    loss += head_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            lr_current = scheduler.get_last_lr()[0]
            print(f"BC Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, LR: {lr_current:.2e}")

    return losses


def evaluate_policy(
    env: HHIShipyardEnv,
    policy: ActorCriticPolicy,
    encoder: HeterogeneousGNNEncoder,
    n_episodes: int = 5,
    max_steps: int = 1000,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate trained policy."""
    total_reward = 0.0
    total_throughput = 0.0
    total_tardiness = 0.0

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
        total_tardiness += env.metrics.get("total_tardiness", 0)

    return {
        "avg_reward": total_reward / n_episodes,
        "avg_throughput": total_throughput / n_episodes,
        "avg_tardiness": total_tardiness / n_episodes,
        "blocks_completed": env.metrics["blocks_completed"],
    }


def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning for Shipyard Scheduling")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="BC training epochs")
    parser.add_argument("--demo-episodes", type=int, default=50, help="Expert demo episodes")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="data/checkpoints/bc/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config and create environment
    cfg = load_config(args.config)
    env = HHIShipyardEnv(cfg)

    # Create encoder and policy
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=args.hidden_dim,
        num_layers=2,
    ).to(args.device)

    state_dim = args.hidden_dim * 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2)),
        max_requests=env.n_blocks,
        hidden_dim=args.policy_hidden,
        epsilon=0.0,  # No exploration during BC evaluation
    ).to(args.device)

    print("=" * 60)
    print("Pure Behavioral Cloning Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Demo episodes: {args.demo_episodes}")
    print(f"BC epochs: {args.epochs}")
    print(f"Architecture: GNN hidden={args.hidden_dim}, Policy hidden={args.policy_hidden}")

    # Phase 1: Collect expert demonstrations
    print("\n" + "=" * 60)
    print("PHASE 1: Collecting Expert Demonstrations")
    print("=" * 60)

    states, actions = collect_expert_demos(
        env, encoder, n_episodes=args.demo_episodes, device=args.device
    )

    # Phase 2: Train via behavioral cloning
    print("\n" + "=" * 60)
    print("PHASE 2: Behavioral Cloning Training")
    print("=" * 60)

    bc_losses = train_behavioral_cloning(
        policy, states, actions,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Phase 3: Evaluate
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluation")
    print("=" * 60)

    metrics = evaluate_policy(env, policy, encoder, n_episodes=5, device=args.device)
    print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
    print(f"  Avg Throughput: {metrics['avg_throughput']:.4f}")
    print(f"  Avg Tardiness: {metrics['avg_tardiness']:.2f}")
    print(f"  Blocks Completed: {metrics['blocks_completed']}")

    # Compare with rule-based
    print("\n--- Comparison with Rule-Based Expert ---")
    expert = RuleBasedScheduler()
    expert_metrics = {"throughput": 0, "tardiness": 0}
    for _ in range(5):
        obs, _ = env.reset()
        for _ in range(1000):
            action = expert.decide(env)
            obs, reward, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        if env.sim_time > 0:
            expert_metrics["throughput"] += env.metrics["blocks_completed"] / env.sim_time
        expert_metrics["tardiness"] += env.metrics.get("total_tardiness", 0)

    expert_metrics["throughput"] /= 5
    expert_metrics["tardiness"] /= 5

    print(f"  Expert Throughput: {expert_metrics['throughput']:.4f}")
    print(f"  BC Throughput: {metrics['avg_throughput']:.4f}")
    print(f"  BC vs Expert: {100 * metrics['avg_throughput'] / expert_metrics['throughput']:.1f}%")

    # Save checkpoint
    os.makedirs(args.save, exist_ok=True)
    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(checkpoint, os.path.join(args.save, "bc_final.pt"))
    print(f"\nCheckpoint saved to {args.save}")


if __name__ == "__main__":
    main()
