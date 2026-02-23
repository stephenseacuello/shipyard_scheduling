#!/usr/bin/env python3
"""DAgger training script with Wandb sweep support.

This script is designed for wandb hyperparameter sweeps.
It logs all metrics to wandb and supports hidden_dim as a tunable parameter.

Usage:
    wandb sweep experiments/sweep_dagger_small.yaml
    wandb agent <sweep_id>
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
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

from src.simulation.shipyard_env import HHIShipyardEnv
from src.agent.gnn_encoder import HeterogeneousGNNEncoder
from src.agent.policy import ActorCriticPolicy
from src.baselines.rule_based import RuleBasedScheduler


def load_config(path: str) -> Dict[str, Any]:
    """Load config with inheritance support."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


class DAggerTrainerSweep:
    """DAgger trainer with wandb logging support."""

    def __init__(
        self,
        env: HHIShipyardEnv,
        encoder: HeterogeneousGNNEncoder,
        policy: ActorCriticPolicy,
        expert: RuleBasedScheduler,
        device: str = "cpu",
        lr: float = 3e-4,
    ):
        self.env = env
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        self.expert = expert
        self.device = device

        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(policy.parameters()),
            lr=lr, weight_decay=1e-5
        )

        # Aggregated dataset
        self.states: List[torch.Tensor] = []
        self.expert_actions: List[Dict[str, int]] = []

        # Action key mapping
        self.action_keys = {
            "action_type": "action_type",
            "spmt": "spmt_idx",
            "request": "request_idx",
            "crane": "crane_idx",
            "lift": "lift_idx",
            "equipment": "equipment_idx",
        }

    def collect_expert_demos(self, n_episodes: int, max_steps: int = 1000):
        """Collect initial expert demonstrations."""
        for ep in range(n_episodes):
            obs, info = self.env.reset()

            for step in range(max_steps):
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)

                expert_action = self.expert.decide(self.env)

                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)

                obs, reward, terminated, truncated, info = self.env.step(expert_action)

                if terminated or truncated:
                    break

        print(f"Collected {len(self.states)} expert state-action pairs")

    def collect_dagger_data(self, n_episodes: int, beta: float = 0.5, max_steps: int = 1000):
        """Collect DAgger data: roll out policy but query expert for labels."""
        new_samples = 0

        for ep in range(n_episodes):
            obs, info = self.env.reset()

            for step in range(max_steps):
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)

                expert_action = self.expert.decide(self.env)

                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)
                new_samples += 1

                if random.random() < beta:
                    action = expert_action
                else:
                    with torch.no_grad():
                        policy_action, _, _ = self.policy.get_action(state_emb)
                    action = {k: int(v.item()) for k, v in policy_action.items()}

                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    break

        return new_samples

    def train_epoch(self, batch_size: int = 128) -> float:
        """Train policy on aggregated dataset for one epoch."""
        n_samples = len(self.states)
        indices = list(range(n_samples))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < 2:
                continue

            batch_states = torch.cat(
                [self.states[j] for j in batch_indices], dim=0
            ).to(self.device)

            action_dist, _ = self.policy.forward(batch_states)

            loss = 0.0
            for head_name, action_key in self.action_keys.items():
                target = torch.tensor(
                    [self.expert_actions[j].get(action_key, 0) for j in batch_indices],
                    device=self.device
                )
                max_idx = action_dist[head_name].probs.shape[-1] - 1
                target = target.clamp(0, max_idx)

                head_loss = F.cross_entropy(action_dist[head_name].logits, target)

                if head_name == "action_type":
                    loss += 2.0 * head_loss
                else:
                    loss += head_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.encoder.parameters()), 1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, n_episodes: int = 5, max_steps: int = 1000) -> Dict[str, float]:
        """Evaluate current policy."""
        total_throughput = 0.0
        total_reward = 0.0
        total_blocks = 0

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            ep_reward = 0.0

            for step in range(max_steps):
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)
                    action, _, _ = self.policy.get_action(state_emb, deterministic=True)

                action_cpu = {k: int(v.item()) for k, v in action.items()}
                obs, reward, terminated, truncated, info = self.env.step(action_cpu)
                ep_reward += reward

                if terminated or truncated:
                    break

            total_reward += ep_reward
            blocks = self.env.metrics.get("blocks_erected", self.env.metrics.get("blocks_completed", 0))
            total_blocks += blocks
            if self.env.sim_time > 0:
                total_throughput += blocks / self.env.sim_time

        return {
            "avg_reward": total_reward / n_episodes,
            "avg_throughput": total_throughput / n_episodes,
            "avg_blocks": total_blocks / n_episodes,
        }

    def evaluate_expert(self, n_episodes: int = 5, max_steps: int = 1000) -> float:
        """Evaluate expert throughput."""
        total_throughput = 0.0

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            for _ in range(max_steps):
                action = self.expert.decide(self.env)
                obs, reward, term, trunc, _ = self.env.step(action)
                if term or trunc:
                    break
            if self.env.sim_time > 0:
                blocks = self.env.metrics.get("blocks_erected", self.env.metrics.get("blocks_completed", 0))
                total_throughput += blocks / self.env.sim_time

        return total_throughput / n_episodes


def main():
    parser = argparse.ArgumentParser(description="DAgger Sweep Training")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--init-episodes", type=int, default=30)
    parser.add_argument("--dagger-episodes", type=int, default=15)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-end", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Initialize wandb
    run = wandb.init(
        project="shipyard-scheduling",
        config=vars(args),
        reinit=True,
    )

    # Override args with wandb config (for sweeps)
    config = wandb.config

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load config and create environment
    cfg = load_config(config.config)
    env = HHIShipyardEnv(cfg)

    # Create networks with configurable hidden_dim
    hidden_dim = config.hidden_dim
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
        n_cranes=getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2)),
        max_requests=env.n_blocks,
        hidden_dim=hidden_dim * 2,
        epsilon=0.0,
    )

    expert = RuleBasedScheduler()

    trainer = DAggerTrainerSweep(
        env=env,
        encoder=encoder,
        policy=policy,
        expert=expert,
        device=config.device,
        lr=config.lr,
    )

    print("=" * 60)
    print("DAgger Sweep Training")
    print("=" * 60)
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Learning rate: {config.lr}")
    print(f"DAgger iterations: {config.iterations}")

    # Phase 1: Collect initial expert demonstrations
    print("\nPhase 1: Collecting expert demonstrations...")
    trainer.collect_expert_demos(config.init_episodes)

    # Initial BC training
    print("Initial BC training...")
    for epoch in range(config.train_epochs):
        loss = trainer.train_epoch()
    
    # Evaluate initial policy
    metrics = trainer.evaluate()
    wandb.log({
        "iteration": 0,
        "loss": loss,
        "throughput": metrics["avg_throughput"],
        "reward": metrics["avg_reward"],
        "blocks": metrics["avg_blocks"],
    })
    print(f"Initial - Loss: {loss:.4f}, Throughput: {metrics['avg_throughput']:.4f}")

    # Phase 2: DAgger iterations
    print("\nPhase 2: DAgger iterations...")
    best_throughput = metrics["avg_throughput"]
    best_state = {
        "encoder": encoder.state_dict(),
        "policy": policy.state_dict(),
    }

    for iteration in range(config.iterations):
        beta = config.beta_start - (config.beta_start - config.beta_end) * iteration / max(config.iterations - 1, 1)

        # Collect DAgger data
        new_samples = trainer.collect_dagger_data(config.dagger_episodes, beta=beta)

        # Train on aggregated dataset
        for epoch in range(config.train_epochs):
            loss = trainer.train_epoch()

        # Evaluate
        metrics = trainer.evaluate()
        
        wandb.log({
            "iteration": iteration + 1,
            "beta": beta,
            "loss": loss,
            "throughput": metrics["avg_throughput"],
            "reward": metrics["avg_reward"],
            "blocks": metrics["avg_blocks"],
            "dataset_size": len(trainer.states),
        })

        print(f"Iter {iteration+1}/{config.iterations} - Beta: {beta:.2f}, Loss: {loss:.4f}, Throughput: {metrics['avg_throughput']:.4f}")

        if metrics["avg_throughput"] > best_throughput:
            best_throughput = metrics["avg_throughput"]
            best_state = {
                "encoder": encoder.state_dict(),
                "policy": policy.state_dict(),
            }

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    encoder.load_state_dict(best_state["encoder"])
    policy.load_state_dict(best_state["policy"])

    final_metrics = trainer.evaluate(n_episodes=10)
    expert_throughput = trainer.evaluate_expert(n_episodes=10)

    dagger_vs_expert = 0.0
    if expert_throughput > 0:
        dagger_vs_expert = 100 * final_metrics["avg_throughput"] / expert_throughput

    print(f"Best Policy Throughput: {final_metrics['avg_throughput']:.4f}")
    print(f"Expert Throughput: {expert_throughput:.4f}")
    print(f"DAgger vs Expert: {dagger_vs_expert:.1f}%")

    # Log final metrics
    wandb.log({
        "final_throughput": final_metrics["avg_throughput"],
        "final_reward": final_metrics["avg_reward"],
        "expert_throughput": expert_throughput,
        "dagger_vs_expert_percent": dagger_vs_expert,
        "best_throughput": best_throughput,
    })

    # Log summary metrics
    wandb.summary["dagger_vs_expert_percent"] = dagger_vs_expert
    wandb.summary["best_throughput"] = best_throughput
    wandb.summary["expert_throughput"] = expert_throughput

    wandb.finish()
    print("\nSweep run complete!")


if __name__ == "__main__":
    main()
