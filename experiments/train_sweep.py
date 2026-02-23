"""Training script for wandb hyperparameter sweeps.

This script is designed to be called by wandb agent during sweeps.
It reads hyperparameters from wandb.config and runs training.

Usage:
    wandb sweep experiments/sweep_config.yaml
    wandb agent <sweep_id>
"""

from __future__ import annotations

import yaml
import os
import random
from typing import Dict, Any

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    raise ImportError("wandb is required for sweep training. Install with: pip install wandb")

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from agent.ppo import PPOTrainer
from utils.metrics import compute_kpis


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


def train():
    """Main training function for sweep runs."""
    # Initialize wandb run
    run = wandb.init()
    config = wandb.config

    # Set random seeds
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load environment config
    cfg = load_config("config/hhi_ulsan.yaml")

    # Apply warmup_steps from sweep config if present
    warmup_steps = config.get("warmup_steps", 10)

    # Create environment
    env = ShipyardEnv(cfg)

    # Get dimensions
    n_spmts = env.n_spmts
    n_cranes = env.n_cranes
    hidden_dim = config.get("hidden_dim", 128)

    # Create encoder
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=config.get("gnn_layers", 2),
    )

    # Create policy
    state_dim = hidden_dim * 4  # four pooled node types
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=n_spmts,
        n_cranes=n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=hidden_dim * 2,  # policy hidden dim
        epsilon=config.get("epsilon", 0.15),
    )

    # Create trainer with sweep hyperparameters
    trainer = PPOTrainer(
        policy=policy,
        encoder=encoder,
        lr=config.get("lr", 3e-4),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_epsilon=config.get("clip_epsilon", 0.2),
        entropy_coef=config.get("entropy_coef", 0.1),
        value_coef=config.get("value_coef", 0.5),
        max_grad_norm=0.5,
        n_epochs=config.get("n_epochs", 4),
        batch_size=config.get("batch_size", 64),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Training parameters
    epochs = config.get("epochs", 50)
    steps_per_epoch = config.get("steps_per_epoch", 500)

    # Training loop
    best_throughput = 0.0
    reset_next = True

    for epoch in range(epochs):
        rollout = trainer.collect_rollout(env, steps_per_epoch, reset=reset_next)
        reset_next = False  # Continue episode across epochs
        metrics = trainer.update(rollout)

        # Compute KPIs
        kpis = compute_kpis(env.metrics, env.sim_time)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/policy_loss": metrics["policy_loss"],
            "train/value_loss": metrics["value_loss"],
            "train/entropy": metrics["entropy"],
            "kpi/throughput": kpis["throughput"],
            "kpi/average_tardiness": kpis["average_tardiness"],
            "kpi/on_time_rate": kpis.get("on_time_rate", 0),
            "kpi/spmt_utilization": kpis.get("spmt_utilization", 0),
            "env/blocks_completed": env.metrics["blocks_completed"],
            "env/stage_advances": env.metrics.get("stage_advances", 0),
            "env/sim_time": env.sim_time,
        })

        # Track best throughput
        if kpis["throughput"] > best_throughput:
            best_throughput = kpis["throughput"]

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Throughput: {kpis['throughput']:.3f}, "
              f"Entropy: {metrics['entropy']:.3f}, "
              f"Completed: {env.metrics['blocks_completed']}")

    # Log final summary
    wandb.run.summary["final/throughput"] = kpis["throughput"]
    wandb.run.summary["final/best_throughput"] = best_throughput
    wandb.run.summary["final/blocks_completed"] = env.metrics["blocks_completed"]
    wandb.run.summary["final/entropy"] = metrics["entropy"]

    wandb.finish()


if __name__ == "__main__":
    train()
