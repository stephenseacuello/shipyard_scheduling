"""Training script for the shipyard scheduling RL agent.

This script loads a configuration, constructs the environment, encoder and
policy, then trains the agent using PPO. It logs training metrics to
standard output, wandb, and optionally saves checkpoints.

Supports:
- Weights & Biases (wandb) for experiment tracking
- Ray for distributed training (see train_ray.py)
"""

from __future__ import annotations

import argparse
import yaml
import os
import random
from typing import Dict, Any, Optional

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from agent.ppo import PPOTrainer
from utils.metrics import compute_kpis
from utils.logging import log_results_csv
from agent.curriculum import CurriculumScheduler


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Handle inheritance
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


def init_wandb(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialize Weights & Biases logging if enabled."""
    if not args.wandb or not WANDB_AVAILABLE:
        return None

    # Build config dict for wandb
    wandb_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps,
        "device": args.device,
        "curriculum": args.curriculum,
        "config_file": args.config,
        # PPO hyperparameters
        "lr": cfg.get("ppo", {}).get("lr", 3e-4),
        "gamma": cfg.get("ppo", {}).get("gamma", 0.99),
        "gae_lambda": cfg.get("ppo", {}).get("gae_lambda", 0.95),
        "clip_epsilon": cfg.get("ppo", {}).get("clip_epsilon", 0.2),
        "entropy_coef": cfg.get("ppo", {}).get("entropy_coef", 0.01),
        "value_coef": cfg.get("ppo", {}).get("value_coef", 0.5),
        # Environment config
        "n_blocks": cfg.get("environment", {}).get("n_blocks", 50),
        "n_spmts": cfg.get("environment", {}).get("n_spmts", 6),
        "n_cranes": cfg.get("environment", {}).get("n_cranes", 2),
        # GNN config
        "hidden_dim": cfg.get("gnn", {}).get("hidden_dim", 128),
        "n_layers": cfg.get("gnn", {}).get("n_layers", 2),
    }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=wandb_config,
        tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        notes=args.wandb_notes,
        save_code=True,
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Shipyard RL Agent")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps per epoch")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--save", type=str, default="", help="Directory to save checkpoints")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--no-db-log", action="store_true", help="Disable MES database logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="shipyard-scheduling", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, default=None, help="W&B tags (comma-separated)")
    parser.add_argument("--wandb-notes", type=str, default=None, help="W&B run notes")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)

    # Initialize wandb
    wandb_run = init_wandb(args, cfg)
    if wandb_run:
        print(f"Wandb run initialized: {wandb_run.url}")

    # Curriculum learning setup
    curriculum = None
    if args.curriculum and "curriculum" in cfg:
        curriculum = CurriculumScheduler.from_config(cfg["curriculum"])
        cfg = curriculum.get_config(cfg, 0)
    # Initialize MES database logging
    db_logging = not args.no_db_log
    if db_logging:
        from mes.database import init_db, clear_db
        init_db()
        clear_db()

    # Create environment
    env = ShipyardEnv(cfg)
    if db_logging:
        env.db_logging_enabled = True
    # Create encoder and policy
    hidden_dim = 128
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
    )
    state_dim = hidden_dim * 4  # four pooled node types
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=256,
    )
    # Trainer
    trainer = PPOTrainer(
        policy=policy,
        encoder=encoder,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        device=args.device,
    )
    # Training loop
    all_metrics = []
    base_cfg = load_config(args.config)
    for epoch in range(args.epochs):
        # Update curriculum difficulty
        if curriculum is not None:
            epoch_cfg = curriculum.get_config(base_cfg, epoch)
            env = ShipyardEnv(epoch_cfg)
            if db_logging:
                env.db_logging_enabled = True
        rollout = trainer.collect_rollout(env, args.steps)
        # Log simulation state to MES database
        if db_logging:
            env.log_state_to_db()
        metrics = trainer.update(rollout)
        # Compute KPIs for this epoch
        kpis = compute_kpis(env.metrics, env.sim_time)
        print(f"Epoch {epoch + 1}/{args.epochs} - PolicyLoss: {metrics['policy_loss']:.3f}, ValueLoss: {metrics['value_loss']:.3f}, Entropy: {metrics['entropy']:.3f}, Throughput: {kpis['throughput']:.3f}, Tardiness: {kpis['average_tardiness']:.3f}")
        result_row = {
            "epoch": epoch + 1,
            **metrics,
            **kpis,
        }
        all_metrics.append(result_row)

        # Log to wandb
        if wandb_run:
            wandb.log({
                "epoch": epoch + 1,
                "train/policy_loss": metrics["policy_loss"],
                "train/value_loss": metrics["value_loss"],
                "train/entropy": metrics["entropy"],
                "kpi/throughput": kpis["throughput"],
                "kpi/average_tardiness": kpis["average_tardiness"],
                "kpi/on_time_rate": kpis.get("on_time_rate", 0),
                "kpi/breakdown_count": kpis.get("breakdown_count", 0),
                "kpi/spmt_utilization": kpis.get("spmt_utilization", 0),
            })
        # Save checkpoint
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "encoder": encoder.state_dict(),
                },
                os.path.join(args.save, f"checkpoint_epoch_{epoch + 1}.pt"),
            )
    # Write results
    if args.save:
        log_results_csv(os.path.join(args.save, "training_metrics.csv"), all_metrics)

        # Log final model to wandb as artifact
        if wandb_run:
            artifact = wandb.Artifact(
                name=f"model-{wandb_run.id}",
                type="model",
                description="Trained GNN-PPO model for shipyard scheduling",
            )
            artifact.add_dir(args.save)
            wandb_run.log_artifact(artifact)

    # Finish wandb run
    if wandb_run:
        # Log final summary metrics
        if all_metrics:
            final = all_metrics[-1]
            wandb.run.summary["final/policy_loss"] = final.get("policy_loss", 0)
            wandb.run.summary["final/value_loss"] = final.get("value_loss", 0)
            wandb.run.summary["final/throughput"] = final.get("throughput", 0)
            wandb.run.summary["final/on_time_rate"] = final.get("on_time_rate", 0)
        wandb.finish()


if __name__ == "__main__":
    main()