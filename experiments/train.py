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

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from agent.ppo import PPOTrainer
from utils.metrics import compute_kpis
from utils.logging import log_results_csv
from utils.data_splits import ShipyardDataSplits
from agent.curriculum import CurriculumScheduler


def evaluate(
    env: HHIShipyardEnv,
    policy: ActorCriticPolicy,
    encoder: HeterogeneousGNNEncoder,
    n_episodes: int = 3,
    max_steps: int = 200,
    device: str = "cpu",
) -> Dict[str, float]:
    """Run evaluation episodes without training.

    Args:
        env: Environment for evaluation
        policy: Policy network
        encoder: GNN encoder
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        device: Device to run on

    Returns:
        Dict with mean metrics across episodes
    """
    policy.eval()
    encoder.eval()

    total_rewards = []
    throughputs = []
    tardiness_values = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        graph_data = env.get_graph_data()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < max_steps:
            with torch.no_grad():
                # Encode graph
                graph_data = graph_data.to(device)
                state = encoder(graph_data)

                # Get action mask and convert to tensors
                action_mask = env.get_action_mask()
                action_mask = {k: torch.tensor(v, device=device) for k, v in action_mask.items()}
                action, _, _ = policy.get_action(state, action_mask, deterministic=True)

                # Convert action tensors to CPU ints for environment
                action_dict = {k: int(v.item()) for k, v in action.items()}

            obs, reward, terminated, truncated, info = env.step(action_dict)
            graph_data = env.get_graph_data()
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        total_rewards.append(total_reward)
        kpis = compute_kpis(env.metrics, env.sim_time)
        throughputs.append(kpis.get("throughput", 0))
        tardiness_values.append(kpis.get("average_tardiness", 0))

    policy.train()
    encoder.train()

    return {
        "val_reward": float(np.mean(total_rewards)),
        "val_throughput": float(np.mean(throughputs)),
        "val_tardiness": float(np.mean(tardiness_values)),
    }


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
    # Validation and early stopping
    parser.add_argument("--val-interval", type=int, default=10, help="Validate every N epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0 to disable)")
    # Architecture options
    parser.add_argument("--hidden-dim", type=int, default=256, help="GNN hidden dimension")
    parser.add_argument("--policy-hidden", type=int, default=512, help="Policy network hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--adaptive-entropy", action="store_true", help="Enable SAC-style adaptive entropy")
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1, help="Encoder LR scale (0.1 = 10x slower)")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable potential-based reward shaping")
    parser.add_argument("--reward-shaping-weight", type=float, default=0.2, help="Weight for shaped rewards (0.2 = 20 percent)")
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
        from mes.database import init_db, clear_db, log_position_snapshot
        init_db()
        clear_db()

    # Create train/val data splits
    splits = ShipyardDataSplits(cfg, seed=args.seed)
    train_cfg = splits.get_train_config()
    val_cfg = splits.get_val_config()

    env = HHIShipyardEnv(train_cfg)
    if db_logging:
        env.db_logging_enabled = True

    # Create validation environment (separate seed, no randomization)
    val_env = HHIShipyardEnv(val_cfg)

    n_spmts = env.n_spmts
    n_cranes = getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2))

    # Create encoder and policy with upgraded architecture
    # Phase 3B: Increase capacity (hidden_dim 128→256, layers 2→4, policy 256→512)
    hidden_dim = args.hidden_dim
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=args.num_layers,
    )
    state_dim = hidden_dim * 4  # four pooled node types (256*4 = 1024)
    n_action_types = 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=n_action_types,
        n_spmts=n_spmts,
        n_cranes=n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=args.policy_hidden,
        epsilon=0.30,  # Epsilon-greedy exploration (30% random actions)
        temperature=1.0,  # Standard temperature (no exploration boost)
    )
    # Trainer with adaptive entropy and enhanced config
    # Phase 3B/3C: Enable differential LRs and adaptive entropy tuning
    trainer = PPOTrainer(
        policy=policy,
        encoder=encoder,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=2.0,  # Very high entropy to prevent collapse
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        device=args.device,
        total_epochs=args.epochs,  # Pass actual epoch count for proper entropy schedule
        entropy_schedule="cosine",
        entropy_coef_final=0.1,  # Keep minimum 10% exploration throughout training
        use_adaptive_entropy=args.adaptive_entropy,
        encoder_lr_scale=args.encoder_lr_scale,
        use_reward_shaping=args.reward_shaping,
        reward_shaping_weight=args.reward_shaping_weight,
    )

    # Log architecture details
    lrs = trainer.get_learning_rates()
    print(f"Architecture:")
    print(f"  GNN: hidden_dim={hidden_dim}, layers={args.num_layers}")
    print(f"  Policy: state_dim={state_dim}, hidden_dim={args.policy_hidden}")
    print(f"  Learning rates: encoder={lrs['encoder_lr']:.2e}, policy={lrs['policy_lr']:.2e}")
    print(f"  Adaptive entropy: {args.adaptive_entropy}")
    print(f"  Reward shaping: {args.reward_shaping} (weight: {args.reward_shaping_weight})")
    # Training loop with validation and early stopping
    all_metrics = []
    base_cfg = load_config(args.config)
    cumulative_time = 0.0  # Track cumulative time across epochs for playback
    reset_next = True  # Reset on first epoch, then continue unless episode ends

    # Early stopping state
    best_val_throughput = -float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Update curriculum difficulty
        if curriculum is not None:
            epoch_cfg = curriculum.get_config(base_cfg, epoch)
            env = HHIShipyardEnv(epoch_cfg)
            if db_logging:
                env.db_logging_enabled = True
            reset_next = True  # Reset when curriculum changes environment
        rollout = trainer.collect_rollout(env, args.steps, reset=reset_next)
        reset_next = False  # Don't reset on subsequent epochs (continue episode)
        # Log simulation state to MES database
        if db_logging:
            env.log_state_to_db()
            # Log position snapshot for playback (use cumulative time for continuous timeline)
            cumulative_time += env.sim_time
            log_position_snapshot(
                time=cumulative_time,
                blocks=env.entities['blocks'],
                spmts=env.entities['spmts'],
                cranes=env.entities.get('goliath_cranes', env.entities.get('cranes', [])),
            )
        metrics = trainer.update(rollout)
        # Compute KPIs for this epoch
        kpis = compute_kpis(env.metrics, env.sim_time)
        print(f"Epoch {epoch + 1}/{args.epochs} - PolicyLoss: {metrics['policy_loss']:.3f}, ValueLoss: {metrics['value_loss']:.3f}, Entropy: {metrics['entropy']:.3f}, Throughput: {kpis['throughput']:.3f}, Tardiness: {kpis['average_tardiness']:.3f}")
        result_row = {
            "epoch": epoch + 1,
            **metrics,
            **kpis,
        }

        # Validation every val_interval epochs
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = evaluate(
                val_env, policy, encoder,
                n_episodes=3, max_steps=args.steps, device=args.device
            )
            result_row.update(val_metrics)
            print(f"  Validation - Reward: {val_metrics['val_reward']:.3f}, Throughput: {val_metrics['val_throughput']:.3f}")

            # Early stopping check
            if args.patience > 0:
                if val_metrics["val_throughput"] > best_val_throughput:
                    best_val_throughput = val_metrics["val_throughput"]
                    patience_counter = 0
                    # Save best model
                    if args.save:
                        os.makedirs(args.save, exist_ok=True)
                        torch.save(
                            {
                                "policy": policy.state_dict(),
                                "encoder": encoder.state_dict(),
                                "epoch": epoch + 1,
                                "val_throughput": best_val_throughput,
                            },
                            os.path.join(args.save, "best_model.pt"),
                        )
                        print(f"  New best model saved (throughput: {best_val_throughput:.3f})")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience // args.val_interval:
                        print(f"Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
                        break

        all_metrics.append(result_row)

        # Enhanced wandb logging
        if wandb_run:
            lrs = trainer.get_learning_rates()
            log_dict = {
                "epoch": epoch + 1,
                # Training metrics
                "train/policy_loss": metrics["policy_loss"],
                "train/value_loss": metrics["value_loss"],
                "train/entropy": metrics["entropy"],
                "train/entropy_coef": metrics.get("entropy_coef", 0),
                "train/kl_divergence": metrics.get("kl_divergence", 0),
                "train/clip_fraction": metrics.get("clip_fraction", 0),
                "train/grad_norm": metrics.get("grad_norm", 0),
                # Learning rates
                "lr/encoder": lrs["encoder_lr"],
                "lr/policy": lrs["policy_lr"],
                # KPIs
                "kpi/throughput": kpis["throughput"],
                "kpi/average_tardiness": kpis["average_tardiness"],
                "kpi/on_time_rate": kpis.get("on_time_rate", 0),
                "kpi/spmt_utilization": kpis.get("spmt_utilization", 0),
                "kpi/crane_utilization": kpis.get("crane_utilization", 0),
                # Equipment health
                "health/breakdown_count": kpis.get("breakdown_count", 0),
                "health/planned_maintenance": kpis.get("planned_maintenance_rate", 0),
                # Block progress
                "blocks/completed": env.metrics.get("blocks_completed", 0),
                "blocks/stage_advances": env.metrics.get("stage_advances", 0),
            }
            # Add validation metrics if available
            if "val_throughput" in result_row:
                log_dict["val/throughput"] = result_row["val_throughput"]
                log_dict["val/reward"] = result_row["val_reward"]
                log_dict["val/tardiness"] = result_row["val_tardiness"]

            wandb.log(log_dict)

            # Log model checkpoint as artifact periodically
            if args.save and (epoch + 1) % 25 == 0:
                artifact = wandb.Artifact(
                    name=f"checkpoint-epoch-{epoch + 1}",
                    type="model",
                )
                artifact.add_file(os.path.join(args.save, f"checkpoint_epoch_{epoch + 1}.pt"))
                wandb_run.log_artifact(artifact)

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