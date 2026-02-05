"""Ray-based distributed training and hyperparameter tuning for shipyard scheduling.

This script provides:
- Distributed training across multiple workers using Ray
- Hyperparameter optimization using Ray Tune
- Integration with Weights & Biases for experiment tracking
- Configurable search spaces and schedulers

Usage:
    # Single run with specific hyperparameters
    python experiments/train_ray.py --config config/small_instance.yaml --epochs 100

    # Hyperparameter sweep with 20 trials
    python experiments/train_ray.py --tune --num-samples 20 --epochs 100

    # Distributed training with 4 workers
    python experiments/train_ray.py --num-workers 4 --epochs 100

    # Combine with wandb logging
    python experiments/train_ray.py --tune --wandb --wandb-project shipyard-hpo
"""

from __future__ import annotations

import argparse
import os
import yaml
import random
from typing import Dict, Any, Optional

import numpy as np
import torch

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air.integrations.wandb import WandbLoggerCallback

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from agent.ppo import PPOTrainer
from utils.metrics import compute_kpis


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with inheritance support."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


def train_shipyard(config: Dict[str, Any]) -> Dict[str, float]:
    """Training function for a single trial.

    Args:
        config: Dictionary containing hyperparameters and settings.

    Returns:
        Dictionary with final metrics.
    """
    # Extract hyperparameters
    seed = config.get("seed", 42)
    epochs = config.get("epochs", 100)
    steps_per_epoch = config.get("steps_per_epoch", 200)
    device = config.get("device", "cpu")
    env_config_path = config.get("config_path", "config/small_instance.yaml")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load environment config
    env_cfg = load_config(env_config_path)

    # Create environment
    env = ShipyardEnv(env_cfg)

    # Extract tunable hyperparameters with defaults
    hidden_dim = config.get("hidden_dim", 128)
    lr = config.get("lr", 3e-4)
    gamma = config.get("gamma", 0.99)
    gae_lambda = config.get("gae_lambda", 0.95)
    clip_epsilon = config.get("clip_epsilon", 0.2)
    entropy_coef = config.get("entropy_coef", 0.01)
    value_coef = config.get("value_coef", 0.5)
    n_ppo_epochs = config.get("n_ppo_epochs", 4)
    batch_size = config.get("batch_size", 64)

    # Create encoder and policy
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
    )
    state_dim = hidden_dim * 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=hidden_dim * 2,
    )

    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        encoder=encoder,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=0.5,
        n_epochs=n_ppo_epochs,
        batch_size=batch_size,
        device=device,
    )

    # Training loop
    best_on_time_rate = 0.0
    for epoch in range(epochs):
        rollout = trainer.collect_rollout(env, steps_per_epoch)
        metrics = trainer.update(rollout)
        kpis = compute_kpis(env.metrics, env.sim_time)

        on_time_rate = kpis.get("on_time_rate", 0)
        best_on_time_rate = max(best_on_time_rate, on_time_rate)

        # Report to Ray Tune
        tune.report(
            epoch=epoch + 1,
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            entropy=metrics["entropy"],
            throughput=kpis["throughput"],
            average_tardiness=kpis["average_tardiness"],
            on_time_rate=on_time_rate,
            breakdown_count=kpis.get("breakdown_count", 0),
            best_on_time_rate=best_on_time_rate,
        )

    return {
        "final_on_time_rate": on_time_rate,
        "best_on_time_rate": best_on_time_rate,
        "final_throughput": kpis["throughput"],
    }


def get_search_space() -> Dict[str, Any]:
    """Define the hyperparameter search space for Ray Tune."""
    return {
        # Learning rate (log uniform)
        "lr": tune.loguniform(1e-5, 1e-3),
        # Discount factor
        "gamma": tune.uniform(0.95, 0.999),
        # GAE lambda
        "gae_lambda": tune.uniform(0.9, 0.99),
        # PPO clip epsilon
        "clip_epsilon": tune.uniform(0.1, 0.3),
        # Entropy coefficient (log uniform for small values)
        "entropy_coef": tune.loguniform(1e-4, 0.1),
        # Value loss coefficient
        "value_coef": tune.uniform(0.25, 1.0),
        # GNN hidden dimension
        "hidden_dim": tune.choice([64, 128, 256]),
        # PPO epochs per update
        "n_ppo_epochs": tune.choice([2, 4, 8]),
        # Mini-batch size
        "batch_size": tune.choice([32, 64, 128]),
    }


def run_hyperparameter_search(
    args: argparse.Namespace,
) -> tune.ResultGrid:
    """Run hyperparameter optimization using Ray Tune.

    Args:
        args: Command line arguments.

    Returns:
        Ray Tune ResultGrid with trial results.
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    # Search space
    search_space = get_search_space()
    search_space.update({
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps,
        "device": args.device,
        "config_path": args.config,
    })

    # Scheduler (early stopping of bad trials)
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            metric="on_time_rate",
            mode="max",
            max_t=args.epochs,
            grace_period=args.epochs // 10,
            reduction_factor=3,
        )
    elif args.scheduler == "pbt":
        scheduler = PopulationBasedTraining(
            metric="on_time_rate",
            mode="max",
            perturbation_interval=args.epochs // 5,
            hyperparam_mutations={
                "lr": tune.loguniform(1e-5, 1e-3),
                "entropy_coef": tune.loguniform(1e-4, 0.1),
            },
        )
    else:
        scheduler = None

    # Search algorithm
    if args.search_algo == "optuna":
        search_alg = OptunaSearch(metric="on_time_rate", mode="max")
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.num_workers)
    else:
        search_alg = None

    # Callbacks
    callbacks = []
    if args.wandb:
        callbacks.append(
            WandbLoggerCallback(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group or "ray-tune",
                log_config=True,
            )
        )

    # Run tuning
    tuner = tune.Tuner(
        tune.with_resources(
            train_shipyard,
            resources={"cpu": 1, "gpu": 0.25 if args.device == "cuda" else 0},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=ray.air.RunConfig(
            name=args.experiment_name or "shipyard-hpo",
            storage_path=args.output_dir,
            callbacks=callbacks,
        ),
    )

    results = tuner.fit()
    return results


def run_single_training(args: argparse.Namespace) -> Dict[str, float]:
    """Run a single training with specified hyperparameters.

    Args:
        args: Command line arguments.

    Returns:
        Dictionary with final metrics.
    """
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps,
        "device": args.device,
        "config_path": args.config,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_epsilon": args.clip_epsilon,
        "entropy_coef": args.entropy_coef,
        "hidden_dim": args.hidden_dim,
    }
    return train_shipyard(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray-based Shipyard RL Training")

    # Basic training arguments
    parser.add_argument("--config", type=str, default="config/small_instance.yaml",
                        help="Path to environment config")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--steps", type=int, default=200, help="Steps per epoch")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Hyperparameter arguments (for single runs)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coef")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GNN hidden dim")

    # Ray Tune arguments
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of HP trials")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--num-cpus", type=int, default=None, help="Total CPUs for Ray")
    parser.add_argument("--num-gpus", type=int, default=0, help="Total GPUs for Ray")
    parser.add_argument("--scheduler", type=str, default="asha",
                        choices=["asha", "pbt", "none"], help="Tune scheduler")
    parser.add_argument("--search-algo", type=str, default="optuna",
                        choices=["optuna", "random"], help="Search algorithm")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="./ray_results",
                        help="Output directory for Ray results")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="shipyard-scheduling",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name")

    args = parser.parse_args()

    if args.tune:
        print(f"Starting hyperparameter search with {args.num_samples} trials...")
        results = run_hyperparameter_search(args)

        # Print best result
        best_result = results.get_best_result(metric="on_time_rate", mode="max")
        print("\n" + "=" * 60)
        print("Best trial configuration:")
        print("=" * 60)
        for key, value in best_result.config.items():
            if key not in ["seed", "epochs", "steps_per_epoch", "device", "config_path"]:
                print(f"  {key}: {value}")
        print(f"\nBest on-time rate: {best_result.metrics['best_on_time_rate']:.2%}")
        print(f"Best throughput: {best_result.metrics['final_throughput']:.3f}")
    else:
        print("Running single training...")
        metrics = run_single_training(args)
        print(f"\nFinal metrics: {metrics}")


if __name__ == "__main__":
    main()
