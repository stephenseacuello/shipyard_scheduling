"""Train Multi-Objective PPO agent for shipyard scheduling.

This script demonstrates training with multiple objectives simultaneously:
- Throughput maximization
- Tardiness minimization
- Equipment health preservation
- Operational efficiency

The agent learns Pareto-optimal policies that trade off between objectives.

Usage:
    # Basic multi-objective training
    python experiments/train_mo_ppo.py --config config/small_instance.yaml --epochs 100

    # With Chebyshev scalarization
    python experiments/train_mo_ppo.py --scalarization chebyshev

    # With adaptive weight sampling
    python experiments/train_mo_ppo.py --weight-sampling adaptive

    # Visualize Pareto front
    python experiments/train_mo_ppo.py --plot-pareto
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import yaml

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import create_encoder
from agent.policy import ActorCriticPolicy
from agent.mo_ppo import MultiObjectivePPO
from utils.metrics import compute_kpis


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_mo_ppo(args: argparse.Namespace) -> Dict[str, Any]:
    """Train Multi-Objective PPO agent.

    Args:
        args: Command line arguments.

    Returns:
        Training results including Pareto front.
    """
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Create environment
    env = ShipyardEnv(config)

    # Objective names
    objective_names = [
        "throughput",
        "negative_tardiness",
        "health",
        "efficiency",
    ]

    # Create encoder
    encoder = create_encoder(
        encoder_type=args.encoder,
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=args.hidden_dim,
    )

    # Create policy
    policy = ActorCriticPolicy(
        state_dim=args.hidden_dim * 4,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=args.hidden_dim * 2,
    )

    # Create MO-PPO trainer
    trainer = MultiObjectivePPO(
        policy=policy,
        encoder=encoder,
        n_objectives=4,
        objective_names=objective_names,
        scalarization=args.scalarization,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        n_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        device=args.device,
        weight_sampling=args.weight_sampling,
        archive_size=args.archive_size,
    )

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"mo_ppo_{args.scalarization}",
            config=vars(args),
        )

    # Training loop
    print(f"\nMulti-Objective PPO Training")
    print(f"  Objectives: {objective_names}")
    print(f"  Scalarization: {args.scalarization}")
    print(f"  Weight sampling: {args.weight_sampling}")
    print(f"  Archive size: {args.archive_size}")
    print("=" * 60)

    epoch_history = []

    for epoch in range(args.epochs):
        # Collect rollout with current weight
        rollout_data = trainer.collect_rollout(env, n_steps=args.steps)

        # Update policy
        metrics = trainer.update(rollout_data)

        # Evaluate and potentially add to Pareto archive
        if (epoch + 1) % args.eval_interval == 0:
            objectives = trainer.evaluate_and_archive(env, n_episodes=3)

            epoch_info = {
                "epoch": epoch + 1,
                "weight": trainer.current_weight.copy(),
                "objectives": objectives.copy(),
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "archive_size": len(trainer.pareto_archive.solutions),
            }
            epoch_history.append(epoch_info)

            # Logging
            print(f"Epoch {epoch + 1:4d} | "
                  f"Obj: [{objectives[0]:.2f}, {objectives[1]:.2f}, {objectives[2]:.2f}, {objectives[3]:.2f}] | "
                  f"Archive: {len(trainer.pareto_archive.solutions)}")

            if args.wandb and WANDB_AVAILABLE:
                log_dict = {
                    "epoch": epoch + 1,
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "archive_size": len(trainer.pareto_archive.solutions),
                }
                for i, name in enumerate(objective_names):
                    log_dict[f"objective/{name}"] = objectives[i]
                for i, w in enumerate(trainer.current_weight):
                    log_dict[f"weight/{i}"] = w
                wandb.log(log_dict)

    # Final Pareto front
    print("\n" + "=" * 60)
    print("Final Pareto Front")
    print("=" * 60)

    pareto_front = trainer.get_pareto_front()
    print(f"Found {len(pareto_front)} Pareto-optimal solutions:\n")

    for i, (obj, weight) in enumerate(pareto_front):
        print(f"Solution {i + 1}:")
        print(f"  Weight: [{', '.join(f'{w:.2f}' for w in weight)}]")
        print(f"  Objectives: {dict(zip(objective_names, obj))}")
        print()

    # Compute hypervolume
    if len(pareto_front) > 0:
        ref_point = np.array([0, -1000, 0, 0])  # Reference point for hypervolume
        hv = trainer.pareto_archive.get_hypervolume(ref_point)
        print(f"Hypervolume indicator: {hv:.4f}")

    # Plot Pareto front
    if args.plot_pareto and MATPLOTLIB_AVAILABLE and len(pareto_front) > 0:
        plot_pareto_front(pareto_front, objective_names, args.output_dir)

    # Save results
    if args.save_archive:
        os.makedirs(args.output_dir, exist_ok=True)
        save_pareto_archive(trainer.pareto_archive, objective_names,
                           f"{args.output_dir}/pareto_archive.npz")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    return {
        "pareto_front": pareto_front,
        "epoch_history": epoch_history,
    }


def plot_pareto_front(
    pareto_front: List[Tuple[np.ndarray, np.ndarray]],
    objective_names: List[str],
    output_dir: str,
) -> None:
    """Plot 2D projections of Pareto front.

    Args:
        pareto_front: List of (objectives, weight) tuples.
        objective_names: Names of objectives.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    objectives = np.array([p[0] for p in pareto_front])

    n_obj = len(objective_names)

    # Create pairwise plots
    fig, axes = plt.subplots(n_obj - 1, n_obj - 1, figsize=(12, 12))

    for i in range(n_obj - 1):
        for j in range(i + 1, n_obj):
            ax = axes[j - 1, i] if n_obj > 2 else axes

            ax.scatter(objectives[:, i], objectives[:, j], c='blue', alpha=0.7)
            ax.set_xlabel(objective_names[i])
            ax.set_ylabel(objective_names[j])
            ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(n_obj - 1):
        for j in range(i):
            if n_obj > 2:
                axes[j, i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pareto_front.png", dpi=150)
    plt.close()

    print(f"Pareto front plot saved to {output_dir}/pareto_front.png")


def save_pareto_archive(archive, objective_names: List[str], path: str) -> None:
    """Save Pareto archive to file.

    Args:
        archive: ParetoArchive object.
        objective_names: Names of objectives.
        path: Output file path.
    """
    objectives = np.array([s.objectives for s in archive.solutions])
    weights = np.array([s.weight_vector for s in archive.solutions])

    np.savez(
        path,
        objectives=objectives,
        weights=weights,
        objective_names=objective_names,
    )
    print(f"Pareto archive saved to {path}")


def load_and_evaluate_solution(args: argparse.Namespace) -> None:
    """Load a saved solution and evaluate with specific weight preference."""
    # Load archive
    data = np.load(args.load_archive)
    objectives = data["objectives"]
    weights = data["weights"]
    objective_names = data["objective_names"]

    print(f"\nLoaded {len(objectives)} Pareto-optimal solutions")
    print(f"Objectives: {list(objective_names)}")

    # Find solution closest to user preference
    user_weight = np.array(args.preference)
    user_weight = user_weight / user_weight.sum()  # Normalize

    # Compute scalarized values
    values = objectives @ user_weight
    best_idx = np.argmax(values)

    print(f"\nUser preference: {user_weight}")
    print(f"Best matching solution (index {best_idx}):")
    print(f"  Objectives: {dict(zip(objective_names, objectives[best_idx]))}")
    print(f"  Training weight: {weights[best_idx]}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Objective PPO Training")

    # Environment
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--encoder", type=str, default="gat",
                        choices=["gat", "transformer", "temporal"])
    parser.add_argument("--hidden-dim", type=int, default=128)

    # Multi-objective settings
    parser.add_argument("--scalarization", type=str, default="weighted_sum",
                        choices=["weighted_sum", "chebyshev", "hypernetwork"])
    parser.add_argument("--weight-sampling", type=str, default="uniform",
                        choices=["uniform", "dirichlet", "adaptive"])
    parser.add_argument("--archive-size", type=int, default=50)

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")

    # Output
    parser.add_argument("--output-dir", type=str, default="./mo_results")
    parser.add_argument("--plot-pareto", action="store_true")
    parser.add_argument("--save-archive", action="store_true", default=True)

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="shipyard-mo-ppo")
    parser.add_argument("--wandb-name", type=str, default=None)

    # Evaluation mode
    parser.add_argument("--load-archive", type=str, default=None,
                        help="Load and evaluate saved Pareto archive")
    parser.add_argument("--preference", type=float, nargs=4, default=[1, 1, 1, 1],
                        help="Weight preference for solution selection")

    args = parser.parse_args()

    if args.load_archive:
        load_and_evaluate_solution(args)
    else:
        train_mo_ppo(args)


if __name__ == "__main__":
    main()
