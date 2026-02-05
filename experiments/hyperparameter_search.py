"""Hyperparameter search for PPO in the shipyard environment.

Supports both grid search and random search over learning rate, clip
epsilon, entropy coefficient, hidden dimension, and PPO epochs.
"""

from __future__ import annotations

import argparse
import yaml
import os
import random
from typing import Dict, Any, List

import torch
import numpy as np

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


def sample_hyperparams(method: str = "random") -> Dict[str, Any]:
    """Sample a hyperparameter configuration."""
    if method == "grid":
        # Will be called iteratively from grid
        raise ValueError("Use grid iteration instead")
    return {
        "lr": 10 ** np.random.uniform(-4, -2.5),
        "clip_epsilon": np.random.choice([0.1, 0.2, 0.3]),
        "entropy_coef": 10 ** np.random.uniform(-3, -1),
        "hidden_dim": int(np.random.choice([64, 128, 256])),
        "n_epochs": int(np.random.choice([2, 4, 8])),
    }


def grid_configs() -> List[Dict[str, Any]]:
    """Generate grid search configurations."""
    configs = []
    for lr in [1e-4, 3e-4, 1e-3]:
        for hd in [64, 128, 256]:
            for clip in [0.1, 0.2]:
                for ent in [0.001, 0.01]:
                    configs.append({
                        "lr": lr,
                        "hidden_dim": hd,
                        "clip_epsilon": clip,
                        "entropy_coef": ent,
                        "n_epochs": 4,
                    })
    return configs


def evaluate_config(
    cfg: Dict[str, Any], hp: Dict[str, Any], episodes: int, steps: int, device: str
) -> Dict[str, float]:
    """Train with given hyperparams and return KPIs."""
    env = ShipyardEnv(cfg)
    hd = hp["hidden_dim"]
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hd,
        num_layers=2,
    )
    state_dim = hd * 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=hd * 2,
    )
    trainer = PPOTrainer(
        policy, encoder,
        lr=hp["lr"],
        clip_epsilon=hp["clip_epsilon"],
        entropy_coef=hp["entropy_coef"],
        n_epochs=hp["n_epochs"],
        batch_size=32,
        device=device,
    )
    for _ in range(episodes):
        rollout = trainer.collect_rollout(env, steps)
        trainer.update(rollout)
    return compute_kpis(env.metrics, env.sim_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="random", choices=["grid", "random"])
    parser.add_argument("--n-trials", type=int, default=10, help="Number of random trials")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)

    if args.method == "grid":
        configs = grid_configs()
    else:
        configs = [sample_hyperparams("random") for _ in range(args.n_trials)]

    best_cfg = None
    best_throughput = -float("inf")
    results = []

    print(f"{'Trial':<6} {'LR':<10} {'Hidden':<8} {'Clip':<6} {'Ent':<8} {'Epochs':<7} {'Throughput':>10}")
    print("-" * 65)

    for i, hp in enumerate(configs):
        kpis = evaluate_config(cfg, hp, args.episodes, args.steps, args.device)
        throughput = kpis.get("throughput", 0.0)
        print(
            f"{i+1:<6} {hp['lr']:<10.5f} {hp['hidden_dim']:<8} "
            f"{hp['clip_epsilon']:<6.2f} {hp['entropy_coef']:<8.4f} "
            f"{hp['n_epochs']:<7} {throughput:>10.4f}"
        )
        results.append({"hp": hp, "kpis": kpis})
        if throughput > best_throughput:
            best_throughput = throughput
            best_cfg = hp

    print(f"\nBest config: {best_cfg}")
    print(f"Best throughput: {best_throughput:.4f}")


if __name__ == "__main__":
    main()
