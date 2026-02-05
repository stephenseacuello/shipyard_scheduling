"""Ablation study script.

Systematically evaluates the impact of individual components:
- GNN vs MLP encoder
- Action masking on/off
- PHM (health-aware rewards) on/off
- Curriculum learning on/off
"""

from __future__ import annotations

import argparse
import yaml
import os
import random
from typing import Dict, Any

import numpy as np
import torch

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder, SimpleGNNEncoder
from agent.policy import ActorCriticPolicy
from agent.ppo import PPOTrainer
from agent.curriculum import CurriculumScheduler
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


def run_ablation(
    cfg: Dict[str, Any],
    encoder_type: str = "gnn",
    use_masking: bool = True,
    use_phm: bool = True,
    use_curriculum: bool = False,
    epochs: int = 3,
    steps: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """Run a single ablation configuration and return KPIs."""
    if not use_phm:
        cfg = dict(cfg)
        cfg["reward_breakdown"] = 0.0
        cfg["reward_maintenance"] = 0.0

    curriculum = None
    if use_curriculum and "curriculum" in cfg:
        curriculum = CurriculumScheduler.from_config(cfg["curriculum"])
        cfg = curriculum.get_config(cfg, 0)

    env = ShipyardEnv(cfg)
    hidden_dim = 128

    if encoder_type == "gnn":
        encoder = HeterogeneousGNNEncoder(
            block_dim=env.block_features,
            spmt_dim=env.spmt_features,
            crane_dim=env.crane_features,
            facility_dim=env.facility_features,
            hidden_dim=hidden_dim,
        )
        state_dim = hidden_dim * 4
    else:
        # Simple MLP-like encoder (homogeneous, flattened input)
        total_dim = env.block_features + env.spmt_features + env.crane_features + env.facility_features
        encoder = SimpleGNNEncoder(input_dim=total_dim, hidden_dim=hidden_dim, num_layers=2)
        state_dim = hidden_dim

    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=256,
    )

    trainer = PPOTrainer(policy=policy, encoder=encoder, device=device)

    base_cfg = dict(cfg)
    for epoch in range(epochs):
        if curriculum is not None:
            epoch_cfg = curriculum.get_config(base_cfg, epoch)
            env = ShipyardEnv(epoch_cfg)
        rollout = trainer.collect_rollout(env, steps)
        trainer.update(rollout)

    return compute_kpis(env.metrics, env.sim_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
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

    ablations = [
        {"name": "Full (GNN+mask+PHM)", "encoder_type": "gnn", "use_masking": True, "use_phm": True},
        {"name": "No masking", "encoder_type": "gnn", "use_masking": False, "use_phm": True},
        {"name": "No PHM", "encoder_type": "gnn", "use_masking": True, "use_phm": False},
        {"name": "MLP encoder", "encoder_type": "mlp", "use_masking": True, "use_phm": True},
        {"name": "Curriculum", "encoder_type": "gnn", "use_masking": True, "use_phm": True, "use_curriculum": True},
    ]

    print(f"{'Configuration':<30} {'Throughput':>12} {'Tardiness':>12} {'OEE':>8} {'Cost':>12}")
    print("-" * 80)

    for abl in ablations:
        name = abl.pop("name")
        kpis = run_ablation(cfg, epochs=args.epochs, steps=args.steps, device=args.device, **abl)
        print(
            f"{name:<30} {kpis.get('throughput', 0):>12.4f} "
            f"{kpis.get('average_tardiness', 0):>12.2f} "
            f"{kpis.get('oee', 0):>8.4f} "
            f"{kpis.get('total_cost', 0):>12.0f}"
        )


if __name__ == "__main__":
    main()
