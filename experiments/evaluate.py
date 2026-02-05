"""Evaluation script for shipyard scheduling agents and baselines.

This script runs a trained RL agent or baseline heuristic for a number of
episodes and reports average performance metrics. It supports selecting
between the RL agent, rule‑based heuristic, myopic RL and siloed
optimization.
"""

from __future__ import annotations

import argparse
import yaml
import os
import random
from typing import Any, Dict

import numpy as np
import torch

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from baselines.rule_based import RuleBasedScheduler
from baselines.myopic_rl import MyopicRLScheduler
from baselines.siloed_opt import SiloedOptimizationScheduler
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Shipyard Scheduling Agents")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml", help="Path to YAML config file")
    parser.add_argument("--agent", type=str, default="rl", choices=["rl", "rule", "myopic", "siloed"], help="Agent type to evaluate")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to agent checkpoint (for RL)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-db-log", action="store_true", help="Disable MES database logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)

    # Initialize MES database logging
    db_logging = not args.no_db_log
    if db_logging:
        from mes.database import init_db, clear_db
        init_db()
        clear_db()

    env = ShipyardEnv(cfg)
    if db_logging:
        env.db_logging_enabled = True
    if args.agent == "rl":
        # Build encoder and policy, load weights
        hidden_dim = 128
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
            hidden_dim=256,
        )
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=args.device)
            policy.load_state_dict(ckpt["policy"])
            encoder.load_state_dict(ckpt["encoder"])
        policy.to(args.device)
        encoder.to(args.device)
    else:
        # Initialize baseline scheduler
        if args.agent == "rule":
            scheduler = RuleBasedScheduler()
        elif args.agent == "myopic":
            scheduler = MyopicRLScheduler()
        else:
            scheduler = SiloedOptimizationScheduler()
    # Evaluate episodes
    total_kpis = {"throughput": 0.0, "average_tardiness": 0.0, "unplanned_breakdown_rate": 0.0, "planned_maintenance_rate": 0.0}
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done:
            if args.agent == "rl":
                graph_data = env.get_graph_data().to(args.device)
                with torch.no_grad():
                    state_emb = encoder(graph_data)
                    mask = env.get_action_mask()
                    torch_mask = {k: torch.tensor(v, device=args.device) for k, v in mask.items()}
                    action, _, _ = policy.get_action(state_emb, torch_mask, deterministic=True)
                action_cpu = {k: int(v.item()) for k, v in action.items()}
                obs, reward, terminated, truncated, info = env.step(action_cpu)
                done = terminated or truncated
            else:
                act = scheduler.decide(env)
                obs, reward, terminated, truncated, info = env.step(act)
                done = terminated or truncated
        # Log final state to MES database
        if db_logging:
            env.log_state_to_db()
        # Compute KPIs
        kpis = compute_kpis(env.metrics, env.sim_time)
        for k in total_kpis:
            total_kpis[k] += kpis.get(k, 0.0)
        print(f"Episode {ep + 1}: {kpis}")
    # Average
    avg_kpis = {k: v / args.episodes for k, v in total_kpis.items()}
    print("\nAverage KPIs over", args.episodes, "episodes:")
    for k, v in avg_kpis.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()