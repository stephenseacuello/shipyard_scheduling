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

import pandas as pd
from pathlib import Path

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from baselines.rule_based import RuleBasedScheduler
from baselines.myopic_rl import MyopicRLScheduler
from baselines.siloed_opt import SiloedOptimizationScheduler
from utils.metrics import compute_kpis
from utils.data_splits import ShipyardDataSplits


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
    parser.add_argument("--n-scenarios", type=int, default=1, help="Number of test scenarios (uses data_splits)")
    parser.add_argument("--output", type=str, default="", help="Path to save results CSV")
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

    env = HHIShipyardEnv(cfg)
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
            n_cranes=getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2)),
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
    # Generate test scenarios if requested
    if args.n_scenarios > 1:
        splits = ShipyardDataSplits(cfg, seed=args.seed)
        test_configs = splits.get_test_configs(n_scenarios=args.n_scenarios)
        print(f"Evaluating across {args.n_scenarios} test scenarios...")
    else:
        test_configs = [cfg]

    # Evaluate episodes across all scenarios
    all_results = []
    total_kpis = {"throughput": 0.0, "average_tardiness": 0.0, "unplanned_breakdown_rate": 0.0, "planned_maintenance_rate": 0.0}
    total_episodes = 0

    for scenario_idx, scenario_cfg in enumerate(test_configs):
        if args.n_scenarios > 1:
            n_blocks = scenario_cfg.get("n_blocks", cfg.get("n_blocks", 50))
            print(f"\nScenario {scenario_idx + 1}/{args.n_scenarios} (n_blocks={n_blocks})")
            env = HHIShipyardEnv(scenario_cfg)
            if db_logging:
                env.db_logging_enabled = True

        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0

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
                total_reward += reward
                step_count += 1

            # Log final state to MES database
            if db_logging:
                env.log_state_to_db()

            # Compute KPIs
            kpis = compute_kpis(env.metrics, env.sim_time)
            for k in total_kpis:
                total_kpis[k] += kpis.get(k, 0.0)
            total_episodes += 1

            # Store result for CSV output
            result = {
                "scenario": scenario_idx,
                "episode": ep,
                "agent": args.agent,
                "n_blocks": scenario_cfg.get("n_blocks", cfg.get("n_blocks", 50)),
                "total_reward": total_reward,
                "steps": step_count,
                "blocks_completed": env.metrics.get("blocks_completed", 0),
                **kpis,
            }
            all_results.append(result)
            print(f"  Episode {ep + 1}: throughput={kpis.get('throughput', 0):.3f}, tardiness={kpis.get('average_tardiness', 0):.1f}")

    # Save results to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    # Average
    avg_kpis = {k: v / total_episodes for k, v in total_kpis.items()}
    print(f"\nAverage KPIs over {total_episodes} episodes ({args.n_scenarios} scenarios):")
    for k, v in avg_kpis.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()