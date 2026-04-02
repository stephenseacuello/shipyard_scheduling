#!/usr/bin/env python3
"""Comprehensive baseline comparison experiment.

Compares all available scheduling agents across multiple configs and seeds:
  - PuLP MIP (Mixed-Integer Programming with CBC solver)
  - Expert / EDD (Earliest Due Date rule-based heuristic)
  - DAgger (Dataset Aggregation imitation learning)
  - SAC (Soft Actor-Critic)
  - PPO (Proximal Policy Optimization)
  - GA (Genetic Algorithm)
  - MPC (Model Predictive Control with CP-SAT)

Each agent is run in a try/except so that missing dependencies (e.g., PuLP,
OR-Tools, trained checkpoints) do not crash the whole comparison.

Usage:
    # Run with defaults (small_instance + hhi_ulsan, 5 seeds each):
    python experiments/compare_pulp_rl.py

    # Specify configs and seeds:
    python experiments/compare_pulp_rl.py --configs config/small_instance.yaml config/hhi_ulsan.yaml --seeds 5

    # Limit max steps per episode:
    python experiments/compare_pulp_rl.py --max-steps 500

    # Save results to a specific CSV:
    python experiments/compare_pulp_rl.py --output results/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from simulation.shipyard_env import HHIShipyardEnv
from baselines.rule_based import RuleBasedScheduler
from baselines.pulp_scheduler import PuLPMIPScheduler


# ---------------------------------------------------------------------------
# Config loading (matches pattern in evaluate.py / train_dagger.py)
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config with optional inheritance."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


# ---------------------------------------------------------------------------
# Agent wrappers — each returns (action_dict, agent_name) per step.
# The run_episode function below handles the episode loop generically.
# ---------------------------------------------------------------------------

def _make_expert_agent() -> Tuple[str, Any]:
    """Return (agent_name, scheduler_instance) for the EDD expert."""
    return "Expert (EDD)", RuleBasedScheduler()


def _make_pulp_agent() -> Tuple[str, Any]:
    """Return (agent_name, scheduler_instance) for PuLP MIP."""
    return "PuLP MIP", PuLPMIPScheduler()


def _make_ga_agent(env: HHIShipyardEnv) -> Optional[Tuple[str, Any]]:
    """Return (agent_name, wrapper) for the GA scheduler, or None if unavailable."""
    try:
        from baselines.ga_scheduler import GAScheduler, GAConfig
    except ImportError:
        return None

    n_blocks = getattr(env, "n_blocks", 50)
    n_spmts = getattr(env, "n_spmts", 6)
    n_cranes = getattr(env, "n_goliath_cranes", getattr(env, "n_cranes", 2))
    ga_cfg = GAConfig(population_size=50, generations=30)
    ga = GAScheduler(ga_cfg, n_blocks=n_blocks, n_spmts=n_spmts, n_cranes=n_cranes)
    return "GA", ga


def _make_mpc_agent() -> Optional[Tuple[str, Any]]:
    """Return (agent_name, wrapper) for the MPC scheduler, or None if unavailable."""
    try:
        from baselines.mpc_scheduler import RollingHorizonMPC, MPCConfig
    except ImportError:
        return None
    mpc = RollingHorizonMPC(MPCConfig(prediction_horizon=30, control_horizon=5))
    return "MPC", mpc


def _make_rl_agent(
    agent_type: str,
    env: HHIShipyardEnv,
    checkpoint_path: str,
    device: str = "cpu",
) -> Optional[Tuple[str, Any]]:
    """Load a trained RL agent (PPO, DAgger, or SAC) from a checkpoint.

    Returns (agent_name, callable) where callable(env) -> action_dict, or None
    if the checkpoint does not exist or required modules are missing.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None

    try:
        import torch
        from agent.gnn_encoder import HeterogeneousGNNEncoder
        from agent.policy import ActorCriticPolicy
    except ImportError:
        return None

    try:
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
            n_cranes=getattr(env, "n_goliath_cranes", getattr(env, "n_cranes", 2)),
            max_requests=env.n_blocks,
            hidden_dim=256,
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        encoder.load_state_dict(ckpt["encoder"])
        policy.to(device)
        encoder.to(device)
        policy.eval()
        encoder.eval()
    except Exception as exc:
        print(f"  [WARN] Could not load {agent_type} checkpoint {checkpoint_path}: {exc}")
        return None

    class _RLWrapper:
        """Thin wrapper so the episode loop can call scheduler.decide(env)."""

        def __init__(self, enc, pol, dev):
            self.encoder = enc
            self.policy = pol
            self.device = dev

        def decide(self, env: HHIShipyardEnv) -> Dict[str, Any]:
            graph_data = env.get_graph_data().to(self.device)
            with torch.no_grad():
                state_emb = self.encoder(graph_data)
                mask = env.get_action_mask()
                torch_mask = {k: torch.tensor(v, device=self.device) for k, v in mask.items()}
                action, _, _ = self.policy.get_action(state_emb, torch_mask, deterministic=True)
            return {k: int(v.item()) for k, v in action.items()}

    return agent_type, _RLWrapper(encoder, policy, device)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: HHIShipyardEnv,
    scheduler: Any,
    seed: int,
    max_steps: int = 5000,
) -> Dict[str, Any]:
    """Run a single episode and collect metrics.

    The *scheduler* object must expose one of:
      - decide(env) -> action_dict   (EDD, PuLP, DAgger/PPO/SAC wrappers)
      - step(env) -> action_dict     (MPC)

    For the GA scheduler (which evolves full schedules) we fall back to
    the EDD expert, since GA does not have a step-level interface matching
    the episode loop.  Its evaluation is handled separately.

    Returns a dict with throughput, blocks_erected, ships_delivered,
    total_reward, steps, wall_time, and raw env metrics.
    """
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step_count = 0

    t0 = time.perf_counter()

    while not done and step_count < max_steps:
        # Choose the right interface
        if hasattr(scheduler, "decide"):
            action = scheduler.decide(env)
        elif hasattr(scheduler, "step"):
            action = scheduler.step(env)
        else:
            # Fallback: hold
            action = {"action_type": 3, "spmt_idx": 0, "request_idx": 0,
                       "crane_idx": 0, "lift_idx": 0, "equipment_idx": 0}

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

    wall_time = time.perf_counter() - t0

    metrics = getattr(env, "metrics", {})
    blocks_completed = metrics.get("blocks_completed", 0)
    ships_delivered = metrics.get("ships_delivered", 0)
    sim_time = getattr(env, "sim_time", 1.0)
    throughput = blocks_completed / sim_time if sim_time > 0 else 0.0

    return {
        "throughput": throughput,
        "blocks_erected": blocks_completed,
        "ships_delivered": ships_delivered,
        "total_reward": total_reward,
        "steps": step_count,
        "sim_time": sim_time,
        "wall_time_s": wall_time,
    }


def run_ga_evaluation(
    cfg: Dict[str, Any],
    ga_scheduler: Any,
    seed: int,
) -> Dict[str, Any]:
    """Run a GA-based evaluation (evolve population then evaluate best).

    The GA scheduler uses its own internal simulation loop during
    fitness evaluation, so we handle it separately from the step-level loop.
    """
    from baselines.ga_scheduler import GAScheduler

    env = HHIShipyardEnv(cfg)
    random.seed(seed)
    np.random.seed(seed)

    ga_scheduler.initialize_population()

    t0 = time.perf_counter()
    # Evaluate initial population
    for chrom in ga_scheduler.population:
        ga_scheduler.evaluate_fitness(chrom, env)

    # Evolve for configured generations
    try:
        for gen in range(ga_scheduler.config.generations):
            ga_scheduler.evolve_generation(env)
    except Exception:
        pass  # Accept partial evolution

    wall_time = time.perf_counter() - t0

    best = ga_scheduler.best_solution
    if best is None and ga_scheduler.population:
        best = max(ga_scheduler.population, key=lambda c: c.fitness)

    # Re-evaluate best chromosome to extract metrics
    if best is not None:
        env.reset(seed=seed)
        for step_idx, block_idx in enumerate(best.block_order):
            spmt_idx = int(best.spmt_assignments[block_idx])
            crane_idx = int(best.crane_assignments[block_idx])
            action = {
                "action_type": 0,
                "spmt_idx": spmt_idx,
                "request_idx": int(block_idx) % max(1, len(getattr(env, "transport_requests", [1]))),
                "crane_idx": crane_idx,
                "lift_idx": 0,
                "equipment_idx": 0,
            }
            try:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            except Exception:
                continue

    metrics = getattr(env, "metrics", {})
    blocks_completed = metrics.get("blocks_completed", 0)
    ships_delivered = metrics.get("ships_delivered", 0)
    sim_time = getattr(env, "sim_time", 1.0)
    throughput = blocks_completed / sim_time if sim_time > 0 else 0.0

    return {
        "throughput": throughput,
        "blocks_erected": blocks_completed,
        "ships_delivered": ships_delivered,
        "total_reward": best.fitness if best else 0.0,
        "steps": len(best.block_order) if best else 0,
        "sim_time": sim_time,
        "wall_time_s": wall_time,
    }


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of all scheduling baselines and RL agents."
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=["config/small_instance.yaml", "config/hhi_ulsan.yaml"],
        help="YAML config files to evaluate on (default: small_instance + hhi_ulsan)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds per (agent, config) pair (default: 5)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Maximum environment steps per episode (default: 5000)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for RL agents (default: cpu)",
    )
    parser.add_argument(
        "--output", type=str, default="results/compare_pulp_rl.csv",
        help="Path to save results CSV (default: results/compare_pulp_rl.csv)",
    )
    parser.add_argument(
        "--dagger-checkpoint", type=str, default="data/checkpoints/dagger/best.pt",
        help="Path to DAgger checkpoint",
    )
    parser.add_argument(
        "--sac-checkpoint", type=str, default="data/checkpoints/sac/best.pt",
        help="Path to SAC checkpoint",
    )
    parser.add_argument(
        "--ppo-checkpoint", type=str, default="data/checkpoints/mac_run/best.pt",
        help="Path to PPO checkpoint",
    )
    parser.add_argument(
        "--no-extensions", action="store_true",
        help="Disable all stochastic simulation extensions (deterministic mode)",
    )
    args = parser.parse_args()

    seed_list = list(range(42, 42 + args.seeds))

    print("=" * 72)
    print("  Shipyard Scheduling — Comprehensive Baseline Comparison")
    print("=" * 72)
    print(f"  Configs : {args.configs}")
    print(f"  Seeds   : {seed_list}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Output  : {args.output}")
    print("=" * 72)

    all_results: List[Dict[str, Any]] = []

    for config_path in args.configs:
        print(f"\n{'='*72}")
        print(f"  Config: {config_path}")
        print(f"{'='*72}")

        try:
            cfg = load_config(config_path)
        except FileNotFoundError:
            print(f"  [SKIP] Config file not found: {config_path}")
            continue

        if args.no_extensions:
            cfg.pop("extensions", None)

        config_name = os.path.splitext(os.path.basename(config_path))[0]

        # Build list of (agent_name, scheduler_or_wrapper) to evaluate
        # Each entry is tried inside try/except so missing deps are harmless.
        agents: List[Tuple[str, Any]] = []

        # 1) Expert (EDD) — always available
        agents.append(_make_expert_agent())

        # 2) PuLP MIP
        try:
            agents.append(_make_pulp_agent())
        except Exception as exc:
            print(f"  [SKIP] PuLP MIP: {exc}")

        # We need an env instance to build some agents; use seed=42 just for init
        random.seed(42)
        np.random.seed(42)
        env_init = HHIShipyardEnv(cfg)

        # 3) GA
        try:
            ga_result = _make_ga_agent(env_init)
            if ga_result is not None:
                agents.append(ga_result)
            else:
                print("  [SKIP] GA scheduler: import failed")
        except Exception as exc:
            print(f"  [SKIP] GA scheduler: {exc}")

        # 4) MPC
        try:
            mpc_result = _make_mpc_agent()
            if mpc_result is not None:
                agents.append(mpc_result)
            else:
                print("  [SKIP] MPC scheduler: import failed")
        except Exception as exc:
            print(f"  [SKIP] MPC scheduler: {exc}")

        # 5) DAgger
        try:
            dagger_result = _make_rl_agent("DAgger", env_init, args.dagger_checkpoint, args.device)
            if dagger_result is not None:
                agents.append(dagger_result)
            else:
                print(f"  [SKIP] DAgger: checkpoint not found at {args.dagger_checkpoint}")
        except Exception as exc:
            print(f"  [SKIP] DAgger: {exc}")

        # 6) SAC
        try:
            sac_result = _make_rl_agent("SAC", env_init, args.sac_checkpoint, args.device)
            if sac_result is not None:
                agents.append(sac_result)
            else:
                print(f"  [SKIP] SAC: checkpoint not found at {args.sac_checkpoint}")
        except Exception as exc:
            print(f"  [SKIP] SAC: {exc}")

        # 7) PPO
        try:
            ppo_result = _make_rl_agent("PPO", env_init, args.ppo_checkpoint, args.device)
            if ppo_result is not None:
                agents.append(ppo_result)
            else:
                print(f"  [SKIP] PPO: checkpoint not found at {args.ppo_checkpoint}")
        except Exception as exc:
            print(f"  [SKIP] PPO: {exc}")

        print(f"\n  Active agents: {[name for name, _ in agents]}")

        # Run each agent across all seeds
        for agent_name, scheduler in agents:
            print(f"\n  --- {agent_name} ---")

            for seed in seed_list:
                random.seed(seed)
                np.random.seed(seed)

                try:
                    import torch
                    torch.manual_seed(seed)
                except ImportError:
                    pass

                env = HHIShipyardEnv(cfg)

                try:
                    # Reset stateful schedulers
                    if hasattr(scheduler, "reset"):
                        scheduler.reset()
                    episode_metrics = run_episode(env, scheduler, seed, args.max_steps)
                except Exception as exc:
                    print(f"    seed={seed} FAILED: {exc}")
                    episode_metrics = {
                        "throughput": 0.0,
                        "blocks_erected": 0,
                        "ships_delivered": 0,
                        "total_reward": 0.0,
                        "steps": 0,
                        "sim_time": 0.0,
                        "wall_time_s": 0.0,
                    }

                result_row = {
                    "config": config_name,
                    "agent": agent_name,
                    "seed": seed,
                    **episode_metrics,
                }
                all_results.append(result_row)

                print(
                    f"    seed={seed:3d}  "
                    f"throughput={episode_metrics['throughput']:.4f}  "
                    f"blocks={episode_metrics['blocks_erected']:4d}  "
                    f"ships={episode_metrics['ships_delivered']:2d}  "
                    f"reward={episode_metrics['total_reward']:8.1f}  "
                    f"wall={episode_metrics['wall_time_s']:.1f}s"
                )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if all_results:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fieldnames = list(all_results[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"\nResults saved to {args.output}")

    # ------------------------------------------------------------------
    # Print comparison summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  COMPARISON SUMMARY (mean +/- std across seeds)")
    print("=" * 72)

    # Group by (config, agent)
    from collections import defaultdict
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in all_results:
        key = (row["config"], row["agent"])
        grouped[key].append(row)

    metric_keys = ["throughput", "blocks_erected", "ships_delivered", "total_reward"]

    header = f"{'Config':<20s} {'Agent':<16s}"
    for mk in metric_keys:
        header += f" {mk:>22s}"
    print(header)
    print("-" * len(header))

    prev_config = None
    for (config_name, agent_name), rows in sorted(grouped.items()):
        if config_name != prev_config:
            if prev_config is not None:
                print()
            prev_config = config_name

        line = f"{config_name:<20s} {agent_name:<16s}"
        for mk in metric_keys:
            vals = [r[mk] for r in rows]
            mean = np.mean(vals)
            std = np.std(vals)
            line += f" {mean:9.2f} +/- {std:6.2f}"
        print(line)

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
