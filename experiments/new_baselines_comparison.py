#!/usr/bin/env python3
"""Compare new baselines (MCTS, Random, FIFO, CPM) against Expert (EDD).

Runs each agent on tiny and small configs for 3 seeds, 500 steps each.
Reports throughput (blocks_completed / steps) and saves to CSV.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import yaml
from simulation.shipyard_env import HHIShipyardEnv
from baselines.rule_based import RuleBasedScheduler
from baselines.random_policy import RandomScheduler
from baselines.fifo_policy import FIFOScheduler
from baselines.cpm_scheduler import CPMScheduler
from agent.mcts import MCTSScheduler


def run_episode(env, scheduler, seed: int, max_steps: int = 500) -> Dict[str, Any]:
    """Run a single episode and collect metrics."""
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step_count = 0

    t0 = time.perf_counter()
    if hasattr(scheduler, "reset"):
        scheduler.reset()

    while not done and step_count < max_steps:
        if hasattr(scheduler, "decide"):
            action = scheduler.decide(env)
        elif hasattr(scheduler, "step"):
            action = scheduler.step(env)
        else:
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

    return {
        "blocks_completed": blocks_completed,
        "ships_delivered": ships_delivered,
        "total_reward": total_reward,
        "steps": step_count,
        "sim_time": sim_time,
        "wall_time_s": wall_time,
        "throughput_per_step": blocks_completed / max(step_count, 1),
        "throughput_per_simtime": blocks_completed / max(sim_time, 1),
    }


def main():
    configs = {
        "tiny": os.path.join(PROJECT_ROOT, "config", "tiny_instance.yaml"),
        "small": os.path.join(PROJECT_ROOT, "config", "small_instance.yaml"),
    }
    n_seeds = 3
    max_steps = 500

    # Build agent factories (functions that return fresh agents)
    agent_factories = {
        "Expert": lambda seed: RuleBasedScheduler(),
        "Random": lambda seed: RandomScheduler(seed=seed),
        "FIFO": lambda seed: FIFOScheduler(),
        "CPM": lambda seed: CPMScheduler(),
        "MCTS": lambda seed: MCTSScheduler(n_simulations=10, max_rollout_depth=10),
    }

    all_results: List[Dict[str, Any]] = []

    # First, do a quick MCTS timing test on tiny to see if we need to reduce n_simulations
    print("=" * 60)
    print("Quick MCTS timing test on tiny (1 seed, 20 steps)...")
    print("=" * 60)
    cfg_path = configs["tiny"]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.pop("extensions", None)  # disable stochastic extensions for speed
    test_env = HHIShipyardEnv(cfg)
    mcts_agent = MCTSScheduler(n_simulations=10, max_rollout_depth=10)
    test_result = run_episode(test_env, mcts_agent, seed=0, max_steps=20)
    mcts_time_per_step = test_result["wall_time_s"] / max(test_result["steps"], 1)
    estimated_500 = mcts_time_per_step * 500
    print(f"  MCTS: {test_result['wall_time_s']:.1f}s for {test_result['steps']} steps "
          f"({mcts_time_per_step:.2f}s/step)")
    print(f"  Estimated time for 500 steps: {estimated_500:.0f}s")

    if estimated_500 > 30:
        print("  -> MCTS too slow with n_simulations=10, reducing to 5")
        agent_factories["MCTS"] = lambda seed: MCTSScheduler(n_simulations=5, max_rollout_depth=10)
        # Re-test
        mcts_agent = MCTSScheduler(n_simulations=5, max_rollout_depth=10)
        test_result = run_episode(test_env, mcts_agent, seed=0, max_steps=20)
        mcts_time_per_step = test_result["wall_time_s"] / max(test_result["steps"], 1)
        estimated_500 = mcts_time_per_step * 500
        print(f"  MCTS (n=5): {test_result['wall_time_s']:.1f}s for {test_result['steps']} steps "
              f"({mcts_time_per_step:.2f}s/step)")
        print(f"  Estimated time for 500 steps: {estimated_500:.0f}s")

    del test_env

    # Main comparison loop
    for config_name, config_path in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Config: {config_name} ({config_path})")
        print(f"Seeds: {n_seeds}, Max steps: {max_steps}")
        print(f"{'=' * 60}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("extensions", None)  # deterministic for reproducibility

        for agent_name, factory in agent_factories.items():
            print(f"\n  Agent: {agent_name}")
            for seed in range(n_seeds):
                env = HHIShipyardEnv(cfg)
                agent = factory(seed)
                result = run_episode(env, agent, seed=seed, max_steps=max_steps)
                result["config"] = config_name
                result["agent"] = agent_name
                result["seed"] = seed
                all_results.append(result)
                print(f"    Seed {seed}: blocks={result['blocks_completed']:>4}, "
                      f"throughput/step={result['throughput_per_step']:.4f}, "
                      f"reward={result['total_reward']:.1f}, "
                      f"wall={result['wall_time_s']:.1f}s")
                del env

    # Save CSV
    output_path = os.path.join(PROJECT_ROOT, "data", "new_baselines_comparison.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if all_results:
        fieldnames = ["config", "agent", "seed", "blocks_completed", "ships_delivered",
                      "total_reward", "steps", "sim_time", "wall_time_s",
                      "throughput_per_step", "throughput_per_simtime"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<8} {'Agent':<10} {'Blocks (mean±std)':<22} "
          f"{'Throughput/step':<22} {'Wall time (s)':<15}")
    print("-" * 80)

    for config_name in configs:
        for agent_name in agent_factories:
            rows = [r for r in all_results
                    if r["config"] == config_name and r["agent"] == agent_name]
            if not rows:
                continue
            blocks = np.array([r["blocks_completed"] for r in rows])
            tp = np.array([r["throughput_per_step"] for r in rows])
            wt = np.array([r["wall_time_s"] for r in rows])
            print(f"{config_name:<8} {agent_name:<10} "
                  f"{np.mean(blocks):>6.1f} ± {np.std(blocks):>5.1f}      "
                  f"{np.mean(tp):>8.4f} ± {np.std(tp):>6.4f}    "
                  f"{np.mean(wt):>6.1f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
