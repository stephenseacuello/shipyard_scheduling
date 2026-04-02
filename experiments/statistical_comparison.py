#!/usr/bin/env python3
"""Statistical comparison of scheduling agents.

Runs Expert, MPC, GA, and (optionally) DAgger on specified configs with
multiple seeds, then computes:
  - Mann-Whitney U tests (non-parametric)
  - Cohen's d effect sizes
  - 95% confidence intervals

Outputs: CSV results + LaTeX table fragment for paper.

Usage:
    python experiments/statistical_comparison.py \
        --configs config/small_instance.yaml config/medium_hhi.yaml \
        --seeds 10 --max-steps 2000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import yaml
from simulation.shipyard_env import HHIShipyardEnv
from baselines.rule_based import RuleBasedScheduler


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using t-distribution."""
    from scipy import stats
    n = len(data)
    if n < 2:
        return (float(np.mean(data)), float(np.mean(data)))
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (float(mean - h), float(mean + h))


def mann_whitney_u(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Compute Mann-Whitney U statistic and p-value."""
    from scipy import stats
    if len(group1) < 2 or len(group2) < 2:
        return (0.0, 1.0)
    stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    return (float(stat), float(p))


# ---------------------------------------------------------------------------
# Episode runner (mirrors compare_pulp_rl.run_episode)
# ---------------------------------------------------------------------------

def run_episode(
    env: HHIShipyardEnv,
    scheduler: Any,
    seed: int,
    max_steps: int = 5000,
) -> Dict[str, Any]:
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
    throughput = blocks_completed / sim_time if sim_time > 0 else 0.0

    return {
        "throughput": throughput,
        "blocks_completed": blocks_completed,
        "ships_delivered": ships_delivered,
        "total_reward": total_reward,
        "steps": step_count,
        "sim_time": sim_time,
        "wall_time_s": wall_time,
    }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_agents(env: HHIShipyardEnv, skip_ga: bool = False) -> Dict[str, Any]:
    """Create all available agents for a given environment."""
    agents = {}

    # Expert (EDD)
    agents["Expert"] = RuleBasedScheduler()

    # MPC
    try:
        from baselines.mpc_scheduler import RollingHorizonMPC
        agents["MPC"] = RollingHorizonMPC()
    except Exception as e:
        print(f"  MPC not available: {e}")

    # GA
    if not skip_ga:
        try:
            from baselines.ga_scheduler import GAScheduler, GAConfig
            n_blocks = getattr(env, "n_blocks", 50)
            n_spmts = getattr(env, "n_spmts", 6)
            n_cranes = getattr(env, "n_goliath_cranes", getattr(env, "n_cranes", 2))
            agents["GA"] = GAScheduler(GAConfig(), n_blocks, n_spmts, n_cranes)
        except Exception as e:
            print(f"  GA not available: {e}")
    else:
        print("  GA skipped (--skip-ga)")

    return agents


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    config_paths: List[str],
    n_seeds: int = 10,
    max_steps: int = 5000,
    disable_extensions: bool = False,
    skip_ga: bool = False,
) -> List[Dict[str, Any]]:
    """Run all agents on all configs with multiple seeds."""
    all_results = []

    for config_path in config_paths:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        print(f"\n{'=' * 60}")
        print(f"Config: {config_name}")
        print(f"{'=' * 60}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        if disable_extensions:
            cfg.pop("extensions", None)
        env = HHIShipyardEnv(cfg)

        agents = create_agents(env, skip_ga=skip_ga)

        for agent_name, agent in agents.items():
            print(f"\n  Agent: {agent_name}")
            for seed in range(n_seeds):
                if hasattr(agent, "reset"):
                    agent.reset()
                result = run_episode(env, agent, seed=seed, max_steps=max_steps)
                result["config"] = config_name
                result["agent"] = agent_name
                result["seed"] = seed
                all_results.append(result)
                print(f"    Seed {seed}: blocks={result['blocks_completed']}, "
                      f"ships={result['ships_delivered']}, "
                      f"throughput={result['throughput']:.4f}, "
                      f"time={result['wall_time_s']:.1f}s")

    return all_results


def compute_statistics(results: List[Dict[str, Any]]) -> str:
    """Compute pairwise statistical tests and format output."""
    from collections import defaultdict

    # Group results by (config, agent)
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in results:
        grouped[(r["config"], r["agent"])].append(r)

    configs = sorted(set(r["config"] for r in results))
    agents = sorted(set(r["agent"] for r in results))

    output_lines = []

    for config in configs:
        output_lines.append(f"\n{'=' * 60}")
        output_lines.append(f"Statistical Analysis: {config}")
        output_lines.append(f"{'=' * 60}")

        # Summary table
        output_lines.append(f"\n{'Agent':<12} {'Blocks':>8} {'Ships':>6} {'Throughput':>12} {'95% CI':>22}")
        output_lines.append("-" * 62)

        agent_throughputs = {}
        for agent in agents:
            key = (config, agent)
            if key not in grouped:
                continue
            runs = grouped[key]
            tp = np.array([r["throughput"] for r in runs])
            blocks = np.array([r["blocks_completed"] for r in runs])
            ships = np.array([r["ships_delivered"] for r in runs])
            agent_throughputs[agent] = tp

            try:
                ci = confidence_interval(tp)
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            except Exception:
                ci_str = "N/A"

            output_lines.append(
                f"{agent:<12} {np.mean(blocks):>8.1f} {np.mean(ships):>6.1f} "
                f"{np.mean(tp):>12.4f} {ci_str:>22}"
            )

        # Pairwise tests
        agent_list = [a for a in agents if a in agent_throughputs]
        if len(agent_list) >= 2:
            output_lines.append(f"\nPairwise Comparisons (Mann-Whitney U):")
            output_lines.append(f"{'Pair':<25} {'U-stat':>10} {'p-value':>10} {'Cohen d':>10} {'Significant':>12}")
            output_lines.append("-" * 70)

            for i, a1 in enumerate(agent_list):
                for a2 in agent_list[i + 1:]:
                    tp1 = agent_throughputs[a1]
                    tp2 = agent_throughputs[a2]
                    u_stat, p_val = mann_whitney_u(tp1, tp2)
                    d = cohens_d(tp1, tp2)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    output_lines.append(
                        f"{a1} vs {a2:<15} {u_stat:>10.1f} {p_val:>10.4f} {d:>10.2f} {sig:>12}"
                    )

    return "\n".join(output_lines)


def generate_latex_table(results: List[Dict[str, Any]]) -> str:
    """Generate LaTeX table fragment for paper."""
    from collections import defaultdict

    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in results:
        grouped[(r["config"], r["agent"])].append(r)

    configs = sorted(set(r["config"] for r in results))
    agents = sorted(set(r["agent"] for r in results))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Agent comparison across instance sizes (mean $\pm$ std, $n$=10 seeds)}",
        r"\label{tab:statistical_comparison}",
        r"\begin{tabular}{ll" + "c" * len(configs) + "}",
        r"\toprule",
        r"Agent & Metric & " + " & ".join(configs) + r" \\",
        r"\midrule",
    ]

    for agent in agents:
        for metric, label in [("blocks_completed", "Blocks"), ("throughput", "Throughput")]:
            row = f"{agent if metric == 'blocks_completed' else ''} & {label}"
            for config in configs:
                key = (config, agent)
                if key in grouped:
                    vals = np.array([r[metric] for r in grouped[key]])
                    if metric == "throughput":
                        row += f" & ${np.mean(vals):.4f} \\pm {np.std(vals):.4f}$"
                    else:
                        row += f" & ${np.mean(vals):.0f} \\pm {np.std(vals):.0f}$"
                else:
                    row += " & --"
            row += r" \\"
            lines.append(row)
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of scheduling agents")
    parser.add_argument("--configs", nargs="+",
                        default=[
                            os.path.join(PROJECT_ROOT, "config", "small_instance.yaml"),
                        ])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "statistical_comparison.csv"))
    parser.add_argument("--no-extensions", action="store_true",
                        help="Disable all stochastic simulation extensions (deterministic mode)")
    parser.add_argument("--skip-ga", action="store_true",
                        help="Skip GA agent (very slow on large instances)")
    args = parser.parse_args()

    print("Statistical Comparison of Scheduling Agents")
    print(f"Configs: {args.configs}")
    print(f"Seeds: {args.seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Extensions: {'OFF' if args.no_extensions else 'ON'}")

    results = run_comparison(args.configs, n_seeds=args.seeds, max_steps=args.max_steps,
                             disable_extensions=args.no_extensions,
                             skip_ga=args.skip_ga)

    # Save raw results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\nRaw results saved to {args.output}")

    # Statistical analysis
    stats_output = compute_statistics(results)
    print(stats_output)

    # LaTeX table
    latex = generate_latex_table(results)
    latex_path = os.path.join(PROJECT_ROOT, "paper", "data", "statistical_comparison.tex")
    os.makedirs(os.path.dirname(latex_path), exist_ok=True)
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
