#!/usr/bin/env python3
"""Extended statistical validation for shipyard scheduling with 20 seeds.

Runs the expert (rule-based) scheduler on HHI Ulsan configuration with
seeds 0-19, 1000 steps each. Records comprehensive metrics and computes
statistical analysis including confidence intervals.

For publication-quality results.
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np
import yaml
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from simulation.shipyard_env import HHIShipyardEnv
from baselines.rule_based import RuleBasedScheduler


@dataclass
class SeedResult:
    """Results from a single seed run."""
    seed: int
    blocks_completed: int
    ships_delivered: int
    total_reward: float
    breakdowns: int
    steps_run: int
    wall_time: float
    on_time_blocks: int = 0
    total_tardiness: float = 0.0
    avg_equipment_health: float = 100.0
    transport_dispatches: int = 0
    crane_dispatches: int = 0
    maintenance_actions: int = 0
    hold_actions: int = 0
    blocks_in_progress: int = 0
    sim_time: float = 0.0


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with inheritance."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


def run_single_seed(config: Dict[str, Any], seed: int, max_steps: int) -> SeedResult:
    """Run expert scheduler for one seed."""
    import random
    import torch
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = HHIShipyardEnv(config)
    scheduler = RuleBasedScheduler()
    
    obs, info = env.reset(seed=seed)
    
    total_reward = 0.0
    blocks_completed = 0
    ships_delivered = 0
    breakdowns = 0
    on_time_blocks = 0
    total_tardiness = 0.0
    transport_dispatches = 0
    crane_dispatches = 0
    maintenance_actions = 0
    hold_actions = 0
    
    start_time = time.time()
    
    for step in range(max_steps):
        action = scheduler.decide(env)
        
        # Track action types
        action_type = action.get("action_type", 3)
        if action_type == 0:
            transport_dispatches += 1
        elif action_type == 1:
            crane_dispatches += 1
        elif action_type == 2:
            maintenance_actions += 1
        else:
            hold_actions += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Track metrics from info
        if info.get("block_completed"):
            blocks_completed += 1
            # Check if on time
            block_info = info.get("completed_block_info", {})
            tardiness = block_info.get("tardiness", 0)
            if tardiness <= 0:
                on_time_blocks += 1
            else:
                total_tardiness += tardiness
                
        if info.get("ship_delivered"):
            ships_delivered += 1
            
        if info.get("breakdown_occurred"):
            breakdowns += 1
            
        if terminated or truncated:
            break
    
    wall_time = time.time() - start_time
    
    # Calculate average equipment health at end
    avg_health = 100.0
    blocks_in_progress = 0
    try:
        spmts = env.entities.get("spmts", [])
        cranes = env.entities.get("goliath_cranes", [])
        healths = []
        for s in spmts:
            healths.append(s.get_min_health())
        for c in cranes:
            healths.append(c.get_min_health())
        if healths:
            avg_health = np.mean(healths)
        
        # Count blocks in various processing stages
        blocks = env.entities.get("blocks", [])
        for b in blocks:
            if hasattr(b, 'status') and str(b.status) not in ['PENDING', 'DELIVERED']:
                blocks_in_progress += 1
    except Exception:
        pass
    
    return SeedResult(
        seed=seed,
        blocks_completed=blocks_completed,
        ships_delivered=ships_delivered,
        total_reward=total_reward,
        breakdowns=breakdowns,
        steps_run=step + 1,
        wall_time=wall_time,
        on_time_blocks=on_time_blocks,
        total_tardiness=total_tardiness,
        avg_equipment_health=avg_health,
        transport_dispatches=transport_dispatches,
        crane_dispatches=crane_dispatches,
        maintenance_actions=maintenance_actions,
        hold_actions=hold_actions,
        blocks_in_progress=blocks_in_progress,
        sim_time=env.sim_time,
    )


def compute_confidence_interval(data: np.ndarray, confidence: float) -> tuple:
    """Compute confidence interval for the mean."""
    n = len(data)
    if n < 2:
        return (np.mean(data), np.mean(data))
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return (mean, mean)
    
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def main():
    print("=" * 70)
    print("EXTENDED STATISTICAL VALIDATION - SHIPYARD SCHEDULING")
    print("Expert Scheduler on HHI Ulsan Configuration")
    print("20 Seeds, 1000 Steps Each")
    print("=" * 70)
    print()
    
    # Load config
    config_path = "config/hhi_ulsan.yaml"
    full_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config_path
    )
    
    print(f"Loading config: {full_config_path}")
    config = load_config(full_config_path)
    
    # Run parameters
    n_seeds = 20
    max_steps = 1000
    
    print(f"Seeds: 0-{n_seeds-1}")
    print(f"Steps per seed: {max_steps}")
    print()
    
    # Run all seeds
    results: List[SeedResult] = []
    
    print("-" * 90)
    print(f"{'Seed':>4} | {'Transport':>9} | {'Crane':>7} | {'Maint':>6} | {'Hold':>6} | {'Reward':>10} | {'SimTime':>8} | {'Time':>6}")
    print("-" * 90)
    
    total_start = time.time()
    
    for seed in range(n_seeds):
        result = run_single_seed(config, seed, max_steps)
        results.append(result)
        
        print(f"{result.seed:>4} | {result.transport_dispatches:>9} | {result.crane_dispatches:>7} | "
              f"{result.maintenance_actions:>6} | {result.hold_actions:>6} | {result.total_reward:>10.1f} | "
              f"{result.sim_time:>8.1f} | {result.wall_time:>5.1f}s")
    
    total_time = time.time() - total_start
    
    print("-" * 90)
    print(f"Total wall time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    
    # Extract arrays for statistical analysis
    rewards = np.array([r.total_reward for r in results])
    transport = np.array([r.transport_dispatches for r in results])
    crane = np.array([r.crane_dispatches for r in results])
    maintenance = np.array([r.maintenance_actions for r in results])
    hold = np.array([r.hold_actions for r in results])
    breakdowns = np.array([r.breakdowns for r in results])
    healths = np.array([r.avg_equipment_health for r in results])
    blocks_prog = np.array([r.blocks_in_progress for r in results])
    sim_times = np.array([r.sim_time for r in results])
    blocks = np.array([r.blocks_completed for r in results])
    ships = np.array([r.ships_delivered for r in results])
    
    # Print comprehensive statistics
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print()
    
    metrics = {
        "Total Reward": rewards,
        "Transport Dispatches": transport,
        "Crane Dispatches": crane,
        "Maintenance Actions": maintenance,
        "Hold Actions": hold,
        "Breakdowns": breakdowns,
        "Avg Equipment Health (%)": healths,
        "Blocks In Progress": blocks_prog,
        "Simulation Time (hrs)": sim_times,
        "Blocks Completed": blocks,
        "Ships Delivered": ships,
    }
    
    for name, data in metrics.items():
        print(f"\n{name}:")
        print(f"  Mean +/- Std:    {np.mean(data):.4f} +/- {np.std(data):.4f}")
        print(f"  Min / Max:       {np.min(data):.4f} / {np.max(data):.4f}")
        print(f"  Median:          {np.median(data):.4f}")
        
        ci_95 = compute_confidence_interval(data, 0.95)
        ci_99 = compute_confidence_interval(data, 0.99)
        print(f"  95% CI:          [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"  99% CI:          [{ci_99[0]:.4f}, {ci_99[1]:.4f}]")
    
    # Summary table for publication
    print()
    print("=" * 100)
    print("PUBLICATION-READY SUMMARY TABLE")
    print("=" * 100)
    print()
    print(f"{'Metric':<30} | {'Mean':>12} | {'Std':>10} | {'95% CI':>24} | {'99% CI':>24}")
    print("-" * 110)
    
    for name, data in metrics.items():
        mean = np.mean(data)
        std = np.std(data)
        ci_95 = compute_confidence_interval(data, 0.95)
        ci_99 = compute_confidence_interval(data, 0.99)
        
        ci_95_str = f"[{ci_95[0]:.2f}, {ci_95[1]:.2f}]"
        ci_99_str = f"[{ci_99[0]:.2f}, {ci_99[1]:.2f}]"
        
        print(f"{name:<30} | {mean:>12.2f} | {std:>10.2f} | {ci_95_str:>24} | {ci_99_str:>24}")
    
    print("-" * 110)
    print()
    
    # Additional statistics
    print("=" * 70)
    print("ADDITIONAL STATISTICS")
    print("=" * 70)
    print()
    
    # Action distribution
    total_actions = transport + crane + maintenance + hold
    print("Action Distribution (Mean %):")
    print(f"  Transport:   {np.mean(transport/total_actions)*100:.2f}%")
    print(f"  Crane:       {np.mean(crane/total_actions)*100:.2f}%")
    print(f"  Maintenance: {np.mean(maintenance/total_actions)*100:.2f}%")
    print(f"  Hold:        {np.mean(hold/total_actions)*100:.2f}%")
    
    # Throughput metrics
    print()
    print("Throughput Metrics:")
    avg_dispatches_per_hour = np.mean((transport + crane) / sim_times)
    print(f"  Avg Dispatches per Sim-Hour: {avg_dispatches_per_hour:.4f}")
    
    # Breakdown frequency
    total_steps = sum(r.steps_run for r in results)
    breakdown_freq = np.sum(breakdowns) / total_steps * 1000 if total_steps > 0 else 0
    print(f"  Breakdown Frequency: {breakdown_freq:.4f} per 1000 steps")
    
    # Coefficient of variation
    print()
    print("Coefficient of Variation (CV = std/mean):")
    for name, data in metrics.items():
        mean = np.mean(data)
        if mean != 0:
            cv = np.std(data) / abs(mean) * 100
            print(f"  {name}: {cv:.2f}%")
    
    # Inter-quartile range
    print()
    print("Inter-Quartile Range (IQR):")
    for name, data in metrics.items():
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        print(f"  {name}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    
    # Reward per simulation hour (efficiency metric)
    print()
    print("Efficiency Metrics:")
    reward_per_hour = rewards / sim_times
    print(f"  Reward per Sim-Hour: {np.mean(reward_per_hour):.4f} +/- {np.std(reward_per_hour):.4f}")
    
    print()
    print("=" * 70)
    print("Statistical validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
