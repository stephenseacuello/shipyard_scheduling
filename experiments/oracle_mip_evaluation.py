"""Oracle MIP Evaluation: Quantify the optimality gap between EDD and optimal.

This script solves the shipyard block scheduling problem as a static MIP
(Mixed-Integer Program) using PuLP/CBC to establish a lower bound on the
optimal makespan and total tardiness. It then compares this bound against
the EDD expert's actual simulation performance.

The static MIP optimizes block-to-equipment assignment and sequencing
with precedence constraints, providing a lower bound on the dynamic problem.

Usage:
    PYTHONPATH=src python experiments/oracle_mip_evaluation.py \
        --configs config/tiny_instance.yaml config/small_instance.yaml \
        --time-limit 600 --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simulation.shipyard_env import HHIShipyardEnv
from baselines.rule_based import RuleBasedScheduler

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


@dataclass
class OracleResult:
    """Results from oracle MIP evaluation."""
    config: str
    seed: int
    # MIP results
    mip_makespan: float
    mip_tardiness: float
    mip_objective: float
    mip_status: str
    mip_gap: float
    mip_solve_time: float
    # EDD simulation results
    edd_makespan: float
    edd_tardiness: float
    edd_blocks_completed: int
    edd_throughput: float
    edd_steps: int
    # Gap analysis
    makespan_gap_pct: float  # (EDD - MIP) / MIP * 100
    tardiness_gap_pct: float


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with inheritance support."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "inherit_from" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg["inherit_from"])
        base = load_config(base_path)
        base.update(cfg)
        return base
    return cfg


def run_edd_simulation(cfg: Dict, seed: int, max_steps: int) -> Dict[str, float]:
    """Run EDD expert through full simulation, collect metrics."""
    # Disable stochastic extensions for fair comparison with static MIP
    cfg_det = dict(cfg)
    cfg_det["extensions"] = {
        "enable_spatial": False,
        "enable_shifts": False,
        "enable_weather": False,
        "labor_leveling": False,
        "duration_uncertainty": False,
    }

    env = HHIShipyardEnv(cfg_det)
    obs, info = env.reset(seed=seed)
    scheduler = RuleBasedScheduler()

    blocks_completed = 0
    step_count = 0
    last_completion_time = 0.0

    done = False
    while not done and step_count < max_steps:
        action = scheduler.decide(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        # Track block completions from metrics dict
        metrics = info.get("metrics", info)
        current_completed = metrics.get("blocks_completed", 0)
        if current_completed > blocks_completed:
            last_completion_time = env.sim_time
            blocks_completed = current_completed

    # Get final metrics
    metrics = info.get("metrics", info)
    total_tardiness = metrics.get("total_tardiness", 0.0)
    makespan = last_completion_time if last_completion_time > 0 else env.sim_time
    throughput = blocks_completed / max(step_count, 1)

    return {
        "makespan": makespan,
        "tardiness": total_tardiness,
        "blocks_completed": blocks_completed,
        "throughput": throughput,
        "steps": step_count,
        "sim_time": env.sim_time,
    }


def solve_static_mip(
    cfg: Dict, seed: int, time_limit: float = 600.0
) -> Dict[str, float]:
    """Solve the static scheduling problem as a MIP using PuLP/CBC.

    Formulation:
    - Decision vars: x[i,s] = block i assigned to SPMT s; start[i] = start time
    - Minimize: alpha * makespan + (1-alpha) * weighted_tardiness
    - Subject to: assignment, precedence, disjunctive (SPMT sequencing)

    This provides a LOWER BOUND on the dynamic simulation problem.
    """
    if not HAS_PULP:
        return {
            "makespan": float("inf"),
            "tardiness": float("inf"),
            "objective": float("inf"),
            "status": "no_solver",
            "gap": 1.0,
            "solve_time": 0.0,
        }

    # Create deterministic environment to extract problem data
    cfg_det = dict(cfg)
    cfg_det["extensions"] = {
        "enable_spatial": False,
        "enable_shifts": False,
        "enable_weather": False,
        "labor_leveling": False,
        "duration_uncertainty": False,
    }

    env = HHIShipyardEnv(cfg_det)
    env.reset(seed=seed)

    blocks = env.entities.get("blocks", [])
    spmts = env.entities.get("spmts", [])

    n_blocks = len(blocks)
    n_spmts = len(spmts)

    # Build stage -> mean processing time mapping from facility configs
    all_facilities = env._get_all_facilities()
    stage_to_mean_pt = {}
    for fac in all_facilities:
        fac_name = fac.get("name", "")
        mean_pt = fac.get("processing_time_mean", 10.0)
        # Map facility to stage via _STAGE_MAP
        if fac_name in env._STAGE_MAP:
            stage = env._STAGE_MAP[fac_name]
            if stage not in stage_to_mean_pt:
                stage_to_mean_pt[stage] = mean_pt

    # Extract problem data
    processing_times = []
    due_dates = []
    predecessors_list = []

    for block in blocks:
        # Use stage-appropriate mean processing time
        pt = stage_to_mean_pt.get(block.current_stage, 10.0)
        # Apply processing multiplier (curved blocks take longer)
        multiplier = block.get_processing_multiplier() if hasattr(block, "get_processing_multiplier") else 1.0
        pt *= multiplier
        processing_times.append(pt)
        due_dates.append(block.due_date)
        preds = getattr(block, "predecessors", [])
        predecessors_list.append(preds if preds else [])

    # Estimate total time horizon
    M = max(due_dates) * 2 if due_dates else 10000.0

    print(f"  MIP: {n_blocks} blocks, {n_spmts} SPMTs, horizon={M:.0f}")
    print(f"  Decision variables: ~{n_blocks * n_spmts + n_blocks * 3} continuous + "
          f"{n_blocks * n_spmts + n_blocks * (n_blocks - 1) * n_spmts // 2} binary")

    start_time = time.time()

    # Build MIP
    prob = pulp.LpProblem("ShipyardOracle", pulp.LpMinimize)

    # Decision variables
    # x[i, s] = 1 if block i assigned to SPMT s
    x = {}
    for i in range(n_blocks):
        for s in range(n_spmts):
            x[i, s] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")

    # start[i] = start time of block i
    start_vars = [pulp.LpVariable(f"start_{i}", lowBound=0, upBound=M) for i in range(n_blocks)]
    # completion[i] = completion time of block i
    completion = [pulp.LpVariable(f"comp_{i}", lowBound=0, upBound=M) for i in range(n_blocks)]
    # tardiness[i] = max(0, completion[i] - due_date[i])
    tardiness = [pulp.LpVariable(f"tard_{i}", lowBound=0) for i in range(n_blocks)]
    # makespan
    makespan = pulp.LpVariable("makespan", lowBound=0)

    # Sequencing variables z[i, j, s] - only needed for blocks on same SPMT
    z = {}
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            for s in range(n_spmts):
                z[i, j, s] = pulp.LpVariable(f"z_{i}_{j}_{s}", cat="Binary")

    # Constraints

    # 1. Each block assigned to exactly one SPMT
    for i in range(n_blocks):
        prob += pulp.lpSum(x[i, s] for s in range(n_spmts)) == 1, f"assign_{i}"

    # 2. Completion time = start time + processing time
    for i in range(n_blocks):
        prob += completion[i] == start_vars[i] + processing_times[i], f"comp_{i}"

    # 3. Tardiness definition
    for i in range(n_blocks):
        prob += tardiness[i] >= completion[i] - due_dates[i], f"tard_{i}"

    # 4. Precedence constraints
    for i in range(n_blocks):
        for pred_id in predecessors_list[i]:
            if isinstance(pred_id, int) and 0 <= pred_id < n_blocks:
                prob += start_vars[i] >= completion[pred_id], f"prec_{pred_id}_{i}"

    # 5. Disjunctive constraints (if both blocks on same SPMT, they can't overlap)
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            for s in range(n_spmts):
                # If x[i,s] = 1 and x[j,s] = 1, then either i before j or j before i
                prob += (
                    start_vars[j] >= completion[i] - M * (1 - z[i, j, s]) - M * (2 - x[i, s] - x[j, s]),
                    f"seq1_{i}_{j}_{s}",
                )
                prob += (
                    start_vars[i] >= completion[j] - M * z[i, j, s] - M * (2 - x[i, s] - x[j, s]),
                    f"seq2_{i}_{j}_{s}",
                )

    # 6. Makespan definition
    for i in range(n_blocks):
        prob += makespan >= completion[i], f"mkspan_{i}"

    # Objective: weighted combination
    alpha = 0.5  # Balance makespan vs tardiness
    prob += alpha * makespan + (1 - alpha) * pulp.lpSum(tardiness[i] for i in range(n_blocks))

    # Solve
    solver = pulp.PULP_CBC_CMD(
        msg=1,  # Show solver output
        timeLimit=time_limit,
        gapRel=0.0,  # Try for optimality
    )

    print(f"  Solving with CBC (time limit: {time_limit}s)...")
    status = prob.solve(solver)
    solve_time = time.time() - start_time

    status_str = pulp.LpStatus[status]
    print(f"  Status: {status_str}, solve time: {solve_time:.1f}s")

    if status in (pulp.constants.LpStatusOptimal, 1):
        mip_makespan = makespan.varValue or float("inf")
        mip_tardiness = sum(
            (tardiness[i].varValue or 0.0) for i in range(n_blocks)
        )
        mip_objective = pulp.value(prob.objective) or float("inf")

        # Estimate gap from CBC
        best_bound = prob.bestBound if hasattr(prob, "bestBound") else mip_objective
        gap = abs(mip_objective - best_bound) / max(abs(mip_objective), 1e-10) if mip_objective != float("inf") else 1.0

        print(f"  Makespan: {mip_makespan:.1f}, Tardiness: {mip_tardiness:.1f}")
        print(f"  Objective: {mip_objective:.2f}, Gap: {gap:.4f}")

        return {
            "makespan": mip_makespan,
            "tardiness": mip_tardiness,
            "objective": mip_objective,
            "status": "optimal" if gap < 0.001 else "feasible",
            "gap": gap,
            "solve_time": solve_time,
        }
    else:
        print(f"  Solver failed: {status_str}")
        return {
            "makespan": float("inf"),
            "tardiness": float("inf"),
            "objective": float("inf"),
            "status": status_str,
            "gap": 1.0,
            "solve_time": solve_time,
        }


def run_random_simulation(cfg: Dict, seed: int, max_steps: int) -> Dict[str, float]:
    """Run random valid action selection as lower-bound baseline."""
    cfg_det = dict(cfg)
    cfg_det["extensions"] = {
        "enable_spatial": False,
        "enable_shifts": False,
        "enable_weather": False,
        "labor_leveling": False,
        "duration_uncertainty": False,
    }

    env = HHIShipyardEnv(cfg_det)
    obs, info = env.reset(seed=seed)
    rng = np.random.RandomState(seed)

    blocks_completed = 0
    step_count = 0
    last_completion_time = 0.0
    total_tardiness = 0.0

    done = False
    while not done and step_count < max_steps:
        # Get valid actions from mask
        mask = env.get_action_mask()
        action_type_mask = mask.get("action_type", np.ones(4, dtype=bool))
        valid_types = np.where(action_type_mask)[0]

        if len(valid_types) == 0:
            action_type = 3  # Hold
        else:
            action_type = rng.choice(valid_types)

        action = {
            "action_type": int(action_type),
            "spmt_idx": 0,
            "request_idx": 0,
            "crane_idx": 0,
            "lift_idx": 0,
            "erection_idx": 0,
            "equipment_idx": 0,
        }

        # Try to pick valid sub-actions
        if action_type == 0:  # Transport
            spmt_mask = mask.get("spmt_dispatch", None)
            if spmt_mask is not None and spmt_mask.any():
                valid = np.where(spmt_mask.any(axis=1))[0]
                if len(valid) > 0:
                    s = rng.choice(valid)
                    valid_reqs = np.where(spmt_mask[s])[0]
                    if len(valid_reqs) > 0:
                        action["spmt_idx"] = int(s)
                        action["request_idx"] = int(rng.choice(valid_reqs))
        elif action_type == 1:  # Crane
            crane_mask = mask.get("crane_dispatch", None)
            if crane_mask is not None and crane_mask.any():
                valid = np.where(crane_mask.any(axis=1))[0]
                if len(valid) > 0:
                    c = rng.choice(valid)
                    valid_reqs = np.where(crane_mask[c])[0]
                    if len(valid_reqs) > 0:
                        action["crane_idx"] = int(c)
                        action["lift_idx"] = int(rng.choice(valid_reqs))
                        action["erection_idx"] = action["lift_idx"]
        elif action_type == 2:  # Maintenance
            maint_mask = mask.get("maintenance", None)
            if maint_mask is not None and maint_mask.any():
                valid = np.where(maint_mask)[0]
                if len(valid) > 0:
                    action["equipment_idx"] = int(rng.choice(valid))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        metrics = info.get("metrics", info)
        current_completed = metrics.get("blocks_completed", 0)
        if current_completed > blocks_completed:
            last_completion_time = env.sim_time
            blocks_completed = current_completed

    makespan = last_completion_time if last_completion_time > 0 else env.sim_time
    throughput = blocks_completed / max(step_count, 1)

    return {
        "makespan": makespan,
        "tardiness": 0.0,
        "blocks_completed": blocks_completed,
        "throughput": throughput,
        "steps": step_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle MIP Evaluation")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config/tiny_instance.yaml", "config/small_instance.yaml"],
        help="Config files to evaluate",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument(
        "--time-limit", type=float, default=300.0, help="MIP solver time limit (seconds)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Max simulation steps for EDD"
    )
    parser.add_argument(
        "--output", default="data/oracle_gap_results.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    results: List[OracleResult] = []

    for config_path in args.configs:
        config_name = Path(config_path).stem
        cfg = load_config(config_path)
        n_blocks = cfg.get("n_blocks_per_ship", cfg.get("n_blocks", 50)) * cfg.get("n_ships", 1)
        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({n_blocks} blocks)")
        print(f"{'='*60}")

        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")

            # Run EDD simulation
            print("  Running EDD expert simulation...")
            edd = run_edd_simulation(cfg, seed, args.max_steps)
            print(f"  EDD: makespan={edd['makespan']:.1f}, blocks={edd['blocks_completed']}, "
                  f"throughput={edd['throughput']:.4f}")

            # Run random baseline
            print("  Running Random baseline...")
            rnd = run_random_simulation(cfg, seed, args.max_steps)
            print(f"  Random: makespan={rnd['makespan']:.1f}, blocks={rnd['blocks_completed']}, "
                  f"throughput={rnd['throughput']:.4f}")

            # Solve static MIP
            print("  Solving static MIP oracle...")
            mip = solve_static_mip(cfg, seed, args.time_limit)

            # Compute gaps
            if mip["makespan"] > 0 and mip["makespan"] < float("inf"):
                makespan_gap = (edd["makespan"] - mip["makespan"]) / mip["makespan"] * 100
                tardiness_gap = (
                    (edd["tardiness"] - mip["tardiness"]) / max(mip["tardiness"], 1.0) * 100
                    if mip["tardiness"] > 0
                    else 0.0
                )
            else:
                makespan_gap = float("nan")
                tardiness_gap = float("nan")

            result = OracleResult(
                config=config_name,
                seed=seed,
                mip_makespan=mip["makespan"],
                mip_tardiness=mip["tardiness"],
                mip_objective=mip["objective"],
                mip_status=mip["status"],
                mip_gap=mip["gap"],
                mip_solve_time=mip["solve_time"],
                edd_makespan=edd["makespan"],
                edd_tardiness=edd["tardiness"],
                edd_blocks_completed=edd["blocks_completed"],
                edd_throughput=edd["throughput"],
                edd_steps=edd["steps"],
                makespan_gap_pct=makespan_gap,
                tardiness_gap_pct=tardiness_gap,
            )
            results.append(result)

            if not np.isnan(makespan_gap):
                print(f"  >>> Makespan gap: {makespan_gap:+.1f}% (EDD vs MIP optimal)")
                print(f"  >>> Tardiness gap: {tardiness_gap:+.1f}%")
            else:
                print(f"  >>> MIP failed, no gap computed")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Oracle Gap Analysis")
    print(f"{'='*60}")

    for config_path in args.configs:
        config_name = Path(config_path).stem
        config_results = [r for r in results if r.config == config_name]
        if not config_results:
            continue

        valid = [r for r in config_results if not np.isnan(r.makespan_gap_pct)]
        if valid:
            gaps = [r.makespan_gap_pct for r in valid]
            statuses = [r.mip_status for r in valid]
            print(f"\n{config_name}:")
            print(f"  MIP status: {', '.join(statuses)}")
            print(f"  Makespan gap (EDD vs MIP): {np.mean(gaps):+.1f}% ± {np.std(gaps):.1f}%")
            print(f"  EDD throughput: {np.mean([r.edd_throughput for r in valid]):.4f}")
            print(f"  MIP solve time: {np.mean([r.mip_solve_time for r in valid]):.1f}s")

            if np.mean(gaps) < 10:
                print(f"  --> EDD is NEAR-OPTIMAL (<10% gap)")
            elif np.mean(gaps) < 20:
                print(f"  --> MODERATE gap (10-20%): some room for learned policies")
            else:
                print(f"  --> LARGE gap (>20%): significant headroom for improvement")
        else:
            print(f"\n{config_name}: MIP failed on all seeds")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "seed",
            "mip_makespan", "mip_tardiness", "mip_objective",
            "mip_status", "mip_gap", "mip_solve_time",
            "edd_makespan", "edd_tardiness", "edd_blocks_completed",
            "edd_throughput", "edd_steps",
            "makespan_gap_pct", "tardiness_gap_pct",
        ])
        for r in results:
            writer.writerow([
                r.config, r.seed,
                r.mip_makespan, r.mip_tardiness, r.mip_objective,
                r.mip_status, r.mip_gap, r.mip_solve_time,
                r.edd_makespan, r.edd_tardiness, r.edd_blocks_completed,
                r.edd_throughput, r.edd_steps,
                r.makespan_gap_pct, r.tardiness_gap_pct,
            ])
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
