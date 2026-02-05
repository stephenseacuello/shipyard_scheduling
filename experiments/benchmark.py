"""Benchmark suite for systematic comparison of algorithms and baselines.

This module provides a comprehensive benchmarking framework for comparing
different scheduling algorithms across multiple instance sizes and metrics.

Features:
- Multiple algorithm implementations (GNN-PPO, DQN, SAC, Rule-based, MIP, CP)
- Configurable instance sizes (small, medium, large, extra-large)
- Multiple random seeds for statistical significance
- Comprehensive metrics (on-time rate, tardiness, breakdowns, OEE, runtime)
- Results export to CSV and wandb
- Statistical analysis and visualization

Usage:
    # Run full benchmark suite
    python experiments/benchmark.py --all

    # Compare specific algorithms
    python experiments/benchmark.py --methods ppo dqn rule --instances small medium

    # Quick smoke test
    python experiments/benchmark.py --quick

    # With wandb logging
    python experiments/benchmark.py --all --wandb --wandb-project shipyard-benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import warnings

import numpy as np
import pandas as pd
import yaml

import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class InstanceConfig:
    """Configuration for a benchmark instance size."""
    name: str
    n_blocks: int
    n_spmts: int
    n_cranes: int
    max_time: int
    description: str = ""


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    instance: str
    seed: int
    on_time_rate: float
    average_tardiness: float
    total_tardiness: float
    throughput: float
    breakdown_count: int
    oee: float  # Overall Equipment Effectiveness
    runtime_seconds: float
    episodes_completed: int
    additional_metrics: Dict[str, float] = field(default_factory=dict)


# Standard instance configurations
INSTANCE_CONFIGS = {
    "tiny": InstanceConfig(
        name="tiny",
        n_blocks=10,
        n_spmts=2,
        n_cranes=1,
        max_time=500,
        description="Minimal instance for debugging",
    ),
    "small": InstanceConfig(
        name="small",
        n_blocks=20,
        n_spmts=3,
        n_cranes=2,
        max_time=2000,
        description="Small instance for quick experiments",
    ),
    "medium": InstanceConfig(
        name="medium",
        n_blocks=50,
        n_spmts=5,
        n_cranes=3,
        max_time=5000,
        description="Medium instance for standard experiments",
    ),
    "large": InstanceConfig(
        name="large",
        n_blocks=100,
        n_spmts=8,
        n_cranes=4,
        max_time=10000,
        description="Large instance for scalability testing",
    ),
    "xlarge": InstanceConfig(
        name="xlarge",
        n_blocks=200,
        n_spmts=12,
        n_cranes=6,
        max_time=20000,
        description="Extra-large instance for stress testing",
    ),
}


def load_base_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """Load base configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_instance_config(
    base_config: Dict[str, Any],
    instance: InstanceConfig,
) -> Dict[str, Any]:
    """Create configuration for a specific instance size."""
    config = dict(base_config)
    config.update({
        "n_blocks": instance.n_blocks,
        "n_spmts": instance.n_spmts,
        "n_cranes": instance.n_cranes,
        "max_time": instance.max_time,
    })
    return config


class BenchmarkRunner:
    """Orchestrates benchmark runs across methods and instances.

    Args:
        output_dir: Directory for results output.
        n_seeds: Number of random seeds to run.
        n_eval_episodes: Episodes per evaluation.
        device: Computation device.
        wandb_enabled: Whether to log to wandb.
        wandb_project: Wandb project name.
    """

    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        n_seeds: int = 5,
        n_eval_episodes: int = 10,
        device: str = "cpu",
        wandb_enabled: bool = False,
        wandb_project: str = "shipyard-benchmark",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_seeds = n_seeds
        self.n_eval_episodes = n_eval_episodes
        self.device = device
        self.wandb_enabled = wandb_enabled
        self.wandb_project = wandb_project

        self.results: List[BenchmarkResult] = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Method registry
        self._methods: Dict[str, Callable] = {}
        self._register_methods()

        # Initialize wandb if enabled
        if self.wandb_enabled:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=f"benchmark_{self.run_timestamp}",
                    config={
                        "n_seeds": n_seeds,
                        "n_eval_episodes": n_eval_episodes,
                        "device": device,
                    },
                )
            except ImportError:
                print("Warning: wandb not available, disabling logging")
                self.wandb_enabled = False

    def _register_methods(self) -> None:
        """Register available benchmark methods."""
        self._methods = {
            "ppo": self._run_ppo,
            "dqn": self._run_dqn,
            "rule_fifo": self._run_rule_based_fifo,
            "rule_edd": self._run_rule_based_edd,
            "rule_slack": self._run_rule_based_slack,
            "random": self._run_random,
        }

        # Optional methods that require additional dependencies
        try:
            import gurobipy
            self._methods["mip"] = self._run_mip
        except ImportError:
            pass

        try:
            from ortools.sat.python import cp_model
            self._methods["cp"] = self._run_cp
        except ImportError:
            pass

    @property
    def available_methods(self) -> List[str]:
        """List of available benchmark methods."""
        return list(self._methods.keys())

    def run_benchmark(
        self,
        methods: Optional[List[str]] = None,
        instances: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Run benchmark suite.

        Args:
            methods: List of method names to benchmark (default: all).
            instances: List of instance names to test (default: all).
            seeds: List of random seeds (default: range(n_seeds)).

        Returns:
            DataFrame with all benchmark results.
        """
        methods = methods or list(self._methods.keys())
        instances = instances or list(INSTANCE_CONFIGS.keys())
        seeds = seeds or list(range(self.n_seeds))

        # Validate inputs
        for method in methods:
            if method not in self._methods:
                raise ValueError(f"Unknown method: {method}. Available: {self.available_methods}")
        for instance in instances:
            if instance not in INSTANCE_CONFIGS:
                raise ValueError(f"Unknown instance: {instance}. Available: {list(INSTANCE_CONFIGS.keys())}")

        total_runs = len(methods) * len(instances) * len(seeds)
        print(f"\nBenchmark Suite: {total_runs} total runs")
        print(f"  Methods: {methods}")
        print(f"  Instances: {instances}")
        print(f"  Seeds: {seeds}")
        print("=" * 60)

        run_idx = 0
        for instance_name in instances:
            instance = INSTANCE_CONFIGS[instance_name]
            base_config = load_base_config()
            config = create_instance_config(base_config, instance)

            for method in methods:
                for seed in seeds:
                    run_idx += 1
                    print(f"\n[{run_idx}/{total_runs}] {method} | {instance_name} | seed={seed}")

                    try:
                        result = self._methods[method](config, instance_name, seed)
                        self.results.append(result)
                        self._print_result(result)

                        if self.wandb_enabled:
                            self._log_to_wandb(result)

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        # Record failure
                        self.results.append(BenchmarkResult(
                            method=method,
                            instance=instance_name,
                            seed=seed,
                            on_time_rate=-1,
                            average_tardiness=-1,
                            total_tardiness=-1,
                            throughput=-1,
                            breakdown_count=-1,
                            oee=-1,
                            runtime_seconds=-1,
                            episodes_completed=0,
                            additional_metrics={"error": str(e)},
                        ))

        # Save results
        df = self._save_results()
        self._print_summary(df)

        return df

    def _print_result(self, result: BenchmarkResult) -> None:
        """Print a single result."""
        print(f"  On-time: {result.on_time_rate:.2%} | "
              f"Tardiness: {result.average_tardiness:.1f} | "
              f"Throughput: {result.throughput:.3f} | "
              f"Time: {result.runtime_seconds:.1f}s")

    def _log_to_wandb(self, result: BenchmarkResult) -> None:
        """Log result to wandb."""
        try:
            import wandb
            wandb.log({
                "method": result.method,
                "instance": result.instance,
                "seed": result.seed,
                "on_time_rate": result.on_time_rate,
                "average_tardiness": result.average_tardiness,
                "throughput": result.throughput,
                "breakdown_count": result.breakdown_count,
                "oee": result.oee,
                "runtime_seconds": result.runtime_seconds,
            })
        except Exception:
            pass

    def _save_results(self) -> pd.DataFrame:
        """Save results to CSV and JSON."""
        df = pd.DataFrame([asdict(r) for r in self.results])

        # Save CSV
        csv_path = self.output_dir / f"benchmark_{self.run_timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Save JSON for detailed analysis
        json_path = self.output_dir / f"benchmark_{self.run_timestamp}.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Filter out failed runs
        df_valid = df[df["on_time_rate"] >= 0]

        if len(df_valid) == 0:
            print("No valid results to summarize.")
            return

        # Summary by method
        summary = df_valid.groupby("method").agg({
            "on_time_rate": ["mean", "std"],
            "average_tardiness": ["mean", "std"],
            "throughput": ["mean", "std"],
            "runtime_seconds": ["mean"],
        }).round(3)

        print("\nBy Method:")
        print(summary.to_string())

        # Summary by instance
        summary_instance = df_valid.groupby("instance").agg({
            "on_time_rate": ["mean", "std"],
            "average_tardiness": ["mean", "std"],
        }).round(3)

        print("\nBy Instance:")
        print(summary_instance.to_string())

        # Best method per instance
        print("\nBest Method per Instance (by on-time rate):")
        for instance in df_valid["instance"].unique():
            inst_df = df_valid[df_valid["instance"] == instance]
            best = inst_df.groupby("method")["on_time_rate"].mean().idxmax()
            best_rate = inst_df.groupby("method")["on_time_rate"].mean().max()
            print(f"  {instance}: {best} ({best_rate:.2%})")

    # ==================== Method Implementations ====================

    def _run_ppo(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run PPO agent benchmark."""
        from simulation.environment import ShipyardEnv
        from agent.gnn_encoder import HeterogeneousGNNEncoder
        from agent.policy import ActorCriticPolicy
        from agent.ppo import PPOTrainer
        from utils.metrics import compute_kpis

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        start_time = time.time()

        # Create environment
        env = ShipyardEnv(config)

        # Create encoder and policy
        encoder = HeterogeneousGNNEncoder(
            block_dim=env.block_features,
            spmt_dim=env.spmt_features,
            crane_dim=env.crane_features,
            facility_dim=env.facility_features,
            hidden_dim=128,
        )
        policy = ActorCriticPolicy(
            state_dim=128 * 4,
            n_action_types=4,
            n_spmts=env.n_spmts,
            n_cranes=env.n_cranes,
            max_requests=env.n_blocks,
        )

        # Train
        trainer = PPOTrainer(
            policy=policy,
            encoder=encoder,
            device=self.device,
            total_epochs=50,  # Quick training for benchmark
        )

        # Training loop
        for epoch in range(50):
            trainer.collect_rollout(env, n_steps=100)
            trainer.update(trainer._compute_returns_and_advantages())

        # Evaluation
        metrics_list = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                graph_data = env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state = encoder(graph_data)
                    mask = env.get_action_mask()
                    from agent.action_masking import flatten_env_mask_to_policy_mask, to_torch_mask
                    policy_mask = flatten_env_mask_to_policy_mask(
                        mask, policy.n_spmts, policy.n_cranes, policy.max_requests
                    )
                    torch_mask = to_torch_mask(policy_mask, device=self.device)
                    action, _, _ = policy.get_action(state, torch_mask)
                action_dict = {k: int(v.item()) for k, v in action.items()}
                _, _, terminated, truncated, _ = env.step(action_dict)
                done = terminated or truncated
            metrics_list.append(compute_kpis(env.metrics, env.sim_time))

        runtime = time.time() - start_time
        avg_metrics = self._average_metrics(metrics_list)

        return BenchmarkResult(
            method="ppo",
            instance=instance_name,
            seed=seed,
            on_time_rate=avg_metrics.get("on_time_rate", 0),
            average_tardiness=avg_metrics.get("average_tardiness", 0),
            total_tardiness=avg_metrics.get("total_tardiness", 0),
            throughput=avg_metrics.get("throughput", 0),
            breakdown_count=int(avg_metrics.get("breakdown_count", 0)),
            oee=avg_metrics.get("oee", 0),
            runtime_seconds=runtime,
            episodes_completed=self.n_eval_episodes,
        )

    def _run_dqn(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run Double DQN agent benchmark."""
        from simulation.environment import ShipyardEnv
        from agent.gnn_encoder import HeterogeneousGNNEncoder
        from agent.dqn import DoubleDQNAgent
        from utils.metrics import compute_kpis

        np.random.seed(seed)
        torch.manual_seed(seed)

        start_time = time.time()

        env = ShipyardEnv(config)

        encoder = HeterogeneousGNNEncoder(
            block_dim=env.block_features,
            spmt_dim=env.spmt_features,
            crane_dim=env.crane_features,
            facility_dim=env.facility_features,
            hidden_dim=128,
        )

        agent = DoubleDQNAgent(
            encoder=encoder,
            state_dim=128 * 4,
            n_spmts=env.n_spmts,
            n_cranes=env.n_cranes,
            max_requests=env.n_blocks,
            device=self.device,
        )

        # Training
        for episode in range(100):  # Quick training
            obs, _ = env.reset()
            done = False
            while not done:
                graph_data = env.get_graph_data()
                mask = env.get_action_mask()
                action = agent.select_action(graph_data, mask)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_graph = env.get_graph_data()
                from agent.action_masking import flatten_env_mask_to_policy_mask
                flat_mask = flatten_env_mask_to_policy_mask(
                    mask, agent.n_spmts, agent.n_cranes, agent.max_requests
                )
                agent.store_transition(
                    graph_data, action, reward, next_graph, done,
                    np.concatenate([flat_mask[k] for k in ["action_type", "spmt", "crane", "request"]])
                )
                agent.update()

        # Evaluation
        metrics_list = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                graph_data = env.get_graph_data()
                mask = env.get_action_mask()
                action = agent.select_action(graph_data, mask, training=False)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            metrics_list.append(compute_kpis(env.metrics, env.sim_time))

        runtime = time.time() - start_time
        avg_metrics = self._average_metrics(metrics_list)

        return BenchmarkResult(
            method="dqn",
            instance=instance_name,
            seed=seed,
            on_time_rate=avg_metrics.get("on_time_rate", 0),
            average_tardiness=avg_metrics.get("average_tardiness", 0),
            total_tardiness=avg_metrics.get("total_tardiness", 0),
            throughput=avg_metrics.get("throughput", 0),
            breakdown_count=int(avg_metrics.get("breakdown_count", 0)),
            oee=avg_metrics.get("oee", 0),
            runtime_seconds=runtime,
            episodes_completed=self.n_eval_episodes,
        )

    def _run_rule_based_fifo(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run FIFO rule-based scheduler benchmark."""
        return self._run_rule_based(config, instance_name, seed, "fifo")

    def _run_rule_based_edd(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run EDD (Earliest Due Date) rule-based scheduler benchmark."""
        return self._run_rule_based(config, instance_name, seed, "edd")

    def _run_rule_based_slack(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run Slack-based rule scheduler benchmark."""
        return self._run_rule_based(config, instance_name, seed, "slack")

    def _run_rule_based(
        self, config: Dict[str, Any], instance_name: str, seed: int, rule: str
    ) -> BenchmarkResult:
        """Run rule-based scheduler benchmark."""
        from simulation.environment import ShipyardEnv
        from baselines.rule_based import RuleBasedScheduler
        from utils.metrics import compute_kpis

        np.random.seed(seed)

        start_time = time.time()

        env = ShipyardEnv(config)
        scheduler = RuleBasedScheduler(rule=rule)

        metrics_list = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.get_action_mask()
                action = scheduler.select_action(env, mask)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            metrics_list.append(compute_kpis(env.metrics, env.sim_time))

        runtime = time.time() - start_time
        avg_metrics = self._average_metrics(metrics_list)

        return BenchmarkResult(
            method=f"rule_{rule}",
            instance=instance_name,
            seed=seed,
            on_time_rate=avg_metrics.get("on_time_rate", 0),
            average_tardiness=avg_metrics.get("average_tardiness", 0),
            total_tardiness=avg_metrics.get("total_tardiness", 0),
            throughput=avg_metrics.get("throughput", 0),
            breakdown_count=int(avg_metrics.get("breakdown_count", 0)),
            oee=avg_metrics.get("oee", 0),
            runtime_seconds=runtime,
            episodes_completed=self.n_eval_episodes,
        )

    def _run_random(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run random action baseline benchmark."""
        from simulation.environment import ShipyardEnv
        from utils.metrics import compute_kpis

        np.random.seed(seed)

        start_time = time.time()

        env = ShipyardEnv(config)

        metrics_list = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            metrics_list.append(compute_kpis(env.metrics, env.sim_time))

        runtime = time.time() - start_time
        avg_metrics = self._average_metrics(metrics_list)

        return BenchmarkResult(
            method="random",
            instance=instance_name,
            seed=seed,
            on_time_rate=avg_metrics.get("on_time_rate", 0),
            average_tardiness=avg_metrics.get("average_tardiness", 0),
            total_tardiness=avg_metrics.get("total_tardiness", 0),
            throughput=avg_metrics.get("throughput", 0),
            breakdown_count=int(avg_metrics.get("breakdown_count", 0)),
            oee=avg_metrics.get("oee", 0),
            runtime_seconds=runtime,
            episodes_completed=self.n_eval_episodes,
        )

    def _run_mip(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run MIP solver benchmark (requires Gurobi)."""
        try:
            from baselines.mip_scheduler import MIPScheduler
        except ImportError:
            raise ImportError("MIP scheduler requires baselines.mip_scheduler module")

        from simulation.environment import ShipyardEnv
        from utils.metrics import compute_kpis

        np.random.seed(seed)

        start_time = time.time()

        env = ShipyardEnv(config)
        scheduler = MIPScheduler(time_limit=60)  # 60 second time limit

        metrics_list = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            # MIP computes full schedule upfront
            schedule = scheduler.solve(env)
            # Execute schedule
            for action in schedule:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            metrics_list.append(compute_kpis(env.metrics, env.sim_time))

        runtime = time.time() - start_time
        avg_metrics = self._average_metrics(metrics_list)

        return BenchmarkResult(
            method="mip",
            instance=instance_name,
            seed=seed,
            on_time_rate=avg_metrics.get("on_time_rate", 0),
            average_tardiness=avg_metrics.get("average_tardiness", 0),
            total_tardiness=avg_metrics.get("total_tardiness", 0),
            throughput=avg_metrics.get("throughput", 0),
            breakdown_count=int(avg_metrics.get("breakdown_count", 0)),
            oee=avg_metrics.get("oee", 0),
            runtime_seconds=runtime,
            episodes_completed=self.n_eval_episodes,
        )

    def _run_cp(
        self, config: Dict[str, Any], instance_name: str, seed: int
    ) -> BenchmarkResult:
        """Run CP-SAT solver benchmark (requires OR-Tools)."""
        try:
            from baselines.cp_scheduler import CPScheduler
        except ImportError:
            raise ImportError("CP scheduler requires baselines.cp_scheduler module")

        from simulation.environment import ShipyardEnv
        from utils.metrics import compute_kpis

        np.random.seed(seed)

        start_time = time.time()

        env = ShipyardEnv(config)
        scheduler = CPScheduler(time_limit=60)

        metrics_list = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            schedule = scheduler.solve(env)
            for action in schedule:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            metrics_list.append(compute_kpis(env.metrics, env.sim_time))

        runtime = time.time() - start_time
        avg_metrics = self._average_metrics(metrics_list)

        return BenchmarkResult(
            method="cp",
            instance=instance_name,
            seed=seed,
            on_time_rate=avg_metrics.get("on_time_rate", 0),
            average_tardiness=avg_metrics.get("average_tardiness", 0),
            total_tardiness=avg_metrics.get("total_tardiness", 0),
            throughput=avg_metrics.get("throughput", 0),
            breakdown_count=int(avg_metrics.get("breakdown_count", 0)),
            oee=avg_metrics.get("oee", 0),
            runtime_seconds=runtime,
            episodes_completed=self.n_eval_episodes,
        )

    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across episodes."""
        if not metrics_list:
            return {}

        keys = metrics_list[0].keys()
        return {
            k: np.mean([m.get(k, 0) for m in metrics_list])
            for k in keys
        }


def main():
    parser = argparse.ArgumentParser(description="Shipyard Scheduling Benchmark Suite")

    # Run configuration
    parser.add_argument("--all", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to benchmark (default: all available)"
    )
    parser.add_argument(
        "--instances", nargs="+", default=None,
        help="Instance sizes to test (default: all)"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes per run")

    # Output
    parser.add_argument("--output-dir", type=str, default="./benchmark_results")
    parser.add_argument("--device", type=str, default="cpu")

    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="shipyard-benchmark")

    args = parser.parse_args()

    # Configure run
    if args.quick:
        methods = ["random", "rule_fifo"]
        instances = ["tiny"]
        n_seeds = 1
        n_episodes = 2
    elif args.all:
        methods = None  # All available
        instances = None  # All instances
        n_seeds = args.seeds
        n_episodes = args.episodes
    else:
        methods = args.methods
        instances = args.instances
        n_seeds = args.seeds
        n_episodes = args.episodes

    # Run benchmark
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        n_seeds=n_seeds,
        n_eval_episodes=n_episodes,
        device=args.device,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
    )

    print(f"Available methods: {runner.available_methods}")

    df = runner.run_benchmark(methods=methods, instances=instances)

    print("\nBenchmark complete!")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
