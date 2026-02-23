#!/usr/bin/env python3
"""DAgger (Dataset Aggregation) for Shipyard Scheduling.

Iterative imitation learning that addresses distribution mismatch:
1. Train initial policy via BC on expert demos
2. Roll out current policy, but collect expert labels
3. Aggregate new data with old dataset
4. Retrain policy on aggregated data
5. Repeat

This helps the policy learn to recover from its own mistakes.

Reference: Ross et al., "A Reduction of Imitation Learning and Structured
           Prediction to No-Regret Online Learning" (2011)

Usage:
    python experiments/train_dagger.py --config config/small_instance.yaml --iterations 10
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from baselines.rule_based import RuleBasedScheduler

# Database logging for dashboard visualization
try:
    from mes.database import (
        init_db, clear_db, log_position_snapshot, log_metrics,
        log_ships, log_goliath_cranes, log_hhi_blocks, log_spmts,
        log_health_snapshot, log_queue_depths,
    )
    DB_LOGGING_AVAILABLE = True
except ImportError:
    DB_LOGGING_AVAILABLE = False


def create_env(config: Dict[str, Any]):
    """Create the appropriate environment based on config."""
    return HHIShipyardEnv(config)


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


class DAggerTrainer:
    """DAgger trainer for imitation learning."""

    def __init__(
        self,
        env: HHIShipyardEnv,
        encoder: HeterogeneousGNNEncoder,
        policy: ActorCriticPolicy,
        expert: RuleBasedScheduler,
        device: str = "cpu",
        lr: float = 3e-4,
        db_logging: bool = False,
        db_log_interval: int = 10,
    ):
        self.env = env
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        self.expert = expert
        self.device = device
        self.db_logging = db_logging
        self.db_log_interval = db_log_interval
        self.cumulative_time = 0.0  # Track cumulative sim time for continuous timeline

        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(policy.parameters()),
            lr=lr, weight_decay=1e-5
        )

        # Aggregated dataset
        self.states: List[torch.Tensor] = []
        self.expert_actions: List[Dict[str, int]] = []

        # Action key mapping
        self.action_keys = {
            "action_type": "action_type",
            "spmt": "spmt_idx",
            "request": "request_idx",
            "crane": "crane_idx",
            "lift": "lift_idx",
            "equipment": "equipment_idx",
            "supplier": "supplier_idx",
            "material": "material_idx",
            "labor_pool": "labor_pool_idx",
            "target_block": "target_block_idx",
        }

    def _log_to_db(self, step: int):
        """Log current state to database for dashboard visualization."""
        if not self.db_logging or step % self.db_log_interval != 0:
            return

        self.cumulative_time += self.env.sim_time / max(step, 1)

        blocks = self.env.entities.get("blocks", [])
        spmts = self.env.entities.get("spmts", [])
        cranes = self.env.entities.get("goliath_cranes", self.env.entities.get("cranes", []))
        ships = self.env.entities.get("ships", [])

        # Log position snapshot for playback
        log_position_snapshot(
            time=self.cumulative_time,
            blocks=blocks,
            spmts=spmts,
            cranes=cranes,
            ships=ships,
        )

        # Log current state tables
        if ships:
            log_ships(ships)
        if cranes:
            log_goliath_cranes(cranes)
        if blocks:
            log_hhi_blocks(blocks)
        if spmts:
            log_spmts(spmts)
        log_metrics(self.cumulative_time, self.env.metrics)

        # Log health history
        log_health_snapshot(self.cumulative_time, spmts, cranes)

        # Log queue depths
        log_queue_depths(
            self.cumulative_time,
            getattr(self.env, 'facility_queues', {}),
            getattr(self.env, 'facility_processing', {}),
        )

    def collect_expert_demos(self, n_episodes: int, max_steps: int = 1000):
        """Collect initial expert demonstrations."""
        print(f"Collecting {n_episodes} expert demonstration episodes...")

        for ep in range(n_episodes):
            obs, info = self.env.reset()

            for step in range(max_steps):
                # Encode state
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)

                # Get expert action
                expert_action = self.expert.decide(self.env)

                # Store
                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)

                # Step with expert action
                obs, reward, terminated, truncated, info = self.env.step(expert_action)

                # Log to database for dashboard visualization
                self._log_to_db(step)

                if terminated or truncated:
                    break

        print(f"  Collected {len(self.states)} state-action pairs")

    def collect_dagger_data(self, n_episodes: int, beta: float = 0.5, max_steps: int = 1000):
        """
        Collect data using DAgger: roll out policy but query expert for labels.

        Args:
            n_episodes: Number of episodes to collect
            beta: Probability of using expert action (annealed over iterations)
            max_steps: Maximum steps per episode
        """
        print(f"Collecting DAgger data (beta={beta:.2f})...")
        new_samples = 0

        for ep in range(n_episodes):
            obs, info = self.env.reset()

            for step in range(max_steps):
                # Encode state
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)

                # Get expert action (this is the label we're collecting)
                expert_action = self.expert.decide(self.env)

                # Store state and expert action
                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)
                new_samples += 1

                # Decide whether to use expert or policy action for execution
                if random.random() < beta:
                    # Use expert action
                    action = expert_action
                else:
                    # Use policy action
                    with torch.no_grad():
                        policy_action, _, _ = self.policy.get_action(state_emb)
                    action = {k: int(v.item()) for k, v in policy_action.items()}

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Log to database for dashboard visualization
                self._log_to_db(step)

                if terminated or truncated:
                    break

        print(f"  Added {new_samples} new samples, total: {len(self.states)}")

    def train_epoch(self, batch_size: int = 128) -> float:
        """Train policy on aggregated dataset for one epoch."""
        n_samples = len(self.states)
        indices = list(range(n_samples))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < 2:
                continue

            # Batch states
            batch_states = torch.cat(
                [self.states[j] for j in batch_indices], dim=0
            ).to(self.device)

            # Get policy distributions
            action_dist, _ = self.policy.forward(batch_states)

            # Compute cross-entropy loss for each head
            loss = 0.0
            for head_name, action_key in self.action_keys.items():
                target = torch.tensor(
                    [self.expert_actions[j].get(action_key, 0) for j in batch_indices],
                    device=self.device
                )
                max_idx = action_dist[head_name].probs.shape[-1] - 1
                target = target.clamp(0, max_idx)

                head_loss = F.cross_entropy(action_dist[head_name].logits, target)

                # Weight action_type more heavily
                if head_name == "action_type":
                    loss += 2.0 * head_loss
                else:
                    loss += head_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.encoder.parameters()), 1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, n_episodes: int = 5, max_steps: int = 1000) -> Dict[str, float]:
        """Evaluate current policy."""
        total_throughput = 0.0
        total_reward = 0.0

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            ep_reward = 0.0

            for step in range(max_steps):
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)
                    action, _, _ = self.policy.get_action(state_emb, deterministic=True)

                action_cpu = {k: int(v.item()) for k, v in action.items()}
                obs, reward, terminated, truncated, info = self.env.step(action_cpu)
                ep_reward += reward

                # Log to database for dashboard visualization
                self._log_to_db(step)

                if terminated or truncated:
                    break

            total_reward += ep_reward
            if self.env.sim_time > 0:
                total_throughput += self.env.metrics.get("blocks_erected", self.env.metrics.get("blocks_completed", 0)) / self.env.sim_time

        return {
            "avg_reward": total_reward / n_episodes,
            "avg_throughput": total_throughput / n_episodes,
        }


def main():
    parser = argparse.ArgumentParser(description="DAgger for Shipyard Scheduling")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--iterations", type=int, default=10, help="DAgger iterations")
    parser.add_argument("--init-episodes", type=int, default=20, help="Initial expert demos")
    parser.add_argument("--dagger-episodes", type=int, default=10, help="Episodes per iteration")
    parser.add_argument("--train-epochs", type=int, default=20, help="Training epochs per iteration")
    parser.add_argument("--beta-start", type=float, default=1.0, help="Initial expert probability")
    parser.add_argument("--beta-end", type=float, default=0.1, help="Final expert probability")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="data/checkpoints/dagger/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-log", action="store_true", help="Enable database logging for dashboard")
    parser.add_argument("--db-log-interval", type=int, default=10, help="Log every N steps")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config and create environment
    cfg = load_config(args.config)
    env = HHIShipyardEnv(cfg)

    # Initialize database logging if requested
    db_logging = args.db_log and DB_LOGGING_AVAILABLE
    if args.db_log and not DB_LOGGING_AVAILABLE:
        print("Warning: --db-log requested but mes.database module not available")
    if db_logging:
        init_db()
        clear_db()
        print("Database logging enabled - view progress at http://localhost:8050")

    # Create networks
    hidden_dim = 128

    # Supply chain feature dims (0 if disabled)
    supplier_dim = env.supplier_features if env.n_suppliers > 0 else 0
    inventory_dim = env.inventory_features if env.n_inventory_nodes > 0 else 0
    labor_dim = env.labor_features if env.n_labor_pools > 0 else 0

    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=2,
        supplier_dim=supplier_dim,
        inventory_dim=inventory_dim,
        labor_dim=labor_dim,
    )

    # Output dim: hidden_dim * number of active node types
    n_active_types = 4  # block, spmt, crane, facility
    if supplier_dim > 0:
        n_active_types += 1
    if inventory_dim > 0:
        n_active_types += 1
    if labor_dim > 0:
        n_active_types += 1
    state_dim = hidden_dim * n_active_types

    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=env.n_action_types,
        n_spmts=env.n_spmts,
        n_cranes=getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2)),
        max_requests=env.n_blocks,
        hidden_dim=256,
        epsilon=0.0,  # No exploration during evaluation
        n_suppliers=env.n_suppliers,
        n_inventory=env.n_inventory_nodes,
        n_labor_pools=env.n_labor_pools,
    )

    expert = RuleBasedScheduler()

    trainer = DAggerTrainer(
        env=env,
        encoder=encoder,
        policy=policy,
        expert=expert,
        device=args.device,
        lr=args.lr,
        db_logging=db_logging,
        db_log_interval=args.db_log_interval,
    )

    print("=" * 60)
    print("DAgger Training for Shipyard Scheduling")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Initial demos: {args.init_episodes}")
    print(f"DAgger episodes per iteration: {args.dagger_episodes}")
    print(f"Training epochs per iteration: {args.train_epochs}")
    print(f"Beta schedule: {args.beta_start} -> {args.beta_end}")

    # Phase 1: Collect initial expert demonstrations
    print("\n" + "-" * 40)
    print("Phase 1: Initial Expert Demonstrations")
    print("-" * 40)
    trainer.collect_expert_demos(args.init_episodes)

    # Initial training
    print("\nInitial BC training...")
    for epoch in range(args.train_epochs):
        loss = trainer.train_epoch()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.train_epochs} - Loss: {loss:.4f}")

    # Evaluate initial policy
    metrics = trainer.evaluate()
    print(f"\nInitial Policy: Throughput={metrics['avg_throughput']:.4f}")

    # Phase 2: DAgger iterations
    print("\n" + "-" * 40)
    print("Phase 2: DAgger Iterations")
    print("-" * 40)

    best_throughput = metrics["avg_throughput"]
    best_state = {
        "encoder": encoder.state_dict(),
        "policy": policy.state_dict(),
    }

    for iteration in range(args.iterations):
        # Compute beta (annealing from beta_start to beta_end)
        beta = args.beta_start - (args.beta_start - args.beta_end) * iteration / max(args.iterations - 1, 1)

        print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")

        # Collect DAgger data
        trainer.collect_dagger_data(args.dagger_episodes, beta=beta)

        # Train on aggregated dataset
        print(f"Training for {args.train_epochs} epochs...")
        for epoch in range(args.train_epochs):
            loss = trainer.train_epoch()

        print(f"  Final loss: {loss:.4f}")

        # Evaluate
        metrics = trainer.evaluate()
        print(f"  Throughput: {metrics['avg_throughput']:.4f}")

        # Track best
        if metrics["avg_throughput"] > best_throughput:
            best_throughput = metrics["avg_throughput"]
            best_state = {
                "encoder": encoder.state_dict(),
                "policy": policy.state_dict(),
            }
            print(f"  *** New best! ***")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Restore best model
    encoder.load_state_dict(best_state["encoder"])
    policy.load_state_dict(best_state["policy"])

    metrics = trainer.evaluate(n_episodes=10)
    print(f"Best Policy Throughput: {metrics['avg_throughput']:.4f}")

    # Compare with expert
    expert_throughput = 0.0
    for _ in range(10):
        obs, _ = env.reset()
        for _ in range(1000):
            action = expert.decide(env)
            obs, reward, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        if env.sim_time > 0:
            expert_throughput += env.metrics.get("blocks_erected", env.metrics.get("blocks_completed", 0)) / env.sim_time
    expert_throughput /= 10

    print(f"Expert Throughput: {expert_throughput:.4f}")
    if expert_throughput > 0:
        print(f"DAgger vs Expert: {100 * metrics['avg_throughput'] / expert_throughput:.1f}%")
    else:
        print("Expert throughput is 0 - blocks may not be completing. Check environment config.")

    # Save
    os.makedirs(args.save, exist_ok=True)
    torch.save({
        "encoder": best_state["encoder"],
        "policy": best_state["policy"],
        "metrics": metrics,
        "args": vars(args),
    }, os.path.join(args.save, "dagger_final.pt"))
    print(f"\nCheckpoint saved to {args.save}")


if __name__ == "__main__":
    main()
