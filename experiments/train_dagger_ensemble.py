#!/usr/bin/env python3
"""DAgger Ensemble Training for Shipyard Scheduling.

Trains multiple DAgger policies with different seeds and creates an ensemble
for more robust scheduling decisions. Includes feature normalization for
improved learning stability.

Key improvements over basic DAgger:
1. Feature normalization with running statistics
2. Ensemble of N policies for variance reduction
3. Scaled-up training (more iterations, demos, epochs)
4. Majority voting or probability averaging for action selection

Reference: Ross et al., "A Reduction of Imitation Learning and Structured
           Prediction to No-Regret Online Learning" (2011)

Usage:
    python experiments/train_dagger_ensemble.py --config config/small_instance.yaml \
        --n-ensemble 3 --iterations 20 --init-episodes 50
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
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from baselines.rule_based import RuleBasedScheduler


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with inheritance support."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


class RunningNormalizer:
    """Running mean/std normalizer for feature standardization.

    Computes online statistics and normalizes features to zero mean, unit variance.
    Essential for stable imitation learning when features have different scales.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = dim
        self.eps = eps
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 0

    def update(self, x: np.ndarray):
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Welford's online algorithm for numerical stability
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count

        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using current statistics."""
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(np.sqrt(self.var + self.eps), dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, Any]):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


@dataclass
class DAggerConfig:
    """Configuration for DAgger training."""
    iterations: int = 20
    init_episodes: int = 50
    dagger_episodes: int = 20
    train_epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-5
    beta_start: float = 1.0
    beta_end: float = 0.1
    max_steps: int = 1000
    hidden_dim: int = 128
    policy_hidden: int = 256


class DAggerTrainer:
    """DAgger trainer with feature normalization."""

    def __init__(
        self,
        env: HHIShipyardEnv,
        encoder: HeterogeneousGNNEncoder,
        policy: ActorCriticPolicy,
        expert: RuleBasedScheduler,
        config: DAggerConfig,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.env = env
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        self.expert = expert
        self.config = config
        self.device = device
        self.seed = seed

        # Set seed for this trainer
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(policy.parameters()),
            lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.iterations * config.train_epochs
        )

        # Aggregated dataset
        self.states: List[torch.Tensor] = []
        self.expert_actions: List[Dict[str, int]] = []

        # Feature normalizer (initialized after first batch)
        self.normalizer: Optional[RunningNormalizer] = None

        # Action key mapping
        self.action_keys = {
            "action_type": "action_type",
            "spmt": "spmt_idx",
            "request": "request_idx",
            "crane": "crane_idx",
            "lift": "lift_idx",
            "equipment": "equipment_idx",
        }

    def _encode_state(self) -> torch.Tensor:
        """Encode current environment state using GNN."""
        graph_data = self.env.get_graph_data().to(self.device)
        with torch.no_grad():
            state_emb = self.encoder(graph_data)
        return state_emb

    def collect_expert_demos(self, n_episodes: int):
        """Collect initial expert demonstrations."""
        print(f"  Collecting {n_episodes} expert demos (seed={self.seed})...")
        raw_states = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()

            for step in range(self.config.max_steps):
                state_emb = self._encode_state()
                expert_action = self.expert.decide(self.env)

                raw_states.append(state_emb.cpu().numpy().flatten())
                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)

                obs, reward, terminated, truncated, info = self.env.step(expert_action)

                if terminated or truncated:
                    break

        # Initialize normalizer with collected data
        raw_states_np = np.array(raw_states)
        state_dim = raw_states_np.shape[1]
        self.normalizer = RunningNormalizer(state_dim)
        self.normalizer.update(raw_states_np)

        print(f"    Collected {len(self.states)} state-action pairs")

    def collect_dagger_data(self, n_episodes: int, beta: float):
        """Collect DAgger data: roll out policy but query expert for labels."""
        new_samples = 0
        raw_states = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()

            for step in range(self.config.max_steps):
                state_emb = self._encode_state()
                expert_action = self.expert.decide(self.env)

                # Store for normalizer update
                raw_states.append(state_emb.cpu().numpy().flatten())
                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)
                new_samples += 1

                # Choose action: expert with prob beta, policy otherwise
                if random.random() < beta:
                    action = expert_action
                else:
                    # Normalize state before policy forward pass
                    if self.normalizer:
                        state_norm = self.normalizer.normalize(state_emb)
                    else:
                        state_norm = state_emb
                    with torch.no_grad():
                        policy_action, _, _ = self.policy.get_action(state_norm)
                    action = {k: int(v.item()) for k, v in policy_action.items()}

                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    break

        # Update normalizer with new data
        if raw_states and self.normalizer:
            self.normalizer.update(np.array(raw_states))

        return new_samples

    def train_epoch(self) -> float:
        """Train policy on aggregated dataset for one epoch."""
        n_samples = len(self.states)
        indices = list(range(n_samples))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, self.config.batch_size):
            batch_indices = indices[i:i+self.config.batch_size]
            if len(batch_indices) < 2:
                continue

            # Batch and normalize states
            batch_states = torch.cat(
                [self.states[j] for j in batch_indices], dim=0
            ).to(self.device)

            if self.normalizer:
                batch_states = self.normalizer.normalize(batch_states)

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
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy."""
        total_throughput = 0.0
        total_reward = 0.0

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            ep_reward = 0.0

            for step in range(self.config.max_steps):
                state_emb = self._encode_state()
                if self.normalizer:
                    state_emb = self.normalizer.normalize(state_emb)

                with torch.no_grad():
                    action, _, _ = self.policy.get_action(state_emb, deterministic=True)

                action_cpu = {k: int(v.item()) for k, v in action.items()}
                obs, reward, terminated, truncated, info = self.env.step(action_cpu)
                ep_reward += reward

                if terminated or truncated:
                    break

            total_reward += ep_reward
            if self.env.sim_time > 0:
                total_throughput += self.env.metrics["blocks_completed"] / self.env.sim_time

        return {
            "avg_reward": total_reward / n_episodes,
            "avg_throughput": total_throughput / n_episodes,
        }

    def train(self) -> Dict[str, Any]:
        """Run full DAgger training."""
        # Phase 1: Initial expert demonstrations
        self.collect_expert_demos(self.config.init_episodes)

        # Initial BC training
        print(f"  Initial BC training...")
        for epoch in range(self.config.train_epochs):
            loss = self.train_epoch()

        metrics = self.evaluate()
        print(f"    Initial throughput: {metrics['avg_throughput']:.4f}")

        best_throughput = metrics["avg_throughput"]
        best_state = {
            "encoder": {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()},
            "policy": {k: v.cpu().clone() for k, v in self.policy.state_dict().items()},
            "normalizer": self.normalizer.state_dict() if self.normalizer else None,
        }

        # Phase 2: DAgger iterations
        for iteration in range(self.config.iterations):
            beta = self.config.beta_start - (
                self.config.beta_start - self.config.beta_end
            ) * iteration / max(self.config.iterations - 1, 1)

            new_samples = self.collect_dagger_data(self.config.dagger_episodes, beta)

            # Train on aggregated dataset
            for epoch in range(self.config.train_epochs):
                loss = self.train_epoch()

            # Evaluate
            metrics = self.evaluate()

            if metrics["avg_throughput"] > best_throughput:
                best_throughput = metrics["avg_throughput"]
                best_state = {
                    "encoder": {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()},
                    "policy": {k: v.cpu().clone() for k, v in self.policy.state_dict().items()},
                    "normalizer": self.normalizer.state_dict() if self.normalizer else None,
                }

            if (iteration + 1) % 5 == 0:
                print(f"    Iter {iteration+1}/{self.config.iterations}: "
                      f"throughput={metrics['avg_throughput']:.4f}, "
                      f"samples={len(self.states)}, beta={beta:.2f}")

        # Restore best
        self.encoder.load_state_dict(best_state["encoder"])
        self.policy.load_state_dict(best_state["policy"])
        if best_state["normalizer"]:
            self.normalizer.load_state_dict(best_state["normalizer"])

        return {"best_throughput": best_throughput, "total_samples": len(self.states)}


class DAggerEnsemble:
    """Ensemble of DAgger-trained policies for robust scheduling."""

    def __init__(
        self,
        encoders: List[HeterogeneousGNNEncoder],
        policies: List[ActorCriticPolicy],
        normalizers: List[Optional[RunningNormalizer]],
        device: str = "cpu",
        voting: str = "majority",  # "majority" or "average"
    ):
        self.encoders = encoders
        self.policies = policies
        self.normalizers = normalizers
        self.device = device
        self.voting = voting
        self.n_models = len(policies)

    def get_action(
        self, env: HHIShipyardEnv, deterministic: bool = True
    ) -> Dict[str, int]:
        """Get ensemble action using voting or averaging."""
        graph_data = env.get_graph_data().to(self.device)

        all_actions = []
        all_probs = []

        for encoder, policy, normalizer in zip(
            self.encoders, self.policies, self.normalizers
        ):
            with torch.no_grad():
                state_emb = encoder(graph_data)
                if normalizer:
                    state_emb = normalizer.normalize(state_emb)

                action, _, _ = policy.get_action(state_emb, deterministic=deterministic)
                action_cpu = {k: int(v.item()) for k, v in action.items()}
                all_actions.append(action_cpu)

                # Get probabilities for averaging
                action_dist, _ = policy.forward(state_emb)
                probs = {k: d.probs.cpu().numpy() for k, d in action_dist.items()}
                all_probs.append(probs)

        if self.voting == "majority":
            # Majority voting for each action head
            final_action = {}
            for key in all_actions[0].keys():
                votes = [a[key] for a in all_actions]
                counter = Counter(votes)
                final_action[key] = counter.most_common(1)[0][0]
            return final_action
        else:
            # Probability averaging
            final_action = {}
            for key in all_probs[0].keys():
                avg_prob = np.mean([p[key] for p in all_probs], axis=0)
                final_action[key] = int(np.argmax(avg_prob))
            return final_action

    def evaluate(
        self, env: HHIShipyardEnv, n_episodes: int = 10, max_steps: int = 1000
    ) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        total_throughput = 0.0
        total_reward = 0.0

        for ep in range(n_episodes):
            obs, info = env.reset()
            ep_reward = 0.0

            for step in range(max_steps):
                action = self.get_action(env, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                if terminated or truncated:
                    break

            total_reward += ep_reward
            if env.sim_time > 0:
                total_throughput += env.metrics["blocks_completed"] / env.sim_time

        return {
            "avg_reward": total_reward / n_episodes,
            "avg_throughput": total_throughput / n_episodes,
        }


def train_ensemble(
    env: HHIShipyardEnv,
    config: DAggerConfig,
    n_ensemble: int = 3,
    base_seed: int = 42,
    device: str = "cpu",
) -> Tuple[DAggerEnsemble, List[Dict[str, Any]]]:
    """Train an ensemble of DAgger policies."""
    encoders = []
    policies = []
    normalizers = []
    results = []

    expert = RuleBasedScheduler()

    for i in range(n_ensemble):
        seed = base_seed + i * 1000
        print(f"\n{'='*50}")
        print(f"Training ensemble member {i+1}/{n_ensemble} (seed={seed})")
        print(f"{'='*50}")

        # Create fresh networks for each member
        encoder = HeterogeneousGNNEncoder(
            block_dim=env.block_features,
            spmt_dim=env.spmt_features,
            crane_dim=env.crane_features,
            facility_dim=env.facility_features,
            hidden_dim=config.hidden_dim,
            num_layers=2,
        )

        state_dim = config.hidden_dim * 4
        policy = ActorCriticPolicy(
            state_dim=state_dim,
            n_action_types=4,
            n_spmts=env.n_spmts,
            n_cranes=getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2)),
            max_requests=env.n_blocks,
            hidden_dim=config.policy_hidden,
            epsilon=0.0,
        )

        trainer = DAggerTrainer(
            env=env,
            encoder=encoder,
            policy=policy,
            expert=expert,
            config=config,
            device=device,
            seed=seed,
        )

        result = trainer.train()
        results.append(result)

        encoders.append(trainer.encoder)
        policies.append(trainer.policy)
        normalizers.append(trainer.normalizer)

        print(f"  Member {i+1} best throughput: {result['best_throughput']:.4f}")

    ensemble = DAggerEnsemble(
        encoders=encoders,
        policies=policies,
        normalizers=normalizers,
        device=device,
        voting="majority",
    )

    return ensemble, results


def main():
    parser = argparse.ArgumentParser(description="DAgger Ensemble Training")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--n-ensemble", type=int, default=3, help="Number of ensemble members")
    parser.add_argument("--iterations", type=int, default=20, help="DAgger iterations per member")
    parser.add_argument("--init-episodes", type=int, default=50, help="Initial expert demos")
    parser.add_argument("--dagger-episodes", type=int, default=20, help="Episodes per iteration")
    parser.add_argument("--train-epochs", type=int, default=30, help="Epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="data/checkpoints/dagger_ensemble/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config and create environment
    cfg = load_config(args.config)
    env = HHIShipyardEnv(cfg)

    # Create DAgger config
    dagger_config = DAggerConfig(
        iterations=args.iterations,
        init_episodes=args.init_episodes,
        dagger_episodes=args.dagger_episodes,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        policy_hidden=args.policy_hidden,
    )

    print("=" * 60)
    print("DAgger Ensemble Training for Shipyard Scheduling")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Ensemble size: {args.n_ensemble}")
    print(f"Iterations per member: {args.iterations}")
    print(f"Initial demos: {args.init_episodes}")
    print(f"DAgger episodes per iteration: {args.dagger_episodes}")
    print(f"Training epochs per iteration: {args.train_epochs}")

    # Train ensemble
    ensemble, member_results = train_ensemble(
        env=env,
        config=dagger_config,
        n_ensemble=args.n_ensemble,
        base_seed=args.seed,
        device=args.device,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Evaluate ensemble
    ensemble_metrics = ensemble.evaluate(env, n_episodes=10)
    print(f"Ensemble Throughput: {ensemble_metrics['avg_throughput']:.4f}")

    # Compare with individual members
    individual_throughputs = [r["best_throughput"] for r in member_results]
    print(f"Individual member throughputs: {[f'{t:.4f}' for t in individual_throughputs]}")
    print(f"Individual mean: {np.mean(individual_throughputs):.4f}")
    print(f"Individual std: {np.std(individual_throughputs):.4f}")

    # Compare with expert
    expert = RuleBasedScheduler()
    expert_throughput = 0.0
    for _ in range(10):
        obs, _ = env.reset()
        for _ in range(1000):
            action = expert.decide(env)
            obs, reward, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        if env.sim_time > 0:
            expert_throughput += env.metrics["blocks_completed"] / env.sim_time
    expert_throughput /= 10

    print(f"\nExpert Throughput: {expert_throughput:.4f}")
    print(f"Ensemble vs Expert: {100 * ensemble_metrics['avg_throughput'] / expert_throughput:.1f}%")

    # Save checkpoint
    os.makedirs(args.save, exist_ok=True)
    checkpoint = {
        "n_ensemble": args.n_ensemble,
        "config": vars(dagger_config),
        "args": vars(args),
        "ensemble_metrics": ensemble_metrics,
        "member_results": member_results,
        "expert_throughput": expert_throughput,
    }

    # Save ensemble state dicts
    for i, (encoder, policy, normalizer) in enumerate(
        zip(ensemble.encoders, ensemble.policies, ensemble.normalizers)
    ):
        checkpoint[f"encoder_{i}"] = encoder.state_dict()
        checkpoint[f"policy_{i}"] = policy.state_dict()
        checkpoint[f"normalizer_{i}"] = normalizer.state_dict() if normalizer else None

    torch.save(checkpoint, os.path.join(args.save, "ensemble_final.pt"))
    print(f"\nCheckpoint saved to {args.save}")


if __name__ == "__main__":
    main()
