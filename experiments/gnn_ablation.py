#!/usr/bin/env python3
"""GNN Architecture Ablation Study.

Compares different GNN configurations for DAgger training:
1. 2-layer GNN, 128 hidden dim (baseline)
2. 4-layer GNN, 128 hidden dim
3. 2-layer GNN, 256 hidden dim
4. 4-layer GNN, 256 hidden dim (full)

For each configuration:
- Train DAgger for N iterations on specified config
- Evaluate throughput and training time
- Report parameter counts

Usage:
    python experiments/gnn_ablation.py --config config/small_instance.yaml --iterations 10
"""

import argparse
import os
import sys
import random
import time
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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_mask_to_tensor(mask: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    """Convert numpy mask dict to tensor dict."""
    if mask is None:
        return None
    result = {}
    for k, v in mask.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).to(device)
        else:
            result[k] = v
    return result


class GNNAblationTrainer:
    """Simplified DAgger trainer for ablation study."""

    def __init__(
        self,
        env: HHIShipyardEnv,
        encoder: HeterogeneousGNNEncoder,
        policy: ActorCriticPolicy,
        expert: RuleBasedScheduler,
        device: str = "cpu",
        lr: float = 3e-4,
    ):
        self.env = env
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        self.expert = expert
        self.device = device

        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(policy.parameters()),
            lr=lr, weight_decay=1e-5
        )

        # Aggregated dataset
        self.states: List[torch.Tensor] = []
        self.expert_actions: List[Dict[str, int]] = []

    def collect_expert_demos(self, n_episodes: int, max_steps: int = 500):
        """Collect initial expert demonstrations."""
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            for step in range(max_steps):
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)
                expert_action = self.expert.decide(self.env)
                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)
                obs, reward, terminated, truncated, info = self.env.step(expert_action)
                if terminated or truncated:
                    break

    def collect_dagger_data(self, n_episodes: int, beta: float = 0.5, max_steps: int = 500):
        """Collect data using DAgger."""
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            for step in range(max_steps):
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)
                expert_action = self.expert.decide(self.env)
                self.states.append(state_emb.cpu())
                self.expert_actions.append(expert_action)
                
                # Choose action based on beta
                if random.random() < beta:
                    action = expert_action
                else:
                    with torch.no_grad():
                        mask = convert_mask_to_tensor(self.env.get_action_mask(), self.device)
                        action_dist, _ = self.policy(state_emb, mask)
                        action = {
                            "action_type": action_dist["action_type"].sample().item(),
                            "spmt_idx": action_dist["spmt"].sample().item(),
                            "request_idx": action_dist["request"].sample().item(),
                            "crane_idx": action_dist["crane"].sample().item(),
                        }
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break

    def train_epoch(self) -> float:
        """Train one epoch on aggregated dataset."""
        if not self.states:
            return 0.0

        self.encoder.train()
        self.policy.train()

        indices = list(range(len(self.states)))
        random.shuffle(indices)

        total_loss = 0.0
        batch_size = 64
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            states = torch.stack([self.states[j] for j in batch_indices]).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute loss for each action type
            loss = torch.tensor(0.0, device=self.device)
            for j, idx in enumerate(batch_indices):
                expert_action = self.expert_actions[idx]
                action_dist, _ = self.policy(states[j].unsqueeze(0))
                
                # Negative log-likelihood loss for each action component
                if "action_type" in expert_action:
                    target = torch.tensor(expert_action["action_type"], device=self.device)
                    loss = loss - action_dist["action_type"].log_prob(target)
                if "spmt_idx" in expert_action:
                    target = torch.tensor(expert_action["spmt_idx"], device=self.device)
                    loss = loss - action_dist["spmt"].log_prob(target)
                if "request_idx" in expert_action:
                    target = torch.tensor(expert_action["request_idx"], device=self.device)
                    target = target.clamp(0, action_dist["request"].logits.shape[-1] - 1)
                    loss = loss - action_dist["request"].log_prob(target)
                if "crane_idx" in expert_action:
                    target = torch.tensor(expert_action["crane_idx"], device=self.device)
                    loss = loss - action_dist["crane"].log_prob(target)

            loss = loss / len(batch_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.policy.parameters()), 1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy."""
        self.encoder.eval()
        self.policy.eval()

        total_throughput = 0.0
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            for step in range(500):
                graph_data = self.env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb = self.encoder(graph_data)
                    mask = convert_mask_to_tensor(self.env.get_action_mask(), self.device)
                    action_dist, _ = self.policy(state_emb, mask)
                    action = {
                        "action_type": action_dist["action_type"].sample().item(),
                        "spmt_idx": action_dist["spmt"].sample().item(),
                        "request_idx": action_dist["request"].sample().item(),
                        "crane_idx": action_dist["crane"].sample().item(),
                    }
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break
            if self.env.sim_time > 0:
                total_throughput += self.env.metrics.get("blocks_erected", 
                    self.env.metrics.get("blocks_completed", 0)) / self.env.sim_time

        return {"avg_throughput": total_throughput / n_episodes}


def run_gnn_ablation(
    cfg: Dict[str, Any],
    num_layers: int,
    hidden_dim: int,
    iterations: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Run a single GNN configuration ablation."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = HHIShipyardEnv(cfg)

    # Create encoder with specified configuration
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    state_dim = hidden_dim * 4  # 4 node types
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2)),
        max_requests=env.n_blocks,
        hidden_dim=hidden_dim * 2,  # Policy hidden dim scales with encoder
    )

    expert = RuleBasedScheduler()

    trainer = GNNAblationTrainer(
        env=env,
        encoder=encoder,
        policy=policy,
        expert=expert,
        device=device,
    )

    # Count parameters
    encoder_params = count_parameters(encoder)
    policy_params = count_parameters(policy)
    total_params = encoder_params + policy_params

    # Training
    start_time = time.time()

    # Phase 1: Initial expert demos
    trainer.collect_expert_demos(n_episodes=10)
    for _ in range(10):
        trainer.train_epoch()

    # Phase 2: DAgger iterations
    for iteration in range(iterations):
        beta = 1.0 - 0.9 * iteration / max(iterations - 1, 1)
        trainer.collect_dagger_data(n_episodes=5, beta=beta)
        for _ in range(10):
            trainer.train_epoch()

    training_time = time.time() - start_time

    # Final evaluation
    metrics = trainer.evaluate(n_episodes=10)

    return {
        "throughput": metrics["avg_throughput"],
        "training_time": training_time,
        "total_params": total_params,
        "encoder_params": encoder_params,
        "policy_params": policy_params,
    }


def main():
    parser = argparse.ArgumentParser(description="GNN Architecture Ablation Study")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--iterations", type=int, default=10, help="DAgger iterations")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # GNN configurations to test
    configurations = [
        {"name": "2-layer, 128 dim (baseline)", "num_layers": 2, "hidden_dim": 128},
        {"name": "4-layer, 128 dim", "num_layers": 4, "hidden_dim": 128},
        {"name": "2-layer, 256 dim", "num_layers": 2, "hidden_dim": 256},
        {"name": "4-layer, 256 dim (full)", "num_layers": 4, "hidden_dim": 256},
    ]

    print("=" * 80)
    print("GNN Architecture Ablation Study")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"DAgger iterations: {args.iterations}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print()
    sys.stdout.flush()

    results = []
    for config in configurations:
        name = config["name"]
        print(f"Running: {name}...")
        sys.stdout.flush()
        
        result = run_gnn_ablation(
            cfg=cfg,
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            iterations=args.iterations,
            device=args.device,
            seed=args.seed,
        )
        result["name"] = name
        results.append(result)
        
        print(f"  Throughput: {result['throughput']:.6f}")
        print(f"  Training time: {result['training_time']:.1f}s")
        print(f"  Parameters: {result['total_params']:,}")
        print()
        sys.stdout.flush()

    # Print final table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Throughput':>12} {'Training Time':>15} {'Parameters':>12}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<30} {r['throughput']:>12.6f} {r['training_time']:>12.1f}s {r['total_params']:>12,}")

    print("-" * 80)

    # Find best configuration
    best = max(results, key=lambda x: x["throughput"])
    print(f"\nBest configuration: {best['name']}")
    print(f"  Throughput: {best['throughput']:.6f}")
    print(f"  Training time: {best['training_time']:.1f}s")
    print(f"  Parameters: {best['total_params']:,}")


if __name__ == "__main__":
    main()
