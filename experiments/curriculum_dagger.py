#!/usr/bin/env python3
"""Curriculum Learning with DAgger for Shipyard Scheduling.

Progressively trains on increasingly complex instances:
1. Tiny instance (10 blocks, 1 ship)
2. Small instance (50 blocks, 1 ship)  
3. Medium HHI (100 blocks, 2 ships)
4. Full HHI Ulsan (200 blocks, 8 ships) - evaluation only

Usage:
    python experiments/curriculum_dagger.py --iterations 5
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
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from torch_geometric.data import Batch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.simulation.shipyard_env import HHIShipyardEnv
from src.agent.gnn_encoder import HeterogeneousGNNEncoder
from src.agent.policy import ActorCriticPolicy
from src.baselines.rule_based import RuleBasedScheduler

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, metrics will only be printed")


class RunningMeanStd:
    """Online normalization using Welford's algorithm.

    Tracks running mean and variance to normalize observations,
    preventing distribution shift across curriculum stages.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # avoid division by zero

    def update(self, x: np.ndarray) -> None:
        """Update running stats with a batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor using running stats."""
        mean_t = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std_t = torch.tensor(np.sqrt(self.var + 1e-8), dtype=x.dtype, device=x.device)
        return (x - mean_t) / std_t


def load_config(path: str) -> Dict[str, Any]:
    """Load config with inheritance support."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


class CurriculumDAggerTrainer:
    """DAgger trainer with curriculum learning support."""
    
    def __init__(
        self,
        encoder: HeterogeneousGNNEncoder,
        policy: ActorCriticPolicy,
        device: str = "cpu",
        lr: float = 3e-4,
    ):
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(policy.parameters()),
            lr=lr, weight_decay=1e-5
        )

        # Aggregated dataset — store raw graph data for encoder gradient flow
        self.graph_data_list: List[Any] = []  # HeteroData objects
        self.expert_actions: List[Dict[str, int]] = []

        # Stage-aware replay buffers for curriculum learning
        # Prevents catastrophic forgetting by maintaining data from all stages
        self.stage_buffers: List[Tuple[List[Any], List[Dict[str, int]]]] = []
        self.current_stage_start: int = 0  # index where current stage data begins

        # Observation normalizer (initialized lazily on first embedding)
        self.obs_normalizer: Optional[RunningMeanStd] = None

        # Action key mapping
        self.action_keys = {
            "action_type": "action_type",
            "spmt": "spmt_idx",
            "request": "request_idx",
            "crane": "crane_idx",
            "lift": "lift_idx",
            "equipment": "equipment_idx",
        }
        
    def reset_dataset(self):
        """Clear the aggregated dataset."""
        self.graph_data_list = []
        self.expert_actions = []

    def advance_stage(self):
        """Save current stage data to replay buffer and prepare for next stage.

        Instead of discarding data, maintains separate buffers per stage
        for mixed sampling during training. This prevents catastrophic
        forgetting across curriculum stages.
        """
        # Save current stage data
        current_data = self.graph_data_list[self.current_stage_start:]
        current_actions = self.expert_actions[self.current_stage_start:]
        if current_data:
            self.stage_buffers.append((list(current_data), list(current_actions)))

        # Mark where new stage data begins
        self.current_stage_start = len(self.graph_data_list)

    def trim_dataset(self, keep_fraction: float = 0.5):
        """Retain most recent fraction of dataset (for stage transitions).

        Also saves data to stage buffer before trimming to prevent
        catastrophic forgetting.
        """
        # Save current stage to replay buffer first
        self.advance_stage()

        n = len(self.graph_data_list)
        if n == 0:
            return
        keep = max(1, int(n * keep_fraction))
        self.graph_data_list = self.graph_data_list[-keep:]
        self.expert_actions = self.expert_actions[-keep:]
        self.current_stage_start = 0

    def _init_normalizer(self, dim: int):
        """Lazily initialize the observation normalizer."""
        if self.obs_normalizer is None:
            self.obs_normalizer = RunningMeanStd(shape=(dim,))

    def _update_and_normalize(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Update normalizer stats and return normalized embedding."""
        arr = state_emb.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._init_normalizer(arr.shape[-1])
        self.obs_normalizer.update(arr)
        return self.obs_normalizer.normalize(state_emb)

    def _normalize(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Normalize without updating stats (for evaluation)."""
        if self.obs_normalizer is None:
            return state_emb
        return self.obs_normalizer.normalize(state_emb)

    def collect_expert_demos(
        self,
        env: HHIShipyardEnv,
        expert: RuleBasedScheduler,
        n_episodes: int,
        max_steps: int = 500
    ) -> Dict[str, float]:
        """Collect expert demonstrations (stores raw graph data, not embeddings)."""
        total_reward = 0.0
        total_throughput = 0.0

        for ep in range(n_episodes):
            obs, info = env.reset()
            ep_reward = 0.0

            for step in range(max_steps):
                # Store raw graph data for re-encoding during training
                graph_data = env.get_graph_data().cpu()

                # Get expert action
                expert_action = expert.decide(env)

                # Store graph data and expert action
                self.graph_data_list.append(graph_data)
                self.expert_actions.append(expert_action)

                # Step with expert action
                obs, reward, terminated, truncated, info = env.step(expert_action)
                ep_reward += reward

                if terminated or truncated:
                    break

            total_reward += ep_reward
            if env.sim_time > 0:
                completed = env.metrics.get("blocks_erected", env.metrics.get("blocks_completed", 0))
                total_throughput += completed / env.sim_time

        return {
            "avg_reward": total_reward / n_episodes,
            "avg_throughput": total_throughput / n_episodes,
            "n_samples": len(self.graph_data_list),
        }
    
    def collect_dagger_data(
        self,
        env: HHIShipyardEnv,
        expert: RuleBasedScheduler,
        n_episodes: int,
        beta: float = 0.5,
        max_steps: int = 500
    ) -> int:
        """Collect DAgger data (stores raw graph data for encoder gradient flow)."""
        new_samples = 0

        for ep in range(n_episodes):
            obs, info = env.reset()

            for step in range(max_steps):
                # Store raw graph data
                graph_data = env.get_graph_data()

                expert_action = expert.decide(env)

                self.graph_data_list.append(graph_data.cpu())
                self.expert_actions.append(expert_action)
                new_samples += 1

                if random.random() < beta:
                    action = expert_action
                else:
                    # Encode with masking + entity embeddings for valid action selection
                    with torch.no_grad():
                        state_emb, block_embs = self.encoder(
                            graph_data.to(self.device), return_entity_embeddings=True
                        )
                        env_mask = env.get_action_mask()
                        from agent.action_masking import flatten_env_mask_to_policy_mask
                        pmask = flatten_env_mask_to_policy_mask(
                            env_mask, self.policy.n_spmts,
                            self.policy.n_cranes, self.policy.max_requests
                        )
                        tmask = {k: torch.tensor(v, device=self.device) for k, v in pmask.items()}
                        entity_embs = block_embs.unsqueeze(0) if block_embs is not None else None
                        policy_action, _, _ = self.policy.get_action(
                            state_emb, mask=tmask, entity_embeddings=entity_embs
                        )
                    action = {k: int(v.item()) for k, v in policy_action.items()}

                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        return new_samples
    
    def _sample_replay_batch(self, batch_size: int, replay_fraction: float = 0.3):
        """Sample a batch mixing current data with replay from previous stages.

        Allocates replay_fraction of the batch to previous stages (split
        evenly among them), with the remainder from the current dataset.
        Prevents catastrophic forgetting during curriculum transitions.
        """
        replay_graphs = []
        replay_actions = []

        if self.stage_buffers and replay_fraction > 0:
            n_replay = max(1, int(batch_size * replay_fraction))
            per_stage = max(1, n_replay // len(self.stage_buffers))

            for stage_data, stage_acts in self.stage_buffers:
                if not stage_data:
                    continue
                idxs = random.sample(range(len(stage_data)), min(per_stage, len(stage_data)))
                replay_graphs.extend([stage_data[i] for i in idxs])
                replay_actions.extend([stage_acts[i] for i in idxs])

        return replay_graphs, replay_actions

    def train_epoch(self, batch_size: int = 64) -> float:
        """Train policy AND encoder on aggregated dataset for one epoch.

        Re-encodes raw graph data through the GNN encoder during training,
        allowing encoder gradients to flow. Mixes in replay samples from
        previous curriculum stages to prevent catastrophic forgetting.
        """
        n_samples = len(self.graph_data_list)
        if n_samples < batch_size:
            batch_size = max(1, n_samples // 2)

        indices = list(range(n_samples))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < 2:
                continue

            # Collect current-stage data
            batch_graph_list = [self.graph_data_list[j] for j in batch_indices]
            batch_action_list = [self.expert_actions[j] for j in batch_indices]

            # Mix in replay from previous stages (30% of batch)
            replay_graphs, replay_actions = self._sample_replay_batch(
                len(batch_indices), replay_fraction=0.3
            )
            if replay_graphs:
                batch_graph_list.extend(replay_graphs)
                batch_action_list.extend(replay_actions)

            # Re-encode through GNN encoder (gradients flow to encoder!)
            batch_graphs = Batch.from_data_list(batch_graph_list).to(self.device)
            batch_states = self.encoder(batch_graphs)

            action_dist, _ = self.policy.forward(batch_states)

            loss = 0.0
            for head_name, action_key in self.action_keys.items():
                target = torch.tensor(
                    [a.get(action_key, 0) for a in batch_action_list],
                    device=self.device
                )
                max_idx = action_dist[head_name].probs.shape[-1] - 1
                target = target.clamp(0, max_idx)
                
                head_loss = F.cross_entropy(action_dist[head_name].logits, target)
                
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
    
    def evaluate(
        self,
        env: HHIShipyardEnv,
        n_episodes: int = 3,
        max_steps: int = 500
    ) -> Dict[str, float]:
        """Evaluate current policy."""
        total_throughput = 0.0
        total_reward = 0.0
        total_completed = 0

        for ep in range(n_episodes):
            obs, info = env.reset()
            ep_reward = 0.0

            for step in range(max_steps):
                graph_data = env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state_emb, block_embs = self.encoder(graph_data, return_entity_embeddings=True)
                    # Apply masking + entity embeddings
                    env_mask = env.get_action_mask()
                    from agent.action_masking import flatten_env_mask_to_policy_mask
                    pmask = flatten_env_mask_to_policy_mask(
                        env_mask, self.policy.n_spmts,
                        self.policy.n_cranes, self.policy.max_requests
                    )
                    tmask = {k: torch.tensor(v, device=self.device) for k, v in pmask.items()}
                    entity_embs = block_embs.unsqueeze(0) if block_embs is not None else None
                    action, _, _ = self.policy.get_action(
                        state_emb, mask=tmask, entity_embeddings=entity_embs, deterministic=True
                    )

                action_cpu = {k: int(v.item()) for k, v in action.items()}
                obs, reward, terminated, truncated, info = env.step(action_cpu)
                ep_reward += reward

                if terminated or truncated:
                    break

            total_reward += ep_reward
            completed = env.metrics.get("blocks_erected", env.metrics.get("blocks_completed", 0))
            total_completed += completed

            if env.sim_time > 0:
                total_throughput += completed / env.sim_time

        return {
            "avg_reward": total_reward / n_episodes,
            "avg_throughput": total_throughput / n_episodes,
            "total_completed": total_completed,
        }


def create_networks(env: HHIShipyardEnv, hidden_dim: int = 128) -> Tuple[HeterogeneousGNNEncoder, ActorCriticPolicy]:
    """Create encoder and policy networks."""
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=2,
    )
    
    state_dim = hidden_dim * 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=max(24, getattr(env, 'n_spmts', 6)),
        n_cranes=max(9, getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2))),
        max_requests=max(1600, getattr(env, 'n_blocks', 50)),
        hidden_dim=256,
        epsilon=0.0,
        entity_dim=hidden_dim,  # Match GNN encoder hidden_dim for attention action heads
    )
    
    return encoder, policy


def run_curriculum_training(
    config_paths: List[str],
    stage_names: List[str],
    iterations_per_stage: int = 5,
    init_episodes: int = 5,
    dagger_episodes: int = 3,
    train_epochs: int = 10,
    max_steps: int = 300,
    beta_start: float = 1.0,
    beta_end: float = 0.1,
    device: str = "cpu",
    use_wandb: bool = True,
    run_name: str = "curriculum_dagger",
) -> Dict[str, Any]:
    """Run curriculum learning experiment."""
    
    results = {
        "stages": [],
        "final_metrics": None,
        "total_time": 0.0,
    }
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="shipyard-curriculum",
            name=run_name,
            config={
                "method": "curriculum_dagger",
                "stages": stage_names,
                "iterations_per_stage": iterations_per_stage,
                "init_episodes": init_episodes,
                "dagger_episodes": dagger_episodes,
                "train_epochs": train_epochs,
            }
        )
    
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING WITH DAGGER")
    print("=" * 70)
    print(f"Stages: {' -> '.join(stage_names)}")
    print(f"Iterations per stage: {iterations_per_stage}")
    
    # Initialize with first environment
    first_cfg = load_config(config_paths[0])
    first_env = HHIShipyardEnv(first_cfg)
    encoder, policy = create_networks(first_env, hidden_dim=128)
    
    expert = RuleBasedScheduler()
    
    trainer = CurriculumDAggerTrainer(
        encoder=encoder,
        policy=policy,
        device=device,
        lr=3e-4,
    )
    
    # Train through curriculum stages
    for stage_idx, (config_path, stage_name) in enumerate(zip(config_paths, stage_names)):
        stage_start = time.time()

        print("\n" + "-" * 70)
        print(f"STAGE {stage_idx + 1}/{len(config_paths)}: {stage_name}")
        print("-" * 70)

        cfg = load_config(config_path)
        env = HHIShipyardEnv(cfg)

        n_blocks = getattr(env, 'n_blocks', cfg.get('n_blocks', 'unknown'))
        n_spmts = getattr(env, 'n_spmts', cfg.get('n_spmts', 'unknown'))
        n_cranes = getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', cfg.get('n_cranes', 'unknown')))

        print(f"Environment: {n_blocks} blocks, {n_spmts} SPMTs, {n_cranes} cranes")

        # Per-stage max_steps: scale with instance size
        if isinstance(n_blocks, int):
            stage_max_steps = max(max_steps, n_blocks * 10)
        else:
            stage_max_steps = max_steps
        print(f"  max_steps for this stage: {stage_max_steps}")

        # Trim dataset at stage boundaries (keep 50% from prior stage)
        if stage_idx > 0 and len(trainer.graph_data_list) > 0:
            pre_trim = len(trainer.graph_data_list)
            trainer.trim_dataset(keep_fraction=0.5)
            print(f"  Dataset trimmed: {pre_trim} -> {len(trainer.graph_data_list)} samples")

        # Collect expert demos
        print(f"\nCollecting {init_episodes} expert demonstrations...")
        demo_metrics = trainer.collect_expert_demos(env, expert, init_episodes, max_steps=stage_max_steps)
        print(f"  Samples collected: {demo_metrics['n_samples']}")
        print(f"  Expert throughput: {demo_metrics['avg_throughput']:.6f}")

        # Initial BC training
        print(f"\nInitial BC training ({train_epochs} epochs)...")
        for epoch in range(train_epochs):
            loss = trainer.train_epoch()
        print(f"  Final loss: {loss:.4f}")

        # Evaluate after BC
        metrics = trainer.evaluate(env, n_episodes=2, max_steps=stage_max_steps)
        print(f"After BC: Throughput = {metrics['avg_throughput']:.6f}")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f"stage_{stage_idx}/bc_throughput": metrics['avg_throughput'],
                f"stage_{stage_idx}/bc_reward": metrics['avg_reward'],
                "stage": stage_idx,
            })

        # DAgger iterations — per-stage beta schedule (reset each stage)
        stage_metrics = []
        stage_beta_start = 0.8 if stage_idx > 0 else beta_start
        stage_beta_end = 0.2
        for iteration in range(iterations_per_stage):
            beta = stage_beta_start - (stage_beta_start - stage_beta_end) * iteration / max(iterations_per_stage - 1, 1)

            print(f"\n  DAgger Iteration {iteration + 1}/{iterations_per_stage} (beta={beta:.2f})")

            new_samples = trainer.collect_dagger_data(env, expert, dagger_episodes, beta=beta, max_steps=stage_max_steps)
            print(f"    New samples: {new_samples}, Total: {len(trainer.graph_data_list)}")

            for epoch in range(train_epochs):
                loss = trainer.train_epoch()
            print(f"    Final loss: {loss:.4f}")

            metrics = trainer.evaluate(env, n_episodes=2, max_steps=stage_max_steps)
            print(f"    Throughput: {metrics['avg_throughput']:.6f}")

            stage_metrics.append({
                "iteration": iteration + 1,
                "beta": beta,
                "loss": loss,
                "throughput": metrics['avg_throughput'],
                "reward": metrics['avg_reward'],
            })

            global_iter = stage_idx * iterations_per_stage + iteration
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"stage_{stage_idx}/iteration": iteration + 1,
                    f"stage_{stage_idx}/throughput": metrics['avg_throughput'],
                    f"stage_{stage_idx}/reward": metrics['avg_reward'],
                    f"stage_{stage_idx}/loss": loss,
                    f"stage_{stage_idx}/beta": beta,
                    "global_iteration": global_iter,
                })
        
        stage_time = time.time() - stage_start
        
        final_stage_metrics = trainer.evaluate(env, n_episodes=3, max_steps=max_steps)
        
        results["stages"].append({
            "name": stage_name,
            "config": config_path,
            "n_blocks": n_blocks,
            "n_spmts": n_spmts,
            "n_cranes": n_cranes,
            "iterations": stage_metrics,
            "final_throughput": final_stage_metrics['avg_throughput'],
            "final_reward": final_stage_metrics['avg_reward'],
            "time_seconds": stage_time,
        })
        
        print(f"\n  Stage complete in {stage_time:.1f}s")
        print(f"  Final throughput: {final_stage_metrics['avg_throughput']:.6f}")
    
    # Final evaluation on full HHI
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HHI ULSAN (full scale)")
    print("=" * 70)
    
    hhi_cfg = load_config(os.path.join(PROJECT_ROOT, "config", "hhi_ulsan.yaml"))
    hhi_env = HHIShipyardEnv(hhi_cfg)
    
    print(f"HHI Environment: {hhi_env.n_blocks} blocks, {hhi_env.n_spmts} SPMTs")
    
    final_metrics = trainer.evaluate(hhi_env, n_episodes=2, max_steps=1000)
    results["final_metrics"] = final_metrics
    
    print(f"Curriculum DAgger - HHI Throughput: {final_metrics['avg_throughput']:.6f}")
    print(f"Curriculum DAgger - HHI Reward: {final_metrics['avg_reward']:.2f}")
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final/hhi_throughput": final_metrics['avg_throughput'],
            "final/hhi_reward": final_metrics['avg_reward'],
            "final/total_time_seconds": total_time,
        })
    
    # Save checkpoint
    save_dir = os.path.join(PROJECT_ROOT, "data", "checkpoints", "curriculum")
    os.makedirs(save_dir, exist_ok=True)
    # Save normalizer state alongside model weights for reproducible inference
    normalizer_state = None
    if trainer.obs_normalizer is not None:
        normalizer_state = {
            "mean": trainer.obs_normalizer.mean,
            "var": trainer.obs_normalizer.var,
            "count": trainer.obs_normalizer.count,
        }
    torch.save({
        "encoder": encoder.state_dict(),
        "policy": policy.state_dict(),
        "normalizer": normalizer_state,
        "results": results,
    }, os.path.join(save_dir, "curriculum_dagger_final.pt"))
    
    return results, encoder, policy


def run_direct_training(
    config_path: str,
    iterations: int = 15,
    init_episodes: int = 5,
    dagger_episodes: int = 3,
    train_epochs: int = 10,
    max_steps: int = 500,
    beta_start: float = 1.0,
    beta_end: float = 0.1,
    device: str = "cpu",
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """Run direct training on HHI (baseline)."""
    
    results = {
        "iterations": [],
        "final_metrics": None,
        "total_time": 0.0,
    }
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="shipyard-curriculum",
            name="direct_hhi_dagger",
            config={
                "method": "direct_dagger",
                "iterations": iterations,
            }
        )
    
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("DIRECT TRAINING ON HHI ULSAN (BASELINE)")
    print("=" * 70)
    
    cfg = load_config(config_path)
    env = HHIShipyardEnv(cfg)
    
    encoder, policy = create_networks(env, hidden_dim=128)
    expert = RuleBasedScheduler()
    
    trainer = CurriculumDAggerTrainer(
        encoder=encoder,
        policy=policy,
        device=device,
        lr=3e-4,
    )
    
    print(f"Environment: {env.n_blocks} blocks, {env.n_spmts} SPMTs")
    
    # Collect expert demos
    print(f"\nCollecting {init_episodes} expert demonstrations...")
    demo_metrics = trainer.collect_expert_demos(env, expert, init_episodes, max_steps=max_steps)
    print(f"  Samples: {demo_metrics['n_samples']}, Expert throughput: {demo_metrics['avg_throughput']:.6f}")
    
    # Initial BC
    print(f"\nInitial BC training ({train_epochs} epochs)...")
    for epoch in range(train_epochs):
        loss = trainer.train_epoch()
    
    metrics = trainer.evaluate(env, n_episodes=2, max_steps=max_steps)
    print(f"After BC: Throughput = {metrics['avg_throughput']:.6f}")
    
    # DAgger iterations
    for iteration in range(iterations):
        beta = beta_start - (beta_start - beta_end) * iteration / max(iterations - 1, 1)
        
        print(f"\nDAgger Iteration {iteration + 1}/{iterations} (beta={beta:.2f})")
        
        new_samples = trainer.collect_dagger_data(env, expert, dagger_episodes, beta=beta, max_steps=max_steps)
        print(f"  New samples: {new_samples}, Total: {len(trainer.graph_data_list)}")
        
        for epoch in range(train_epochs):
            loss = trainer.train_epoch()
        print(f"  Final loss: {loss:.4f}")
        
        metrics = trainer.evaluate(env, n_episodes=2, max_steps=max_steps)
        print(f"  Throughput: {metrics['avg_throughput']:.6f}")
        
        results["iterations"].append({
            "iteration": iteration + 1,
            "throughput": metrics['avg_throughput'],
            "reward": metrics['avg_reward'],
        })
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "direct/iteration": iteration + 1,
                "direct/throughput": metrics['avg_throughput'],
                "direct/loss": loss,
            })
    
    final_metrics = trainer.evaluate(env, n_episodes=3, max_steps=1000)
    results["final_metrics"] = final_metrics
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    
    print(f"\nDirect DAgger - HHI Throughput: {final_metrics['avg_throughput']:.6f}")
    print(f"Total time: {total_time:.1f}s")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "direct/final_throughput": final_metrics['avg_throughput'],
            "direct/total_time_seconds": total_time,
        })
        wandb.finish()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Curriculum DAgger for Shipyard Scheduling")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=10, help="DAgger iterations per stage")
    parser.add_argument("--init-episodes", type=int, default=15)
    parser.add_argument("--dagger-episodes", type=int, default=8)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    
    # Define curriculum stages
    config_dir = os.path.join(PROJECT_ROOT, "config")
    curriculum_configs = [
        os.path.join(config_dir, "tiny_instance.yaml"),
        os.path.join(config_dir, "small_instance.yaml"),
        os.path.join(config_dir, "medium_hhi.yaml"),
    ]
    stage_names = ["Tiny (10 blocks)", "Small (50 blocks)", "Medium HHI (200 blocks)"]
    
    # Run curriculum training
    curriculum_results, encoder, policy = run_curriculum_training(
        config_paths=curriculum_configs,
        stage_names=stage_names,
        iterations_per_stage=args.iterations,
        init_episodes=args.init_episodes,
        dagger_episodes=args.dagger_episodes,
        train_epochs=args.train_epochs,
        max_steps=args.max_steps,
        device=args.device,
        use_wandb=use_wandb,
        run_name=f"curriculum_dagger_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Run direct training baseline
    if not args.skip_direct:
        direct_results = run_direct_training(
            config_path=os.path.join(config_dir, "hhi_ulsan.yaml"),
            iterations=args.iterations * len(curriculum_configs),
            init_episodes=args.init_episodes,
            dagger_episodes=args.dagger_episodes,
            train_epochs=args.train_epochs,
            max_steps=args.max_steps,
            device=args.device,
            use_wandb=use_wandb,
        )
    else:
        direct_results = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print("\n--- Curriculum Learning Results ---")
    for stage in curriculum_results["stages"]:
        print(f"\n{stage['name']}:")
        print(f"  Blocks: {stage['n_blocks']}, SPMTs: {stage['n_spmts']}, Cranes: {stage['n_cranes']}")
        print(f"  Final throughput: {stage['final_throughput']:.6f}")
        print(f"  Time: {stage['time_seconds']:.1f}s")
    
    print(f"\nCurriculum - Final HHI Throughput: {curriculum_results['final_metrics']['avg_throughput']:.6f}")
    print(f"Curriculum - Total Time: {curriculum_results['total_time']:.1f}s")
    
    if direct_results:
        print("\n--- Direct Training Results ---")
        print(f"Direct - Final HHI Throughput: {direct_results['final_metrics']['avg_throughput']:.6f}")
        print(f"Direct - Total Time: {direct_results['total_time']:.1f}s")
        
        print("\n--- Comparison ---")
        curriculum_tp = curriculum_results['final_metrics']['avg_throughput']
        direct_tp = direct_results['final_metrics']['avg_throughput']
        
        if direct_tp > 0:
            improvement = (curriculum_tp - direct_tp) / direct_tp * 100
            print(f"Curriculum vs Direct: {improvement:+.1f}% throughput")
        
        time_ratio = curriculum_results['total_time'] / max(direct_results['total_time'], 1)
        print(f"Time ratio (curriculum/direct): {time_ratio:.2f}x")
        
        if curriculum_tp > direct_tp:
            print("\n*** Curriculum learning HELPS! ***")
        else:
            print("\n*** Direct training performs better on this run ***")


if __name__ == "__main__":
    main()
