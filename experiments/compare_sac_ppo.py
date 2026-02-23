#!/usr/bin/env python3
"""Compare SAC vs PPO for Shipyard Scheduling.

Runs both algorithms and compares:
- Final throughput
- Entropy over training
- Sample efficiency (epochs to reach 50% of expert)
- Wall clock time

Usage:
    python experiments/compare_sac_ppo.py --epochs 20
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
from torch.distributions import Categorical
from collections import deque
import yaml
from typing import Dict, List, Any, Tuple, Optional
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from agent.ppo import PPOTrainer
from agent.action_masking import flatten_env_mask_to_policy_mask, to_torch_mask
from utils.metrics import compute_kpis


# Expert throughput baseline (from default config analysis)
EXPERT_THROUGHPUT = 0.01  # blocks per time unit (adjust based on actual expert performance)


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


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return states, actions, rewards, next_states, dones, masks

    def __len__(self):
        return len(self.buffer)


class SACPolicy(nn.Module):
    """SAC-style policy for hierarchical action space."""
    def __init__(self, state_dim: int, n_action_types: int = 4, n_spmts: int = 1,
                 n_cranes: int = 1, max_requests: int = 50, hidden_dim: int = 256):
        super().__init__()
        self.n_action_types = n_action_types
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_type_head = nn.Linear(hidden_dim, n_action_types)
        self.spmt_head = nn.Linear(hidden_dim, n_spmts)
        self.request_head = nn.Linear(hidden_dim, max_requests)
        self.crane_head = nn.Linear(hidden_dim, n_cranes)
        self.lift_head = nn.Linear(hidden_dim, max_requests)
        self.equipment_head = nn.Linear(hidden_dim, n_spmts + n_cranes)

    def forward(self, state: torch.Tensor, mask: Optional[Dict] = None):
        features = self.shared(state)
        logits = {
            "action_type": self.action_type_head(features),
            "spmt": self.spmt_head(features),
            "request": self.request_head(features),
            "crane": self.crane_head(features),
            "lift": self.lift_head(features),
            "equipment": self.equipment_head(features),
        }
        
        if mask is not None:
            for key in logits:
                if key in mask:
                    m = mask[key]
                    if isinstance(m, torch.Tensor):
                        if m.dim() == 1 and logits[key].dim() == 2:
                            m = m.unsqueeze(0)
                        logits[key] = logits[key].masked_fill(~m.bool(), -1e9)
        return logits

    def get_action(self, state: torch.Tensor, mask: Optional[Dict] = None, deterministic: bool = False):
        logits = self.forward(state, mask)
        dists = {k: Categorical(logits=v) for k, v in logits.items()}
        if deterministic:
            action = {k: d.probs.argmax(dim=-1) for k, d in dists.items()}
        else:
            action = {k: d.sample() for k, d in dists.items()}
        log_prob = sum(d.log_prob(action[k]) for k, d in dists.items())
        entropy = sum(d.entropy() for d in dists.values())
        return action, log_prob, entropy


class SACTrainer:
    """Simplified SAC trainer for comparison."""
    def __init__(self, policy, state_dim, n_action_types=4, device="cpu", 
                 lr=3e-4, gamma=0.99, tau=0.005, alpha_init=0.2):
        self.policy = policy
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Q-networks (simplified - using same architecture as policy)
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + n_action_types, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + n_action_types, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        self.q1_target = nn.Sequential(
            nn.Linear(state_dim + n_action_types, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        self.q2_target = nn.Sequential(
            nn.Linear(state_dim + n_action_types, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Automatic entropy tuning
        self.target_entropy = -np.log(1.0 / n_action_types) * 0.98
        self.log_alpha = torch.tensor(np.log(alpha_init), requires_grad=True, device=device)
        
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.replay_buffer = ReplayBuffer()
        
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return {"policy_loss": 0, "q_loss": 0, "entropy": 0, "alpha": self.alpha.item()}
        
        states, actions, rewards, next_states, dones, masks = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.stack([s.squeeze(0) if s.dim() > 1 else s for s in states]).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack([s.squeeze(0) if s.dim() > 1 else s for s in next_states]).to(self.device)
        
        # Extract action type as one-hot for Q-network input
        action_types = torch.stack([a["action_type"] for a in actions]).to(self.device)
        action_one_hot = F.one_hot(action_types, num_classes=4).float()
        
        # Q-network update
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.get_action(next_states)
            next_action_one_hot = F.one_hot(next_action["action_type"], num_classes=4).float()
            
            q1_next = self.q1_target(torch.cat([next_states, next_action_one_hot], dim=-1))
            q2_next = self.q2_target(torch.cat([next_states, next_action_one_hot], dim=-1))
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_prob.unsqueeze(1))
        
        q1_pred = self.q1(torch.cat([states, action_one_hot], dim=-1))
        q2_pred = self.q2(torch.cat([states, action_one_hot], dim=-1))
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Policy update
        action, log_prob, entropy = self.policy.get_action(states)
        action_one_hot = F.one_hot(action["action_type"], num_classes=4).float()
        
        q1_val = self.q1(torch.cat([states, action_one_hot], dim=-1))
        q2_val = self.q2(torch.cat([states, action_one_hot], dim=-1))
        q_val = torch.min(q1_val, q2_val)
        
        policy_loss = (self.alpha.detach() * log_prob - q_val.squeeze()).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Alpha (entropy coefficient) update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "policy_loss": policy_loss.item(),
            "q_loss": q_loss.item(),
            "entropy": entropy.mean().item(),
            "alpha": self.alpha.item()
        }


def run_ppo_training(cfg: Dict, epochs: int, steps: int, device: str, seed: int, 
                     use_wandb: bool = False) -> Dict[str, Any]:
    """Run PPO training and return metrics."""
    print("\n" + "="*60)
    print("TRAINING PPO")
    print("="*60)
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    start_time = time.time()
    
    # Initialize wandb for PPO
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="shipyard-scheduling",
            name="PPO-comparison",
            tags=["ppo", "comparison"],
            config={"algorithm": "PPO", "epochs": epochs, "seed": seed},
            reinit=True
        )
    
    # Create environment
    env = HHIShipyardEnv(cfg)
    
    # Get feature dimensions from environment
    n_spmts = env.n_spmts
    n_cranes = getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2))
    hidden_dim = 256
    
    # Create encoder with correct feature dimensions
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=4,
    ).to(device)
    
    state_dim = hidden_dim * 4  # four pooled node types
    
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_spmts=n_spmts,
        n_cranes=n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=512
    ).to(device)
    
    # Create trainer with high initial entropy
    trainer = PPOTrainer(
        policy=policy,
        encoder=encoder,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=2.0,  # High initial entropy
        value_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        total_epochs=epochs,
        entropy_schedule="cosine",
        entropy_coef_final=0.1
    )
    
    # Training metrics
    entropy_history = []
    throughput_history = []
    epoch_to_50_pct = None
    expert_50_pct = EXPERT_THROUGHPUT * 0.5
    
    reset_next = True
    for epoch in range(epochs):
        # Collect rollout using PPOTrainer's method
        rollout = trainer.collect_rollout(env, steps, reset=reset_next)
        reset_next = False  # Continue episode
        
        # Update policy
        metrics = trainer.update(rollout)
        
        # Compute KPIs
        kpis = compute_kpis(env.metrics, env.sim_time)
        throughput = kpis.get("throughput", 0)
        
        entropy_history.append(metrics.get("entropy", 0))
        throughput_history.append(throughput)
        
        # Check for 50% expert threshold
        if epoch_to_50_pct is None and throughput >= expert_50_pct:
            epoch_to_50_pct = epoch + 1
        
        print(f"PPO Epoch {epoch+1}/{epochs} - Entropy: {metrics.get('entropy', 0):.4f}, "
              f"Throughput: {throughput:.4f}, PolicyLoss: {metrics.get('policy_loss', 0):.4f}")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "entropy": metrics.get("entropy", 0),
                "throughput": throughput,
                "policy_loss": metrics.get("policy_loss", 0),
                "value_loss": metrics.get("value_loss", 0)
            })
    
    wall_time = time.time() - start_time
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return {
        "algorithm": "PPO",
        "final_throughput": throughput_history[-1] if throughput_history else 0,
        "max_throughput": max(throughput_history) if throughput_history else 0,
        "entropy_history": entropy_history,
        "throughput_history": throughput_history,
        "epoch_to_50_pct": epoch_to_50_pct,
        "wall_time": wall_time,
        "entropy_collapse": entropy_history[-1] < 0.1 if entropy_history else False,
        "final_entropy": entropy_history[-1] if entropy_history else 0
    }


def run_sac_training(cfg: Dict, epochs: int, steps: int, device: str, seed: int,
                     use_wandb: bool = False) -> Dict[str, Any]:
    """Run SAC training and return metrics."""
    print("\n" + "="*60)
    print("TRAINING SAC")
    print("="*60)
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    start_time = time.time()
    
    # Initialize wandb for SAC
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="shipyard-scheduling",
            name="SAC-comparison",
            tags=["sac", "comparison"],
            config={"algorithm": "SAC", "epochs": epochs, "seed": seed},
            reinit=True
        )
    
    # Create environment
    env = HHIShipyardEnv(cfg)
    
    # Get feature dimensions from environment
    n_spmts = env.n_spmts
    n_cranes = getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 2))
    hidden_dim = 256
    
    # Create encoder with correct feature dimensions
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=4,
    ).to(device)
    
    state_dim = hidden_dim * 4  # four pooled node types
    
    policy = SACPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=n_spmts,
        n_cranes=n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Create SAC trainer
    trainer = SACTrainer(
        policy=policy,
        state_dim=state_dim,
        n_action_types=4,
        device=device,
        lr=3e-4,
        gamma=0.99,
        alpha_init=0.2
    )
    
    # Training metrics
    entropy_history = []
    throughput_history = []
    alpha_history = []
    epoch_to_50_pct = None
    expert_50_pct = EXPERT_THROUGHPUT * 0.5
    
    for epoch in range(epochs):
        obs, info = env.reset()
        graph_data = env.get_graph_data().to(device)
        done = False
        step = 0
        episode_reward = 0
        
        while not done and step < steps:
            with torch.no_grad():
                state = encoder(graph_data)
                action_mask = env.get_action_mask()
                action_mask_tensor = {k: torch.tensor(v, device=device) for k, v in action_mask.items()}
                action, log_prob, entropy = policy.get_action(state, action_mask_tensor)
                action_dict = {k: int(v.item()) for k, v in action.items()}
            
            obs, reward, terminated, truncated, info = env.step(action_dict)
            next_graph_data = env.get_graph_data().to(device)
            done = terminated or truncated
            
            with torch.no_grad():
                next_state = encoder(next_graph_data)
            
            # Store in replay buffer
            trainer.replay_buffer.push(
                state.detach(),
                {k: v.detach() for k, v in action.items()},
                reward,
                next_state.detach(),
                done,
                action_mask_tensor
            )
            
            graph_data = next_graph_data
            episode_reward += reward
            step += 1
            
            # Update SAC every step (if enough samples)
            if len(trainer.replay_buffer) >= 64:
                _ = trainer.update(batch_size=64)
        
        # Final update for epoch
        metrics = trainer.update(batch_size=64)
        
        # Compute KPIs
        kpis = compute_kpis(env.metrics, env.sim_time)
        throughput = kpis.get("throughput", 0)
        
        entropy_history.append(metrics.get("entropy", 0))
        throughput_history.append(throughput)
        alpha_history.append(metrics.get("alpha", 0))
        
        # Check for 50% expert threshold
        if epoch_to_50_pct is None and throughput >= expert_50_pct:
            epoch_to_50_pct = epoch + 1
        
        print(f"SAC Epoch {epoch+1}/{epochs} - Entropy: {metrics.get('entropy', 0):.4f}, "
              f"Alpha: {metrics.get('alpha', 0):.4f}, Throughput: {throughput:.4f}, "
              f"QLoss: {metrics.get('q_loss', 0):.4f}")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "entropy": metrics.get("entropy", 0),
                "alpha": metrics.get("alpha", 0),
                "throughput": throughput,
                "policy_loss": metrics.get("policy_loss", 0),
                "q_loss": metrics.get("q_loss", 0)
            })
    
    wall_time = time.time() - start_time
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return {
        "algorithm": "SAC",
        "final_throughput": throughput_history[-1] if throughput_history else 0,
        "max_throughput": max(throughput_history) if throughput_history else 0,
        "entropy_history": entropy_history,
        "throughput_history": throughput_history,
        "alpha_history": alpha_history,
        "epoch_to_50_pct": epoch_to_50_pct,
        "wall_time": wall_time,
        "entropy_maintained": entropy_history[-1] > 0.1 if entropy_history else False,
        "final_entropy": entropy_history[-1] if entropy_history else 0
    }


def print_comparison_report(ppo_results: Dict, sac_results: Dict):
    """Print detailed comparison report."""
    print("\n" + "="*70)
    print("COMPARISON REPORT: SAC vs PPO for Shipyard Scheduling")
    print("="*70)
    
    print("\n1. FINAL THROUGHPUT")
    print("-" * 40)
    print(f"  PPO: {ppo_results['final_throughput']:.6f}")
    print(f"  SAC: {sac_results['final_throughput']:.6f}")
    throughput_winner = "PPO" if ppo_results['final_throughput'] > sac_results['final_throughput'] else "SAC"
    print(f"  Winner: {throughput_winner}")
    
    print("\n2. MAX THROUGHPUT ACHIEVED")
    print("-" * 40)
    print(f"  PPO: {ppo_results['max_throughput']:.6f}")
    print(f"  SAC: {sac_results['max_throughput']:.6f}")
    
    print("\n3. ENTROPY BEHAVIOR")
    print("-" * 40)
    print(f"  PPO Initial Entropy: {ppo_results['entropy_history'][0]:.4f}")
    print(f"  PPO Final Entropy: {ppo_results['final_entropy']:.4f}")
    print(f"  PPO Entropy Collapse: {ppo_results['entropy_collapse']}")
    print(f"  SAC Initial Entropy: {sac_results['entropy_history'][0]:.4f}")
    print(f"  SAC Final Entropy: {sac_results['final_entropy']:.4f}")
    print(f"  SAC Entropy Maintained: {sac_results['entropy_maintained']}")
    
    print("\n4. SAMPLE EFFICIENCY (Epochs to 50% Expert)")
    print("-" * 40)
    ppo_eff = ppo_results['epoch_to_50_pct'] if ppo_results['epoch_to_50_pct'] else "Not reached"
    sac_eff = sac_results['epoch_to_50_pct'] if sac_results['epoch_to_50_pct'] else "Not reached"
    print(f"  PPO: {ppo_eff}")
    print(f"  SAC: {sac_eff}")
    
    print("\n5. WALL CLOCK TIME")
    print("-" * 40)
    print(f"  PPO: {ppo_results['wall_time']:.2f} seconds")
    print(f"  SAC: {sac_results['wall_time']:.2f} seconds")
    time_winner = "PPO" if ppo_results['wall_time'] < sac_results['wall_time'] else "SAC"
    print(f"  Faster: {time_winner}")
    
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATION")
    print("="*70)
    
    # Determine overall winner
    scores = {"PPO": 0, "SAC": 0}
    
    # Throughput comparison
    if ppo_results['final_throughput'] > sac_results['final_throughput']:
        scores["PPO"] += 2
    else:
        scores["SAC"] += 2
    
    # Entropy maintenance (important for exploration)
    if not ppo_results['entropy_collapse']:
        scores["PPO"] += 1
    if sac_results['entropy_maintained']:
        scores["SAC"] += 1
    
    # Sample efficiency
    if ppo_results['epoch_to_50_pct'] and sac_results['epoch_to_50_pct']:
        if ppo_results['epoch_to_50_pct'] < sac_results['epoch_to_50_pct']:
            scores["PPO"] += 1
        else:
            scores["SAC"] += 1
    
    # Wall time
    if ppo_results['wall_time'] < sac_results['wall_time']:
        scores["PPO"] += 1
    else:
        scores["SAC"] += 1
    
    print(f"\nScoring Summary:")
    print(f"  PPO Score: {scores['PPO']}")
    print(f"  SAC Score: {scores['SAC']}")
    
    winner = "PPO" if scores["PPO"] > scores["SAC"] else "SAC" if scores["SAC"] > scores["PPO"] else "TIE"
    
    print(f"\nOVERALL WINNER: {winner}")
    
    print("\nREASONING:")
    if winner == "SAC":
        print("  - SAC maintains entropy through automatic tuning, preventing collapse")
        print("  - Off-policy learning enables better sample reuse")
        print("  - Maximum entropy framework provides more robust exploration")
    elif winner == "PPO":
        print("  - PPO is more sample efficient for on-policy updates")
        print("  - Simpler to tune and more stable during training")
        print("  - Lower wall clock time due to simpler update procedure")
    else:
        print("  - Both algorithms perform comparably")
        print("  - Choice depends on specific requirements (stability vs exploration)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Compare SAC vs PPO")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()
    
    print("="*70)
    print("SAC vs PPO Comparison for Shipyard Scheduling")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {args.steps}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"W&B Logging: {args.wandb and WANDB_AVAILABLE}")
    
    # Load config
    cfg = load_config(args.config)
    
    # Run PPO training
    ppo_results = run_ppo_training(cfg, args.epochs, args.steps, args.device, args.seed, args.wandb)
    
    # Run SAC training
    sac_results = run_sac_training(cfg, args.epochs, args.steps, args.device, args.seed, args.wandb)
    
    # Print comparison report
    print_comparison_report(ppo_results, sac_results)
    
    # Save results to JSON
    results = {
        "ppo": {k: v if not isinstance(v, list) else v for k, v in ppo_results.items()},
        "sac": {k: v if not isinstance(v, list) else v for k, v in sac_results.items()}
    }
    
    results_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
