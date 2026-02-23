#!/usr/bin/env python3
"""Run a trained DAgger/BC policy and optionally visualize in dashboard.

This script loads a trained checkpoint and runs episodes, printing metrics
and optionally logging to the MES database for dashboard visualization.

Usage:
    # Quick evaluation
    python experiments/run_trained_policy.py --checkpoint data/checkpoints/dagger/dagger_final.pt

    # Run and log to dashboard database
    python experiments/run_trained_policy.py --checkpoint data/checkpoints/dagger/dagger_final.pt --log-db

    # Run with visualization output
    python experiments/run_trained_policy.py --checkpoint data/checkpoints/dagger/dagger_final.pt --verbose
"""

import argparse
import os
import sys
import torch
import yaml
import numpy as np
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy


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


def load_checkpoint(checkpoint_path: str, env: ShipyardEnv, device: str = "cpu"):
    """Load encoder and policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine architecture from checkpoint
    args = checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", args.get("hidden-dim", 128))
    policy_hidden = args.get("policy_hidden", args.get("policy-hidden", 256))

    # Create encoder
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=2,
    ).to(device)

    # Create policy
    state_dim = hidden_dim * 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=policy_hidden,
        epsilon=0.0,
    ).to(device)

    # Load weights - handle different checkpoint formats
    if "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
    elif "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "encoder_0" in checkpoint:  # Ensemble checkpoint
        encoder.load_state_dict(checkpoint["encoder_0"])

    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    elif "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    elif "policy_0" in checkpoint:  # Ensemble checkpoint
        policy.load_state_dict(checkpoint["policy_0"])

    # Load normalizer if available
    normalizer = None
    if "normalizer" in checkpoint and checkpoint["normalizer"] is not None:
        from experiments.train_dagger_ensemble import RunningNormalizer
        normalizer = RunningNormalizer(state_dim)
        normalizer.load_state_dict(checkpoint["normalizer"])
    elif "normalizer_0" in checkpoint and checkpoint["normalizer_0"] is not None:
        from experiments.train_dagger_ensemble import RunningNormalizer
        normalizer = RunningNormalizer(state_dim)
        normalizer.load_state_dict(checkpoint["normalizer_0"])

    return encoder, policy, normalizer, checkpoint.get("metrics", {})


def run_episode(env, encoder, policy, normalizer, device, max_steps=1000, verbose=False):
    """Run a single episode with the trained policy."""
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        # Encode state
        graph_data = env.get_graph_data().to(device)
        with torch.no_grad():
            state_emb = encoder(graph_data)

            # Normalize if available
            if normalizer is not None:
                state_emb = normalizer.normalize(state_emb)

            # Get action
            action, _, _ = policy.get_action(state_emb, deterministic=True)

        action_cpu = {k: int(v.item()) for k, v in action.items()}

        if verbose and step % 50 == 0:
            print(f"  Step {step}: action_type={action_cpu.get('action_type', 'N/A')}, "
                  f"blocks_completed={env.metrics['blocks_completed']}")

        obs, reward, terminated, truncated, info = env.step(action_cpu)
        total_reward += reward

        if terminated or truncated:
            break

    throughput = env.metrics["blocks_completed"] / max(env.sim_time, 1.0)

    return {
        "total_reward": total_reward,
        "throughput": throughput,
        "blocks_completed": env.metrics["blocks_completed"],
        "breakdowns": env.metrics.get("breakdowns", 0),
        "sim_time": env.sim_time,
        "steps": step + 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Run trained policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config override (default: from checkpoint)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress")
    parser.add_argument("--log-db", action="store_true", help="Log to MES database")
    args = parser.parse_args()

    # Load checkpoint to get config
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})

    # Determine config file
    if args.config:
        config_path = args.config
    elif "config" in ckpt_args:
        config_path = ckpt_args["config"]
    else:
        config_path = "config/tiny_instance.yaml"

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Using config: {config_path}")

    # Load config and create environment
    cfg = load_config(config_path)
    env = ShipyardEnv(cfg)

    # Load model
    encoder, policy, normalizer, saved_metrics = load_checkpoint(
        args.checkpoint, env, args.device
    )
    encoder.eval()
    policy.eval()

    print(f"\nSaved metrics from training: {saved_metrics}")
    print(f"\nRunning {args.episodes} evaluation episodes...")
    print("=" * 50)

    all_metrics = []
    for ep in range(args.episodes):
        metrics = run_episode(env, encoder, policy, normalizer, args.device, verbose=args.verbose)
        all_metrics.append(metrics)
        print(f"Episode {ep+1}: throughput={metrics['throughput']:.4f}, "
              f"blocks={metrics['blocks_completed']}, reward={metrics['total_reward']:.1f}")

    # Summary
    print("=" * 50)
    avg_throughput = np.mean([m["throughput"] for m in all_metrics])
    avg_reward = np.mean([m["total_reward"] for m in all_metrics])
    avg_blocks = np.mean([m["blocks_completed"] for m in all_metrics])

    print(f"\nSummary ({args.episodes} episodes):")
    print(f"  Avg Throughput: {avg_throughput:.4f}")
    print(f"  Avg Reward: {avg_reward:.1f}")
    print(f"  Avg Blocks Completed: {avg_blocks:.1f}")

    # Compare with expert
    from baselines.rule_based import RuleBasedScheduler
    expert = RuleBasedScheduler()
    expert_throughput = 0.0
    for _ in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(1000):
            action = expert.decide(env)
            obs, reward, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        if env.sim_time > 0:
            expert_throughput += env.metrics["blocks_completed"] / env.sim_time
    expert_throughput /= args.episodes

    print(f"\n  Expert Throughput: {expert_throughput:.4f}")
    print(f"  Policy vs Expert: {100 * avg_throughput / expert_throughput:.1f}%")

    if args.log_db:
        print("\n[DB logging not yet implemented - dashboard shows simulation data]")


if __name__ == "__main__":
    main()
