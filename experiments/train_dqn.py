"""Train Double DQN agent for shipyard scheduling.

This script demonstrates training the Double DQN baseline agent with
prioritized experience replay and dueling architecture.

Usage:
    # Basic training
    python experiments/train_dqn.py --config config/small_instance.yaml --episodes 500

    # With wandb logging
    python experiments/train_dqn.py --config config/small_instance.yaml --wandb

    # Evaluation only (load checkpoint)
    python experiments/train_dqn.py --eval --checkpoint checkpoints/dqn_best.pt
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from simulation.environment import ShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder, create_encoder
from agent.dqn import DoubleDQNAgent
from utils.metrics import compute_kpis


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_dqn(args: argparse.Namespace) -> Dict[str, float]:
    """Train Double DQN agent.

    Args:
        args: Command line arguments.

    Returns:
        Final evaluation metrics.
    """
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Create environment
    env = ShipyardEnv(config)

    # Create encoder
    encoder = create_encoder(
        encoder_type=args.encoder,
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=args.hidden_dim,
    )

    # Create DQN agent
    agent = DoubleDQNAgent(
        encoder=encoder,
        state_dim=args.hidden_dim * 4,  # 4 node types
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        n_step=args.n_step,
        prioritized=args.prioritized,
        dueling=args.dueling,
        factorized=args.factorized,
        device=args.device,
    )

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"dqn_{args.encoder}",
            config=vars(args),
        )

    # Training loop
    best_on_time_rate = 0.0
    episode_rewards = []

    print(f"\nTraining DQN for {args.episodes} episodes...")
    print(f"  Encoder: {args.encoder}")
    print(f"  Prioritized replay: {args.prioritized}")
    print(f"  Dueling architecture: {args.dueling}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            # Get graph data and action mask
            graph_data = env.get_graph_data()
            mask = env.get_action_mask()

            # Select action
            action = agent.select_action(graph_data, mask, training=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            # Store transition
            next_graph = env.get_graph_data()
            from agent.action_masking import flatten_env_mask_to_policy_mask
            flat_mask = flatten_env_mask_to_policy_mask(
                mask, agent.n_spmts, agent.n_cranes, agent.max_requests
            )
            mask_array = np.concatenate([
                flat_mask["action_type"],
                flat_mask["spmt"],
                flat_mask["crane"],
                flat_mask["request"],
            ])
            agent.store_transition(graph_data, action, reward, next_graph, done, mask_array)

            # Update agent
            update_info = agent.update()

        episode_rewards.append(episode_reward)

        # Compute KPIs
        kpis = compute_kpis(env.metrics, env.sim_time)
        on_time_rate = kpis.get("on_time_rate", 0)

        # Track best model
        if on_time_rate > best_on_time_rate:
            best_on_time_rate = on_time_rate
            if args.save_best:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                agent.save_checkpoint(f"{args.checkpoint_dir}/dqn_best.pt")

        # Logging
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"On-time: {on_time_rate:5.1%} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer)}")

            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "episode": episode + 1,
                    "reward": episode_reward,
                    "avg_reward": avg_reward,
                    "on_time_rate": on_time_rate,
                    "epsilon": agent.epsilon,
                    "buffer_size": len(agent.buffer),
                    "throughput": kpis.get("throughput", 0),
                    "average_tardiness": kpis.get("average_tardiness", 0),
                })

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (10 episodes)")
    print("=" * 60)

    eval_metrics = evaluate_agent(agent, env, n_episodes=10)

    print(f"  On-time rate:    {eval_metrics['on_time_rate']:.1%}")
    print(f"  Avg tardiness:   {eval_metrics['average_tardiness']:.1f}")
    print(f"  Throughput:      {eval_metrics['throughput']:.3f}")
    print(f"  Best on-time:    {best_on_time_rate:.1%}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.log({"final/" + k: v for k, v in eval_metrics.items()})
        wandb.finish()

    return eval_metrics


def evaluate_agent(
    agent: DoubleDQNAgent,
    env: ShipyardEnv,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate trained agent.

    Args:
        agent: Trained DQN agent.
        env: Environment for evaluation.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dictionary of averaged metrics.
    """
    all_metrics = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            graph_data = env.get_graph_data()
            mask = env.get_action_mask()
            action = agent.select_action(graph_data, mask, training=False)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        all_metrics.append(compute_kpis(env.metrics, env.sim_time))

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Double DQN Agent")

    # Environment
    parser.add_argument("--config", type=str, default="config/small_instance.yaml",
                        help="Environment configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model
    parser.add_argument("--encoder", type=str, default="gat",
                        choices=["gat", "transformer", "temporal"],
                        help="GNN encoder type")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")

    # DQN hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=int, default=50000, help="Epsilon decay steps")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--n-step", type=int, default=3, help="N-step returns")

    # DQN variants
    parser.add_argument("--prioritized", action="store_true", default=True,
                        help="Use prioritized experience replay")
    parser.add_argument("--no-prioritized", action="store_false", dest="prioritized")
    parser.add_argument("--dueling", action="store_true", default=True,
                        help="Use dueling architecture")
    parser.add_argument("--no-dueling", action="store_false", dest="dueling")
    parser.add_argument("--factorized", action="store_true", default=True,
                        help="Use factorized Q-network")
    parser.add_argument("--no-factorized", action="store_false", dest="factorized")

    # Training
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

    # Checkpointing
    parser.add_argument("--save-best", action="store_true", default=True,
                        help="Save best model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load checkpoint for evaluation")
    parser.add_argument("--eval", action="store_true", help="Evaluation only mode")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="shipyard-dqn")
    parser.add_argument("--wandb-name", type=str, default=None)

    args = parser.parse_args()

    if args.eval and args.checkpoint:
        # Evaluation mode
        print(f"Loading checkpoint: {args.checkpoint}")
        config = load_config(args.config)
        env = ShipyardEnv(config)

        encoder = create_encoder(
            encoder_type=args.encoder,
            block_dim=env.block_features,
            spmt_dim=env.spmt_features,
            crane_dim=env.crane_features,
            facility_dim=env.facility_features,
            hidden_dim=args.hidden_dim,
        )

        agent = DoubleDQNAgent(
            encoder=encoder,
            state_dim=args.hidden_dim * 4,
            n_spmts=env.n_spmts,
            n_cranes=env.n_cranes,
            max_requests=env.n_blocks,
            device=args.device,
        )
        agent.load_checkpoint(args.checkpoint)

        metrics = evaluate_agent(agent, env, n_episodes=20)
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        # Training mode
        train_dqn(args)


if __name__ == "__main__":
    main()
