#!/usr/bin/env python3
"""Live simulation mode for dashboard visualization.

Runs the shipyard simulation in real-time (or accelerated time) while
writing position snapshots to the database. The dashboard can then
show blocks, SPMTs, cranes, and ships moving in real-time.

Usage:
    # Run with random policy (for visualization only)
    python experiments/live_simulation.py --config config/small_instance.yaml --speed 10

    # Run with trained DAgger policy
    python experiments/live_simulation.py --config config/small_instance.yaml \
        --checkpoint data/checkpoints/dagger/dagger_final.pt --speed 5

    # Run HHI Ulsan shipyard with expert policy
    python experiments/live_simulation.py --config config/hhi_ulsan.yaml \
        --policy expert --speed 20
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.environment import ShipyardEnv
from src.simulation.shipyard_env import HHIShipyardEnv
from src.mes.database import (
    init_db, clear_db, log_position_snapshot, log_ships,
    log_goliath_cranes, log_hhi_blocks, log_metrics,
    log_health_snapshot, log_queue_depths, log_spmts, log_dry_docks,
    create_simulation_run, update_simulation_run,
)
from baselines.rule_based import RuleBasedScheduler


def load_config(path: str) -> dict:
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


def load_checkpoint(checkpoint_path: str, env, device: str = "cpu"):
    """Load a trained DAgger/BC checkpoint."""
    import torch
    from agent.gnn_encoder import HeterogeneousGNNEncoder
    from agent.policy import ActorCriticPolicy

    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", args.get("hidden-dim", 128))
    policy_hidden = args.get("policy_hidden", args.get("policy-hidden", 256))

    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=hidden_dim,
        num_layers=2,
    ).to(device)

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

    # Load weights
    if "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
    elif "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])

    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    elif "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])

    encoder.eval()
    policy.eval()

    # Load normalizer if available
    normalizer = None
    if "normalizer" in checkpoint and checkpoint["normalizer"] is not None:
        from experiments.train_dagger_ensemble import RunningNormalizer
        normalizer = RunningNormalizer(state_dim)
        normalizer.load_state_dict(checkpoint["normalizer"])

    return encoder, policy, normalizer


def run_live_simulation(
    config_path: str,
    checkpoint_path: str = None,
    policy_type: str = "expert",
    speed: float = 10.0,
    max_steps: int = 5000,
    log_interval: int = 1,
    verbose: bool = False,
    device: str = "cpu",
):
    """Run live simulation with database logging.

    Parameters
    ----------
    config_path : str
        Path to YAML config file
    checkpoint_path : str, optional
        Path to trained checkpoint (DAgger/BC)
    policy_type : str
        Policy type: "expert", "random", or "trained"
    speed : float
        Simulation speed multiplier (10 = 10x faster than real-time)
    max_steps : int
        Maximum simulation steps
    log_interval : int
        Log to database every N steps
    verbose : bool
        Print step-by-step progress
    device : str
        PyTorch device (cpu/mps/cuda)
    """
    import torch

    # Load config and create environment
    cfg = load_config(config_path)

    # Use HHIShipyardEnv for HHI configs, ShipyardEnv for standard configs
    is_hhi_config = (
        "hhi" in config_path.lower() or
        cfg.get("yard_type") == "hhi" or
        "outfitting_quays" in cfg  # HHI-specific key
    )

    if is_hhi_config:
        env = HHIShipyardEnv(cfg)
        print(f"Using HHI Shipyard Environment")
    else:
        env = ShipyardEnv(cfg)
        print(f"Using Standard Shipyard Environment")

    # Initialize database and enable environment DB logging
    init_db()
    # Note: Don't clear_db() to preserve historical runs for playback
    # clear_db()  # Commented out to preserve history
    env.db_logging_enabled = True  # Enable block event logging for Gantt chart

    # Create a new simulation run for tracking
    run_name = f"Live Sim - {os.path.basename(config_path).replace('.yaml', '')}"
    current_run_id = create_simulation_run(
        name=run_name,
        config_path=config_path,
        policy_type=policy_type,
    )
    print(f"Created simulation run #{current_run_id}")

    # Set up policy
    if checkpoint_path and os.path.exists(checkpoint_path):
        encoder, policy, normalizer = load_checkpoint(checkpoint_path, env, device)
        policy_type = "trained"
        print(f"Loaded trained policy from {checkpoint_path}")
    elif policy_type == "expert":
        expert = RuleBasedScheduler()
        encoder, policy, normalizer = None, None, None
        print("Using expert (rule-based) policy")
    else:
        encoder, policy, normalizer = None, None, None
        print("Using random policy")

    # Reset environment
    obs, info = env.reset()

    print(f"\nStarting live simulation:")
    print(f"  Config: {config_path}")
    print(f"  Policy: {policy_type}")
    print(f"  Speed: {speed}x")
    print(f"  Max steps: {max_steps}")
    print(f"  Log interval: every {log_interval} steps")
    print(f"\nOpen the dashboard to watch: python -m src.mes.app")
    print("=" * 60)

    # Calculate real-time delay between steps
    # If speed=10, we run 10 sim steps per real second
    real_delay = 1.0 / speed if speed > 0 else 0

    total_reward = 0.0
    start_time = time.time()

    for step in range(max_steps):
        # Get action
        if policy_type == "trained" and encoder is not None:
            graph_data = env.get_graph_data().to(device)
            with torch.no_grad():
                state_emb = encoder(graph_data)
                if normalizer is not None:
                    state_emb = normalizer.normalize(state_emb)
                action, _, _ = policy.get_action(state_emb, deterministic=True)
            action_dict = {k: int(v.item()) for k, v in action.items()}
        elif policy_type == "expert":
            action_dict = expert.decide(env)
        else:
            # Random policy - handle both HHI (n_goliath_cranes) and standard (n_cranes)
            n_cranes = getattr(env, 'n_goliath_cranes', getattr(env, 'n_cranes', 1))
            action_dict = {
                "action_type": np.random.randint(0, 4),
                "spmt_idx": np.random.randint(0, max(env.n_spmts, 1)),
                "request_idx": 0,
                "crane_idx": np.random.randint(0, max(n_cranes, 1)),
                "lift_idx": 0,
                "equipment_idx": 0,
            }

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_dict)
        total_reward += reward

        # Log to database
        if step % log_interval == 0:
            _log_state(env, step, current_run_id)

        # Print progress
        if verbose or step % 100 == 0:
            ships_delivered = env.metrics.get("ships_delivered", 0)
            # Use blocks_completed or blocks_erected (different env versions)
            blocks_erected = env.metrics.get("blocks_erected", env.metrics.get("blocks_completed", 0))
            print(f"Step {step:5d} | Time {env.sim_time:7.1f} | "
                  f"Blocks: {blocks_erected:3d} | Ships: {ships_delivered} | "
                  f"Reward: {total_reward:8.1f}")

        # Check termination
        if terminated or truncated:
            print(f"\nSimulation ended at step {step}")
            break

        # Real-time delay
        if real_delay > 0:
            time.sleep(real_delay)

    # Final logging
    _log_state(env, step, current_run_id)

    # Update simulation run with final statistics
    blocks_completed = env.metrics.get('blocks_erected', env.metrics.get('blocks_completed', 0))
    ships_delivered = env.metrics.get('ships_delivered', 0)
    update_simulation_run(
        run_id=current_run_id,
        total_steps=step + 1,
        blocks_completed=blocks_completed,
        ships_delivered=ships_delivered,
        total_reward=total_reward,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Simulation complete!")
    print(f"  Run ID: #{current_run_id}")
    print(f"  Steps: {step + 1}")
    print(f"  Sim time: {env.sim_time:.1f}")
    print(f"  Real time: {elapsed:.1f}s")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Blocks completed: {blocks_completed}")
    print(f"  Ships delivered: {ships_delivered}")


def _log_state(env, step: int, run_id: int = None):
    """Log current simulation state to database."""
    # Get entities - handle both HHIShipyardEnv and ShipyardEnv naming
    blocks = env.entities.get("blocks", [])
    spmts = env.entities.get("spmts", [])
    # HHI uses "goliath_cranes", standard uses "cranes"
    cranes = env.entities.get("goliath_cranes", env.entities.get("cranes", []))
    ships = env.entities.get("ships", [])

    # Log position history for playback
    log_position_snapshot(
        time=env.sim_time,
        blocks=blocks,
        spmts=spmts,
        cranes=cranes,
        ships=ships,
        run_id=run_id,
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

    # Log dry docks
    docks = env.entities.get("docks", [])
    if docks:
        log_dry_docks(docks)

    log_metrics(env.sim_time, env.metrics)

    # Log health history
    log_health_snapshot(env.sim_time, spmts, cranes)

    # Log queue depths
    log_queue_depths(
        env.sim_time,
        getattr(env, 'facility_queues', {}),
        getattr(env, 'facility_processing', {}),
    )


def main():
    parser = argparse.ArgumentParser(description="Live shipyard simulation for dashboard")
    parser.add_argument("--config", type=str, default="config/small_instance.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint")
    parser.add_argument("--policy", type=str, default="expert",
                        choices=["expert", "random", "trained"],
                        help="Policy type (expert, random, or trained)")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Simulation speed multiplier (0 = as fast as possible)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum simulation steps")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Log to database every N steps")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every step")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (cpu/mps/cuda)")
    args = parser.parse_args()

    run_live_simulation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        policy_type=args.policy,
        speed=args.speed,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        verbose=args.verbose,
        device=args.device,
    )


if __name__ == "__main__":
    main()
