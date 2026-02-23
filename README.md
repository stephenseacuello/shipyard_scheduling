# Health-Aware LNG Carrier Block Scheduling via Graph Reinforcement Learning

A reinforcement learning framework for integrated shipyard scheduling with predictive health management. This project combines discrete-event simulation of shipyard operations with a heterogeneous graph neural network (GNN) agent trained using Proximal Policy Optimization (PPO).

**Case Study: HD Hyundai Heavy Industries (HHI) Ulsan Shipyard**

This project models LNG carrier production at the world's largest shipyard located in Ulsan, South Korea:
- **1,780 acres** of shipbuilding facilities along Mipo Bay
- **10 dry docks** (260m - 490m length)
- **9 Goliath cranes** (109m tall, up to 900 tons capacity)
- **200 blocks per ship** with 37.5% curved blocks
- **11-stage Korean shipbuilding workflow** from steel cutting to delivery

## Problem Overview

LNG carrier block scheduling is a complex combinatorial optimization problem involving:
- **Production scheduling**: Routing 200 blocks through 11 manufacturing stages
- **SPMT routing**: Dispatching Self-Propelled Modular Transporters for block transport
- **Goliath crane scheduling**: Coordinating erection lifts with precedence constraints
- **Predictive maintenance**: Managing equipment health to minimize unplanned breakdowns

We formulate this as a Markov Decision Process (MDP) where the state is represented as a heterogeneous graph with 6 node types: blocks, SPMTs, Goliath cranes, ships, dry docks, and facilities. The agent learns to make scheduling decisions that minimize tardiness and breakdowns while maximizing throughput.

**Key Features:**
- Heterogeneous GNN encoder with graph attention for 6 node types
- Hierarchical action masking ensuring only valid actions are sampled
- Wiener process degradation model with load-dependent drift (3 crane components: hoist, trolley, gantry)
- Health-aware reward shaping for proactive maintenance decisions
- Curriculum learning for progressive difficulty scaling
- **HHI Ulsan shipyard** modeling with 10 dry docks and 9 Goliath cranes
- **Interactive Leaflet maps** with Mipo Bay satellite overlay
- **Simulation playback** with timeline scrubber for historical analysis

**Research-Grade RL Algorithms:**
- PPO with entropy annealing (linear, exponential, cosine schedules)
- SAC-style adaptive entropy tuning (automatic temperature adjustment)
- Differential learning rates (encoder 0.1x policy rate for stable representations)
- Soft Actor-Critic (SAC) with automatic temperature tuning
- PPO-LSTM for temporal memory and sequence dependencies
- Multi-Agent PPO (MAPPO) for coordinated SPMT scheduling
- Hierarchical RL with options framework (macro-actions)

**Imitation Learning (Best Performing):**
- **DAgger (Dataset Aggregation)** - Iterative imitation learning that addresses distribution mismatch; achieves **103% of expert throughput**
- **Behavioral Cloning (BC)** - Pure supervised learning from expert demonstrations; achieves **86.5% of expert throughput**
- **DAgger Ensemble** - Multiple DAgger policies with majority voting for robust decisions
- Feature normalization with running statistics for stable learning
- Beta-annealing schedule (expert → policy) for progressive autonomy

**Advanced GNN Features:**
- Upgraded architecture: 4-layer encoder (256 dim), 512-dim policy network
- Edge-aware encoding (travel time, capacity, urgency features)
- Temporal attention mechanism for due-date-aware scheduling
- Hierarchical pooling (zone → global aggregation)
- Sparse edge construction for scalability

**Training Stabilization:**
- Softened action masking (-20 logit penalty for exploration)
- Reward clipping ([-10, +2]) for gradient stability
- Reduced domain randomization variance (±20% degradation rate)
- Entropy > 0.1 sustained throughout training
- **Incremental tardiness calculation** - prevents reward explosion (was accumulating `sim_time - due_date` every step; now adds `dt` per tardy block)
- **SQLite WAL mode** with 30s timeout for concurrent dashboard access

**Multi-Objective Optimization:**
- Weighted sum, Chebyshev, and hypernetwork scalarization
- Constraint-based and hypervolume-aware reward shaping
- Pareto archive for non-dominated solutions

**OR Baselines:**
- Priority dispatch rules (EDD, SPT, CR, WSJF, FIFO)
- Genetic Algorithm scheduler with order crossover
- Rolling Horizon MPC with CP-SAT optimization
- MIP and CP-SAT exact solvers

**Robustness:**
- Domain randomization for sim-to-real transfer
- Adversarial training with state perturbations
- Noise injection for robust policies

**Experiment Tracking:**
- Weights & Biases (wandb) integration
- Bayesian hyperparameter sweeps

See [docs/FORMULATION.md](docs/FORMULATION.md) for the formal MDP definition.

## Production Stages (Korean Shipbuilding)

| Stage | Name | Description | Duration |
|-------|------|-------------|----------|
| 0 | STEEL_CUTTING | NC/plasma cutting of steel plates | 4-8 hrs |
| 1 | PART_FABRICATION | Marking, bending, edge preparation | 8-16 hrs |
| 2 | PANEL_ASSEMBLY | Flat/curved panel sub-assembly | 16-24 hrs |
| 3 | BLOCK_ASSEMBLY | 3D block construction (~300 tons) | 40-60 hrs |
| 4 | BLOCK_OUTFITTING | Piping, electrical, HVAC, insulation | 30-50 hrs |
| 5 | PAINTING | Block painting (pre-erection) | 12-20 hrs |
| 6 | PRE_ERECTION | Grand block staging/joining | 24-40 hrs |
| 7 | ERECTION | Goliath crane places block in dock | 8-16 hrs |
| 8 | QUAY_OUTFITTING | Post-launch systems installation | 100-200 hrs |
| 9 | SEA_TRIALS | Testing at sea | 48-72 hrs |
| 10 | DELIVERY | Handover to owner | - |

## Project Structure

```
config/                     # YAML configurations
  hhi_ulsan.yaml            # HD HHI Ulsan shipyard configuration
  small_instance.yaml       # Small test instance (50 blocks)
  medium_instance.yaml      # Medium instance (150 blocks)
  large_instance.yaml       # Large instance (300 blocks)
experiments/                # Runnable scripts
  train.py                  # Main PPO training script with wandb support
  train_dagger.py           # DAgger iterative imitation learning (RECOMMENDED)
  train_dagger_ensemble.py  # DAgger ensemble with majority voting
  train_bc.py               # Pure behavioral cloning
  train_sac.py              # Soft Actor-Critic training
  train_imitation.py        # BC + RL fine-tuning (deprecated)
  sweep_config.yaml         # PPO hyperparameter sweep
  sweep_bc.yaml             # Behavioral cloning sweep
  sweep_dagger.yaml         # DAgger hyperparameter sweep
  evaluate.py               # Evaluation script
src/
  simulation/               # Gym environment, entities, degradation model
    environment.py          # Gymnasium environment for HHI shipyard
    entities.py             # Block, SPMT, GoliathCrane, LNGCarrier, DryDock
    shipyard.py             # HHIShipyardGraph for Ulsan facility
    domain_randomization.py # Domain randomization for robust training
    noise_injection.py      # Observation/action noise injection
  agent/                    # RL algorithms and neural network components
    gnn_encoder.py          # Heterogeneous GNN encoders (GAT, HGT, Temporal)
    policy.py               # Actor-critic policy with hierarchical actions
    ppo.py                  # PPO trainer with entropy annealing
    sac.py                  # Soft Actor-Critic implementation
    recurrent_policy.py     # PPO-LSTM with temporal memory
    mappo.py                # Multi-Agent PPO (CTDE)
    hierarchical_rl.py      # Options framework for macro-actions
    mo_ppo.py               # Multi-objective PPO with Pareto archive
    entropy_tuning.py       # Adaptive entropy (SAC-style)
    reward_shaping.py       # Potential-based reward shaping
    adversarial.py          # Adversarial training
    action_masking.py       # Hierarchical action masking utilities
  baselines/                # OR and heuristic baselines
    rule_based.py           # EDD heuristic scheduler
    priority_rules.py       # Priority dispatch rules (SPT, CR, WSJF, etc.)
    ga_scheduler.py         # Genetic Algorithm scheduler
    mpc_scheduler.py        # Rolling Horizon MPC
    mip_scheduler.py        # Mixed Integer Programming (Gurobi/OR-Tools)
    cp_scheduler.py         # Constraint Programming (CP-SAT)
    siloed_opt.py           # Siloed optimization baseline
  phm/                      # Predictive Health Management
    health_model.py         # Equipment degradation modeling
    rul_estimator.py        # Remaining Useful Life prediction
  mes/                      # Dash web dashboard
    app.py                  # Main dashboard application
    layouts.py              # Tab layouts (overview, maps, ships, docks)
    callbacks.py            # Interactive callbacks with playback support
    map_builder.py          # HHI Ulsan map visualization
    map_builder_leaflet.py  # Leaflet-based geographic map (Mipo Bay)
    map_coordinates_geo.py  # HHI facility GPS coordinates
    database.py             # SQLite helpers with HHI-specific tables
  utils/                    # Metrics, logging, visualization, graph utilities
    metrics.py              # KPI computation (throughput, tardiness, OEE)
    logging.py              # CSV logging utilities
tests/                      # Unit and integration tests
  test_or_baselines.py      # Tests for OR baselines
  test_new_rl_components.py # Tests for new RL algorithms
```

## Setup

Activate the virtual environment and install the package in development mode:

```bash
source shipyard/bin/activate
pip install -e .
```

All commands below assume the `shipyard` venv is active.

## Training

Train the RL agent on the small instance (50 blocks, 6 SPMTs, 2 cranes):

```bash
python experiments/train.py \
  --config config/small_instance.yaml \
  --epochs 10 \
  --steps 200 \
  --device cpu \
  --save data/checkpoints/
```

Train on the full HHI Ulsan configuration (200 blocks, 9 Goliath cranes, 10 docks):

```bash
python experiments/train.py \
  --config config/hhi_ulsan.yaml \
  --epochs 50 \
  --steps 500 \
  --device cuda \
  --curriculum \
  --save data/checkpoints/hhi/
```

| Flag | Description |
|------|-------------|
| `--config` | YAML config file |
| `--epochs` | Number of training epochs |
| `--steps` | Environment steps per epoch (rollout length) |
| `--device` | `cpu` or `cuda` |
| `--curriculum` | Enable curriculum learning |
| `--save` | Directory for checkpoints and metrics CSV |
| `--wandb` | Enable Weights & Biases logging |
| `--wandb-project` | W&B project name |
| `--seed` | Random seed for reproducibility |
| `--hidden-dim` | GNN hidden dimension (default: 256) |
| `--policy-hidden` | Policy network hidden dim (default: 512) |
| `--num-layers` | Number of GNN layers (default: 4) |
| `--adaptive-entropy` | Enable SAC-style adaptive entropy tuning |
| `--encoder-lr-scale` | Encoder LR scale (default: 0.1, i.e., 10x slower) |

### Hyperparameter Sweeps with Wandb

Run Bayesian hyperparameter optimization:

```bash
# Initialize sweep
wandb sweep experiments/sweep_config.yaml

# Launch agent (run this on multiple machines for parallel sweeps)
wandb agent <sweep_id>
```

The sweep optimizes:
- Learning rate (log-uniform: 1e-5 to 1e-3)
- Entropy coefficient (log-uniform: 0.001 to 0.5)
- Epsilon-greedy (uniform: 0.05 to 0.3)
- Hidden dimensions (64, 128, 256)
- GNN layers (2, 3, 4)

### Alternative Training Algorithms

```bash
# Train with SAC
python experiments/train_sac.py --config config/hhi_ulsan.yaml

# Train with MAPPO (multi-agent)
python experiments/train_mappo.py --config config/hhi_ulsan.yaml

# Train with multi-objective PPO
python experiments/train_mo_ppo.py --config config/hhi_ulsan.yaml --scalarization chebyshev
```

### Imitation Learning (Recommended for Best Results)

DAgger consistently outperforms pure RL methods on this task due to the hierarchical action space complexity.

```bash
# DAgger (best single-model performance) - achieves >100% of expert throughput
python experiments/train_dagger.py \
  --config config/small_instance.yaml \
  --iterations 20 \
  --init-episodes 50 \
  --dagger-episodes 20 \
  --train-epochs 30

# DAgger Ensemble (most robust) - majority voting across N policies
python experiments/train_dagger_ensemble.py \
  --config config/small_instance.yaml \
  --n-ensemble 3 \
  --iterations 20 \
  --init-episodes 50

# Pure Behavioral Cloning (simpler, still effective)
python experiments/train_bc.py \
  --config config/small_instance.yaml \
  --epochs 100 \
  --demo-episodes 50
```

| Method | vs Expert | Notes |
|--------|-----------|-------|
| DAgger | **103.2%** | Learns to recover from mistakes |
| DAgger Ensemble | **~105%** | Variance reduction via voting |
| Pure BC | 86.5% | Simple, no iteration needed |
| PPO | 0% | Entropy collapse in hierarchical action space |
| SAC | ~28% | Better but still suffers |

**Why DAgger works better:**
1. **Distribution mismatch** - BC trains on expert states but tests on learner states; DAgger iteratively collects data from learner rollouts
2. **Expert labels** - Always queries the expert for the "correct" action, even when the learner drives
3. **Beta annealing** - Progressively shifts from expert (β=1.0) to learner (β=0.1) driving

## Evaluation

Evaluate a trained RL agent:

```bash
python experiments/evaluate.py \
  --config config/small_instance.yaml \
  --agent rl \
  --checkpoint data/checkpoints/checkpoint_epoch_10.pt \
  --episodes 5 \
  --device cpu
```

Evaluate baseline schedulers:

```bash
# Rule-based (Earliest Due Date heuristic)
python experiments/evaluate.py --config config/small_instance.yaml --agent rule --episodes 5

# Myopic RL (random valid actions)
python experiments/evaluate.py --config config/small_instance.yaml --agent myopic --episodes 5

# Siloed optimization (independent per-facility)
python experiments/evaluate.py --config config/small_instance.yaml --agent siloed --episodes 5
```

## Dashboard

Launch the MES monitoring dashboard:

```bash
python -m src.mes.app
```

Available at `http://localhost:8050/`. The dashboard has multiple tabs:

**HHI Ulsan Maps:**
- **HHI Map**: Interactive Plotly visualization of the Ulsan shipyard with 10 dry docks, 9 Goliath cranes, and facility zones
- **Leaflet Map**: Geographic satellite view of Mipo Bay with facility overlays
- **Ship Animation**: Watch LNG carriers swim away from quay during sea trials and delivery

**Ship & Dock Monitoring:**
- **Ships**: LNG carrier construction progress with block completion tracking
- **Ship Life Cycle**: Full 11-stage workflow from steel cutting to delivery
- **Docks**: 10 dry dock status with Goliath crane assignments

**Equipment Health:**
- **Fleet**: Goliath crane table with hoist/trolley/gantry health values
- **Health**: Degradation trend lines with failure/PM thresholds

**Playback Controls:**
- **Timeline scrubber**: Replay historical simulation data
- **Live mode**: Watch simulation in real-time

**Classic Tabs:**
- **Overview**: KPI cards (blocks completed, breakdowns, utilization) + trend chart
- **Blocks**: Status table with stage, location, completion %, due date
- **Operations**: Gantt chart showing block flow through production stages
- **GNN Graph**: Interactive Cytoscape visualization of the heterogeneous graph state

### Live Simulation Mode

Run the simulation with real-time database updates for dashboard visualization:

```bash
# Run with expert policy at 10x speed (watch blocks and ships move)
python experiments/live_simulation.py --config config/small_instance.yaml --policy expert --speed 10

# Run with trained DAgger checkpoint
python experiments/live_simulation.py --config config/small_instance.yaml \
  --checkpoint data/checkpoints/dagger/dagger_final.pt --speed 5

# Run as fast as possible (for data collection)
python experiments/live_simulation.py --config config/hhi_ulsan.yaml --speed 0 --max-steps 5000
```

Open the dashboard in another terminal to watch the simulation in real-time.

## Configuration

The main HHI Ulsan configuration is in `config/hhi_ulsan.yaml`:

```yaml
# Key parameters
n_ships: 8          # Concurrent LNG carriers
n_blocks: 200       # Blocks per ship

# 10 Dry Docks
dry_docks:
  - name: dock_1    # 490m x 115m, VLCC/LNG capable
    cranes: [GC01, GC02]
  - name: dock_2    # 400m x 80m
    cranes: [GC03]
  # ... (10 total)

# 9 Goliath Cranes
goliath_cranes:
  - id: GC01
    capacity_tons: 900
    height_m: 109
  # ... (9 total)

# Reward weights
reward_tardy: 10.0
reward_breakdown: 100.0
reward_erection: 10.0
reward_ship_delivery: 500.0
```

## Architecture

### Simulation
- **Environment** (`simulation/environment.py`): Gymnasium env with 4 action types (SPMT dispatch, Goliath crane dispatch, maintenance, hold)
- **Entities** (`simulation/entities.py`): HHI-specific classes including GoliathCrane (3 health components), LNGCarrier, DryDock, BlockType enum
- **HHI Shipyard Graph** (`simulation/shipyard.py`): HHIShipyardGraph modeling Ulsan facility zones and transport network

### Agent
- **GNN Encoder** (`agent/gnn_encoder.py`): Heterogeneous graph encoder with 6 node types (blocks, SPMTs, Goliath cranes, ships, docks, facilities)
- **Policy** (`agent/policy.py`): Actor-critic with hierarchical action heads
- **PPO Trainer** (`agent/ppo.py`): Proximal Policy Optimization with GAE

### MES Dashboard
- **Map Builder** (`mes/map_builder.py`): Plotly visualizations for HHI Ulsan shipyard
- **Leaflet Map** (`mes/map_builder_leaflet.py`): Geographic satellite view of Mipo Bay (35.5067°N, 129.4133°E)
- **Database** (`mes/database.py`): SQLite with HHI-specific tables (ships, goliath_cranes, dry_docks)

## Tests

```bash
python -m pytest tests/ -v
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/FORMULATION.md](docs/FORMULATION.md) | Formal MDP definition with state/action spaces |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | Pseudocode for GNN-PPO algorithm |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | Literature review |
| [REPRODUCTION.md](REPRODUCTION.md) | Step-by-step reproduction guide |
| [RESULTS.md](RESULTS.md) | Experimental results |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hhi_gnn_ppo_2025,
  title={Health-Aware Block Scheduling for Large-Scale LNG Carrier Production via Graph Reinforcement Learning: A Case Study at HD Hyundai Heavy Industries},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

This project is developed for academic research purposes.
