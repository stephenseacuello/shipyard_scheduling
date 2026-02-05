# Health-Aware Shipyard Block Scheduling via Graph Reinforcement Learning

A reinforcement learning framework for integrated shipyard scheduling with predictive health management. This project combines discrete-event simulation of shipyard operations with a heterogeneous graph neural network (GNN) agent trained using Proximal Policy Optimization (PPO).

## Problem Overview

Shipyard block scheduling is a complex combinatorial optimization problem involving:
- **Production scheduling**: Routing ship blocks through manufacturing stages
- **Vehicle routing**: Dispatching Self-Propelled Modular Transporters (SPMTs) for block transport
- **Crane scheduling**: Coordinating crane lifts for dock placement with precedence constraints
- **Predictive maintenance**: Managing equipment health to minimize unplanned breakdowns

We formulate this as a Markov Decision Process (MDP) where the state is represented as a heterogeneous graph with block, SPMT, crane, and facility nodes. The agent learns to make scheduling decisions that minimize tardiness and breakdowns while maximizing throughput.

**Key Features:**
- Heterogeneous GNN encoder with graph attention for relational reasoning
- Hierarchical action masking ensuring only valid actions are sampled
- Wiener process degradation model with load-dependent drift
- Health-aware reward shaping for proactive maintenance decisions
- Curriculum learning for progressive difficulty scaling
- **Dual-yard workflow** modeling Electric Boat Quonset-Groton operations
- **Interactive shipyard maps** with split-screen views and equipment health overlays
- **Simulation playback** with timeline scrubber for historical analysis

See [docs/FORMULATION.md](docs/FORMULATION.md) for the formal MDP definition.

## Related Work

This project builds on research in:
- Graph neural networks for combinatorial optimization
- Reinforcement learning for scheduling problems
- Condition-based maintenance and prognostics
- Shipyard production planning

See [docs/RELATED_WORK.md](docs/RELATED_WORK.md) for a comprehensive literature review.

## Algorithm

The GNN-PPO algorithm combines:
1. **Heterogeneous GNN Encoder**: Multi-relational message passing over 4 node types and 8 edge types
2. **Hierarchical Action Masking**: Ensures valid actions per equipment status and constraints
3. **PPO with GAE**: Stable policy optimization with generalized advantage estimation

See [docs/ALGORITHMS.md](docs/ALGORITHMS.md) for detailed pseudocode.

## Project Structure

```
config/                     # YAML configurations (small/medium/large instances)
  eb_dual_yard.yaml         # Electric Boat dual-yard configuration
experiments/                # Runnable scripts: train, evaluate, hyperparameter search
src/
  simulation/               # Gym environment, entities, degradation model, shipyard graph
    environment.py          # Single-yard Gymnasium environment
    dual_yard_env.py        # Dual-yard environment (Quonset/Groton)
    entities.py             # Block, SPMT, Crane, Barge, SuperModule classes
    shipyard.py             # ShipyardGraph and DualShipyardGraph
  agent/                    # GNN encoder, actor-critic policy, PPO trainer
  baselines/                # Rule-based, myopic RL, siloed optimization schedulers
  phm/                      # Health model, RUL estimator, feature engineering
  mes/                      # Dash web dashboard
    app.py                  # Main dashboard application
    layouts.py              # Tab layouts (overview, maps, dependencies)
    callbacks.py            # Interactive callbacks with playback support
    map_builder.py          # Quonset/Groton map visualization builders
    map_coordinates.py      # Facility coordinate definitions
    dependency_graph.py     # Block dependency visualization
    database.py             # SQLite helpers with position history
  utils/                    # Metrics, logging, visualization, graph utilities
tests/                      # Unit and integration tests
```

## Setup

Activate the virtual environment and install the package in development mode:

```bash
source shipyard/bin/activate
pip install -e .
```

All commands below assume the `shipyard` venv is active.

## Manual Commands

### Training

Train the RL agent on the small instance (50 blocks, 6 SPMTs, 2 cranes):

```bash
python experiments/train.py \
  --config config/small_instance.yaml \
  --epochs 10 \
  --steps 200 \
  --device cpu \
  --save data/checkpoints/
```

| Flag | Description |
|------|-------------|
| `--config` | YAML config file (`config/small_instance.yaml`, `medium_instance.yaml`, `large_instance.yaml`) |
| `--epochs` | Number of training epochs |
| `--steps` | Environment steps per epoch (rollout length) |
| `--device` | `cpu` or `cuda` |
| `--save` | Directory for checkpoints and metrics CSV |

### Evaluation

Evaluate a trained RL agent:

```bash
python experiments/evaluate.py \
  --config config/small_instance.yaml \
  --agent rl \
  --checkpoint data/checkpoints/checkpoint_epoch_10.pt \
  --episodes 5 \
  --device cpu
```

Evaluate baseline schedulers (no checkpoint needed):

```bash
# Rule-based (Earliest Due Date heuristic)
python experiments/evaluate.py --config config/small_instance.yaml --agent rule --episodes 5

# Myopic RL (random valid actions)
python experiments/evaluate.py --config config/small_instance.yaml --agent myopic --episodes 5

# Siloed optimization (independent per-facility)
python experiments/evaluate.py --config config/small_instance.yaml --agent siloed --episodes 5
```

Train with curriculum learning (progressive difficulty):

```bash
python experiments/train.py \
  --config config/small_instance.yaml \
  --epochs 10 \
  --steps 200 \
  --device cpu \
  --curriculum \
  --save data/checkpoints/
```

### Hyperparameter Search

Random search (default 10 trials):

```bash
python experiments/hyperparameter_search.py \
  --config config/small_instance.yaml \
  --episodes 3 \
  --steps 200 \
  --method random \
  --n-trials 10
```

Grid search:

```bash
python experiments/hyperparameter_search.py \
  --config config/small_instance.yaml \
  --episodes 3 \
  --steps 200 \
  --method grid
```

### Ablation Studies

Run systematic ablation over GNN vs MLP, masking, PHM, and curriculum:

```bash
python experiments/ablation.py \
  --config config/small_instance.yaml \
  --epochs 3 \
  --steps 100
```

### Dashboard

Launch the MES monitoring dashboard:

```bash
python -m mes.app
```

Available at `http://localhost:8050/`. The dashboard has multiple tabs:

**Dual-Yard Maps (New):**
- **Dual View**: Split-screen showing Quonset Point and Groton facilities side-by-side with barge transit visualization
- **Quonset**: Detailed map of EB-Quonset (RI) with steel processing, AFC, Building 9, and pier areas
- **Groton**: Detailed map of EB-Groton (CT) with land-level construction, Building 600, and graving dock
- **Dependencies**: Interactive block dependency graph with critical path highlighting

**Map Features:**
- Toggle health overlay to color-code equipment by condition (green/yellow/red)
- Hover tooltips showing equipment status, health percentages, and current loads
- Barge transit progress visualization between yards
- Queue depth indicators at each facility

**Simulation Playback:**
- Enable playback mode to scrub through historical simulation states
- Play/pause, rewind, fast-forward controls
- Timeline slider for precise navigation
- "Live" button to return to real-time data

**Classic Tabs:**
- **Overview**: KPI cards (blocks completed, breakdowns, utilization, OEE) + trend chart
- **Blocks**: Status table with stage, location, completion %, due date
- **Fleet**: SPMT table with per-component health values + utilization chart
- **Health**: Degradation trend lines over time with failure/PM thresholds, equipment filter dropdown, RUL predictions table
- **Operations**: Gantt chart showing block flow through production stages + facility queue depth chart
- **KPIs**: Full KPI trend lines (completed, breakdowns, maintenance, empty travel)

**Alerts Banner:**
Real-time alerts appear at the top of the dashboard for equipment health warnings and breakdowns.

The dashboard reads from `shipyard.db`, which is automatically populated during training and evaluation (disable with `--no-db-log`).

### Tests

```bash
python -m pytest tests/ -v
```

## Configuration

Configs use YAML with inheritance. Instance configs (e.g. `small_instance.yaml`) inherit from `config/default.yaml` and override specific fields:

- **small**: 50 blocks, 6 SPMTs, 2 cranes, max 5,000 time steps
- **medium**: 150 blocks, 9 SPMTs, 3 cranes, max 15,000 time steps
- **large**: 300 blocks, 12 SPMTs, 4 cranes, max 30,000 time steps

Key parameters in `default.yaml`:
- `shipyard.facilities`: production stages with processing times and capacities
- `reward_*`: reward weight coefficients (tardiness, breakdown, completion, maintenance, empty travel)
- `n_blocks`, `n_spmts`, `n_cranes`: entity counts
- `degradation`: Wiener process parameters (drift, volatility, failure threshold)
- `ppo`: PPO hyperparameters (lr, gamma, clip_epsilon, etc.)
- `curriculum.milestones`: curriculum learning difficulty progression

### Dual-Yard Configuration

For Electric Boat dual-yard operations, use `config/eb_dual_yard.yaml`:

```yaml
dual_yard:
  quonset:
    facilities:
      - name: steel_processing
      - name: afc_facility
      - name: bldg_9a / bldg_9b / bldg_9c
      - name: super_module_assembly
      - name: quonset_pier
  groton:
    facilities:
      - name: groton_pier
      - name: land_level_construction
      - name: building_600
      - name: graving_dock
  transport:
    transit_time_hours: 36.0    # Quonset -> Groton
    return_time_hours: 30.0     # Groton -> Quonset (empty)
    barge_capacity: 2           # Super modules per trip
```

Use `DualShipyardEnv` instead of `ShipyardEnv` for dual-yard training:

```python
from simulation import DualShipyardEnv

env = DualShipyardEnv(config, render_mode="human")
env.db_logging_enabled = True  # Enable dashboard integration
```

## Architecture

### Simulation
- **Environment** (`simulation/environment.py`): Single-yard Gymnasium env wrapping the shipyard simulation. Hierarchical action space (dispatch SPMT, dispatch crane, trigger maintenance, hold).
- **Dual-Yard Environment** (`simulation/dual_yard_env.py`): Extended environment modeling Electric Boat's Quonset-Groton workflow with barge transport. 6 action types including barge load/unload.
- **Entities** (`simulation/entities.py`): Block, SPMT, Crane, Barge, SuperModule classes with EBProductionStage enum for 10-stage dual-yard workflow.
- **Shipyard Graph** (`simulation/shipyard.py`): ShipyardGraph for single-yard and DualShipyardGraph for modeling connected yards with BargeRoute.
- **Degradation** (`simulation/degradation.py`): Wiener process health model with load-dependent drift and stochastic noise.

### Agent
- **GNN Encoder** (`agent/gnn_encoder.py`): Heterogeneous graph encoder with GAT message-passing layers over block/SPMT/crane/facility nodes.
- **Policy** (`agent/policy.py`): Actor-critic with shared MLP trunk and 6 categorical action heads.
- **PPO Trainer** (`agent/ppo.py`): Collects rollouts, computes GAE advantages, performs clipped PPO updates.
- **Action Masking** (`agent/action_masking.py`): Hierarchical masking ensuring only valid actions are sampled per action type.
- **Curriculum Learning** (`agent/curriculum.py`): Progressive difficulty scheduler adjusting block count and time horizon.

### Supporting Modules
- **Baselines** (`baselines/`): Rule-based (EDD + nearest vehicle), myopic RL, and siloed optimization (independent production/routing/maintenance).
- **PHM** (`phm/`): Feature engineering (statistical, trending, frequency), RUL estimation.

### MES Dashboard
- **App** (`mes/app.py`): Main Dash application with tabbed interface.
- **Layouts** (`mes/layouts.py`): Tab layouts including dual-yard maps, dependencies, and playback controls.
- **Callbacks** (`mes/callbacks.py`): Interactive callbacks with simulation playback support.
- **Map Builder** (`mes/map_builder.py`): Plotly figure builders for Quonset, Groton, and transit visualizations.
- **Dependency Graph** (`mes/dependency_graph.py`): Block dependency visualization with critical path highlighting.
- **Database** (`mes/database.py`): SQLite helpers with position_history table for playback.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/FORMULATION.md](docs/FORMULATION.md) | Formal MDP definition with state/action spaces, reward function |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | Pseudocode for GNN-PPO and key algorithms |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | Literature review with 30+ references |
| [REPRODUCTION.md](REPRODUCTION.md) | Step-by-step guide to reproduce results |
| [RESULTS.md](RESULTS.md) | Experimental results and analysis |

## Reproducing Results

For full reproduction instructions, see [REPRODUCTION.md](REPRODUCTION.md).

Quick validation:
```bash
# Run tests
python -m pytest tests/ -v

# Quick training (5 min)
python experiments/train.py --config config/small_instance.yaml --epochs 3 --steps 100 --seed 42 --no-db-log

# Evaluate baseline
python experiments/evaluate.py --config config/small_instance.yaml --agent rule --episodes 5 --seed 42 --no-db-log
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shipyard_gnn_ppo_2025,
  title={Health-Aware Shipyard Block Scheduling via Graph Reinforcement Learning},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

This project is developed for academic research purposes.
