# User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Simulation Environment](#simulation-environment)
5. [Dashboard Guide](#dashboard-guide)
6. [Training & Evaluation](#training--evaluation)
7. [Configuration Reference](#configuration-reference)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

The Shipyard Scheduling System is a reinforcement learning framework for optimizing block scheduling in shipyard operations. It combines discrete-event simulation with graph neural network (GNN) policies to learn scheduling decisions that minimize tardiness, reduce equipment breakdowns, and maximize throughput.

### Key Capabilities

- **HHI Ulsan Shipyard**: Models HD Hyundai Heavy Industries LNG carrier production with 10 dry docks, 9 Goliath cranes
- **Health-Aware Scheduling**: Integrates equipment degradation models for predictive maintenance
- **Interactive Visualization**: Real-time dashboard with shipyard maps, dependency graphs, and playback
- **Flexible Training**: Supports curriculum learning, hyperparameter search, and baseline comparisons

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SHIPYARD SCHEDULING SYSTEM                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   SIMULATION    │    │      AGENT      │    │    DASHBOARD    │ │
│  │                 │    │                 │    │                 │ │
│  │ • HHIShipyardEnv│◄──►│ • GNN Encoder   │    │ • Dash App      │ │
│  │ • Entities      │    │ • Actor-Critic  │    │ • Plotly Maps   │ │
│  │ • Degradation   │    │ • PPO Trainer   │    │ • Real-time     │ │
│  │ • Calibration   │    │ • Action Mask   │    │ • Playback      │ │
│  └────────┬────────┘    └─────────────────┘    └────────┬────────┘ │
│           │                                             │          │
│           ▼                                             ▼          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                        SQLite Database                        │ │
│  │  metrics | blocks | spmts | cranes | position_history | ...   │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### HHI Ulsan Production Flow

```
STEEL CUTTING → PART FABRICATION → PANEL ASSEMBLY → BLOCK ASSEMBLY
       → BLOCK OUTFITTING → PAINTING → PRE-ERECTION
       → ERECTION (Dry Dock) → QUAY OUTFITTING → SEA TRIALS → DELIVERY
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd shipyard_scheduling
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv shipyard
   source shipyard/bin/activate  # Linux/macOS
   # or: shipyard\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Initialize the database:**
   ```bash
   python -c "from mes.database import init_db; init_db()"
   ```

5. **Verify installation:**
   ```bash
   python -m pytest tests/ -v
   ```

---

## Simulation Environment

### Single-Yard Environment (`ShipyardEnv`)

The base environment models a single shipyard with sequential production stages:

```python
from simulation import ShipyardEnv

config = {
    "n_blocks": 50,
    "n_spmts": 6,
    "n_cranes": 2,
    "shipyard": {
        "facilities": [
            {"name": "cutting", "capacity": 4, "processing_time_mean": 10},
            {"name": "panel", "capacity": 3, "processing_time_mean": 15},
            # ...
        ]
    }
}

env = ShipyardEnv(config)
obs, info = env.reset()

# Take an action
action = {
    "action_type": 0,  # Dispatch SPMT
    "spmt_idx": 0,
    "request_idx": 0,
}
obs, reward, terminated, truncated, info = env.step(action)
```

**Action Types:**
| Type | Description |
|------|-------------|
| 0 | Dispatch SPMT to fulfill transport request |
| 1 | Dispatch crane to lift block to dock |
| 2 | Trigger preventive maintenance |
| 3 | Hold (no operation) |

### HHI Ulsan Environment (`HHIShipyardEnv`)

Production environment for HD Hyundai Heavy Industries LNG carrier production:

```python
from simulation.shipyard_env import HHIShipyardEnv
import yaml

# Load config
with open("config/hhi_ulsan.yaml") as f:
    cfg = yaml.safe_load(f)

env = HHIShipyardEnv(cfg)
obs, info = env.reset()

# Run with expert scheduler
from baselines.rule_based import RuleBasedScheduler
expert = RuleBasedScheduler()

for step in range(1000):
    action = expert.decide(env)
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        break
```

**HHI-Specific Features:**
- 11-stage Korean shipbuilding workflow
- 200 blocks per LNG carrier
- 9 Goliath cranes (109m tall, 900-ton capacity)
- 10 dry docks along Mipo Bay
- Ship status transitions: `in_block_production` → `in_erection` → `afloat` → `in_quay_outfitting` → `in_sea_trials` → `delivered`

**Running Live Simulation with Dashboard:**
```bash
# Terminal 1: Start dashboard
python -m src.mes.app

# Terminal 2: Run simulation with expert policy
python experiments/live_simulation.py \
    --config config/hhi_ulsan.yaml \
    --policy expert \
    --speed 10 \
    --max-steps 5000
```

### Entity Classes

**Block:**
```python
@dataclass
class Block:
    id: str                    # Unique identifier
    weight: float              # Weight in tons
    size: Tuple[float, float]  # Length, width in meters
    due_date: float            # Target completion time
    current_stage: ProductionStage
    status: BlockStatus
    location: str
    predecessors: List[str]    # Blocks that must complete first
```

**SPMT (Self-Propelled Modular Transporter):**
```python
@dataclass
class SPMT:
    id: str
    capacity: float = 500.0    # Max load in tons
    current_location: str
    status: SPMTStatus
    health_hydraulic: float    # 0-100 health
    health_tires: float
    health_engine: float
```


### Production Stages

**Single-Yard (`ProductionStage`):**
- CUTTING → PANEL → ASSEMBLY → OUTFITTING → PAINTING → PRE_ERECTION → DOCK

**Dual-Yard (`EBProductionStage`):**
- STEEL_PROCESSING → CYLINDER_FABRICATION → MODULE_OUTFITTING → SUPER_MODULE_ASSEMBLY → BARGE_LOADING → BARGE_TRANSIT → BARGE_UNLOADING → FINAL_ASSEMBLY → SYSTEMS_INTEGRATION → FLOAT_OFF

---

## Dashboard Guide

### Launching the Dashboard

```bash
python -m mes.app
```

Access at: http://localhost:8050

### Tab Reference

#### HHI Map (Default)
Interactive map of HD Hyundai Heavy Industries Ulsan shipyard showing all production zones, equipment, and block positions.

**Controls:**
- **Health Overlay**: Toggle to color equipment by health status
- **Playback**: Enable to scrub through historical simulation states

**Map Elements:**
- Colored rectangles: Facilities with queue indicators
- Diamond markers: SPMTs
- Triangle markers: Goliath cranes

#### Dependencies
Interactive block dependency graph:
- **Node colors**: Indicate block status (green=complete, blue=processing, gray=waiting)
- **Arrows**: Show predecessor relationships
- **Block filter**: Highlight specific block's dependency chain
- **Critical path**: Toggle to show longest dependency chain

#### Simulation Playback

1. Check "Playback: Enable" checkbox
2. Use controls:
   - **▶️ Play / ⏸️ Pause**: Auto-advance through snapshots
   - **⏪ Rewind**: Jump back 10% of timeline
   - **⏩ Forward**: Jump forward 10% of timeline
   - **🔴 Live**: Return to real-time data
3. Drag slider to scrub to specific time

#### Classic Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | KPI cards (blocks completed, breakdowns, OEE) with trend chart |
| **Blocks** | Sortable table of all blocks with status, stage, location |
| **Fleet** | SPMT table with health values and utilization heatmap |
| **Health** | Equipment degradation trends with failure thresholds |
| **Operations** | Gantt chart of block flow + facility queue depths |
| **KPIs** | Full metric trends (tardiness, empty travel, maintenance) |

### Real-Time Alerts

Alerts appear at the top of the dashboard:
- **⚠️ HEALTH**: Equipment component below 30% health
- **🔴 BREAKDOWN**: Equipment has failed
- **⚠️ CRANE**: Crane requires maintenance

---

## Training & Evaluation

### Training Commands

**Basic training:**
```bash
python experiments/train.py \
  --config config/small_instance.yaml \
  --epochs 10 \
  --steps 200 \
  --device cpu \
  --save data/checkpoints/
```

**With curriculum learning:**
```bash
python experiments/train.py \
  --config config/small_instance.yaml \
  --epochs 20 \
  --steps 500 \
  --curriculum \
  --save data/checkpoints/
```

### Evaluation Commands

**Evaluate trained agent:**
```bash
python experiments/evaluate.py \
  --config config/small_instance.yaml \
  --agent rl \
  --checkpoint data/checkpoints/checkpoint_epoch_10.pt \
  --episodes 5
```

**Evaluate baselines:**
```bash
# Rule-based (Earliest Due Date)
python experiments/evaluate.py --config config/small_instance.yaml --agent rule --episodes 5

# Myopic RL (random valid actions)
python experiments/evaluate.py --config config/small_instance.yaml --agent myopic --episodes 5

# Siloed optimization
python experiments/evaluate.py --config config/small_instance.yaml --agent siloed --episodes 5
```

### Hyperparameter Search

```bash
python experiments/hyperparameter_search.py \
  --config config/small_instance.yaml \
  --episodes 3 \
  --steps 200 \
  --method random \
  --n-trials 10
```

---

## Configuration Reference

### Instance Configurations

| Config | Blocks | SPMTs | Cranes | Max Time |
|--------|--------|-------|--------|----------|
| `small_instance.yaml` | 50 | 6 | 2 | 5,000 |
| `medium_instance.yaml` | 150 | 9 | 3 | 15,000 |
| `large_instance.yaml` | 300 | 12 | 4 | 30,000 |

### Key Configuration Parameters

```yaml
# Entity counts
n_blocks: 50
n_spmts: 6
n_cranes: 2

# Reward weights
reward_tardy: 10.0        # Penalty per unit tardiness
reward_empty_travel: 0.1  # Penalty per unit empty distance
reward_breakdown: 100.0   # Penalty per breakdown event
reward_maintenance: 5.0   # Cost of preventive maintenance
reward_completion: 1.0    # Reward per completed block

# Degradation model
degradation:
  drift: 0.05             # Base degradation rate
  volatility: 0.01        # Random component
  failure_threshold: 20.0 # Health level at failure

# PPO hyperparameters
ppo:
  lr: 0.0003
  gamma: 0.99
  clip_epsilon: 0.2
  gae_lambda: 0.95
  entropy_coef: 0.01
```

---

## API Reference

### Database Functions

```python
from mes.database import (
    init_db,                    # Initialize database tables
    clear_db,                   # Clear all data
    log_metrics,                # Log simulation metrics
    log_entities,               # Log block/SPMT/crane states
    log_position_snapshot,      # Log positions for playback
    fetch_position_at_time,     # Get historical state
    fetch_playback_timeline,    # Get min/max timestamps
    fetch_query,                # Execute arbitrary SQL
)
```

### Environment Methods

```python
# Reset environment
obs, info = env.reset(seed=42)

# Take action
obs, reward, terminated, truncated, info = env.step(action)

# Get valid action mask
mask = env.get_action_mask()

# Get graph representation (for GNN)
graph_data = env.get_graph_data()

# Manual database logging
env.db_logging_enabled = True
env.log_state_to_db()
```

### Dashboard Customization

Add new tabs by modifying:
1. `mes/layouts.py` - Add layout function
2. `mes/callbacks.py` - Add callback functions
3. `mes/app.py` - Register tab in `render_tab()` callback

---

## Troubleshooting

### Common Issues

**"No data available" in dashboard:**
- Ensure training ran with database logging enabled
- Check `shipyard.db` exists: `ls -la shipyard.db`
- Initialize database: `python -c "from mes.database import init_db; init_db()"`

**Import errors:**
- Activate virtual environment: `source shipyard/bin/activate`
- Install package: `pip install -e .`

**Playback shows "No data":**
- Position history requires simulation to run at least 50 time steps
- Ensure `env.db_logging_enabled = True`

**Training is slow:**
- Use GPU: `--device cuda`
- Reduce steps per epoch: `--steps 100`
- Use smaller instance: `--config config/small_instance.yaml`

**Dashboard performance issues:**
- Disable auto-refresh when not needed
- Reduce time range slider to limit data
- Close playback mode when not in use

**Database locked errors (sqlite3.OperationalError):**
- The database now uses WAL mode with 30s timeout for concurrent access
- If issues persist, restart both simulation and dashboard
- Check for zombie processes: `ps aux | grep python | grep shipyard`

**Massive negative reward in simulation:**
- This was a known bug with tardiness accumulation (fixed)
- Old code: accumulated `(sim_time - due_date) * dt` per block per step (exponential growth)
- New code: accumulates `dt` per tardy block per step (linear growth)
- Expected reward should be positive (+500 to +2000 for successful runs)

**Ships not appearing on map:**
- Ships are only visible on the map when status is: `in_erection`, `afloat`, `in_quay_outfitting`, `in_sea_trials`, or `delivered`
- Ships in `in_block_production` status are not rendered (blocks are still being processed)
- Check ship status in the Ships tab

**DataTable column errors:**
- If you see "Invalid argument 'data[0].X' passed into DataTable", the query columns don't match layout columns
- Ensure SELECT aliases match the column `id` values in `layouts.py`

### Getting Help

- Check [README.md](README.md) for architecture overview
- Review [docs/FORMULATION.md](docs/FORMULATION.md) for MDP specification
- Run tests: `python -m pytest tests/ -v --tb=short`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025 | Added HHI Ulsan shipyard model, playback, dependency graphs |
| 1.0.0 | 2024 | Initial release with single-yard scheduling |
