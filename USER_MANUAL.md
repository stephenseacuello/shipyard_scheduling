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

- **Dual-Yard Workflow**: Models Electric Boat's Quonset Point (RI) to Groton (CT) submarine production pipeline
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
│  │ • ShipyardEnv   │◄──►│ • GNN Encoder   │    │ • Dash App      │ │
│  │ • DualShipyardEnv│   │ • Actor-Critic  │    │ • Plotly Maps   │ │
│  │ • Entities      │    │ • PPO Trainer   │    │ • Real-time     │ │
│  │ • Degradation   │    │ • Action Mask   │    │ • Playback      │ │
│  └────────┬────────┘    └─────────────────┘    └────────┬────────┘ │
│           │                                             │          │
│           ▼                                             ▼          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                        SQLite Database                        │ │
│  │  metrics | blocks | spmts | cranes | position_history | ...   │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Dual-Yard Workflow

The Electric Boat dual-yard workflow models submarine production across two facilities:

```
QUONSET POINT (RI)                    GROTON (CT)
─────────────────                     ────────────
Steel Processing                      Barge Unloading
       ↓                                    ↓
Cylinder Fabrication                  Land-Level Construction
       ↓                                    ↓
Module Outfitting (Bldg 9)           Building 600 Systems
       ↓                                    ↓
Super-Module Assembly                 Graving Dock Float-Off
       ↓
Barge Loading
       ↓
═══════════════════════════════════════════════════════════
       HOLLAND BARGE TRANSIT (~36 hours)
═══════════════════════════════════════════════════════════
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

### Dual-Yard Environment (`DualShipyardEnv`)

Extended environment for Electric Boat dual-yard operations:

```python
from simulation import DualShipyardEnv

config = {
    "n_blocks": 20,
    "n_super_modules": 6,
    "n_quonset_spmts": 4,
    "n_groton_spmts": 3,
    "n_quonset_cranes": 2,
    "n_groton_cranes": 2,
    "n_barges": 1,
    "dual_yard": {
        "quonset": {...},
        "groton": {...},
        "transport": {
            "transit_time_hours": 36.0,
            "return_time_hours": 30.0,
            "barge_capacity": 2,
        }
    }
}

env = DualShipyardEnv(config)
env.db_logging_enabled = True  # Enable dashboard integration
```

**Extended Action Types:**
| Type | Description |
|------|-------------|
| 0 | Dispatch SPMT (yard-specific) |
| 1 | Dispatch crane (yard-specific) |
| 2 | Trigger maintenance |
| 3 | Hold |
| 4 | Load barge / Start barge transit |
| 5 | Unload barge |

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

**Barge (Dual-Yard only):**
```python
@dataclass
class Barge:
    id: str
    capacity: int = 2          # Super modules per trip
    current_location: str      # "quonset_pier" or "groton_pier"
    status: BargeStatus
    cargo: List[str]           # Module IDs currently loaded
    transit_progress: float    # 0.0 to transit_time
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

#### Dual View (Default)
Split-screen showing both shipyards with barge transit visualization.

**Controls:**
- **Health Overlay**: Toggle to color equipment by health status
- **Playback**: Enable to scrub through historical simulation states

**Map Elements:**
- Colored rectangles: Facilities with queue indicators
- Diamond markers: SPMTs
- Triangle markers: Cranes
- Square marker: Holland barge

#### Quonset Map
Detailed view of EB-Quonset Point (RI) facilities:
- Steel Processing
- AFC Facility (Automated Fiber Composite)
- Building 9A/9B/9C (Module Outfitting)
- Super-Module Assembly Area
- Pier (Barge Loading)

#### Groton Map
Detailed view of EB-Groton (CT) facilities:
- Pier (Barge Unloading)
- Land-Level Construction Area
- Building 600 (Systems Integration)
- Graving Dock (Float-Off)
- Crane Rail with positioned cranes
- Dock Grid showing block placement positions

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

**Dual-yard training:**
```bash
python experiments/train.py \
  --config config/eb_dual_yard.yaml \
  --epochs 15 \
  --steps 300 \
  --dual-yard \
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
| `eb_dual_yard.yaml` | 20 | 7 (4+3) | 4 (2+2) | 20,000 |

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

### Dual-Yard Configuration

```yaml
dual_yard:
  quonset:
    facilities:
      - name: steel_processing
        processing_time_mean: 16.0
        processing_time_std: 4.0
        capacity: 4
      # ... more facilities
  groton:
    facilities:
      - name: groton_pier
        processing_time_mean: 4.0
        capacity: 2
      # ... more facilities
  transport:
    origin_pier: quonset_pier
    destination_pier: groton_pier
    transit_time_hours: 36.0
    return_time_hours: 30.0
    barge_capacity: 2
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

### Getting Help

- Check [README.md](README.md) for architecture overview
- Review [docs/FORMULATION.md](docs/FORMULATION.md) for MDP specification
- Run tests: `python -m pytest tests/ -v --tb=short`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025 | Added dual-yard environment, playback, dependency graphs |
| 1.0.0 | 2024 | Initial release with single-yard scheduling |
