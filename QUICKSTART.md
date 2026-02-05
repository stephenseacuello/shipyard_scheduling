# Quick Start Guide

Get the shipyard scheduling system running in 5 minutes.

## Prerequisites

- Python 3.10+
- Virtual environment set up

## 1. Activate Environment

```bash
cd /Users/stepheneacuello/Projects/shipyard_scheduling
source shipyard/bin/activate
```

## 2. Initialize Database

```bash
python -c "from mes.database import init_db; init_db()"
```

## 3. Run a Quick Training Session

**Single-yard mode (legacy):**
```bash
python experiments/train.py \
  --config config/small_instance.yaml \
  --epochs 3 \
  --steps 100 \
  --device cpu
```

**Dual-yard mode (Electric Boat):**
```bash
python experiments/train.py \
  --config config/eb_dual_yard.yaml \
  --epochs 3 \
  --steps 100 \
  --device cpu \
  --dual-yard
```

## 4. Launch the Dashboard

```bash
python -m mes.app
```

Open http://localhost:8050 in your browser.

## 5. Explore the Dashboard

| Tab | Description |
|-----|-------------|
| **Dual View** | Split-screen Quonset/Groton maps with barge transit |
| **Quonset** | Detailed EB-Quonset Point facility map |
| **Groton** | Detailed EB-Groton facility map |
| **Dependencies** | Block precedence constraint graph |
| **Graph** | GNN heterogeneous graph visualization |
| **Overview** | KPI cards and trends |
| **Blocks** | Block status table |
| **Fleet** | SPMT health and utilization |
| **Health** | Equipment degradation trends |
| **Operations** | Gantt chart and queue depths |
| **KPIs** | Full metric trends |

## Key Features to Try

### GNN Graph Visualization
1. Go to the **Graph** tab
2. View the heterogeneous graph used by the GNN encoder
3. Filter by yard (Quonset/Groton)
4. Toggle edge types (transport, lift, precedence)
5. Click nodes to see feature details

### Health Overlay
Toggle "Health Overlay" checkbox on any map to color-code equipment by health status (green=healthy, yellow=warning, red=critical).

### Simulation Playback
1. Check "Playback: Enable" on Dual View
2. Use timeline scrubber to explore historical states
3. Click "Live" to return to real-time data

### Dependency Graph
1. Go to Dependencies tab
2. Select a block from dropdown to highlight its dependency chain
3. Toggle "Show Critical Path" to see the longest dependency chain

---

## Advanced RL Algorithms

### Double DQN Training

Train with prioritized experience replay and dueling architecture:

```bash
# Basic DQN training
python experiments/train_dqn.py \
  --config config/small_instance.yaml \
  --episodes 500

# With Graph Transformer encoder
python experiments/train_dqn.py \
  --encoder transformer \
  --episodes 500

# With wandb logging
python experiments/train_dqn.py \
  --wandb \
  --wandb-project shipyard-dqn
```

### Multi-Objective PPO

Train policies that optimize multiple objectives simultaneously:

```bash
# Basic multi-objective training
python experiments/train_mo_ppo.py \
  --config config/small_instance.yaml \
  --epochs 200

# With Chebyshev scalarization
python experiments/train_mo_ppo.py \
  --scalarization chebyshev

# With adaptive weight sampling
python experiments/train_mo_ppo.py \
  --weight-sampling adaptive

# Plot Pareto front
python experiments/train_mo_ppo.py \
  --epochs 200 \
  --plot-pareto
```

**Objectives optimized:**
- Throughput (blocks completed per time)
- Tardiness (on-time delivery)
- Health (equipment preservation)
- Efficiency (resource utilization)

---

## GNN Encoder Options

Three encoder architectures are available:

| Encoder | Description | Use Case |
|---------|-------------|----------|
| `gat` | GAT-based heterogeneous GNN | Default, good balance |
| `transformer` | Graph Transformer (HGT-style) | Best performance, slower |
| `temporal` | GNN with GRU temporal tracking | For sequential decisions |

```bash
# Use Graph Transformer
python experiments/train.py --encoder transformer

# Use Temporal GNN
python experiments/train.py --encoder temporal
```

---

## Experiment Tracking with Weights & Biases

Track experiments and compare runs using wandb:

```bash
# Login to wandb (first time only)
wandb login

# Train with wandb logging
python experiments/train.py \
  --config config/small_instance.yaml \
  --epochs 100 \
  --wandb \
  --wandb-project shipyard-scheduling \
  --wandb-name "gnn-ppo-baseline"
```

View your experiments at https://wandb.ai

## Distributed Training with Ray

Use Ray for hyperparameter tuning and distributed training:

```bash
# Hyperparameter search with 20 trials
python experiments/train_ray.py \
  --tune \
  --num-samples 20 \
  --epochs 50 \
  --scheduler asha \
  --wandb

# Single training with custom hyperparameters
python experiments/train_ray.py \
  --epochs 100 \
  --lr 1e-4 \
  --hidden-dim 256

# Distributed training with 4 parallel workers
python experiments/train_ray.py \
  --tune \
  --num-workers 4 \
  --num-samples 40 \
  --output-dir ./ray_results
```

---

## Benchmarking

Compare algorithms systematically:

```bash
# Quick smoke test
python experiments/benchmark.py --quick

# Compare PPO vs DQN vs rule-based
python experiments/benchmark.py \
  --methods ppo dqn rule_fifo \
  --instances small medium \
  --seeds 5

# Full benchmark suite
python experiments/benchmark.py --all --wandb
```

---

## OR Baselines (Optional)

Install OR-Tools for optimal baselines:

```bash
pip install ortools
```

Then run:

```bash
# Use CP-SAT scheduler
python -c "
from baselines.cp_scheduler import CPScheduler
from simulation.environment import ShipyardEnv
import yaml

with open('config/small_instance.yaml') as f:
    config = yaml.safe_load(f)

env = ShipyardEnv(config)
scheduler = CPScheduler(time_limit=30)
solution = scheduler.solve(env)
print(f'Makespan: {solution.makespan}')
print(f'Status: {solution.status}')
"
```

---

## Common Commands

```bash
# Run tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_advanced_agents.py -v

# Evaluate baseline scheduler
python experiments/evaluate.py --config config/small_instance.yaml --agent rule --episodes 5

# Clear database and start fresh
python -c "from mes.database import clear_db; clear_db()"

# Check equipment health
python -c "from mes.database import fetch_query; print(fetch_query('SELECT * FROM spmts'))"
```

## Troubleshooting

**Dashboard shows "No data available":**
- Run a training session with `--db-log` (enabled by default)
- Check that `shipyard.db` exists in the project root

**Import errors:**
- Ensure virtual environment is activated
- Run `pip install -e .` to install the package

**Playback not working:**
- Position history requires `env.db_logging_enabled = True`
- Run training for at least 50 simulation hours to generate snapshots

**OR-Tools not working:**
- Install with: `pip install ortools`
- For Gurobi, you need a license (free academic)

## Next Steps

- Read [USER_MANUAL.md](USER_MANUAL.md) for detailed documentation
- Explore [docs/FORMULATION.md](docs/FORMULATION.md) for the MDP specification
- Check [README.md](README.md) for full architecture overview
- Run the benchmark suite to compare algorithms
