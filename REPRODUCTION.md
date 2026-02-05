# Reproducing Paper Results

This document provides step-by-step instructions for reproducing all experimental results reported in the paper "Health-Aware Shipyard Block Scheduling via Graph Reinforcement Learning."

## Prerequisites

### Hardware Requirements
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 16GB minimum (32GB recommended for large instances)
- GPU: Optional but recommended for faster training (CUDA-compatible)
- Storage: 5GB free space for checkpoints and results

### Software Requirements
- Python 3.10 or 3.11
- pip package manager
- Git (for version control)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/shipyard_scheduling.git
cd shipyard_scheduling
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -m pytest tests/ -v --tb=short
```

All tests should pass before proceeding.

## Reproducing Main Results (Table 1)

### Main Comparison Experiment

Run evaluation for all agents across 10 random seeds with 30 episodes each:

```bash
# Create results directory
mkdir -p results/main_comparison

# GNN-PPO Agent (requires trained checkpoint)
for seed in {0..9}; do
    echo "Evaluating GNN-PPO with seed $seed"
    python experiments/evaluate.py \
        --config config/small_instance.yaml \
        --agent rl \
        --checkpoint data/checkpoints/gnn_ppo_seed_${seed}.pt \
        --episodes 30 \
        --seed $seed \
        --no-db-log \
        > results/main_comparison/gnn_ppo_seed_${seed}.txt 2>&1
done

# Rule-Based Baseline
for seed in {0..9}; do
    echo "Evaluating Rule-Based with seed $seed"
    python experiments/evaluate.py \
        --config config/small_instance.yaml \
        --agent rule \
        --episodes 30 \
        --seed $seed \
        --no-db-log \
        > results/main_comparison/rule_seed_${seed}.txt 2>&1
done

# Myopic RL Baseline
for seed in {0..9}; do
    echo "Evaluating Myopic RL with seed $seed"
    python experiments/evaluate.py \
        --config config/small_instance.yaml \
        --agent myopic \
        --episodes 30 \
        --seed $seed \
        --no-db-log \
        > results/main_comparison/myopic_seed_${seed}.txt 2>&1
done

# Siloed Optimization Baseline
for seed in {0..9}; do
    echo "Evaluating Siloed Opt with seed $seed"
    python experiments/evaluate.py \
        --config config/small_instance.yaml \
        --agent siloed \
        --episodes 30 \
        --seed $seed \
        --no-db-log \
        > results/main_comparison/siloed_seed_${seed}.txt 2>&1
done
```

### Training from Scratch

To train the GNN-PPO agent from scratch:

```bash
# Train with multiple seeds
for seed in {0..9}; do
    echo "Training GNN-PPO with seed $seed"
    python experiments/train.py \
        --config config/small_instance.yaml \
        --epochs 50 \
        --steps 500 \
        --seed $seed \
        --save data/checkpoints/seed_${seed}/ \
        --no-db-log
done
```

**Expected Training Time:**
- Small instance (50 blocks): ~30 min/seed on CPU, ~10 min/seed on GPU
- Medium instance (150 blocks): ~2 hrs/seed on CPU, ~40 min/seed on GPU

### Expected Results (Table 1)

| Agent | Throughput | Avg Tardiness | Breakdowns | OEE |
|-------|------------|---------------|------------|-----|
| GNN-PPO | 0.012 ± 0.002 | 45.3 ± 8.1 | 2.1 ± 0.8 | 0.78 ± 0.05 |
| Rule-based | 0.008 ± 0.001 | 78.2 ± 12.4 | 4.3 ± 1.2 | 0.52 ± 0.08 |
| Myopic RL | 0.005 ± 0.002 | 112.4 ± 18.7 | 6.8 ± 2.1 | 0.31 ± 0.11 |
| Siloed Opt | 0.007 ± 0.001 | 89.1 ± 14.2 | 5.1 ± 1.5 | 0.45 ± 0.09 |

Results are reported as mean ± std across 10 seeds, 30 episodes each.

## Reproducing Ablation Study (Table 2)

```bash
mkdir -p results/ablation

for seed in {0..9}; do
    echo "Running ablation study with seed $seed"
    python experiments/ablation.py \
        --config config/small_instance.yaml \
        --epochs 20 \
        --steps 200 \
        --seed $seed \
        > results/ablation/ablation_seed_${seed}.txt 2>&1
done
```

### Expected Results (Table 2)

| Configuration | Δ Throughput | Δ Tardiness | Δ Breakdowns |
|---------------|--------------|-------------|--------------|
| Full (GNN+mask+PHM) | baseline | baseline | baseline |
| - GNN (use MLP) | -18.2% | +24.1% | +15.3% |
| - Action Masking | -42.3% | +67.8% | +38.9% |
| - PHM Rewards | -8.1% | +12.3% | +22.1% |
| - Curriculum | -5.2% | +8.9% | +7.4% |

## Reproducing Scalability Study (Table 3)

```bash
mkdir -p results/scalability

# Small instance (50 blocks)
for seed in {0..4}; do
    python experiments/train.py \
        --config config/small_instance.yaml \
        --epochs 30 \
        --steps 300 \
        --seed $seed \
        --save results/scalability/small_seed_${seed}/ \
        --no-db-log
done

# Medium instance (150 blocks)
for seed in {0..4}; do
    python experiments/train.py \
        --config config/medium_instance.yaml \
        --epochs 30 \
        --steps 300 \
        --seed $seed \
        --save results/scalability/medium_seed_${seed}/ \
        --no-db-log
done

# Large instance (300 blocks)
for seed in {0..4}; do
    python experiments/train.py \
        --config config/large_instance.yaml \
        --epochs 30 \
        --steps 300 \
        --seed $seed \
        --save results/scalability/large_seed_${seed}/ \
        --no-db-log
done
```

### Expected Results (Table 3)

| Instance | Blocks | SPMTs | Cranes | Training Time | Throughput |
|----------|--------|-------|--------|---------------|------------|
| Small | 50 | 6 | 2 | 28 ± 3 min | 0.012 ± 0.002 |
| Medium | 150 | 9 | 3 | 95 ± 8 min | 0.008 ± 0.001 |
| Large | 300 | 12 | 4 | 185 ± 15 min | 0.006 ± 0.001 |

## Hyperparameter Sensitivity (Figure 3)

```bash
mkdir -p results/hp_search

python experiments/hyperparameter_search.py \
    --config config/small_instance.yaml \
    --method random \
    --n-trials 50 \
    --episodes 5 \
    --steps 200 \
    --seed 42 \
    > results/hp_search/random_search.txt 2>&1
```

## Generating Figures

After running experiments, generate publication figures:

```bash
python -c "
from utils.publication_plots import (
    plot_learning_curves,
    plot_baseline_comparison,
    plot_ablation_heatmap
)

# Generate all figures
plot_learning_curves('results/', 'figures/learning_curves.pdf')
plot_baseline_comparison('results/main_comparison/', 'figures/baseline_comparison.pdf')
plot_ablation_heatmap('results/ablation/', 'figures/ablation_heatmap.pdf')
"
```

## Statistical Analysis

```bash
python -c "
from utils.statistics import (
    compute_confidence_interval,
    paired_ttest,
    generate_results_latex
)

# Compute statistics and generate LaTeX tables
generate_results_latex('results/main_comparison/', 'results/table1.tex')
"
```

## Troubleshooting

### Out of Memory
- Reduce `--steps` parameter
- Use smaller batch size in config
- Use CPU instead of GPU for large instances

### Training Not Converging
- Increase `--epochs`
- Try different `--seed` values
- Check learning rate in config/default.yaml

### Tests Failing
- Ensure all dependencies are correctly installed
- Check Python version (3.10 or 3.11 required)
- Run `pip install -e .` to install package in editable mode

## Contact

For questions about reproducing results, please open an issue on GitHub or contact the authors.

## Citation

If you use this code, please cite:

```bibtex
@article{shipyard2025,
  title={Health-Aware Shipyard Block Scheduling via Graph Reinforcement Learning},
  author={Author Names},
  journal={Journal/Conference Name},
  year={2025}
}
```
