#!/bin/bash
# Master reproduction script for all experiments and figures.
#
# Usage:
#   source shipyard/bin/activate
#   bash scripts/reproduce_all_results.sh
#
# Prerequisites:
#   - Python 3.10+ with venv activated
#   - PYTHONPATH=src (set automatically below)
#   - PuLP installed (pip install pulp)
#   - PyTorch + PyG installed

set -e  # Exit on error

export PYTHONPATH=src
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "Shipyard Scheduling: Full Results Reproduction"
echo "Started: $(date)"
echo "======================================================================"

# 1. Run tests
echo ""
echo "--- Step 1: Run all tests ---"
pytest tests/ -v --tb=short 2>&1 | tail -5
echo "Tests complete."

# 2. Oracle MIP evaluation (tiny instance, 3 seeds)
echo ""
echo "--- Step 2: Oracle MIP evaluation ---"
python experiments/oracle_mip_evaluation.py \
    --configs config/tiny_instance.yaml \
    --seeds 3 --time-limit 120 --max-steps 1000

# 3. Statistical comparison (deterministic)
echo ""
echo "--- Step 3: Statistical comparison (5 seeds, 1000 steps) ---"
python experiments/statistical_comparison.py \
    --configs config/small_instance.yaml config/medium_hhi.yaml \
    --seeds 5 --max-steps 1000 --skip-ga

# 4. Generate all paper figures
echo ""
echo "--- Step 4: Generate paper figures ---"
python paper/generate_figures.py

# 5. Summary
echo ""
echo "======================================================================"
echo "Reproduction complete at $(date)"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  data/oracle_gap_results.csv"
echo "  data/statistical_comparison.csv"
echo "  figures/*.pdf"
echo ""
echo "To run additional experiments:"
echo "  - DAgger training:    python experiments/train_dagger.py --config config/small_instance.yaml"
echo "  - Curriculum DAgger:  python experiments/curriculum_dagger.py"
echo "  - GNN ablation:       python experiments/gnn_component_ablation.py"
echo "  - Supply chain:       python experiments/train_dagger_supply_chain.py"
