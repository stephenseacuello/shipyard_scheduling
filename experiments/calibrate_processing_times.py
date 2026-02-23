"""Calibrate plate-count processing time coefficients.

Usage:
    # Collect data from simulation and fit coefficients:
    python experiments/calibrate_processing_times.py --simulate --config config/hhi_plate_decomposition.yaml

    # Fit from existing CSV data:
    python experiments/calibrate_processing_times.py --data data/calibration/times.csv

    # Validate existing coefficients against data:
    python experiments/calibrate_processing_times.py --validate --coefficients data/calibration/fitted.yaml

Outputs: fitted coefficients YAML, validation report, summary statistics.
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml
import numpy as np

from simulation.calibration import CalibrationDataset, CoefficientFitter


def main():
    parser = argparse.ArgumentParser(description="Calibrate plate processing time coefficients")
    parser.add_argument("--simulate", action="store_true", help="Collect data from simulation")
    parser.add_argument("--data", type=str, help="Path to CSV calibration data")
    parser.add_argument("--validate", action="store_true", help="Validate coefficients")
    parser.add_argument("--coefficients", type=str, help="Path to fitted coefficients YAML")
    parser.add_argument("--config", type=str, default="config/hhi_plate_decomposition.yaml",
                        help="Environment config for simulation mode")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/calibration",
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CalibrationDataset()

    # Step 1: Load or collect data
    if args.data:
        print(f"Loading calibration data from {args.data}...")
        dataset.load_from_csv(args.data)
        print(f"  Loaded {len(dataset)} records")

    elif args.simulate:
        print(f"Collecting calibration data from simulation ({args.steps} steps)...")
        with open(args.config) as f:
            config = yaml.safe_load(f)

        from simulation.shipyard_env import HHIShipyardEnv
        env = HHIShipyardEnv(config)

        dataset.load_from_simulation(env, n_steps=args.steps, seed=args.seed)
        print(f"  Collected {len(dataset)} records")

        # Save collected data
        csv_path = os.path.join(args.output_dir, "simulation_calibration_data.csv")
        dataset.save_to_csv(csv_path)
        print(f"  Saved to {csv_path}")

    else:
        parser.error("Must specify --simulate or --data")

    if len(dataset) == 0:
        print("No calibration records collected. Try running more steps.")
        return

    # Step 2: Print summary
    print("\n=== Calibration Data Summary ===")
    summary = dataset.summary()
    for stage, stats in summary.items():
        print(f"  {stage}: n={stats['count']}, mean={stats['mean']:.2f}h, "
              f"std={stats['std']:.2f}h, range=[{stats['min']:.2f}, {stats['max']:.2f}]")

    # Step 3: Fit or validate
    fitter = CoefficientFitter()

    if args.validate and args.coefficients:
        print(f"\n=== Validating coefficients from {args.coefficients} ===")
        with open(args.coefficients) as f:
            coeff_data = yaml.safe_load(f)
        coefficients = coeff_data.get("plate_time_coefficients", coeff_data)
        metrics = fitter.validate(dataset, coefficients)
        for stage, m in metrics.items():
            print(f"  {stage}: R2={m['r2']:.4f}, RMSE={m['rmse']:.2f}h, "
                  f"MAE={m['mae']:.2f}h (n={m['n_samples']})")

    else:
        print("\n=== Fitting coefficients ===")
        try:
            coefficients = fitter.fit(dataset)
        except ImportError as e:
            print(f"  Error: {e}")
            return

        for stage, coeffs in coefficients.items():
            parts = ", ".join(f"{k}={v:.4f}" for k, v in coeffs.items())
            print(f"  {stage}: {parts}")

        # Validate
        print("\n=== Validation (in-sample) ===")
        metrics = fitter.validate(dataset, coefficients)
        for stage, m in metrics.items():
            print(f"  {stage}: R2={m['r2']:.4f}, RMSE={m['rmse']:.2f}h, "
                  f"MAE={m['mae']:.2f}h (n={m['n_samples']})")

        # Export
        coeff_path = os.path.join(args.output_dir, "fitted_coefficients.yaml")
        fitter.export_coefficients(coefficients, coeff_path)
        print(f"\nFitted coefficients saved to {coeff_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
