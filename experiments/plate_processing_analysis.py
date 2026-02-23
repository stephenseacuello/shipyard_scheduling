#!/usr/bin/env python3
"""Plate-specific processing time analysis.

Runs simulation with plate decomposition enabled, compares observed
processing times to plate-count model predictions, and generates
summary statistics per production stage.

Uses the calibration pipeline (CalibrationDataset, CoefficientFitter)
from src/simulation/calibration.py to:
1. Collect processing observations from simulation (expert scheduler)
2. Fit plate-count regression coefficients
3. Validate predictions (R^2, RMSE, MAE per stage)
4. Output calibration data CSV and fitted coefficients YAML

Usage:
    # Run with default config:
    python experiments/plate_processing_analysis.py

    # Specify config, steps, and output directory:
    python experiments/plate_processing_analysis.py \\
        --config config/hhi_ulsan.yaml --steps 3000 --output-dir results/plate_analysis

    # Validate against pre-fitted coefficients:
    python experiments/plate_processing_analysis.py \\
        --coefficients data/calibration/fitted_coefficients.yaml

    # Use existing CSV data instead of simulation:
    python experiments/plate_processing_analysis.py --data data/calibration/times.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from simulation.calibration import CalibrationDataset, CoefficientFitter


# ---------------------------------------------------------------------------
# Config loading (matches pattern across the project)
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config with optional inheritance."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        base_path = os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base_cfg
    return cfg


# ---------------------------------------------------------------------------
# Synthetic plate statistics
# ---------------------------------------------------------------------------

def compute_plate_statistics(dataset: CalibrationDataset) -> Dict[str, Any]:
    """Compute plate-level summary statistics across the dataset.

    Groups records by stage and computes mean/std/min/max for plate counts,
    curved fractions, areas, weld lengths, and observed processing times.
    """
    from collections import defaultdict

    by_stage: Dict[str, List] = defaultdict(list)
    for rec in dataset.records:
        by_stage[rec.stage].append(rec)

    stats: Dict[str, Any] = {}
    for stage, records in sorted(by_stage.items()):
        n = len(records)
        n_plates = np.array([r.n_plates for r in records])
        n_curved = np.array([r.n_curved for r in records])
        n_stiffened = np.array([r.n_stiffened for r in records])
        areas = np.array([r.total_area_m2 for r in records])
        welds = np.array([r.total_weld_m for r in records])
        times = np.array([r.observed_time_hours for r in records])
        curved_frac = n_curved / np.maximum(n_plates, 1)

        stats[stage] = {
            "count": n,
            "n_plates": {
                "mean": float(np.mean(n_plates)),
                "std": float(np.std(n_plates)),
                "min": int(np.min(n_plates)),
                "max": int(np.max(n_plates)),
            },
            "curved_fraction": {
                "mean": float(np.mean(curved_frac)),
                "std": float(np.std(curved_frac)),
            },
            "n_stiffened": {
                "mean": float(np.mean(n_stiffened)),
                "std": float(np.std(n_stiffened)),
            },
            "total_area_m2": {
                "mean": float(np.mean(areas)),
                "std": float(np.std(areas)),
            },
            "total_weld_m": {
                "mean": float(np.mean(welds)),
                "std": float(np.std(welds)),
            },
            "observed_time_hours": {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
            },
        }
    return stats


def compute_prediction_comparison(
    dataset: CalibrationDataset,
    coefficients: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """Compare observed vs predicted times and compute error distributions.

    For each stage, returns arrays of observed times, predicted times,
    absolute errors, and relative errors (%).
    """
    from collections import defaultdict

    fitter = CoefficientFitter()
    by_stage: Dict[str, List] = defaultdict(list)
    for rec in dataset.records:
        by_stage[rec.stage].append(rec)

    comparison: Dict[str, Dict[str, Any]] = {}
    for stage, records in sorted(by_stage.items()):
        coeffs = coefficients.get(stage)
        if not coeffs:
            continue

        observed = np.array([r.observed_time_hours for r in records])
        predicted = np.array([fitter._predict(r, coeffs) for r in records])
        abs_err = np.abs(observed - predicted)
        rel_err = abs_err / np.maximum(observed, 1e-6) * 100.0

        comparison[stage] = {
            "n_samples": len(records),
            "observed_mean": float(np.mean(observed)),
            "predicted_mean": float(np.mean(predicted)),
            "abs_error_mean": float(np.mean(abs_err)),
            "abs_error_std": float(np.std(abs_err)),
            "abs_error_max": float(np.max(abs_err)),
            "rel_error_mean_pct": float(np.mean(rel_err)),
            "rel_error_std_pct": float(np.std(rel_err)),
            "rel_error_p90_pct": float(np.percentile(rel_err, 90)),
        }

    return comparison


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_plate_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted plate-level statistics."""
    print("\n" + "=" * 72)
    print("  PLATE-LEVEL STATISTICS BY STAGE")
    print("=" * 72)

    for stage, s in stats.items():
        print(f"\n  --- {stage} (n={s['count']}) ---")
        np_ = s["n_plates"]
        print(f"    Plates per block : {np_['mean']:.1f} +/- {np_['std']:.1f}  "
              f"[{np_['min']}, {np_['max']}]")
        cf = s["curved_fraction"]
        print(f"    Curved fraction  : {cf['mean']:.2%} +/- {cf['std']:.2%}")
        ns = s["n_stiffened"]
        print(f"    Stiffened plates : {ns['mean']:.1f} +/- {ns['std']:.1f}")
        a = s["total_area_m2"]
        print(f"    Total area (m^2) : {a['mean']:.1f} +/- {a['std']:.1f}")
        w = s["total_weld_m"]
        print(f"    Weld length (m)  : {w['mean']:.1f} +/- {w['std']:.1f}")
        t = s["observed_time_hours"]
        print(f"    Processing (hrs) : {t['mean']:.2f} +/- {t['std']:.2f}  "
              f"[{t['min']:.2f}, {t['max']:.2f}]")


def print_prediction_comparison(comparison: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted observed-vs-predicted comparison."""
    print("\n" + "=" * 72)
    print("  OBSERVED vs PREDICTED PROCESSING TIMES")
    print("=" * 72)

    header = (f"  {'Stage':<28s} {'n':>5s}  {'Obs(h)':>8s}  {'Pred(h)':>8s}  "
              f"{'MAE(h)':>8s}  {'Rel%':>8s}  {'P90%':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for stage, c in comparison.items():
        print(
            f"  {stage:<28s} {c['n_samples']:5d}  "
            f"{c['observed_mean']:8.2f}  "
            f"{c['predicted_mean']:8.2f}  "
            f"{c['abs_error_mean']:8.2f}  "
            f"{c['rel_error_mean_pct']:7.1f}%  "
            f"{c['rel_error_p90_pct']:7.1f}%"
        )


def print_coefficients(coefficients: Dict[str, Dict[str, float]]) -> None:
    """Print fitted coefficients per stage."""
    print("\n" + "=" * 72)
    print("  FITTED PLATE-TIME COEFFICIENTS")
    print("=" * 72)

    for stage, coeffs in sorted(coefficients.items()):
        parts = ", ".join(f"{k}={v:.4f}" for k, v in coeffs.items())
        print(f"  {stage}: {parts}")


def print_validation_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print R^2, RMSE, MAE per stage."""
    print("\n" + "=" * 72)
    print("  REGRESSION VALIDATION METRICS")
    print("=" * 72)

    header = f"  {'Stage':<28s} {'R^2':>8s}  {'RMSE(h)':>8s}  {'MAE(h)':>8s}  {'n':>5s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for stage, m in sorted(metrics.items()):
        print(
            f"  {stage:<28s} {m['r2']:8.4f}  "
            f"{m['rmse']:8.2f}  "
            f"{m['mae']:8.2f}  "
            f"{m['n_samples']:5d}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plate-specific processing time analysis with calibration."
    )
    parser.add_argument(
        "--config", type=str, default="config/small_instance.yaml",
        help="Environment config YAML (default: config/small_instance.yaml)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to existing calibration CSV (skip simulation if provided)",
    )
    parser.add_argument(
        "--coefficients", type=str, default=None,
        help="Path to pre-fitted coefficients YAML for validation-only mode",
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Number of simulation steps for data collection (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/plate_analysis",
        help="Directory for output files (default: results/plate_analysis)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  Plate Processing Time Analysis")
    print("=" * 72)
    print(f"  Config       : {args.config}")
    print(f"  Output dir   : {args.output_dir}")

    # ------------------------------------------------------------------
    # Step 1: Collect or load calibration data
    # ------------------------------------------------------------------
    dataset = CalibrationDataset()

    if args.data:
        print(f"\n  Loading calibration data from {args.data} ...")
        dataset.load_from_csv(args.data)
        print(f"  Loaded {len(dataset)} records.")

    else:
        print(f"\n  Collecting data via simulation ({args.steps} steps, seed={args.seed}) ...")
        try:
            cfg = load_config(args.config)
        except FileNotFoundError:
            print(f"  [ERROR] Config file not found: {args.config}")
            return

        try:
            from simulation.shipyard_env import HHIShipyardEnv
            env = HHIShipyardEnv(cfg)
            dataset.load_from_simulation(env, n_steps=args.steps, seed=args.seed)
            print(f"  Collected {len(dataset)} records.")
        except Exception as exc:
            print(f"  [ERROR] Simulation failed: {exc}")
            return

        # Save collected data
        csv_path = os.path.join(args.output_dir, "calibration_data.csv")
        dataset.save_to_csv(csv_path)
        print(f"  Saved calibration data to {csv_path}")

    if len(dataset) == 0:
        print("\n  No calibration records collected. Try increasing --steps.")
        return

    # ------------------------------------------------------------------
    # Step 2: Print basic summary (from CalibrationDataset.summary)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  CALIBRATION DATA SUMMARY")
    print("=" * 72)
    summary = dataset.summary()
    for stage, s in summary.items():
        print(
            f"  {stage}: n={s['count']}, "
            f"mean={s['mean']:.2f}h, std={s['std']:.2f}h, "
            f"range=[{s['min']:.2f}, {s['max']:.2f}]"
        )

    # ------------------------------------------------------------------
    # Step 3: Compute and print plate-level statistics
    # ------------------------------------------------------------------
    plate_stats = compute_plate_statistics(dataset)
    print_plate_statistics(plate_stats)

    # Save plate statistics to YAML
    stats_path = os.path.join(args.output_dir, "plate_statistics.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(plate_stats, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Plate statistics saved to {stats_path}")

    # ------------------------------------------------------------------
    # Step 4: Fit or load coefficients
    # ------------------------------------------------------------------
    fitter = CoefficientFitter()

    if args.coefficients:
        # Validation-only mode: load pre-fitted coefficients
        print(f"\n  Loading coefficients from {args.coefficients} ...")
        with open(args.coefficients) as f:
            coeff_data = yaml.safe_load(f)
        coefficients = coeff_data.get("plate_time_coefficients", coeff_data)
        print_coefficients(coefficients)

    else:
        # Fit new coefficients from data
        print("\n  Fitting plate-time regression coefficients ...")
        try:
            coefficients = fitter.fit(dataset)
        except ImportError as exc:
            print(f"  [ERROR] {exc}")
            print("  Install scipy to enable coefficient fitting: pip install scipy")
            return
        except Exception as exc:
            print(f"  [ERROR] Fitting failed: {exc}")
            return

        print_coefficients(coefficients)

        # Export fitted coefficients
        coeff_path = os.path.join(args.output_dir, "fitted_coefficients.yaml")
        fitter.export_coefficients(coefficients, coeff_path)
        print(f"\n  Fitted coefficients saved to {coeff_path}")

    # ------------------------------------------------------------------
    # Step 5: Validate and compare observed vs predicted
    # ------------------------------------------------------------------
    print("\n  Validating coefficients against observed data ...")
    validation_metrics = fitter.validate(dataset, coefficients)
    print_validation_metrics(validation_metrics)

    # ------------------------------------------------------------------
    # Step 6: Detailed prediction comparison
    # ------------------------------------------------------------------
    comparison = compute_prediction_comparison(dataset, coefficients)
    print_prediction_comparison(comparison)

    # Save comparison to CSV
    comparison_csv_path = os.path.join(args.output_dir, "prediction_comparison.csv")
    try:
        import csv as csv_mod
        with open(comparison_csv_path, "w", newline="") as f:
            if comparison:
                first_stage = next(iter(comparison.values()))
                fieldnames = ["stage"] + list(first_stage.keys())
                writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for stage, vals in comparison.items():
                    row = {"stage": stage, **vals}
                    writer.writerow(row)
        print(f"\n  Prediction comparison saved to {comparison_csv_path}")
    except Exception as exc:
        print(f"  [WARN] Could not save comparison CSV: {exc}")

    # ------------------------------------------------------------------
    # Step 7: Overall summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  OVERALL SUMMARY")
    print("=" * 72)
    total_records = len(dataset)
    n_stages = len(summary)
    mean_r2 = np.mean([m["r2"] for m in validation_metrics.values()]) if validation_metrics else 0.0
    mean_rmse = np.mean([m["rmse"] for m in validation_metrics.values()]) if validation_metrics else 0.0
    mean_mae = np.mean([m["mae"] for m in validation_metrics.values()]) if validation_metrics else 0.0

    print(f"  Total calibration records : {total_records}")
    print(f"  Production stages covered : {n_stages}")
    print(f"  Mean R^2 across stages    : {mean_r2:.4f}")
    print(f"  Mean RMSE across stages   : {mean_rmse:.2f} hours")
    print(f"  Mean MAE across stages    : {mean_mae:.2f} hours")

    if comparison:
        all_rel_errors = [c["rel_error_mean_pct"] for c in comparison.values()]
        print(f"  Mean relative error       : {np.mean(all_rel_errors):.1f}%")
        worst_stage = max(comparison.items(), key=lambda x: x[1]["rel_error_mean_pct"])
        best_stage = min(comparison.items(), key=lambda x: x[1]["rel_error_mean_pct"])
        print(f"  Best-predicted stage      : {best_stage[0]} ({best_stage[1]['rel_error_mean_pct']:.1f}%)")
        print(f"  Worst-predicted stage     : {worst_stage[0]} ({worst_stage[1]['rel_error_mean_pct']:.1f}%)")

    print(f"\n  All outputs written to: {args.output_dir}/")
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
