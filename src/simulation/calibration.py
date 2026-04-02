"""Calibration pipeline for plate-count processing time coefficients.

Provides tooling to:
1. Collect processing time observations from simulation or real data (CSV)
2. Fit stage-specific coefficients via least-squares regression
3. Validate fitted coefficients (R^2, RMSE, MAE per stage)
4. Export fitted coefficients to YAML for use in environment config
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulation.shipyard_env import HHIShipyardEnv

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class CalibrationRecord:
    """A single observed processing time from real or simulation data."""
    block_id: str
    stage: str  # HHIProductionStage name
    n_plates: int
    n_curved: int
    n_stiffened: int
    total_area_m2: float
    total_weld_m: float
    observed_time_hours: float
    # Extended features for stages with low plate-only R²
    # Block outfitting: driven by piping/electrical complexity, not plate count
    outfit_complexity: float = 0.0  # Composite outfitting score (0-1)
    n_pipe_connections: int = 0     # Number of pipe penetrations
    n_electrical_runs: int = 0      # Number of cable runs
    # Pre-erection: driven by grand-block assembly complexity
    n_blocks_in_grand_block: int = 1  # Blocks composing the grand block
    alignment_difficulty: float = 0.0  # 0-1 based on curvature/position
    total_weight_tonnes: float = 0.0   # Total grand block weight


class CalibrationDataset:
    """Collection of calibration records for fitting processing time coefficients."""

    def __init__(self) -> None:
        self.records: List[CalibrationRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def load_from_csv(self, filepath: str) -> None:
        """Load calibration data from CSV.

        Expected columns: block_id, stage, n_plates, n_curved, n_stiffened,
        total_area_m2, total_weld_m, observed_time_hours
        """
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(CalibrationRecord(
                    block_id=row["block_id"],
                    stage=row["stage"],
                    n_plates=int(row["n_plates"]),
                    n_curved=int(row["n_curved"]),
                    n_stiffened=int(row["n_stiffened"]),
                    total_area_m2=float(row["total_area_m2"]),
                    total_weld_m=float(row["total_weld_m"]),
                    observed_time_hours=float(row["observed_time_hours"]),
                ))

    def load_from_simulation(self, env: "HHIShipyardEnv", n_steps: int = 1000,
                             seed: int = 42) -> None:
        """Collect processing time observations from a simulation run.

        Runs the expert scheduler and records processing times as blocks
        complete at each facility.
        """
        from baselines.rule_based import RuleBasedScheduler

        env.reset(seed=seed)
        expert = RuleBasedScheduler()

        # Track when blocks start processing at a facility
        start_times: Dict[str, float] = {}  # "block_id@fac" -> start_time

        for step in range(n_steps):
            # Record start times for blocks entering processing
            for fac_name, block_ids in env.facility_processing.items():
                for bid in block_ids:
                    key = f"{bid}@{fac_name}"
                    if key not in start_times:
                        start_times[key] = env.sim_time

            action = expert.decide(env)
            obs, reward, terminated, truncated, info = env.step(action)

            # Check for completed blocks (ones that were in processing but now aren't)
            for key, st in list(start_times.items()):
                bid, fac = key.split("@")
                current_processing = env.facility_processing.get(fac, [])
                if bid not in current_processing:
                    # Block completed
                    elapsed = env.sim_time - st
                    if elapsed > 0:
                        block = env._get_block(bid)
                        if block and block.n_plates > 0:
                            n_curved = sum(1 for p in block.plates if p.is_curved)
                            n_stiffened = sum(1 for p in block.plates if p.has_stiffeners)
                            stage = env._STAGE_MAP.get(fac, block.current_stage)
                            stage_name = stage.name if hasattr(stage, 'name') else str(stage)
                            self.records.append(CalibrationRecord(
                                block_id=bid,
                                stage=stage_name,
                                n_plates=block.n_plates,
                                n_curved=n_curved,
                                n_stiffened=n_stiffened,
                                total_area_m2=block.total_plate_area_m2,
                                total_weld_m=sum(p.weld_length_m for p in block.plates),
                                observed_time_hours=elapsed,
                            ))
                    del start_times[key]

            if terminated or truncated:
                break

    def save_to_csv(self, filepath: str) -> None:
        """Export calibration records to CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "block_id", "stage", "n_plates", "n_curved", "n_stiffened",
                "total_area_m2", "total_weld_m", "observed_time_hours",
            ])
            writer.writeheader()
            for rec in self.records:
                writer.writerow({
                    "block_id": rec.block_id,
                    "stage": rec.stage,
                    "n_plates": rec.n_plates,
                    "n_curved": rec.n_curved,
                    "n_stiffened": rec.n_stiffened,
                    "total_area_m2": f"{rec.total_area_m2:.2f}",
                    "total_weld_m": f"{rec.total_weld_m:.2f}",
                    "observed_time_hours": f"{rec.observed_time_hours:.4f}",
                })

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics per stage."""
        from collections import defaultdict
        by_stage: Dict[str, List[float]] = defaultdict(list)
        for rec in self.records:
            by_stage[rec.stage].append(rec.observed_time_hours)

        summary = {}
        for stage, times in sorted(by_stage.items()):
            arr = np.array(times)
            summary[stage] = {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return summary


class CoefficientFitter:
    """Fit plate-time coefficients via least-squares regression.

    For each stage, fits: time = base + c1*n_plates + c2*n_curved + c3*area + ...
    Uses scipy.optimize.curve_fit with bounded constraints (all coefficients >= 0).
    """

    def fit(self, dataset: CalibrationDataset) -> Dict[str, Dict[str, float]]:
        """Fit coefficients from calibration data.

        Returns dict matching _PLATE_TIME_COEFFICIENTS format:
        {stage_name: {"base_hours": ..., "per_plate": ..., ...}}
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for coefficient fitting. Install with: pip install scipy")

        from collections import defaultdict
        by_stage: Dict[str, List[CalibrationRecord]] = defaultdict(list)
        for rec in dataset.records:
            by_stage[rec.stage].append(rec)

        fitted = {}
        for stage, records in sorted(by_stage.items()):
            if len(records) < 3:
                # Not enough data to fit
                continue

            # Build feature matrix — stage-specific for better R²
            y = np.array([r.observed_time_hours for r in records])

            if stage == "BLOCK_OUTFITTING":
                # Outfitting depends on piping/electrical, not just plate geometry
                X = np.array([
                    [r.n_plates, r.n_curved, r.total_area_m2,
                     r.outfit_complexity, r.n_pipe_connections, r.n_electrical_runs]
                    for r in records
                ])
                feature_names = ["per_plate", "per_curved", "per_area_m2",
                                 "per_outfit_complexity", "per_pipe", "per_electrical"]
                p0 = [5.0, 0.1, 0.1, 0.01, 10.0, 0.5, 0.3]
            elif stage == "PRE_ERECTION":
                # Pre-erection depends on grand-block assembly complexity
                X = np.array([
                    [r.n_plates, r.total_area_m2, r.total_weld_m,
                     r.n_blocks_in_grand_block, r.alignment_difficulty, r.total_weight_tonnes]
                    for r in records
                ])
                feature_names = ["per_plate", "per_area_m2", "per_weld_m",
                                 "per_grand_block", "per_alignment", "per_weight_tonne"]
                p0 = [5.0, 0.1, 0.01, 0.05, 2.0, 5.0, 0.1]
            else:
                # Standard plate-count regression for other stages
                X = np.array([
                    [r.n_plates, r.n_curved, r.n_stiffened, r.total_area_m2, r.total_weld_m]
                    for r in records
                ])
                feature_names = ["per_plate", "per_curved", "per_stiffened",
                                 "per_area_m2", "per_weld_m"]
                p0 = [5.0, 0.2, 0.3, 0.3, 0.01, 0.05]

            n_features = X.shape[1]

            def model(x_data, base, *coeffs):
                result = np.full(x_data.shape[0], base)
                for k, c in enumerate(coeffs):
                    result += c * x_data[:, k]
                return result

            try:
                popt, _ = curve_fit(
                    model, X, y,
                    p0=p0,
                    bounds=(0, np.inf),
                    maxfev=5000,
                )
                fitted[stage] = {"base_hours": float(popt[0])}
                for k, name in enumerate(feature_names):
                    fitted[stage][name] = float(popt[k + 1])
            except Exception:
                # Fallback 1: Ridge regression (handles multicollinearity)
                try:
                    from sklearn.linear_model import Ridge
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X, y)
                    fitted[stage] = {
                        "base_hours": max(0.0, float(ridge.intercept_)),
                        "per_plate": max(0.0, float(ridge.coef_[0])),
                        "per_curved": max(0.0, float(ridge.coef_[1])),
                        "per_stiffened": max(0.0, float(ridge.coef_[2])),
                        "per_area_m2": max(0.0, float(ridge.coef_[3])),
                        "per_weld_m": max(0.0, float(ridge.coef_[4])),
                    }
                except Exception:
                    # Fallback 2: simple per-plate linear fit
                    n_plates = np.array([r.n_plates for r in records], dtype=float)
                    if np.std(n_plates) > 0:
                        slope, intercept = np.polyfit(n_plates, y, 1)
                        fitted[stage] = {
                            "base_hours": max(0.0, float(intercept)),
                            "per_plate": max(0.0, float(slope)),
                        }

        return fitted

    def validate(self, dataset: CalibrationDataset,
                 coefficients: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute R^2, RMSE, MAE per stage."""
        from collections import defaultdict
        by_stage: Dict[str, List[CalibrationRecord]] = defaultdict(list)
        for rec in dataset.records:
            by_stage[rec.stage].append(rec)

        metrics = {}
        for stage, records in sorted(by_stage.items()):
            coeffs = coefficients.get(stage)
            if not coeffs or len(records) < 2:
                continue

            y_true = np.array([r.observed_time_hours for r in records])
            y_pred = np.array([self._predict(r, coeffs) for r in records])

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - (ss_res / max(ss_tot, 1e-10))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))

            metrics[stage] = {
                "r2": float(r2),
                "rmse": rmse,
                "mae": mae,
                "n_samples": len(records),
            }

        return metrics

    def _predict(self, record: CalibrationRecord, coeffs: Dict[str, float]) -> float:
        """Predict processing time from coefficients."""
        t = coeffs.get("base_hours", 0.0)
        t += coeffs.get("per_plate", 0.0) * record.n_plates
        t += coeffs.get("per_curved", 0.0) * record.n_curved
        t += coeffs.get("per_stiffened", 0.0) * record.n_stiffened
        n_flat = max(0, record.n_plates - record.n_curved - record.n_stiffened)
        t += coeffs.get("per_flat", 0.0) * n_flat
        t += coeffs.get("per_area_m2", 0.0) * record.total_area_m2
        t += coeffs.get("per_weld_m", 0.0) * record.total_weld_m
        # Extended features for outfitting/pre-erection
        t += coeffs.get("per_outfit_complexity", 0.0) * record.outfit_complexity
        t += coeffs.get("per_pipe", 0.0) * record.n_pipe_connections
        t += coeffs.get("per_electrical", 0.0) * record.n_electrical_runs
        t += coeffs.get("per_grand_block", 0.0) * record.n_blocks_in_grand_block
        t += coeffs.get("per_alignment", 0.0) * record.alignment_difficulty
        t += coeffs.get("per_weight_tonne", 0.0) * record.total_weight_tonnes
        return t

    def export_coefficients(self, coefficients: Dict[str, Dict[str, float]],
                            filepath: str) -> None:
        """Export fitted coefficients to YAML for use in config."""
        import yaml
        with open(filepath, "w") as f:
            yaml.dump({"plate_time_coefficients": coefficients}, f,
                      default_flow_style=False, sort_keys=False)
