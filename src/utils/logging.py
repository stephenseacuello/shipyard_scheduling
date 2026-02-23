"""Experiment logging utilities.

Provides functions to record training progress and evaluation results to
CSV or other formats. These helpers are intended to be called by
training scripts to persist results for later analysis.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Any


def log_results_csv(filename: str, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    """Write a list of result dictionaries to a CSV file."""
    if not rows:
        return
    if fieldnames is None:
        # Gather all unique keys from all rows to handle varying fields
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        # Sort keys for consistent column ordering
        fieldnames = sorted(all_keys)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)