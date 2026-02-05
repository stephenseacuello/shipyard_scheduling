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
        fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)