"""
Taillard / Fisher-Thompson JSSP Benchmark Instances

Standard job-shop scheduling problem instances used for reproducible
entropy-collapse validation across domains.

Instances included:
  - ft06  (6 jobs x 6 machines)  -- Fisher & Thompson, 1963; optimal makespan = 55
  - ft10  (10 jobs x 10 machines) -- Fisher & Thompson, 1963; optimal makespan = 930

Data format
-----------
Each instance is a dict with keys:
  n_jobs            int            number of jobs
  n_machines        int            number of machines
  processing_times  ndarray (J,M)  processing time of job j at its k-th operation
  machine_order     ndarray (J,M)  machine id for the k-th operation of job j

Source
------
http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/jobshop.dir/
Original reference: J.F. Muth and G.L. Thompson, Industrial Scheduling,
Prentice Hall, 1963.
"""

import numpy as np
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────────────
# ft06 — 6 jobs x 6 machines (optimal makespan = 55)
#
# Raw data (machine, processing_time) pairs per job:
#   Job 0: 2 1  0 3  1 6  3 7  5 3  4 6
#   Job 1: 1 8  2 5  4 10 5 10 0 10 3 4
#   Job 2: 2 5  3 4  5 8  0 9  1 1  4 7
#   Job 3: 1 5  0 5  2 5  3 3  4 8  5 9
#   Job 4: 2 9  1 3  4 5  5 4  0 3  3 1
#   Job 5: 1 3  3 3  5 9  0 10 4 4  2 1
# ──────────────────────────────────────────────────────────────────────

_FT06_MACHINE_ORDER = np.array([
    [2, 0, 1, 3, 5, 4],
    [1, 2, 4, 5, 0, 3],
    [2, 3, 5, 0, 1, 4],
    [1, 0, 2, 3, 4, 5],
    [2, 1, 4, 5, 0, 3],
    [1, 3, 5, 0, 4, 2],
], dtype=np.int32)

_FT06_PROCESSING_TIMES = np.array([
    [1, 3, 6, 7, 3, 6],
    [8, 5, 10, 10, 10, 4],
    [5, 4, 8, 9, 1, 7],
    [5, 5, 5, 3, 8, 9],
    [9, 3, 5, 4, 3, 1],
    [3, 3, 9, 10, 4, 1],
], dtype=np.int32)

_FT06 = {
    "name": "ft06",
    "n_jobs": 6,
    "n_machines": 6,
    "processing_times": _FT06_PROCESSING_TIMES,
    "machine_order": _FT06_MACHINE_ORDER,
    "optimal_makespan": 55,
}

# ──────────────────────────────────────────────────────────────────────
# ft10 — 10 jobs x 10 machines (optimal makespan = 930)
#
# Raw data (machine, processing_time) pairs per job:
#   Job 0: 0 29  1 78  2  9  3 36  4 49  5 11  6 62  7 56  8 44  9 21
#   Job 1: 0 43  2 90  4 75  9 11  3 69  1 28  6 46  5 46  7 72  8 30
#   Job 2: 1 91  0 85  3 39  2 74  8 90  5 10  7 12  6 89  9 45  4 33
#   Job 3: 1 81  2 95  0 71  4 99  6  9  8 52  7 85  3 98  9 22  5 43
#   Job 4: 2 14  0  6  1 22  5 61  3 26  4 69  8 21  7 49  9 72  6 53
#   Job 5: 2 84  1  2  5 52  3 95  8 48  9 72  0 47  6 65  4  6  7 25
#   Job 6: 1 46  0 37  3 61  2 13  6 32  5 21  9 32  8 89  7 30  4 55
#   Job 7: 2 31  0 86  1 46  5 74  4 32  6 88  8 19  9 48  7 36  3 79
#   Job 8: 0 76  1 69  3 76  5 51  2 85  9 11  6 40  7 89  4 26  8 74
#   Job 9: 1 85  0 13  2 61  6  7  8 64  9 76  5 47  3 52  4 90  7 45
# ──────────────────────────────────────────────────────────────────────

_FT10_MACHINE_ORDER = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
    [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
    [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
    [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
    [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
    [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
    [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
    [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
    [1, 0, 2, 6, 8, 9, 5, 3, 4, 7],
], dtype=np.int32)

_FT10_PROCESSING_TIMES = np.array([
    [29, 78,  9, 36, 49, 11, 62, 56, 44, 21],
    [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
    [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
    [81, 95, 71, 99,  9, 52, 85, 98, 22, 43],
    [14,  6, 22, 61, 26, 69, 21, 49, 72, 53],
    [84,  2, 52, 95, 48, 72, 47, 65,  6, 25],
    [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
    [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
    [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
    [85, 13, 61,  7, 64, 76, 47, 52, 90, 45],
], dtype=np.int32)

_FT10 = {
    "name": "ft10",
    "n_jobs": 10,
    "n_machines": 10,
    "processing_times": _FT10_PROCESSING_TIMES,
    "machine_order": _FT10_MACHINE_ORDER,
    "optimal_makespan": 930,
}

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

_INSTANCES: Dict[str, Dict[str, Any]] = {
    "ft06": _FT06,
    "ft10": _FT10,
}


def get_instance(name: str) -> Dict[str, Any]:
    """Return a benchmark JSSP instance by name.

    Parameters
    ----------
    name : str
        Instance identifier. One of: ``"ft06"``, ``"ft10"``.

    Returns
    -------
    dict
        Keys: ``n_jobs``, ``n_machines``, ``processing_times`` (ndarray),
        ``machine_order`` (ndarray), ``optimal_makespan``.

    Raises
    ------
    KeyError
        If *name* is not a known instance.
    """
    key = name.lower()
    if key not in _INSTANCES:
        available = ", ".join(sorted(_INSTANCES))
        raise KeyError(
            f"Unknown JSSP instance '{name}'. Available: {available}"
        )
    return _INSTANCES[key].copy()


def list_instances():
    """Return a list of available instance names."""
    return sorted(_INSTANCES.keys())
