"""
Solomon VRPTW Benchmark Instances

Standard vehicle-routing-with-time-windows instances used for reproducible
entropy-collapse validation across domains.

Instances included:
  - R101 (25 customers) -- first 25 customers of Solomon's R101 (100-customer)
    instance, random geographic distribution, tight time windows.

Data format
-----------
Each instance is a dict with keys:
  name         str                     instance identifier
  n_customers  int                     number of customers (excluding depot)
  capacity     int                     vehicle capacity
  depot        list [x, y]             depot coordinates
  customers    list of lists           [[x, y, demand, ready, due, service], ...]
  best_known   dict | None             best-known solution info (if available)

Source
------
https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/
Original reference: M.M. Solomon, "Algorithms for the Vehicle Routing and
Scheduling Problems with Time Window Constraints," Operations Research,
35(2), 1987, pp. 254-265.
"""

import numpy as np
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────────────
# R101 -- 25 customers (subset of the 100-customer R101 instance)
#
# Full R101 has 100 customers, 25 vehicles, capacity 200.
# We include the first 25 customers for a tractable RL benchmark.
#
# Each row: [x, y, demand, ready_time, due_date, service_time]
# Depot (customer 0 in original): x=35, y=35, demand=0, ready=0, due=230, service=0
# ──────────────────────────────────────────────────────────────────────

_R101_DEPOT = [35, 35]

_R101_CUSTOMERS_25 = [
    # [  x,   y, demand, ready, due, service]
    [41, 49, 10, 161, 171, 10],   # customer  1
    [35, 17,  7,  50,  60, 10],   # customer  2
    [55, 45, 13, 116, 126, 10],   # customer  3
    [55, 20, 19, 149, 159, 10],   # customer  4
    [15, 30, 26,  34,  44, 10],   # customer  5
    [25, 30,  3,  99, 109, 10],   # customer  6
    [20, 50,  5,  81,  91, 10],   # customer  7
    [10, 43,  9,  95, 105, 10],   # customer  8
    [55, 60, 16,  97, 107, 10],   # customer  9
    [30, 60, 16, 124, 134, 10],   # customer 10
    [20, 65, 12,  67,  77, 10],   # customer 11
    [50, 35, 19,  63,  73, 10],   # customer 12
    [30, 25, 23, 159, 169, 10],   # customer 13
    [15, 10, 20,  32,  42, 10],   # customer 14
    [30,  5,  8,  61,  71, 10],   # customer 15
    [10, 20, 19,  75,  85, 10],   # customer 16
    [ 5, 30,  2, 157, 167, 10],   # customer 17
    [20, 40, 12,  87,  97, 10],   # customer 18
    [15, 60, 17,  76,  86, 10],   # customer 19
    [45, 65,  9, 126, 136, 10],   # customer 20
    [45, 20, 11,  62,  72, 10],   # customer 21
    [45, 10, 18,  97, 107, 10],   # customer 22
    [55,  5, 29,  68,  78, 10],   # customer 23
    [65, 35,  3, 153, 163, 10],   # customer 24
    [65, 20,  6, 172, 182, 10],   # customer 25
]

# Full 100-customer R101 instance for reference / larger experiments
_R101_CUSTOMERS_100 = [
    [41, 49, 10, 161, 171, 10],
    [35, 17,  7,  50,  60, 10],
    [55, 45, 13, 116, 126, 10],
    [55, 20, 19, 149, 159, 10],
    [15, 30, 26,  34,  44, 10],
    [25, 30,  3,  99, 109, 10],
    [20, 50,  5,  81,  91, 10],
    [10, 43,  9,  95, 105, 10],
    [55, 60, 16,  97, 107, 10],
    [30, 60, 16, 124, 134, 10],
    [20, 65, 12,  67,  77, 10],
    [50, 35, 19,  63,  73, 10],
    [30, 25, 23, 159, 169, 10],
    [15, 10, 20,  32,  42, 10],
    [30,  5,  8,  61,  71, 10],
    [10, 20, 19,  75,  85, 10],
    [ 5, 30,  2, 157, 167, 10],
    [20, 40, 12,  87,  97, 10],
    [15, 60, 17,  76,  86, 10],
    [45, 65,  9, 126, 136, 10],
    [45, 20, 11,  62,  72, 10],
    [45, 10, 18,  97, 107, 10],
    [55,  5, 29,  68,  78, 10],
    [65, 35,  3, 153, 163, 10],
    [65, 20,  6, 172, 182, 10],
    [45, 30, 17, 132, 142, 10],
    [35, 40, 16,  37,  47, 10],
    [41, 37, 16,  39,  49, 10],
    [64, 42,  9,  63,  73, 10],
    [40, 60, 21,  71,  81, 10],
    [31, 52, 27,  50,  60, 10],
    [35, 69, 23, 141, 151, 10],
    [53, 52, 11,  37,  47, 10],
    [65, 55, 14, 117, 127, 10],
    [63, 65,  8, 143, 153, 10],
    [ 2, 60,  5,  41,  51, 10],
    [20, 20,  8, 134, 144, 10],
    [ 5,  5, 16,  83,  93, 10],
    [60, 12, 31,  44,  54, 10],
    [40, 25,  9,  85,  95, 10],
    [42,  7,  5,  97, 107, 10],
    [24, 12,  5,  31,  41, 10],
    [23,  3,  7, 132, 142, 10],
    [11, 14, 18,  69,  79, 10],
    [ 6, 38, 16,  32,  42, 10],
    [ 2, 48,  1, 117, 127, 10],
    [ 8, 56, 27,  51,  61, 10],
    [13, 52, 36, 165, 175, 10],
    [ 6, 68, 30, 108, 118, 10],
    [47, 47, 13, 124, 134, 10],
    [49, 58, 10,  88,  98, 10],
    [27, 43,  9,  52,  62, 10],
    [37, 31, 14,  95, 105, 10],
    [57, 29, 18, 140, 150, 10],
    [63, 23,  2, 136, 146, 10],
    [53, 12,  6, 130, 140, 10],
    [32, 12,  7, 101, 111, 10],
    [36, 26, 18, 200, 210, 10],
    [21, 24, 28,  18,  28, 10],
    [17, 34,  3, 162, 172, 10],
    [12, 24, 13,  76,  86, 10],
    [24, 58, 19,  58,  68, 10],
    [27, 69, 10,  34,  44, 10],
    [15, 77,  9,  73,  83, 10],
    [62, 77, 20,  51,  61, 10],
    [49, 73, 25, 127, 137, 10],
    [67,  5, 25,  83,  93, 10],
    [56, 39, 36, 142, 152, 10],
    [37, 47,  6,  50,  60, 10],
    [37, 56,  5, 182, 192, 10],
    [57, 68, 15,  77,  87, 10],
    [47, 16, 25,  35,  45, 10],
    [44, 17,  9,  78,  88, 10],
    [46, 13,  8, 149, 159, 10],
    [49, 11, 18,  69,  79, 10],
    [49, 42, 13,  73,  83, 10],
    [53, 43, 14, 179, 189, 10],
    [61, 52,  3,  96, 106, 10],
    [57, 48, 23,  92, 102, 10],
    [56, 37,  6, 182, 192, 10],
    [55, 54, 26,  94, 104, 10],
    [15, 47, 16,  55,  65, 10],
    [14, 37, 11,  44,  54, 10],
    [11, 31,  7, 101, 111, 10],
    [16, 22, 41,  91, 101, 10],
    [ 4, 18, 35,  94, 104, 10],
    [28, 18, 26,  93, 103, 10],
    [26, 52,  9,  74,  84, 10],
    [26, 35, 15, 176, 186, 10],
    [31, 67,  3,  95, 105, 10],
    [15, 19,  1, 160, 170, 10],
    [22, 22,  2,  18,  28, 10],
    [18, 24, 22, 188, 198, 10],
    [26, 27, 27, 100, 110, 10],
    [25, 24, 20,  39,  49, 10],
    [22, 27, 11, 135, 145, 10],
    [25, 21, 12, 133, 143, 10],
    [19, 21, 10,  58,  68, 10],
    [20, 26,  9,  83,  93, 10],
    [18, 18, 17, 185, 195, 10],
]

_R101_25 = {
    "name": "R101_25",
    "n_customers": 25,
    "capacity": 200,
    "depot": _R101_DEPOT,
    "customers": _R101_CUSTOMERS_25,
    "best_known": {
        "note": "Subset of full 100-customer instance; no standard BKS for 25-customer truncation."
    },
}

_R101_100 = {
    "name": "R101_100",
    "n_customers": 100,
    "capacity": 200,
    "depot": _R101_DEPOT,
    "customers": _R101_CUSTOMERS_100,
    "best_known": {
        "vehicles": 19,
        "distance": 1650.80,
        "reference": "Homberger, 2000",
    },
}

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

_INSTANCES: Dict[str, Dict[str, Any]] = {
    "r101_25": _R101_25,
    "r101": _R101_25,       # default alias
    "r101_100": _R101_100,
}


def get_instance(name: str) -> Dict[str, Any]:
    """Return a benchmark VRPTW instance by name.

    Parameters
    ----------
    name : str
        Instance identifier. One of: ``"r101"`` / ``"r101_25"`` (25 customers),
        ``"r101_100"`` (full 100 customers).

    Returns
    -------
    dict
        Keys: ``name``, ``n_customers``, ``capacity``, ``depot``,
        ``customers``, ``best_known``.

    Raises
    ------
    KeyError
        If *name* is not a known instance.
    """
    key = name.lower()
    if key not in _INSTANCES:
        available = ", ".join(sorted(set(_INSTANCES.keys()) - {"r101"}))
        raise KeyError(
            f"Unknown VRPTW instance '{name}'. Available: {available}"
        )
    # Return a shallow copy so callers cannot mutate the canonical data
    return _INSTANCES[key].copy()


def list_instances():
    """Return a list of available instance names (canonical, no aliases)."""
    return sorted(set(_INSTANCES.keys()) - {"r101"})
